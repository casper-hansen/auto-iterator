"""Per-run git worktrees: isolation + apply/revert against the source workspace.

The runner mounts every backend inside ``<run_dir>/worktree/`` on a fresh
``auto-iterator/<run_id>`` branch so the agent can never trample the
source workspace's working tree. Two carry-overs make the worktree feel
like the user's normal cwd:

* ``.env*`` files at the workspace root are *copied* (so secrets are
  available; copies because ``git apply`` shouldn't ship them back).
* Other top-level gitignored entries are *symlinked* (so caches, virtual
  envs, data dirs, build outputs are shared with the source workspace).

Apply / revert is patch-based:

* :func:`make_full_patch` produces a binary diff of every change in the
  worktree (committed + staged + unstaged + untracked) versus the
  ``base_commit`` it was branched from. Untracked files are picked up by
  staging into a temporary index so the worktree's own index is never
  mutated.
* :func:`apply_to_source` applies that patch to the source workspace and
  records it in ``applied.json``; :func:`revert_from_source` re-applies
  the recorded patch in reverse. Both can be called repeatedly as long as
  the source workspace's state doesn't conflict with the recorded patch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .run_dir import RunPaths, atomic_write_json, now_iso, read_json


WORKTREE_DIR_NAME = "worktree"
WORKTREE_META_FILE = "worktree.json"
APPLIED_STATE_FILE = "applied.json"
WORKTREE_BRANCH_PREFIX = "auto-iterator/"


# ── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class WorktreeInfo:
    """What the runner records so later CLI calls can find / use the worktree."""

    path: str            # absolute path to the worktree
    branch: str          # branch name created for the worktree
    base_commit: str     # commit the worktree branched from
    base_branch: str     # branch checked out in source at create time
    source_workspace: str  # git toplevel (where apply/revert target lives)
    created_at: str
    # Original ``cfg.workspace`` value — may be a subdirectory of
    # ``source_workspace`` (the git toplevel). The runner's CWD inside
    # the worktree mirrors this relationship: agents run from
    # ``<worktree>/<requested_subdir>`` so a user pointing ``ai run`` at
    # ``repo/app`` still feels like they're in ``app``, while apply /
    # revert can still cover edits anywhere in the repo.
    requested_workspace: str = ""
    # Paths (relative to the worktree root) we symlinked from the source
    # workspace at create time. Tracked so :func:`make_full_patch` can
    # exclude them — git's gitignore rules treat the symlink itself as
    # untracked, and there's no syntax that masks a symlink-to-directory
    # the way it masks a plain directory. Storing the explicit list is
    # the only reliable way to keep these out of the diff.
    carried_links: list[str] = None
    # Paths (relative to the worktree root) of ``.env*`` files we copied
    # from the source workspace. We exclude them from the diff so secrets
    # don't leak into ``ai diff`` / ``ai apply`` output. Copies are
    # intentional (so an agent reading a value can't write it back), but
    # they should never be part of the patch.
    carried_env: list[str] = None

    def __post_init__(self) -> None:
        if self.carried_links is None:
            self.carried_links = []
        if self.carried_env is None:
            self.carried_env = []
        if not self.requested_workspace:
            self.requested_workspace = self.path

    @property
    def excluded_paths(self) -> list[str]:
        """Paths to exclude from diffs — all carry-over (env + symlinks)."""
        return [*self.carried_env, *self.carried_links]

    @property
    def agent_cwd(self) -> str:
        """Where the agent should ``cd`` to before running.

        Mirrors ``requested_workspace`` relative to the source toplevel,
        so a ``--workspace repo/app`` invocation lands the agent in
        ``<worktree>/app``. Falls back to the worktree root when the
        requested workspace was the toplevel itself or sits outside
        ``source_workspace`` (shouldn't happen, but be defensive)."""
        try:
            rel = Path(self.requested_workspace).resolve().relative_to(
                Path(self.source_workspace).resolve()
            )
        except (ValueError, OSError):
            return self.path
        cwd = Path(self.path) / rel
        return str(cwd)

    @classmethod
    def from_dict(cls, d: dict) -> "WorktreeInfo":
        return cls(
            path=d["path"],
            branch=d["branch"],
            base_commit=d["base_commit"],
            base_branch=d.get("base_branch", ""),
            source_workspace=d["source_workspace"],
            created_at=d.get("created_at", ""),
            requested_workspace=d.get("requested_workspace", ""),
            carried_links=list(d.get("carried_links") or []),
            carried_env=list(d.get("carried_env") or []),
        )


# ── Git helpers ──────────────────────────────────────────────────────────────


def _git(
    *args: str,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    input_text: Optional[str] = None,
) -> tuple[int, str, str]:
    """Run a git subcommand. Returns ``(returncode, stdout, stderr)``."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _git_check(*args: str, cwd: Optional[Path] = None,
               env: Optional[dict] = None) -> str:
    rc, out, err = _git(*args, cwd=cwd, env=env)
    if rc != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (rc={rc}): {err.strip() or out.strip()}"
        )
    return out


def is_git_repo(path: Path) -> bool:
    rc, _, _ = _git("rev-parse", "--git-dir", cwd=path)
    return rc == 0


def git_toplevel(path: Path) -> Optional[Path]:
    """Resolve ``path`` to its enclosing git working tree root.

    Returns ``None`` if ``path`` is not inside a git repo. Used by the
    runner so that pointing ``ai run`` at a subdirectory still produces
    a worktree that captures changes to *every* file in the repo, and so
    that the repo-root ``.env`` is carried over (not just one in the
    subdir, which usually doesn't exist)."""
    rc, out, _ = _git("rev-parse", "--show-toplevel", cwd=path)
    if rc != 0:
        return None
    line = out.strip()
    return Path(line) if line else None


def _current_branch(path: Path) -> str:
    """Return ``HEAD``'s branch name, or empty string if detached."""
    rc, out, _ = _git("symbolic-ref", "--quiet", "--short", "HEAD", cwd=path)
    return out.strip() if rc == 0 else ""


# ── Carry-over from source workspace ─────────────────────────────────────────


def _is_env_name(name: str) -> bool:
    return name == ".env" or name.startswith(".env.")


def _is_inside_or_equal(path: str, ancestors: list[str]) -> bool:
    """True if ``path`` equals or sits under any of ``ancestors``.

    Both ``path`` and entries of ``ancestors`` are forward-slash relative
    paths — typically computed via :func:`os.path.relpath` against the
    source workspace root. We compare both ``==`` and ``startswith
    ancestor + "/"`` so ``cache`` matches both ``cache`` itself and
    ``cache/x/y``, but never ``cache_v2``.
    """
    for a in ancestors:
        a_norm = a.rstrip("/")
        if path == a_norm:
            return True
        if path.startswith(a_norm + "/"):
            return True
    return False


def _list_tracked_env_files(src: Path) -> set[str]:
    """Return the set of paths (relative to ``src``) tracked in git that
    look like env files.

    ``git ls-files`` reports every committed/staged path; we keep only
    those whose basename matches ``.env`` / ``.env.*``. The returned set
    drives the "skip tracked" branch in :func:`_bring_over_env_files`:
    tracked files are already in HEAD and therefore in the worktree
    checkout, so we must never re-copy them — copying would put their
    paths into ``carried_env``, and :func:`make_full_patch` would then
    ``git rm --cached`` the same paths from its temp index, turning a
    benign ``.env.example`` into a deletion in the apply patch.
    """
    rc, out, _ = _git("ls-files", "-z", cwd=src)
    if rc != 0:
        return set()
    tracked: set[str] = set()
    for p in out.split("\0"):
        if not p:
            continue
        if _is_env_name(Path(p).name):
            tracked.add(p)
    return tracked


def _bring_over_env_files(
    src: Path,
    dst: Path,
    *,
    skip_under: Optional[list[str]] = None,
) -> list[str]:
    """Copy untracked ``.env`` / ``.env.*`` files from src into dst.

    Walks the source workspace looking for ``.env*`` files. Three
    classes of file are deliberately skipped:

    * **Tracked** files (e.g. ``.env.example`` committed to the repo) —
      they're already in HEAD, so the worktree checkout already contains
      them. Re-copying would land their paths in ``carried_env``, and
      :func:`make_full_patch` would then ``git rm --cached`` them,
      causing ``ai apply`` to delete them from the source workspace.
    * Files inside any path in ``skip_under`` — those directories are
      about to be symlinked by :func:`_bring_over_ignored_entries`, so
      the env files they contain will already be reachable through the
      symlink. Copying them first would create a real directory and
      block the symlink (since the symlink helper refuses to overwrite
      existing destinations).
    * Anything inside ``.git`` or the worktree itself.

    Symlinks pointing outside the source workspace (e.g. ``.env ->
    ../shared.env`` for shared monorepo secrets) are handled correctly
    in two places: we compute the destination via :func:`os.path.relpath`
    (which does **not** follow symlinks) so the symlink is recorded at
    the same relative location the user has it at; and we copy with
    :func:`shutil.copy2` (which **does** follow symlinks by default), so
    the resulting worktree file contains the secret's actual contents.
    The previous implementation used ``Path.resolve().relative_to(src)``,
    which raises ``ValueError`` for symlinks targeting paths outside the
    repo and silently dropped them — leaving the worktree with no
    ``.env`` and the agent without API keys.

    Returns the list of paths (relative to ``dst``) that were copied so
    :func:`make_full_patch` can exclude them from the diff.
    """
    skip_under = list(skip_under or [])
    copied: list[str] = []
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    tracked_env = _list_tracked_env_files(src)

    for root, dirs, files in os.walk(src_resolved):
        # Compute root's path relative to src so we can prune subtrees.
        try:
            rel_root = os.path.relpath(root, src_resolved)
        except ValueError:
            rel_root = ""
        # Don't descend into .git or the worktree we just created
        # (worktree may live inside src — unusual but legal). Also
        # prune carry-over symlink targets so we don't even consider
        # env files that will already be reachable via the symlink.
        pruned: list[str] = []
        for d in dirs:
            if d == ".git":
                continue
            if (Path(root) / d).resolve() == dst_resolved:
                continue
            child_rel = d if rel_root in ("", ".") else f"{rel_root}/{d}"
            child_rel = child_rel.replace(os.sep, "/")
            if _is_inside_or_equal(child_rel, skip_under):
                continue
            pruned.append(d)
        dirs[:] = pruned

        for fname in files:
            if not _is_env_name(fname):
                continue
            src_file = Path(root) / fname
            # Use os.path.relpath, NOT Path.resolve().relative_to(src) —
            # the latter follows symlinks and fails for any .env that
            # points outside the repo (the common shared-secrets case).
            try:
                rel = os.path.relpath(src_file, src_resolved).replace(os.sep, "/")
            except ValueError:
                continue
            if rel.startswith(".."):
                continue
            if rel in tracked_env:
                continue
            if _is_inside_or_equal(rel, skip_under):
                continue
            target = dst / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                # follow_symlinks=True (default) → copies the actual
                # contents of the symlink target, even when that target
                # lives outside the source workspace.
                shutil.copy2(src_file, target)
                copied.append(rel)
            except OSError:
                pass
    return copied


def _ignored_paths(src: Path) -> list[str]:
    """Return git-ignored entries as paths relative to ``src``.

    Uses ``git ls-files --others --ignored --exclude-standard --directory``
    which collapses fully-ignored directories into a single entry. Unlike
    the previous implementation, we keep the *full* path (e.g.
    ``frontend/node_modules`` rather than just ``frontend``) so nested
    ignore patterns are honored. Filters out ``.git`` and ``.env*``
    so they don't compete with the dedicated env-copy path."""
    rc, out, _ = _git(
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--directory",
        cwd=src,
    )
    if rc != 0:
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for raw in out.splitlines():
        line = raw.strip().rstrip("/")
        if not line:
            continue
        if line in seen:
            continue
        # Skip .git and .env files — handled separately or never wanted.
        first = Path(line).parts[0]
        if first == ".git":
            continue
        if _is_env_name(Path(line).name):
            continue
        seen.add(line)
        paths.append(line)
    return paths


def _bring_over_ignored_entries(
    src: Path,
    dst: Path,
    *,
    paths: Optional[list[str]] = None,
) -> list[str]:
    """Symlink gitignored entries from src into dst, preserving nesting.

    For each ignored path (at any depth), creates parent directories in
    the destination as needed and symlinks the entry itself. So
    ``frontend/node_modules`` becomes ``<dst>/frontend/node_modules``
    pointing back at ``<src>/frontend/node_modules``, with the
    ``frontend`` directory created as a real directory in the worktree
    (the tracked files inside it stay tracked from the worktree's
    HEAD).

    ``paths`` lets the caller pass a precomputed list of ignored paths
    so :func:`create_worktree` can share the result with
    :func:`_bring_over_env_files` (which needs the same list to know
    which subtrees to skip during env walking). Falls back to
    :func:`_ignored_paths` when not supplied.

    Carry-over symlinks pointing at directories outside the worktree
    can't be reliably masked via ``info/exclude`` — git's gitignore
    matching for symlinks is inconsistent across versions. Instead we
    return the list of paths so :func:`make_full_patch` can drop them
    from the diff via ``git rm --cached`` against its temp index.

    Returns the list of paths (relative to ``dst``) that were symlinked,
    for logging and exclusion."""
    if paths is None:
        paths = _ignored_paths(src)
    linked: list[str] = []
    for rel in paths:
        src_entry = src / rel
        if not src_entry.exists() and not src_entry.is_symlink():
            continue
        dst_entry = dst / rel
        if dst_entry.exists() or dst_entry.is_symlink():
            continue
        try:
            dst_entry.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(
                src_entry,
                dst_entry,
                target_is_directory=src_entry.is_dir(),
            )
            linked.append(rel)
        except OSError:
            pass
    return linked


# ── Create / remove ──────────────────────────────────────────────────────────


def create_worktree(
    *,
    source_workspace: Path,
    target_path: Path,
    branch_name: str,
    requested_workspace: Optional[Path] = None,
) -> WorktreeInfo:
    """Create a worktree at ``target_path`` branched from source's HEAD.

    ``source_workspace`` should be the git toplevel — :func:`git_toplevel`
    is the standard way to derive it from a user-supplied workspace path.
    The runner resolves to the toplevel before calling so that edits
    anywhere in the repo are captured by ``ai apply`` / ``ai diff``,
    even when ``ai run --workspace`` pointed at a subdirectory.

    ``requested_workspace`` is the (possibly subdirectory) path the user
    actually asked for. The agent's CWD inside the worktree is rooted at
    the equivalent subdir, so ``ai run --workspace repo/app`` still
    feels like running in ``app``. Defaults to ``source_workspace``.

    The new branch is named ``branch_name`` (caller supplies a unique
    identifier like ``auto-iterator/<run_id>``). After ``git worktree
    add`` succeeds, ``.env*`` are copied and gitignored entries are
    symlinked so the agent inherits the user's local environment.

    Raises ``RuntimeError`` if ``source_workspace`` is not a git repo or
    the worktree already exists. Callers wanting "best effort" semantics
    should catch that explicitly."""
    src = source_workspace.resolve()
    target = target_path.resolve()
    req = (requested_workspace or src).resolve()

    if not is_git_repo(src):
        raise RuntimeError(
            f"source workspace '{src}' is not a git repository — "
            "worktrees require git. Pass --no-worktree to disable."
        )
    if target.exists():
        raise RuntimeError(f"worktree path already exists: {target}")

    base_commit = _git_check("rev-parse", "HEAD", cwd=src).strip()
    base_branch = _current_branch(src)

    target.parent.mkdir(parents=True, exist_ok=True)
    _git_check(
        "worktree", "add", "-b", branch_name, str(target), base_commit,
        cwd=src,
    )

    # Compute ignored paths once and share between the two helpers.
    # Order matters: env carry-over must skip files inside paths we're
    # about to symlink (otherwise it creates a real directory at the
    # symlink target and the symlink helper refuses to overwrite it,
    # leaving the worktree with a half-populated copy of the ignored
    # subtree instead of a live link to the source workspace).
    ignored_paths = _ignored_paths(src)
    carried_env = _bring_over_env_files(src, target, skip_under=ignored_paths)
    carried_links = _bring_over_ignored_entries(src, target, paths=ignored_paths)

    return WorktreeInfo(
        path=str(target),
        branch=branch_name,
        base_commit=base_commit,
        base_branch=base_branch,
        source_workspace=str(src),
        created_at=now_iso(),
        requested_workspace=str(req),
        carried_links=carried_links,
        carried_env=carried_env,
    )


def remove_worktree(info: WorktreeInfo, *, force: bool = True) -> tuple[bool, str]:
    """Remove the git worktree and delete its branch.

    Returns ``(ok, message)`` so callers can decide whether to drop
    metadata. ``force`` survives a worktree with uncommitted changes —
    the expected case here, since the runner deliberately leaves work
    in-progress in the worktree for review.
    """
    src = Path(info.source_workspace)
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(info.path)
    rc, out, err = _git(*args, cwd=src)
    messages = [m for m in ((out + err).strip(),) if m]
    if rc != 0:
        # Try a manual cleanup: prune + rmtree. ``git worktree prune``
        # requires the directory to actually be gone, so we remove first.
        try:
            shutil.rmtree(info.path, ignore_errors=True)
        except OSError:
            pass
        prune_rc, prune_out, prune_err = _git("worktree", "prune", cwd=src)
        branch_rc, branch_out, branch_err = _git("branch", "-D", info.branch, cwd=src)
        messages.extend(
            m for m in (
                (prune_out + prune_err).strip(),
                (branch_out + branch_err).strip(),
            ) if m
        )
        if Path(info.path).exists():
            return False, "\n".join(messages) or "worktree directory still exists"
        if prune_rc != 0:
            return False, "\n".join(messages) or "git worktree prune failed"
        if branch_rc != 0:
            return False, "\n".join(messages) or "git branch cleanup failed"
        return True, "\n".join(messages) or "removed with manual fallback"
    # Drop the branch (best-effort; -D ignores merge state).
    branch_rc, branch_out, branch_err = _git("branch", "-D", info.branch, cwd=src)
    messages.extend(m for m in ((branch_out + branch_err).strip(),) if m)
    if branch_rc != 0:
        return False, "\n".join(messages) or "git branch cleanup failed"
    return True, "\n".join(messages)


# ── Worktree metadata persistence ────────────────────────────────────────────


def worktree_meta_path(paths: RunPaths) -> Path:
    return paths.run_dir / WORKTREE_META_FILE


def applied_state_path(paths: RunPaths) -> Path:
    return paths.run_dir / APPLIED_STATE_FILE


def save_worktree_info(paths: RunPaths, info: WorktreeInfo) -> None:
    atomic_write_json(worktree_meta_path(paths), asdict(info))


def load_worktree_info(paths: RunPaths) -> Optional[WorktreeInfo]:
    p = worktree_meta_path(paths)
    if not p.exists():
        return None
    try:
        return WorktreeInfo.from_dict(read_json(p))
    except (OSError, ValueError, KeyError):
        return None


# ── Diff & status ────────────────────────────────────────────────────────────


def make_full_patch(info: WorktreeInfo) -> str:
    """Return a unified diff of every change in the worktree vs base.

    Builds the patch via a *temporary* index so the worktree's actual
    index is untouched (the agent might still be holding staged changes).
    Steps:
      1. Snapshot ``base_commit``'s tree into a tmp index.
      2. ``git add -A`` against that tmp index — stages the current
         working tree, including untracked files (gitignored entries are
         skipped automatically).
      3. Drop carried-over symlinks AND copied ``.env*`` files from the
         tmp index. Symlinks would show up as new tracked symlinks
         pointing outside the source workspace; ``.env*`` files would
         leak secrets into the diff (we copy them in so the agent can
         read API keys, but the source workspace already has its own
         copy and we don't want them in apply patches either).
      4. ``git diff --cached --binary base_commit`` emits a patch
         covering committed + staged + unstaged + untracked changes in
         one shot, with binary file support."""
    wt = Path(info.path)
    with tempfile.NamedTemporaryFile(prefix="ai-idx-", delete=False) as tf:
        tmp_index = tf.name
    try:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = tmp_index
        _git_check("read-tree", info.base_commit, cwd=wt, env=env)
        _git_check("add", "-A", cwd=wt, env=env)
        # Drop carry-over symlinks (point outside src) and copied .env*
        # files (secrets — the source already has its own copy).
        for path in info.excluded_paths:
            _git("rm", "--cached", "--ignore-unmatch", "-r", "--", path,
                 cwd=wt, env=env)
        out = _git_check(
            "diff", "--cached", "--binary", info.base_commit,
            cwd=wt, env=env,
        )
        return out
    finally:
        try:
            os.unlink(tmp_index)
        except OSError:
            pass


def make_status_short(info: WorktreeInfo) -> str:
    """``git status --short``-style summary of the worktree.

    Mirrors what VS Code's source-control view shows: per-file change
    indicators (M/A/D/?) + path. The agent might leave untracked files
    around, so we run with ``--untracked=all`` to surface them too.
    Carry-over symlinks (caches, build dirs we mirrored from the source
    workspace) AND copied ``.env*`` files are filtered out so they don't
    dominate the report — they are intentional plumbing/secrets, not
    user-visible work."""
    wt = Path(info.path)
    rc, out, err = _git(
        "status", "--short", "--untracked-files=all",
        cwd=wt,
    )
    if rc != 0:
        return f"(git status failed: {err.strip()})"
    skip = set(info.excluded_paths)
    if not skip:
        return out
    kept: list[str] = []
    for line in out.splitlines():
        # ``git status --short`` lines look like ``XY path`` (XY = two
        # status chars, then a space). Strip and parse.
        if len(line) < 4:
            continue
        path = line[3:].split(" -> ", 1)[0]
        if path in skip:
            continue
        # Also skip anything *inside* a carried-link directory — git
        # never emits these because the symlink itself is untracked,
        # but defensive against future behavior changes.
        inside_excluded = any(
            path == ex or path.startswith(ex.rstrip("/") + "/")
            for ex in skip
        )
        if inside_excluded:
            continue
        kept.append(line)
    return ("\n".join(kept) + "\n") if kept else ""


def make_diff_stat(info: WorktreeInfo) -> str:
    """``git diff --stat`` against the base commit, for a high-level view."""
    patch_text = make_full_patch(info)
    if not patch_text.strip():
        return "(no changes)"
    # Pipe the patch through ``git apply --stat`` to summarise — it
    # accepts unified diffs without touching the worktree.
    proc = subprocess.run(
        ["git", "apply", "--stat"],
        input=patch_text,
        capture_output=True,
        text=True,
    )
    return proc.stdout or "(no changes)"


# ── Apply / revert ───────────────────────────────────────────────────────────


def _apply_patch(patch_text: str, target: Path, *, reverse: bool) -> tuple[bool, str]:
    """Run ``git apply`` (optionally ``--reverse``) against *target*.

    Tries ``--3way`` first so a patch that touches lines the source
    workspace has independently moved still applies cleanly via merge
    machinery (matches what ``git am --3way`` does). Falls back to a
    plain non-3way apply for patches that lack 3-way information (e.g.
    pure additions of new files where there's no base blob to look up).

    Each strategy pre-flights with ``--check`` so the actual apply is
    only attempted when we know it will succeed cleanly — leaving the
    source workspace untouched on failure is what makes ``ai apply`` /
    ``ai revert`` safely repeatable.
    """
    def _try(extra: list[str]) -> tuple[bool, str]:
        base = ["git", "apply", "--whitespace=nowarn", *extra]
        if reverse:
            base.append("--reverse")
        check = subprocess.run(
            [*base, "--check"], input=patch_text, cwd=str(target),
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            return False, (check.stderr or check.stdout or "").strip()
        proc = subprocess.run(
            base, input=patch_text, cwd=str(target),
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "").strip()
        return True, "applied"

    # 3-way uses the patch's index lines to look up the base blob and
    # do a real merge. Strictly more permissive than plain apply, so
    # try it first.
    ok, msg_3way = _try(["--3way"])
    if ok:
        return True, "applied"

    ok, msg_plain = _try([])
    if ok:
        return True, "applied"

    # Surface the more informative message — usually 3-way's, since it
    # actually attempted the merge — so the user knows why both failed.
    msg = msg_3way or msg_plain or "git apply failed"
    return False, msg


def _patch_check(patch_text: str, target: Path, *, reverse: bool) -> bool:
    """Return True if ``git apply --check`` accepts the patch."""
    if not patch_text:
        return False
    for extra in (["--3way"], []):
        base = ["git", "apply", "--whitespace=nowarn", *extra]
        if reverse:
            base.append("--reverse")
        proc = subprocess.run(
            [*base, "--check"], input=patch_text, cwd=str(target),
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            return True
    return False


def is_applied(paths: RunPaths) -> bool:
    p = applied_state_path(paths)
    if not p.exists():
        return False
    try:
        st = read_json(p)
    except (OSError, ValueError):
        return False
    if bool(st.get("applied")):
        return True
    if not st.get("apply_in_progress"):
        return False
    patch_text = st.get("patch", "")
    src_path = st.get("source_workspace")
    if not patch_text or not src_path:
        return False
    return _patch_check(patch_text, Path(src_path), reverse=True)


def apply_to_source(paths: RunPaths) -> tuple[bool, str]:
    """Apply the worktree's changes to the source workspace.

    Re-derives the patch from the worktree on every call so subsequent
    work on the worktree branch is captured. The patch is recorded in
    ``applied.json`` (along with the source workspace path) so
    :func:`revert_from_source` can reverse it verbatim later, even if
    the worktree is removed in the meantime."""
    info = load_worktree_info(paths)
    if info is None:
        return False, "no worktree recorded for this run"
    if is_applied(paths):
        return False, (
            "changes are already applied — run `ai revert` first if you "
            "want to refresh from the worktree"
        )

    patch_text = make_full_patch(info)
    if not patch_text.strip():
        return False, "worktree has no changes vs its base commit"

    state_path = applied_state_path(paths)
    apply_started_at = now_iso()
    state = {
        "applied": False,
        "apply_in_progress": True,
        "apply_started_at": apply_started_at,
        "base_commit": info.base_commit,
        "source_workspace": info.source_workspace,
        "patch": patch_text,
    }
    try:
        atomic_write_json(state_path, state)
    except OSError as exc:
        return False, f"could not record apply state before patching source: {exc}"

    ok, msg = _apply_patch(patch_text, Path(info.source_workspace), reverse=False)
    if not ok:
        try:
            atomic_write_json(state_path, {
                **state,
                "applied": False,
                "apply_failed_at": now_iso(),
                "apply_error": msg,
            })
        except OSError:
            pass
        return False, f"apply failed: {msg}"

    try:
        apply_finished_at = now_iso()
        atomic_write_json(state_path, {
            **state,
            "applied": True,
            "apply_in_progress": False,
            "applied_at": apply_finished_at,
            "apply_finished_at": apply_finished_at,
        })
    except OSError:
        # The write-ahead record already contains the patch and source.
        # ``is_applied`` / ``revert_from_source`` can infer the applied
        # state by checking whether the patch reverses cleanly.
        pass
    return True, "applied"


def revert_from_source(paths: RunPaths) -> tuple[bool, str]:
    """Reverse the previously-applied patch in the source workspace.

    Works even after ``ai worktree-remove`` — the patch and source path
    are recorded in ``applied.json`` so the worktree itself isn't
    needed. Falls back to ``WorktreeInfo.source_workspace`` when both
    are present (so older runs that pre-date the recorded path keep
    working)."""
    p = applied_state_path(paths)
    if not p.exists():
        return False, "nothing to revert (no applied.json)"
    try:
        st = read_json(p)
    except (OSError, ValueError) as exc:
        return False, f"could not read applied.json: {exc}"
    patch_text = st.get("patch", "")
    if not patch_text:
        return False, "applied.json missing patch text"

    src_path = st.get("source_workspace")
    if not src_path:
        info = load_worktree_info(paths)
        if info is None:
            return False, (
                "no source workspace recorded; run was created before this "
                "field was added and the worktree has been removed"
            )
        src_path = info.source_workspace

    if not st.get("applied"):
        if not st.get("apply_in_progress"):
            return False, "nothing to revert (last action was a revert)"
        if not _patch_check(patch_text, Path(src_path), reverse=True):
            return False, "nothing to revert (apply did not reach source workspace)"

    ok, msg = _apply_patch(patch_text, Path(src_path), reverse=True)
    if not ok:
        return False, f"revert failed: {msg}"

    atomic_write_json(p, {
        **st,
        "applied": False,
        "apply_in_progress": False,
        "reverted_at": now_iso(),
    })
    return True, "reverted"
