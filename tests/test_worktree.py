"""Integration tests for ``auto_iterator.worktree``.

Each test bootstraps a real tmpdir-backed git repo and drives the
worktree helpers end-to-end against it. The asserts target operator
behavior (the patch is correct, the source workspace ends up modified
the way the user expected, the diff doesn't leak secrets) rather than
internal implementation details, so refactors that keep the surface
intact won't break the suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.cli import EXIT_IO_ERROR, cmd_worktree_remove  # noqa: E402
from auto_iterator.meta import update_meta  # noqa: E402
from auto_iterator.run_dir import RunPaths, create_run_dir, new_run_id  # noqa: E402
from auto_iterator.worktree import (  # noqa: E402
    apply_to_source,
    applied_state_path,
    atomic_write_json,
    create_worktree,
    git_toplevel,
    is_applied,
    is_git_repo,
    load_worktree_info,
    make_diff_stat,
    make_full_patch,
    make_status_short,
    remove_worktree,
    revert_from_source,
    save_worktree_info,
    worktree_meta_path,
)


def _git(*args: str, cwd: Path) -> str:
    """Run git, return stdout, raise on failure (with stderr in the message)."""
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd),
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed in {cwd}: {proc.stderr}\n{proc.stdout}"
        )
    return proc.stdout


def _init_repo(root: Path, *, with_env: bool = False,
               gitignore: str = "") -> None:
    """Create a tiny repo at ``root`` with a baseline commit."""
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "t@t", cwd=root)
    _git("config", "user.name", "Tester", cwd=root)
    if gitignore:
        (root / ".gitignore").write_text(gitignore, encoding="utf-8")
    (root / "tracked.txt").write_text("hello\n", encoding="utf-8")
    if with_env:
        (root / ".env").write_text("API_KEY=topsecret\n", encoding="utf-8")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "init", cwd=root)


def _fake_run_paths(runs_dir: Path) -> RunPaths:
    """Build a RunPaths under ``runs_dir`` for the test."""
    return create_run_dir(runs_dir, new_run_id())


# ── Smoke / basics ───────────────────────────────────────────────────────────


def test_is_git_repo_and_toplevel_resolution() -> None:
    """``is_git_repo`` + ``git_toplevel`` find the repo root from a subdir."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        _init_repo(repo)
        sub = repo / "app"
        sub.mkdir()
        (sub / "x.txt").write_text("x\n", encoding="utf-8")
        _git("add", "-A", cwd=repo)
        _git("commit", "-q", "-m", "add app", cwd=repo)

        assert is_git_repo(sub)
        top = git_toplevel(sub)
        assert top is not None
        assert top.resolve() == repo.resolve()
        assert git_toplevel(Path(tmp)) is None


def test_create_worktree_basic() -> None:
    """``create_worktree`` produces a checked-out worktree on a new branch."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        wt = tmp_p / "wt"

        info = create_worktree(
            source_workspace=repo,
            target_path=wt,
            branch_name="auto-iterator/test1",
        )
        assert Path(info.path).exists()
        assert (Path(info.path) / "tracked.txt").exists()
        assert info.branch == "auto-iterator/test1"
        assert info.source_workspace == str(repo.resolve())

        # cleanup
        ok, _ = remove_worktree(info, force=True)
        assert ok or not Path(info.path).exists()


# ── Carry-over: .env + ignored directories ──────────────────────────────────


def test_env_file_is_copied_into_worktree() -> None:
    """``.env`` at the repo root is copied (not symlinked) into the worktree."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo, with_env=True, gitignore=".env\n")
        wt = tmp_p / "wt"

        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name="auto-iterator/test2",
        )
        env_in_wt = Path(info.path) / ".env"
        assert env_in_wt.exists()
        # Real file, not a symlink — secrets stay isolated.
        assert not env_in_wt.is_symlink()
        assert env_in_wt.read_text(encoding="utf-8") == "API_KEY=topsecret\n"
        assert ".env" in info.carried_env

        remove_worktree(info, force=True)


def test_env_file_is_excluded_from_full_patch() -> None:
    """``make_full_patch`` must NEVER contain ``.env`` contents.

    Regression for the secret-leak case: the agent has a real .env in
    the worktree (so it can read API keys), but ``ai diff`` / ``ai
    apply`` must not propagate it to the source workspace's patch
    output.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo, with_env=True, gitignore=".env\n")
        wt = tmp_p / "wt"
        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name="auto-iterator/test3",
        )
        # Make a tracked-file change so the patch isn't empty.
        (Path(info.path) / "tracked.txt").write_text("changed\n", encoding="utf-8")
        patch = make_full_patch(info)

        assert "topsecret" not in patch, (
            "secret leaked into apply patch: " + patch
        )
        assert "diff --git a/.env" not in patch
        assert "tracked.txt" in patch  # sanity: real change still emitted

        remove_worktree(info, force=True)


def test_nested_ignored_directory_is_symlinked() -> None:
    """``.gitignore`` patterns like ``frontend/node_modules/`` carry over."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo, gitignore="frontend/node_modules/\n")
        # Create the ignored nested directory in the source.
        (repo / "frontend").mkdir()
        (repo / "frontend" / "package.json").write_text("{}\n", encoding="utf-8")
        (repo / "frontend" / "node_modules").mkdir()
        (repo / "frontend" / "node_modules" / "x.js").write_text(
            "module.exports = 1\n", encoding="utf-8",
        )
        _git("add", "-A", cwd=repo)
        _git("commit", "-q", "-m", "add frontend", cwd=repo)

        wt = tmp_p / "wt"
        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name="auto-iterator/test4",
        )
        nm = Path(info.path) / "frontend" / "node_modules"
        assert nm.exists()
        assert nm.is_symlink()
        assert (nm / "x.js").read_text(encoding="utf-8") == "module.exports = 1\n"
        assert "frontend/node_modules" in info.carried_links

        # And critically, it should NOT leak into the patch.
        # Make a change so the patch isn't empty.
        (Path(info.path) / "tracked.txt").write_text("changed\n", encoding="utf-8")
        patch = make_full_patch(info)
        assert "node_modules" not in patch, (
            "nested ignored dir leaked into patch: " + patch
        )

        remove_worktree(info, force=True)


# ── Apply / revert lifecycle ─────────────────────────────────────────────────


def _apply_revert_cycle(repo: Path, wt: Path, runs_dir: Path,
                        edit: callable) -> RunPaths:
    """Build a worktree, apply ``edit`` inside it, and apply→revert→apply."""
    paths = _fake_run_paths(runs_dir)
    info = create_worktree(
        source_workspace=repo, target_path=wt,
        branch_name=f"auto-iterator/{paths.run_id}",
    )
    save_worktree_info(paths, info)
    edit(Path(info.path))
    return paths


def test_apply_and_revert_round_trip() -> None:
    """A worktree edit lands in the source via apply, vanishes via revert."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        wt = tmp_p / "wt"
        runs = tmp_p / "runs"
        runs.mkdir()

        def edit(wt_path: Path) -> None:
            (wt_path / "tracked.txt").write_text("modified\n", encoding="utf-8")

        paths = _apply_revert_cycle(repo, wt, runs, edit)

        ok, msg = apply_to_source(paths)
        assert ok, f"apply failed: {msg}"
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "modified\n"
        assert is_applied(paths)

        ok, msg = revert_from_source(paths)
        assert ok, f"revert failed: {msg}"
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "hello\n"
        assert not is_applied(paths)


def test_apply_state_write_failure_leaves_source_untouched() -> None:
    """If ``applied.json`` cannot be recorded, ``ai apply`` must abort
    before mutating the source workspace.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        wt = tmp_p / "wt"
        runs = tmp_p / "runs"
        runs.mkdir()

        def edit(wt_path: Path) -> None:
            (wt_path / "tracked.txt").write_text("modified\n", encoding="utf-8")

        paths = _apply_revert_cycle(repo, wt, runs, edit)

        with mock.patch(
            "auto_iterator.worktree.atomic_write_json",
            side_effect=OSError("disk full"),
        ):
            ok, msg = apply_to_source(paths)

        assert not ok
        assert "could not record apply state" in msg
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "hello\n"
        assert not is_applied(paths)


def test_apply_final_state_write_failure_still_revertible() -> None:
    """If the final applied-state restamp fails, the write-ahead record
    still lets ``is_applied`` and ``revert`` recover.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        wt = tmp_p / "wt"
        runs = tmp_p / "runs"
        runs.mkdir()

        def edit(wt_path: Path) -> None:
            (wt_path / "tracked.txt").write_text("modified\n", encoding="utf-8")

        paths = _apply_revert_cycle(repo, wt, runs, edit)
        writes = 0

        def flaky_write(path: Path, payload: dict) -> None:
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("late write failed")
            atomic_write_json(path, payload)

        with mock.patch(
            "auto_iterator.worktree.atomic_write_json",
            side_effect=flaky_write,
        ):
            ok, msg = apply_to_source(paths)

        assert ok, msg
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "modified\n"
        assert is_applied(paths)

        ok, msg = revert_from_source(paths)
        assert ok, msg
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "hello\n"
        assert not is_applied(paths)


def test_apply_3way_succeeds_when_plain_apply_would_fail() -> None:
    """A patch that touches lines the source has independently moved
    must still apply via the 3-way fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        # Replace the seed file with a multi-line file so the source can
        # diverge from the worktree.
        (repo / "f.txt").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        _git("add", "-A", cwd=repo)
        _git("commit", "-q", "-m", "seed", cwd=repo)

        wt = tmp_p / "wt"
        runs = tmp_p / "runs"
        runs.mkdir()

        paths = _fake_run_paths(runs)
        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name=f"auto-iterator/{paths.run_id}",
        )
        save_worktree_info(paths, info)

        # Worktree changes line 5 (e -> E).
        (Path(info.path) / "f.txt").write_text(
            "a\nb\nc\nd\nE\n", encoding="utf-8"
        )

        # Source has an unrelated change to line 1 (a -> AA), committed.
        (repo / "f.txt").write_text("AA\nb\nc\nd\ne\n", encoding="utf-8")
        _git("commit", "-q", "-am", "diverge", cwd=repo)

        ok, msg = apply_to_source(paths)
        assert ok, f"apply failed (3-way should have worked): {msg}"
        # Both edits should now be present in the source.
        text = (repo / "f.txt").read_text(encoding="utf-8")
        assert "AA" in text and "E" in text, text


def test_worktree_remove_failure_preserves_metadata() -> None:
    """The CLI must not forget a worktree it failed to remove."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        runs = tmp_p / "runs"
        paths = _fake_run_paths(runs)
        update_meta(
            paths,
            run_id=paths.run_id,
            status="running",
            workspace=str(repo),
        )
        info = create_worktree(
            source_workspace=repo,
            target_path=tmp_p / "wt",
            branch_name=f"auto-iterator/{paths.run_id}",
        )
        save_worktree_info(paths, info)

        args = Namespace(run_id=paths.run_id, force=True)
        with mock.patch(
            "auto_iterator.worktree.remove_worktree",
            return_value=(False, "boom"),
        ):
            rc = cmd_worktree_remove(args, runs)

        assert rc == EXIT_IO_ERROR
        assert worktree_meta_path(paths).exists()

        remove_worktree(info, force=True)


def test_revert_works_after_worktree_removed() -> None:
    """After ``ai apply`` + ``ai worktree-remove``, ``ai revert`` still
    undoes the source-workspace edits via the recorded patch."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        wt = tmp_p / "wt"
        runs = tmp_p / "runs"
        runs.mkdir()

        def edit(wt_path: Path) -> None:
            (wt_path / "tracked.txt").write_text("via worktree\n", encoding="utf-8")

        paths = _apply_revert_cycle(repo, wt, runs, edit)

        ok, _ = apply_to_source(paths)
        assert ok
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "via worktree\n"

        # Simulate ``ai worktree-remove`` (worktree gone, applied.json kept).
        info = load_worktree_info(paths)
        assert info is not None
        remove_worktree(info, force=True)
        # Drop only worktree.json; applied.json must survive.
        from auto_iterator.worktree import worktree_meta_path
        worktree_meta_path(paths).unlink(missing_ok=True)
        assert applied_state_path(paths).exists()

        ok, msg = revert_from_source(paths)
        assert ok, f"revert after worktree-remove failed: {msg}"
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "hello\n"


def test_repeated_apply_revert_cycles() -> None:
    """``apply -> revert -> apply -> revert`` should be idempotent."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        wt = tmp_p / "wt"
        runs = tmp_p / "runs"
        runs.mkdir()

        def edit(wt_path: Path) -> None:
            (wt_path / "tracked.txt").write_text("v2\n", encoding="utf-8")

        paths = _apply_revert_cycle(repo, wt, runs, edit)

        for _ in range(3):
            ok, msg = apply_to_source(paths)
            assert ok, f"apply: {msg}"
            assert (repo / "tracked.txt").read_text(encoding="utf-8") == "v2\n"
            ok, msg = revert_from_source(paths)
            assert ok, f"revert: {msg}"
            assert (repo / "tracked.txt").read_text(encoding="utf-8") == "hello\n"


# ── Subdirectory-workspace handling ──────────────────────────────────────────


def test_subdirectory_workspace_captures_repo_root_changes() -> None:
    """Pointing the runner at ``repo/app`` still captures changes to
    ``repo/root.txt`` and carries over the repo-root ``.env``."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo, with_env=True, gitignore=".env\n")
        # Add a subdir.
        app = repo / "app"
        app.mkdir()
        (app / "main.py").write_text("print('hi')\n", encoding="utf-8")
        (repo / "root.txt").write_text("root-v1\n", encoding="utf-8")
        _git("add", "-A", cwd=repo)
        _git("commit", "-q", "-m", "add app + root", cwd=repo)

        wt = tmp_p / "wt"
        runs = tmp_p / "runs"
        runs.mkdir()
        paths = _fake_run_paths(runs)

        # Resolve toplevel from the requested subdir, like the runner does.
        toplevel = git_toplevel(app)
        assert toplevel is not None and toplevel.resolve() == repo.resolve()

        info = create_worktree(
            source_workspace=toplevel,
            target_path=wt,
            branch_name=f"auto-iterator/{paths.run_id}",
            requested_workspace=app,
        )
        save_worktree_info(paths, info)

        # Worktree spans the whole repo.
        assert (Path(info.path) / "root.txt").exists()
        assert (Path(info.path) / "app" / "main.py").exists()
        # And the repo-root .env was carried over (not just an ``app/.env``).
        assert (Path(info.path) / ".env").read_text(encoding="utf-8") == \
            "API_KEY=topsecret\n"

        # Agent CWD honors the subdir request.
        agent_cwd = info.agent_cwd
        assert Path(agent_cwd).resolve() == (Path(info.path) / "app").resolve()

        # Edit the *repo-root* file from inside the worktree.
        (Path(info.path) / "root.txt").write_text("root-v2\n", encoding="utf-8")

        ok, msg = apply_to_source(paths)
        assert ok, f"apply failed: {msg}"
        # The root-level file in the source workspace must be updated.
        assert (repo / "root.txt").read_text(encoding="utf-8") == "root-v2\n"


# ── Status / diff filtering ──────────────────────────────────────────────────


def test_status_short_filters_carry_over_paths() -> None:
    """``ai diff``'s VS-Code-style summary doesn't show carried env/dirs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo, with_env=True, gitignore=".env\nbuild/\n")
        # Build dir is gitignored — create it AFTER repo init so it
        # exists in the source tree but stays untracked, which is what
        # the carry-over symlink path expects.
        (repo / "build").mkdir()
        (repo / "build" / "out.bin").write_bytes(b"binary")

        wt = tmp_p / "wt"
        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name="auto-iterator/test-status",
        )
        # Make a real change.
        (Path(info.path) / "tracked.txt").write_text("modified\n", encoding="utf-8")

        short = make_status_short(info)
        # The real change must appear; the carry-overs must NOT.
        assert "tracked.txt" in short
        assert ".env" not in short
        assert "build" not in short

        remove_worktree(info, force=True)


def test_diff_stat_excludes_env() -> None:
    """``make_diff_stat`` must not mention .env even when one was copied."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo, with_env=True, gitignore=".env\n")
        wt = tmp_p / "wt"
        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name="auto-iterator/test-diffstat",
        )
        (Path(info.path) / "tracked.txt").write_text("x\n", encoding="utf-8")

        stat = make_diff_stat(info)
        assert ".env" not in stat
        assert "tracked.txt" in stat

        remove_worktree(info, force=True)


# ── Regressions: tracked, symlinked, and ignored-dir env handling ───────────


def test_tracked_env_example_survives_apply() -> None:
    """A tracked ``.env.example`` must not be deleted by ``ai apply``.

    Regression: env carry-over used to copy every ``.env*`` file
    indiscriminately, including ones that were already tracked in HEAD.
    Their paths landed in ``carried_env``, then ``make_full_patch``
    ``git rm --cached``-ed them from its temp index, which made the
    patch describe a deletion. ``apply_to_source`` then dutifully
    deleted ``.env.example`` from the source workspace. This test pins
    the fix: tracked env files stay out of ``carried_env``, the patch
    does not delete them, and they survive a full apply.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo)
        # Track .env.example as a real committed file. (.env stays
        # untracked / gitignored — that is the file we DO want copied.)
        (repo / ".env.example").write_text("EXAMPLE=1\n", encoding="utf-8")
        _git("add", "-A", cwd=repo)
        _git("commit", "-q", "-m", "track example", cwd=repo)

        wt = tmp_p / "wt"
        runs = tmp_p / "runs"
        runs.mkdir()
        paths = _fake_run_paths(runs)

        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name=f"auto-iterator/{paths.run_id}",
        )
        save_worktree_info(paths, info)

        # carried_env must NOT contain a tracked file.
        assert ".env.example" not in info.carried_env, info.carried_env

        # Make a real change so the patch isn't empty.
        (Path(info.path) / "tracked.txt").write_text("changed\n", encoding="utf-8")
        patch = make_full_patch(info)
        assert "diff --git a/.env.example" not in patch, patch

        ok, msg = apply_to_source(paths)
        assert ok, f"apply failed: {msg}"
        # The tracked file must still exist with original contents.
        assert (repo / ".env.example").exists(), \
            "ai apply deleted tracked .env.example"
        assert (repo / ".env.example").read_text(encoding="utf-8") == \
            "EXAMPLE=1\n"


def test_symlinked_env_pointing_outside_repo_is_carried_over() -> None:
    """A ``.env`` symlink whose target lives outside the repo must still
    land in the worktree as a regular file with the target's contents.

    Regression: the previous implementation computed the destination
    via ``Path.resolve().relative_to(src)``, which raises ``ValueError``
    whenever the symlink resolves outside the source workspace. The
    file was silently dropped, ``carried_env`` came back empty, and the
    agent ran without API keys — directly violating the "always bring
    over .env" requirement. The fix uses :func:`os.path.relpath` (no
    symlink resolution) for the destination path and lets
    :func:`shutil.copy2` follow the symlink to copy the target's bytes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        # Shared secret outside the repo (common monorepo / dotfiles
        # pattern: one .env at $HOME, every project symlinks to it).
        shared = tmp_p / "shared.env"
        shared.write_text("API_KEY=shared-secret\n", encoding="utf-8")

        repo = tmp_p / "repo"
        _init_repo(repo, gitignore=".env\n")
        (repo / ".env").symlink_to(shared)

        wt = tmp_p / "wt"
        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name="auto-iterator/test-symlink-env",
        )
        env_in_wt = Path(info.path) / ".env"
        assert env_in_wt.exists(), "symlinked .env was not carried over"
        # Real file, not a symlink — secrets get isolated copies, never
        # back-references that could leak writes to the source path.
        assert not env_in_wt.is_symlink()
        assert env_in_wt.read_text(encoding="utf-8") == \
            "API_KEY=shared-secret\n"
        assert ".env" in info.carried_env

        remove_worktree(info, force=True)


def test_env_inside_ignored_dir_does_not_block_symlink() -> None:
    """An ignored directory containing a ``.env*`` file must still ship
    as a single symlink with the directory's full contents available.

    Regression: env carry-over walked the whole tree first and copied
    every ``.env*`` it found, including files inside gitignored
    directories. ``mkdir(parents=True)`` then created real intermediate
    directories in the worktree, so when :func:`_bring_over_ignored_entries`
    ran second it saw the path already existed and skipped the symlink.
    The result: the worktree had a partial copy of the ignored dir
    (only the ``.env`` files), every other file in the ignored dir was
    missing, and ``carried_links`` was empty. This violates the
    "bring over ignored directories" requirement and makes ignored
    caches drift from the source workspace.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        repo = tmp_p / "repo"
        _init_repo(repo, gitignore="cache/\n")
        (repo / "cache").mkdir()
        (repo / "cache" / ".env.local").write_text(
            "CACHE_SECRET=1\n", encoding="utf-8",
        )
        (repo / "cache" / "artifact.bin").write_bytes(b"binary-blob")

        wt = tmp_p / "wt"
        info = create_worktree(
            source_workspace=repo, target_path=wt,
            branch_name="auto-iterator/test-ignored-env",
        )
        cache = Path(info.path) / "cache"
        assert cache.is_symlink(), \
            "ignored cache/ dir should be a symlink, not a real directory"
        # Both files reachable through the symlink.
        assert (cache / ".env.local").read_text(encoding="utf-8") == \
            "CACHE_SECRET=1\n"
        assert (cache / "artifact.bin").read_bytes() == b"binary-blob"
        assert "cache" in info.carried_links
        # The .env.local inside should NOT be in carried_env — it came
        # along via the symlink; double-tracking it would put it in
        # excluded_paths and confuse make_full_patch's exclusion loop.
        assert "cache/.env.local" not in info.carried_env, info.carried_env

        remove_worktree(info, force=True)
