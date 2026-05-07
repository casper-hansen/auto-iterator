"""Unit tests for :func:`auto_iterator.display.view_log_in_less`.

The pager helper is the robust replacement for the previous
custom-bridge approach: instead of intercepting Esc / mouse / PageUp
escapes through a Python PTY and rewriting them on the way to ``less``,
we let native ``less`` own the terminal end-to-end. The helper is
deliberately small — its job is to build the right argv, decide
whether ``less`` and ``--mouse`` are actually available, and fall
back cleanly when they're not.

These tests pin:

* The argv contract: ``-R``, ``--mouse``, ``+G``, and the log path
  are all present; the brittle bridge-only flags (``+F``, ``-X``,
  lesskey files, ``LESSPROMPT`` overrides) are absent.
* The fallback contract: missing ``less`` or missing ``--mouse``
  surfaces a clear stderr warning and routes through the
  caller-supplied fallback rather than crashing or hanging.
* The exit-code contract: ``less``'s exit code is propagated so a
  caller can distinguish "operator quit cleanly" from "less
  crashed".
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.display import (  # noqa: E402
    _less_argv,
    _less_supports_mouse,
    view_log_in_less,
)


# ── argv construction ──────────────────────────────────────────────────────


def test_less_argv_includes_required_flags(tmp_path) -> None:
    """``_less_argv`` must include the three flags the redesign hinges on.

    * ``-R`` → preserve ANSI colour escapes.
    * ``--mouse`` → wheel/click navigation.
    * ``+G`` → start at the bottom in normal viewing mode.

    Plus the log path itself, so ``less`` reads from the file rather
    than waiting on stdin (the previous bridge approach piped through
    a PTY which was the source of the brittle escape-rewriting)."""
    log = tmp_path / "agent.log"
    log.write_text("hello\n", encoding="utf-8")

    argv = _less_argv("/usr/bin/less", log)

    assert argv[0] == "/usr/bin/less"
    assert "-R" in argv, "ANSI colour escapes must be preserved"
    assert "--mouse" in argv, "mouse wheel navigation is the UX contract"
    assert "+G" in argv, (
        "start at the bottom in normal mode (not +F follow); +F "
        "swallows the first scroll and is the source of the "
        "'mouse looks hung' regression"
    )
    assert str(log) in argv, "less must read from the log file path"


def test_less_argv_omits_brittle_bridge_only_flags(tmp_path) -> None:
    """The custom-bridge approach added flags that the native pager
    redesign deliberately drops:

    * ``+F`` — follow-mode-as-default has the wheel-swallowing UX
      bug. Operators who want follow press ``F`` once they're inside.
    * ``-X`` — letting ``less`` send its terminal init/deinit
      sequences cleans up the alternate screen + mouse tracking on
      exit; suppressing it forced "redraw on return" hacks.
    * lesskey files / ``LESSPROMPT`` overrides — those are operator
      config that should be honoured, not fought; the bridge
      approach broke when operators set ``LESSPROMPT`` because it
      parsed the status row.
    """
    log = tmp_path / "agent.log"
    log.write_text("hello\n", encoding="utf-8")

    argv = _less_argv("/usr/bin/less", log)

    assert "+F" not in argv, (
        "+F follow-mode-as-default is the source of the wheel-"
        "swallowing UX bug; native pager mode starts at the bottom "
        "and lets the operator press F if they want follow"
    )
    assert "-X" not in argv, (
        "-X suppresses less's terminal init/deinit; that's what the "
        "bridge approach used to keep the alt-screen up across the "
        "handoff, which is exactly the brittleness we're avoiding"
    )
    # Lesskey / prompt overrides — none of these bridge-only knobs
    # should appear in the argv.
    forbidden_substrings = ("--lesskey", "lesskey", "LESSPROMPT", "--prompt")
    joined = " ".join(argv)
    for needle in forbidden_substrings:
        assert needle not in joined, (
            f"{needle!r} is a bridge-only knob; native less should "
            "honour the operator's environment, not override it"
        )


# ── view_log_in_less: the dispatcher ───────────────────────────────────────


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def test_view_log_in_less_runs_less_when_available(tmp_path) -> None:
    """Happy path: ``less`` is on PATH, ``--mouse`` is supported,
    ``subprocess.run`` is invoked with the correct argv, and its exit
    code is returned."""
    log = tmp_path / "agent.log"
    log.write_text("payload\n", encoding="utf-8")

    captured: dict = {}

    def fake_runner(argv, *, check):
        captured["argv"] = list(argv)
        captured["check"] = check
        return _FakeCompletedProcess(returncode=0)

    rc = view_log_in_less(
        log,
        runner=fake_runner,
        which=lambda name: "/usr/bin/less" if name == "less" else None,
        mouse_check=lambda _path: True,
        stderr=io.StringIO(),
    )

    assert rc == 0
    assert captured["check"] is False, (
        "we propagate less's exit code rather than raise on non-zero; "
        "less returns non-zero in mostly benign cases (e.g. ``q`` "
        "after a search miss) and forcing CalledProcessError would "
        "convert that into a misleading crash for the operator"
    )
    argv = captured["argv"]
    assert argv[0] == "/usr/bin/less"
    assert "-R" in argv
    assert "--mouse" in argv
    assert "+G" in argv
    assert str(log) in argv


def test_view_log_in_less_propagates_less_exit_code(tmp_path) -> None:
    """``less`` exits non-zero in some routine cases (e.g. operator
    pressed ``q`` after a failed search). The helper should propagate
    that exit code so a caller can distinguish "clean exit" from
    "less crashed", but should not raise."""
    log = tmp_path / "agent.log"
    log.write_text("x\n", encoding="utf-8")

    rc = view_log_in_less(
        log,
        runner=lambda argv, *, check: _FakeCompletedProcess(returncode=2),
        which=lambda _name: "/usr/bin/less",
        mouse_check=lambda _path: True,
        stderr=io.StringIO(),
    )
    assert rc == 2


def test_view_log_in_less_falls_back_when_less_missing(tmp_path) -> None:
    """``shutil.which("less")`` returning ``None`` is the canonical
    "less not installed" signal. The helper must surface a clear
    stderr warning and call the fallback rather than crashing."""
    log = tmp_path / "agent.log"
    log.write_text("x\n", encoding="utf-8")

    fallback_calls = {"n": 0}

    def fake_fallback() -> int:
        fallback_calls["n"] += 1
        return 7

    err = io.StringIO()

    rc = view_log_in_less(
        log,
        runner=lambda *a, **k: _FakeCompletedProcess(returncode=0),
        which=lambda _name: None,
        mouse_check=lambda _path: True,
        fallback=fake_fallback,
        stderr=err,
    )

    assert rc == 7
    assert fallback_calls["n"] == 1, (
        "missing `less` ⇒ exactly one fallback call so the operator "
        "still sees the transcript via the streaming tail"
    )
    msg = err.getvalue()
    assert "less" in msg.lower(), "stderr warning must mention `less`"
    assert "fallback" in msg.lower() or "stream" in msg.lower(), (
        "warning must explain the fallback so the operator isn't "
        "surprised by the streaming-tail UX"
    )


def test_view_log_in_less_falls_back_when_mouse_unsupported(tmp_path) -> None:
    """An older ``less`` may be on PATH without ``--mouse``. Launching
    it would fail with a usage error; the helper must detect this and
    take the fallback path instead."""
    log = tmp_path / "agent.log"
    log.write_text("x\n", encoding="utf-8")

    runs: dict = {"runner": 0, "fallback": 0}

    def fake_runner(*_a, **_k):
        runs["runner"] += 1
        return _FakeCompletedProcess(returncode=0)

    def fake_fallback() -> int:
        runs["fallback"] += 1
        return 11

    err = io.StringIO()

    rc = view_log_in_less(
        log,
        runner=fake_runner,
        which=lambda _name: "/usr/bin/less",
        mouse_check=lambda _path: False,
        fallback=fake_fallback,
        stderr=err,
    )

    assert rc == 11
    assert runs["runner"] == 0, (
        "missing --mouse ⇒ we must NOT spawn less; otherwise it would "
        "exit with a usage error and strand the operator"
    )
    assert runs["fallback"] == 1
    msg = err.getvalue()
    assert "mouse" in msg.lower(), (
        "stderr warning must explicitly call out the missing --mouse "
        "support so the operator knows why the fallback engaged"
    )


def test_view_log_in_less_falls_back_when_runner_raises_oserror(
    tmp_path,
) -> None:
    """A spawn failure (ENOENT race, EACCES, etc.) must not bubble up
    as an unhandled exception — the helper should catch it, warn, and
    take the fallback path."""
    log = tmp_path / "agent.log"
    log.write_text("x\n", encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("simulated spawn failure")

    fallback_calls = {"n": 0}

    def fake_fallback() -> int:
        fallback_calls["n"] += 1
        return 0

    err = io.StringIO()

    rc = view_log_in_less(
        log,
        runner=boom,
        which=lambda _name: "/usr/bin/less",
        mouse_check=lambda _path: True,
        fallback=fake_fallback,
        stderr=err,
    )

    assert rc == 0
    assert fallback_calls["n"] == 1
    assert "less" in err.getvalue().lower()


def test_view_log_in_less_returns_zero_when_no_fallback(tmp_path) -> None:
    """If the caller doesn't pass a fallback (e.g. tests, or future
    callers that just want to know whether the pager could run), the
    helper must still return cleanly — never raise — when the pager
    is unavailable."""
    log = tmp_path / "agent.log"
    log.write_text("x\n", encoding="utf-8")

    rc = view_log_in_less(
        log,
        runner=lambda *a, **k: _FakeCompletedProcess(returncode=0),
        which=lambda _name: None,  # less missing
        mouse_check=lambda _path: True,
        stderr=io.StringIO(),
    )
    assert rc == 0


# ── _less_supports_mouse: the heuristic ────────────────────────────────────


def test_less_supports_mouse_returns_true_when_help_mentions_mouse() -> None:
    """``less --help`` containing ``--mouse`` is the positive signal
    we use in production. Verify the heuristic recognises it."""
    def fake_run(argv, **_kwargs):
        cp = subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="...\n  --mouse  Enable mouse input.\n...\n",
            stderr="",
        )
        return cp

    real_run = subprocess.run
    try:
        subprocess.run = fake_run  # type: ignore[assignment]
        assert _less_supports_mouse("/usr/bin/less") is True
    finally:
        subprocess.run = real_run  # type: ignore[assignment]


def test_less_supports_mouse_returns_false_on_oserror() -> None:
    """A failed spawn must collapse into ``False`` so the caller takes
    the fallback path. The previous bridge approach would silently
    proceed and crash later when ``less`` actually launched; we want a
    clean negative answer here."""
    def boom(*_a, **_k):
        raise OSError("nope")

    real_run = subprocess.run
    try:
        subprocess.run = boom  # type: ignore[assignment]
        assert _less_supports_mouse("/usr/bin/less") is False
    finally:
        subprocess.run = real_run  # type: ignore[assignment]


# ── integration smoke: gated on `less` being installed ────────────────────


def _have_real_less_with_mouse() -> bool:
    """Return ``True`` iff a real ``less`` binary on PATH advertises
    ``--mouse``. Used to gate the pty-driven smoke test below — there
    is no point spawning a process we know doesn't support the flag
    we're testing."""
    path = shutil.which("less")
    if path is None:
        return False
    try:
        return _less_supports_mouse(path)
    except Exception:
        return False


def test_real_less_smoke_quits_on_q(tmp_path) -> None:
    """Optional integration smoke: spawn real ``less -R --mouse +G``
    in a pseudo-terminal, send ``q``, and assert it exits cleanly.

    Skipped automatically when ``less`` (or ``--mouse``) is not
    available, so the test suite stays portable. The point is to
    catch obvious argv / terminal-handling regressions — we are not
    trying to assert specific mouse-wheel behaviour here, just that
    the chosen flags don't make ``less`` itself reject the
    invocation."""
    if not _have_real_less_with_mouse():
        import pytest
        pytest.skip("real `less` with --mouse not available")

    try:
        import pty  # noqa: WPS433 — POSIX-only; the skip above guards Windows.
    except ImportError:
        import pytest
        pytest.skip("pty module not available on this platform")

    log = tmp_path / "agent.log"
    log.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")

    argv = _less_argv(shutil.which("less") or "less", log)

    pid, fd = pty.fork()
    if pid == 0:
        # Child: exec less. If exec fails, exit non-zero so the
        # parent's waitpid surfaces a meaningful status.
        try:
            os.execvp(argv[0], argv)
        except OSError:
            os._exit(127)

    try:
        # Give less a moment to draw its initial frame, then send ``q``.
        # A short read drains any startup output without blocking the
        # parent forever — we don't actually care what less drew, only
        # that it exits when we ask it to.
        try:
            os.set_blocking(fd, False)
        except (AttributeError, OSError):
            pass
        import select
        import time
        deadline = time.time() + 2.0
        while time.time() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.05)
            if not ready:
                break
            try:
                if not os.read(fd, 4096):
                    break
            except OSError:
                break
        os.write(fd, b"q")
        # Wait for less to exit; reap with a generous timeout so we
        # don't hang the test suite on a hung child.
        wait_deadline = time.time() + 5.0
        status = None
        while time.time() < wait_deadline:
            done_pid, status = os.waitpid(pid, os.WNOHANG)
            if done_pid != 0:
                break
            try:
                os.read(fd, 4096)
            except OSError:
                pass
            time.sleep(0.05)
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            raise AssertionError(
                "less did not exit within 5s of receiving `q`; "
                "the chosen argv may be stranding it in a mode "
                "without a quit binding"
            )
        assert status is not None
        # less normally exits 0 on a clean ``q``. Some builds return
        # 1 from a fresh ``+G`` start (no search history); accept
        # both as "the binary quit on demand".
        assert os.WIFEXITED(status), (
            "less must exit normally when sent `q`; got non-exit status"
        )
        assert os.WEXITSTATUS(status) in (0, 1), (
            f"unexpected less exit code: {os.WEXITSTATUS(status)}"
        )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
