"""Unit tests for :func:`auto_iterator.display.stream_log`.

``ai show --stream`` is the high-latency-SSH escape hatch for the
pyratatui detail TUI: instead of redrawing frames in raw mode +
alternate screen (which forces every keystroke through one network
round-trip), it writes plain text to the regular screen buffer and
lets the local terminal's native scrollback own navigation.

These tests pin the behavioural contract:

* The status header is printed once.
* The seed tail of ``agent.log`` is printed before streaming starts.
* New appends surfaced by :class:`LogTailer` end up in the output.
* The function never emits ``\\033[?1049h`` / ``\\033[?1049l``
  (alt-screen toggles) or cursor-movement escapes — those are what
  break native scrollback.
* The ``should_continue`` hook lets tests drive the loop
  deterministically without a real wall clock.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.display import stream_log  # noqa: E402
from auto_iterator.events import EventLog, RunState  # noqa: E402
from auto_iterator.meta import write_meta  # noqa: E402
from auto_iterator.run_dir import (  # noqa: E402
    RunPaths,
    create_run_dir,
    new_run_id,
    now_iso,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _seed_run(runs_dir: Path, *, agent_log_text: str = "") -> RunPaths:
    """Bootstrap a minimal run dir with the same shape used by other tests."""
    paths = create_run_dir(runs_dir, new_run_id())
    write_meta(paths, {
        "run_id": paths.run_id,
        "pid": 999_999_999,
        "status": "running",
        "workspace": "/tmp/ws",
        "started_at": now_iso(),
        "heartbeat_at": now_iso(),
    })
    state = RunState(
        run_id=paths.run_id,
        prompt="Tail this run",
        workspace="/tmp/ws",
    )
    state.outer = 1
    state.inner = 1
    state.phase = "review"
    log = EventLog(paths, state)
    log.emit("run_started", workspace="/tmp/ws")
    if agent_log_text:
        paths.agent_log.write_text(agent_log_text, encoding="utf-8")
    return paths


class _FakeClock:
    """Minimal ``sleep``-compatible callable that records calls.

    Used as a substitute for :func:`time.sleep` so the streaming loop
    runs at test speed without touching the wall clock."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(float(seconds))


# ── tests ──────────────────────────────────────────────────────────────────


def test_stream_log_prints_header_then_seed_then_exits(tmp_path) -> None:
    """One iteration with a pre-existing log: header + seed must appear,
    streaming loop exits cleanly via ``should_continue=False``."""
    paths = _seed_run(tmp_path, agent_log_text="alpha\nbeta\ngamma\n")

    out = io.StringIO()
    rc = stream_log(
        paths,
        log_lines=10,
        out=out,
        sleep=_FakeClock(),
        should_continue=lambda i: False,
    )

    assert rc == 0
    text = out.getvalue()

    assert paths.run_id in text, "header must contain the run id"
    assert "Agent output" in text, "seed section header must appear"
    assert "streaming" in text, "header must mark this as the streaming mode"
    assert "alpha" in text and "beta" in text and "gamma" in text, (
        "seed lines must be printed before streaming starts"
    )


def test_stream_log_surfaces_appends_via_tailer(tmp_path) -> None:
    """Lines written *after* the seed must surface on subsequent ticks
    without the seed contents being re-printed."""
    paths = _seed_run(tmp_path, agent_log_text="seed-line-1\nseed-line-2\n")

    iters = {"n": 0}

    def driver(_seconds: float) -> None:
        iters["n"] += 1
        # Append on the first tick so the second tick's
        # read_new_lines surfaces the appended text.
        if iters["n"] == 1:
            with paths.agent_log.open("a", encoding="utf-8") as fh:
                fh.write("FRESH-APPEND\n")

    out = io.StringIO()
    rc = stream_log(
        paths,
        log_lines=10,
        out=out,
        sleep=driver,
        should_continue=lambda i: i < 3,
    )

    assert rc == 0
    text = out.getvalue()
    assert "FRESH-APPEND" in text, (
        "post-seed appends must be streamed to stdout"
    )
    # The seed should appear exactly once — the streaming step must
    # not double-emit it.
    assert text.count("seed-line-1") == 1
    assert text.count("seed-line-2") == 1


def test_stream_log_does_not_emit_alt_screen_or_cursor_escapes(tmp_path) -> None:
    """The whole point of ``--stream`` is that the local terminal's
    native scrollback works. Emitting ``\\033[?1049h`` or cursor-home
    escapes would defeat that — those are what the live TUI uses to
    own the screen."""
    paths = _seed_run(tmp_path, agent_log_text="line1\nline2\n")

    out = io.StringIO()
    stream_log(
        paths,
        log_lines=10,
        out=out,
        sleep=_FakeClock(),
        should_continue=lambda i: False,
    )

    text = out.getvalue()
    forbidden = (
        "\033[?1049h",  # alt-screen on
        "\033[?1049l",  # alt-screen off
        "\033[?25l",    # hide cursor
        "\033[?25h",    # show cursor
        "\033[H\033[2J",  # home + clear
        "\033[2J",      # clear screen
    )
    for esc in forbidden:
        assert esc not in text, (
            f"stream_log emitted {esc!r}; this defeats native "
            "terminal scrollback (the whole reason for --stream)."
        )


def test_stream_log_handles_missing_agent_log(tmp_path) -> None:
    """Bootstrapped run dirs may not have ``logs/agent.log`` yet; the
    tailer must tolerate the gap and the renderer must still print
    the header without crashing."""
    paths = _seed_run(tmp_path)  # no agent log content
    assert not paths.agent_log.exists()

    out = io.StringIO()
    rc = stream_log(
        paths,
        log_lines=10,
        out=out,
        sleep=_FakeClock(),
        should_continue=lambda i: False,
    )

    assert rc == 0
    text = out.getvalue()
    assert paths.run_id in text
    assert "agent has not produced output yet" in text


def test_stream_log_respects_log_lines_cap(tmp_path) -> None:
    """``log_lines=N`` must seed at most N trailing lines so an old run
    with megabytes of transcript doesn't dump everything before the
    streaming loop even starts."""
    body = "\n".join(f"line-{i:04d}" for i in range(500)) + "\n"
    paths = _seed_run(tmp_path, agent_log_text=body)

    out = io.StringIO()
    stream_log(
        paths,
        log_lines=5,
        out=out,
        sleep=_FakeClock(),
        should_continue=lambda i: False,
    )

    text = out.getvalue()
    # The last five lines must be present; the first ones must not.
    for i in range(495, 500):
        assert f"line-{i:04d}" in text, f"line-{i:04d} should appear in seed"
    for i in (0, 100, 200, 400):
        assert f"line-{i:04d}" not in text, (
            f"line-{i:04d} must be trimmed from a 5-line seed"
        )


def test_stream_log_log_lines_none_dumps_full_transcript(tmp_path) -> None:
    """``log_lines=None`` is the "operator picked a run from the TUI"
    contract: dump the *entire* agent transcript so the local
    terminal's native scrollback owns the full history.

    Anything less defeats the whole point of native-scrollback
    streaming for an interactive picker: bytes older than the cap
    are no longer reachable once the alt-screen TUI tears down,
    and the operator's "I cannot see the full log" complaint maps
    to exactly that truncation."""
    body = "\n".join(f"line-{i:04d}" for i in range(1_000)) + "\n"
    paths = _seed_run(tmp_path, agent_log_text=body)

    out = io.StringIO()
    rc = stream_log(
        paths,
        log_lines=None,
        out=out,
        sleep=_FakeClock(),
        should_continue=lambda i: False,
    )

    assert rc == 0
    text = out.getvalue()
    # Sample across the whole range — first, middle, last lines must
    # all be present in the dumped transcript.
    for i in (0, 1, 250, 500, 750, 998, 999):
        assert f"line-{i:04d}" in text, (
            f"line-{i:04d} must appear when log_lines=None dumps the "
            f"full transcript (operator selected this run from the "
            f"interactive picker and expects to see *everything*)"
        )
    # The header must label the seed mode so the operator knows
    # they're looking at the full file rather than a tail.
    assert "full transcript" in text


def test_stream_log_full_dump_then_streams_appends(tmp_path) -> None:
    """After dumping the full transcript, subsequent agent writes must
    still surface — and the seed must not be re-emitted.

    This is the same race-window contract as the bounded-seed path:
    :func:`read_text_file_with_offset` returns the EOF observed
    during the dump so the tailer can be parked exactly past the
    last seed byte. A naive implementation that re-``stat()``s the
    file would either drop bytes (offset moved forward) or
    double-emit them (offset rewound)."""
    body = "\n".join(f"seed-{i:03d}" for i in range(50)) + "\n"
    paths = _seed_run(tmp_path, agent_log_text=body)

    iters = {"n": 0}

    def driver(_seconds: float) -> None:
        iters["n"] += 1
        if iters["n"] == 1:
            with paths.agent_log.open("a", encoding="utf-8") as fh:
                fh.write("APPEND-AFTER-DUMP\n")

    out = io.StringIO()
    rc = stream_log(
        paths,
        log_lines=None,
        out=out,
        sleep=driver,
        should_continue=lambda i: i < 3,
    )

    assert rc == 0
    text = out.getvalue()
    assert "APPEND-AFTER-DUMP" in text
    # Each seed line should appear exactly once — no double-emit
    # caused by the tailer re-reading bytes the dump already
    # surfaced.
    for i in (0, 25, 49):
        assert text.count(f"seed-{i:03d}") == 1, (
            f"seed-{i:03d} must appear exactly once; double-emission "
            f"would mean the tailer was anchored before the dump's EOF"
        )
    assert text.count("APPEND-AFTER-DUMP") == 1


def test_stream_log_full_dump_handles_missing_log(tmp_path) -> None:
    """``log_lines=None`` against a run that has not yet produced any
    agent output must still print the header + placeholder rather
    than crashing on the missing file."""
    paths = _seed_run(tmp_path)
    assert not paths.agent_log.exists()

    out = io.StringIO()
    rc = stream_log(
        paths,
        log_lines=None,
        out=out,
        sleep=_FakeClock(),
        should_continue=lambda i: False,
    )

    assert rc == 0
    text = out.getvalue()
    assert "full transcript" in text
    assert "agent has not produced output yet" in text


def test_stream_log_exits_when_poll_input_signals(tmp_path) -> None:
    """A simulated Esc / q keypress (poll_input → True) must break the
    streaming loop on the next iteration without waiting for Ctrl-C.

    Pins the bug fix for "press Enter on a run from the bare ``ai``
    TUI, can't get out by pressing Esc": the run-list TUI hands off
    to ``stream_log`` after Enter, and the streaming view used to
    only honour Ctrl-C. This contract proves the same loop now
    short-circuits cleanly on a bare-Esc / q signal."""
    paths = _seed_run(tmp_path, agent_log_text="alpha\nbeta\n")

    presses = iter([False, False, True])

    def poll() -> bool:
        try:
            return next(presses)
        except StopIteration:
            return False

    sleeps = _FakeClock()
    rc = stream_log(
        paths,
        log_lines=10,
        out=io.StringIO(),
        sleep=sleeps,
        # ``should_continue`` is a safety net: if the input watcher
        # fails to break the loop the test would hang forever
        # otherwise. With the watcher working we should exit *before*
        # this many iterations are reached.
        should_continue=lambda i: i < 50,
        poll_input=poll,
    )

    assert rc == 0
    assert len(sleeps.calls) < 5, (
        "stream_log should break out within a few ticks of the "
        "first 'exit' poll signal, not run to should_continue's cap"
    )


def test_stream_log_ignores_falsey_poll_input(tmp_path) -> None:
    """``poll_input`` returning ``False`` for every iteration must NOT
    cause an early exit — the loop should still respect the
    ``should_continue`` budget. Pins the inverse of the Esc-exit
    contract so a no-op watcher (the failure-soft branch when stdin
    isn't a TTY) doesn't accidentally short-circuit the stream."""
    paths = _seed_run(tmp_path, agent_log_text="x\n")

    sleeps = _FakeClock()
    rc = stream_log(
        paths,
        log_lines=10,
        out=io.StringIO(),
        sleep=sleeps,
        should_continue=lambda i: i < 4,
        poll_input=lambda: False,
    )

    assert rc == 0
    # Loop ran exactly 4 iterations → at least the 3 between-iteration
    # sleeps the contract calls for. The exact count is allowed to
    # drift slightly, but a no-op poll must not collapse the loop to
    # zero sleeps.
    assert len(sleeps.calls) >= 3


def test_stream_log_does_not_drop_lines_appended_during_seed(
    tmp_path, monkeypatch,
) -> None:
    """Regression: appends that land *between* the seed read and the
    tailer offset must still surface during streaming.

    The bug is subtle: a naive implementation ``stat``s the file once
    for the seed tail and a second time to seek the tailer to EOF.
    Anything written in that window lives below the tailer's offset
    *and* past the seed bytes — invisible forever.

    We simulate the window deterministically by wrapping
    :func:`tail_text_file_with_offset` so it appends a known marker
    *after* it has computed the seed and offset but *before* it
    returns. A correct implementation anchors the tailer at the
    offset that ``tail_text_file_with_offset`` reported, so the
    marker shows up on the next ``read_new_lines`` tick. A buggy
    implementation that re-stats the file would skip the marker."""
    paths = _seed_run(tmp_path, agent_log_text="seed-A\nseed-B\n")
    log_path = paths.agent_log

    from auto_iterator import display as _display

    real_helper = _display.tail_text_file_with_offset

    def racy_helper(path, *, lines, chunk_per_line=4096):
        seed, end_offset = real_helper(
            path, lines=lines, chunk_per_line=chunk_per_line,
        )
        # Simulate an agent write that lands during the seed handoff.
        if Path(path) == log_path:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write("RACE-LINE-DURING-SEED\n")
        return seed, end_offset

    monkeypatch.setattr(
        _display, "tail_text_file_with_offset", racy_helper, raising=True,
    )

    out = io.StringIO()
    rc = stream_log(
        paths,
        log_lines=10,
        out=out,
        sleep=_FakeClock(),
        # Run two iterations: the first read_new_lines tick must
        # surface the racy line; the second confirms we don't
        # double-emit it.
        should_continue=lambda i: i < 2,
    )

    assert rc == 0
    text = out.getvalue()
    assert "RACE-LINE-DURING-SEED" in text, (
        "stream_log must surface bytes written between the seed read "
        "and the tailer hand-off; otherwise --stream silently drops "
        "exactly the lines an operator most wants to see when starting "
        "to follow a run."
    )
    assert text.count("RACE-LINE-DURING-SEED") == 1, (
        "the racy line should be surfaced once via the tailer, "
        "not duplicated"
    )
    # Seed bytes still appear exactly once.
    assert text.count("seed-A") == 1
    assert text.count("seed-B") == 1


# ── tail_text_file_with_offset (the primitive) ──────────────────────────────


def test_tail_text_file_with_offset_returns_eof_position(tmp_path) -> None:
    """The returned offset must equal ``st_size`` after the read.

    A :class:`LogTailer` started at this offset will then surface
    everything appended *after* the read — the contract :func:`stream_log`
    relies on to be race-free."""
    from auto_iterator.display import tail_text_file_with_offset, LogTailer

    log = tmp_path / "agent.log"
    body = "alpha\nbeta\ngamma\n"
    log.write_text(body, encoding="utf-8")

    seed, end = tail_text_file_with_offset(log, lines=10)
    assert seed == ["alpha", "beta", "gamma"]
    assert end == len(body.encode("utf-8"))

    # Append after the read; a tailer parked at ``end`` must see it.
    with log.open("a", encoding="utf-8") as fh:
        fh.write("delta\n")
    tailer = LogTailer(log)
    tailer.seek_to(end)
    fresh = tailer.read_new_lines()
    assert fresh == ["delta"], (
        "tail_text_file_with_offset's offset must mark exactly the "
        "first byte the tailer should consume"
    )


def test_tail_text_file_with_offset_handles_large_files(tmp_path) -> None:
    """When the file is bigger than the lookback window the offset
    must still point past the bytes the function actually consulted —
    not past the (smaller) returned tail."""
    from auto_iterator.display import tail_text_file_with_offset, LogTailer

    log = tmp_path / "agent.log"
    body = "\n".join(f"line-{i:05d}" for i in range(2_000)) + "\n"
    log.write_text(body, encoding="utf-8")
    file_size = len(body.encode("utf-8"))

    seed, end = tail_text_file_with_offset(log, lines=5)
    assert end == file_size, (
        "the returned offset must mark the file's EOF as observed by "
        "the read, even when the seed only contains the trailing few "
        "lines (the dropped bytes were still consumed)"
    )
    assert seed[-1] == "line-01999"

    with log.open("a", encoding="utf-8") as fh:
        fh.write("after-the-tail\n")
    tailer = LogTailer(log)
    tailer.seek_to(end)
    assert tailer.read_new_lines() == ["after-the-tail"]


def test_tail_text_file_with_offset_missing_or_empty(tmp_path) -> None:
    """Missing and empty files must yield ``([], 0)`` so callers can
    safely park a tailer at the returned offset without special-casing."""
    from auto_iterator.display import tail_text_file_with_offset

    missing = tmp_path / "nope.log"
    assert tail_text_file_with_offset(missing, lines=5) == ([], 0)

    empty = tmp_path / "empty.log"
    empty.write_bytes(b"")
    assert tail_text_file_with_offset(empty, lines=5) == ([], 0)


def test_read_text_file_with_offset_returns_eof_position(tmp_path) -> None:
    """``read_text_file_with_offset`` must report the same EOF position
    that :class:`LogTailer` should be parked at to surface only future
    appends. Same race-free hand-off contract as
    :func:`tail_text_file_with_offset`, but for the *full* dump path
    the run-list TUI hand-off uses."""
    from auto_iterator.display import LogTailer, read_text_file_with_offset

    log = tmp_path / "agent.log"
    body = "alpha\nbeta\ngamma\n"
    log.write_text(body, encoding="utf-8")

    lines, end = read_text_file_with_offset(log)
    assert lines == ["alpha", "beta", "gamma"]
    assert end == len(body.encode("utf-8"))

    with log.open("a", encoding="utf-8") as fh:
        fh.write("delta\n")
    tailer = LogTailer(log)
    tailer.seek_to(end)
    assert tailer.read_new_lines() == ["delta"], (
        "an append after the full dump must surface via the tailer "
        "exactly once, with no overlap or gap"
    )


def test_read_text_file_with_offset_missing_or_empty(tmp_path) -> None:
    """Missing and empty files yield ``([], <offset>)`` so callers can
    safely seed a tailer without special-casing. The offset is 0 for
    both — there are no bytes to skip past."""
    from auto_iterator.display import read_text_file_with_offset

    missing = tmp_path / "nope.log"
    assert read_text_file_with_offset(missing) == ([], 0)

    empty = tmp_path / "empty.log"
    empty.write_bytes(b"")
    assert read_text_file_with_offset(empty) == ([], 0)


def test_logtailer_seek_to_clamps_negative(tmp_path) -> None:
    """``seek_to`` clamps negative offsets to 0 so a buggy caller can't
    make the next ``read_new_lines`` skip past EOF."""
    from auto_iterator.display import LogTailer

    log = tmp_path / "agent.log"
    log.write_text("one\ntwo\n", encoding="utf-8")
    tailer = LogTailer(log)
    assert tailer.seek_to(-50) == 0
    assert tailer.read_new_lines() == ["one", "two"]
