"""Unit tests for :class:`auto_iterator.display.LogTailer`.

The tailer is the heart of the TUI's agent-output panel: every tick
reads only the bytes that have arrived since the last call. The
properties below pin the exact behaviour the TUI relies on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_iterator.display import LogTailer  # noqa: E402


def test_first_read_returns_all_lines_and_advances_offset(tmp_path) -> None:
    log = tmp_path / "agent.log"
    log.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    t = LogTailer(log)

    out = t.read_new_lines()

    assert out == ["alpha", "beta", "gamma"]
    assert t.offset == log.stat().st_size, (
        "first read must advance the offset to EOF so the next call "
        "has nothing to deliver"
    )


def test_no_new_bytes_returns_empty_list(tmp_path) -> None:
    log = tmp_path / "agent.log"
    log.write_text("hello\nworld\n", encoding="utf-8")
    t = LogTailer(log)
    _ = t.read_new_lines()  # drain the seed

    assert t.read_new_lines() == []
    assert t.read_new_lines() == []  # idempotent


def test_appended_bytes_return_only_the_new_lines(tmp_path) -> None:
    log = tmp_path / "agent.log"
    log.write_text("first\n", encoding="utf-8")
    t = LogTailer(log)
    assert t.read_new_lines() == ["first"]

    with log.open("a", encoding="utf-8") as fh:
        fh.write("second\nthird\n")

    assert t.read_new_lines() == ["second", "third"]
    assert t.read_new_lines() == []


def test_truncation_resets_offset_and_rereads(tmp_path) -> None:
    """``ai`` runners can rotate or zero the log; we must not get stuck
    seeking past EOF after a shrink."""
    log = tmp_path / "agent.log"
    log.write_text("one\ntwo\nthree\n", encoding="utf-8")
    t = LogTailer(log)
    assert t.read_new_lines() == ["one", "two", "three"]

    # Shrink the file to a smaller content. The new size is below the
    # cached offset → tailer must reset and re-read from the top.
    log.write_text("renewed\n", encoding="utf-8")

    out = t.read_new_lines()
    assert out == ["renewed"]
    assert t.offset == log.stat().st_size


def test_partial_line_is_buffered_until_newline_arrives(tmp_path) -> None:
    """A poll between two writes can land mid-line; the trailing
    bytes must surface on the next call once the newline lands."""
    log = tmp_path / "agent.log"
    log.write_bytes(b"complete\nstart-of-")
    t = LogTailer(log)

    out = t.read_new_lines()
    # The complete line appears now; ``start-of-`` is held back.
    assert out == ["complete"]
    # Offset advances past everything we read so we don't re-read it,
    # but the partial bytes are stashed.
    assert t.offset == log.stat().st_size

    # Append the rest of the partial line and one more.
    with log.open("ab") as fh:
        fh.write(b"second-half\nthird\n")

    out = t.read_new_lines()
    # The buffered prefix joins the freshly-read suffix to form one
    # complete line, and the next line follows.
    assert out == ["start-of-second-half", "third"]


def test_missing_file_returns_empty(tmp_path) -> None:
    """The runner only opens ``logs/agent.log`` lazily; the polling
    loop must tolerate the gap before the file exists."""
    t = LogTailer(tmp_path / "does-not-exist.log")
    assert t.read_new_lines() == []
    assert t.offset == 0


def test_empty_file_returns_empty(tmp_path) -> None:
    log = tmp_path / "empty.log"
    log.write_bytes(b"")
    t = LogTailer(log)
    assert t.read_new_lines() == []


def test_utf8_replace_on_partial_multibyte(tmp_path) -> None:
    """A partial multi-byte codepoint at the read boundary must not
    raise. ``errors=\"replace\"`` is the contract; a real terminal
    would render the replacement char until the rest of the bytes
    arrive (and our buffer ensures they do, on the next call)."""
    log = tmp_path / "agent.log"
    # Mango emoji 🥭 is U+1F96D, which is 4 UTF-8 bytes:
    # 0xF0 0x9F 0xA5 0xAD. Write the first two bytes alone.
    log.write_bytes(b"prefix\n\xf0\x9f")
    t = LogTailer(log)

    out = t.read_new_lines()
    # The complete prefix line is delivered; the half-codepoint is
    # held back as partial bytes (no newline yet).
    assert out == ["prefix"]

    # Append the remaining bytes to complete the codepoint and a newline.
    with log.open("ab") as fh:
        fh.write(b"\xa5\xad\n")
    out = t.read_new_lines()
    # The reassembled line decodes to a single mango glyph.
    assert out == ["\U0001f96d"]


def test_per_tick_read_is_bounded_for_huge_log(tmp_path) -> None:
    """Reviewer pin: the tailer must never load a multi-MiB file in
    one go. We grow a fake log past the per-tick cap and assert each
    call advances the offset by no more than the cap."""
    log = tmp_path / "agent.log"
    # 5 MiB of newline-terminated content. With a 4 MiB per-tick cap
    # this needs at least two reads to drain.
    payload = (b"x" * 4095 + b"\n") * 1280  # ~5 MiB
    log.write_bytes(payload)

    t = LogTailer(log)
    first = t.read_new_lines()
    cap = 4 * 1024 * 1024
    assert t.offset <= cap, (
        f"first read advanced offset by {t.offset} bytes — must be <= {cap}"
    )
    # Some lines were delivered.
    assert first, "first read must yield at least one line"

    # Drain the rest in subsequent calls — none of them may exceed cap.
    last_offset = t.offset
    while t.offset < log.stat().st_size:
        _ = t.read_new_lines()
        delta = t.offset - last_offset
        assert delta <= cap, (
            f"per-tick read of {delta} bytes exceeds {cap} byte cap"
        )
        last_offset = t.offset
    # Final call after EOF returns empty.
    assert t.read_new_lines() == []


def test_reset_clears_offset_and_partial(tmp_path) -> None:
    log = tmp_path / "agent.log"
    log.write_bytes(b"one\nincomplete-")
    t = LogTailer(log)
    _ = t.read_new_lines()
    assert t.offset == log.stat().st_size
    t.reset()
    assert t.offset == 0
    # After reset, the file is re-read from the start.
    out = t.read_new_lines()
    assert out == ["one"]


def test_truncation_to_empty_file_clears_buffers(tmp_path) -> None:
    log = tmp_path / "agent.log"
    log.write_text("a\nb\n", encoding="utf-8")
    t = LogTailer(log)
    assert t.read_new_lines() == ["a", "b"]

    # Zero out the file; the next call must reset rather than report
    # spurious data from the stale offset.
    log.write_bytes(b"")
    assert t.read_new_lines() == []
    assert t.offset == 0

    log.write_text("c\n", encoding="utf-8")
    assert t.read_new_lines() == ["c"]


def test_seek_to_end_skips_to_eof_without_reading(tmp_path) -> None:
    """``seek_to_end`` parks the offset at EOF in O(1) — even on a
    multi-MiB log that's larger than the per-tick read cap.

    This is the primitive the TUI relies on after rendering its
    initial bounded tail: the next ``read_new_lines`` must surface
    *only* future appends, never historical bytes still sitting at
    or below the cap. Calling ``read_new_lines()`` to advance the
    offset would be wrong for files larger than ~4 MiB."""
    log = tmp_path / "agent.log"
    # 8 MiB — comfortably larger than the 4 MiB per-tick cap, so
    # ``read_new_lines`` would only advance partway and leave the
    # offset short of EOF.
    payload = (b"x" * 4095 + b"\n") * 2048
    log.write_bytes(payload)
    size = log.stat().st_size
    cap = 4 * 1024 * 1024
    assert size > cap, "test setup must exceed the per-tick cap"

    t = LogTailer(log)
    rc = t.seek_to_end()
    assert rc == size
    assert t.offset == size

    # No new bytes since the seek → no lines surface.
    assert t.read_new_lines() == []

    # Append a small chunk; only the appended bytes come back.
    with log.open("ab") as fh:
        fh.write(b"future\n")
    assert t.read_new_lines() == ["future"]


def test_seek_to_end_on_missing_file_is_zero(tmp_path) -> None:
    """A run dir is bootstrapped before the agent has written anything
    — ``seek_to_end`` must tolerate that and leave the tailer at 0."""
    t = LogTailer(tmp_path / "absent.log")
    assert t.seek_to_end() == 0
    assert t.offset == 0
    assert t.read_new_lines() == []


def test_seek_to_end_clears_partial_buffer(tmp_path) -> None:
    """Bytes after the last newline must not leak into future reads
    once the operator has explicitly skipped past them.

    Without ``seek_to_end`` clearing the partial buffer, the bytes
    after the most recent newline (``"start-of-"`` below) would be
    prepended to the next chunk of new bytes, conjuring a phantom
    line like ``"start-of-second-half"``. The TUI's seed path opens
    the screen with the bounded tail and then expects to see *only*
    future content — phantom lines made of historical bytes
    contaminate that contract."""
    log = tmp_path / "agent.log"
    log.write_bytes(b"complete\nstart-of-")
    t = LogTailer(log)
    _ = t.read_new_lines()
    assert t._partial == b"start-of-"
    t.seek_to_end()
    assert t._partial == b""
    # Append a fresh line; with the partial buffer cleared the
    # phantom prefix is gone and we only see the new bytes.
    with log.open("ab") as fh:
        fh.write(b"second-half\nfresh\n")
    assert t.read_new_lines() == ["second-half", "fresh"]
