"""Human-readable one-line summaries for Cursor agent tool calls and results."""

from __future__ import annotations

import json


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _shorten_path(path: str, max_parts: int = 4) -> str:
    parts = path.rstrip("/").split("/")
    if len(parts) <= max_parts:
        return path
    return "…/" + "/".join(parts[-max_parts:])


def tool_summary(tc: dict, *, completed: bool = False) -> str:
    """Return a compact one-line summary of a stream-json tool_call payload."""
    for key in tc:
        if not key.endswith("ToolCall"):
            continue
        name = key.replace("ToolCall", "")
        inner = tc[key]
        args = inner.get("args", {})
        if completed and "result" in inner:
            return _fmt_result(name, args, inner["result"])
        return _fmt_call(name, args)
    return _truncate(json.dumps(tc, separators=(",", ":")), 200)


def _fmt_call(name: str, args: dict) -> str:  # noqa: C901
    """Human-readable one-liner for a tool invocation."""

    if name == "shell":
        cmd = args.get("command", "")
        desc = args.get("description", "")
        label = desc or cmd
        return f"shell: {_truncate(label, 90)}"

    if name == "glob":
        pat = args.get("globPattern", args.get("glob_pattern", ""))
        d = _shorten_path(args.get("targetDirectory", args.get("target_directory", "")))
        return f"glob: {pat}" + (f" in {d}" if d else "")

    if name == "grep":
        pat = _truncate(args.get("pattern", ""), 40)
        p = _shorten_path(args.get("path", ""))
        gl = args.get("glob", "")
        suffix = f" [{gl}]" if gl else ""
        return f'grep: "{pat}"' + (f" in {p}" if p else "") + suffix

    if name == "read":
        p = _shorten_path(args.get("path", ""))
        extras = []
        if args.get("offset"):
            extras.append(f"L{args['offset']}")
        if args.get("limit"):
            extras.append(f"+{args['limit']}")
        return f"read: {p}" + (f" ({', '.join(extras)})" if extras else "")

    if name == "delete":
        return f"delete: {_shorten_path(args.get('path', ''))}"

    if name in ("strReplace", "edit"):
        p = _shorten_path(args.get("path", args.get("filePath", "")))
        ra = " (all)" if args.get("replace_all") or args.get("replaceAll") else ""
        return f"edit: {p}{ra}"

    if name in ("write", "createFile"):
        p = _shorten_path(args.get("path", ""))
        size = len(args.get("contents", args.get("content", "")))
        return f"write: {p} ({size} chars)"

    if name == "editNotebook":
        nb = _shorten_path(args.get("targetNotebook", args.get("target_notebook", "")))
        idx = args.get("cellIdx", args.get("cell_idx", "?"))
        new = "new " if args.get("isNewCell", args.get("is_new_cell")) else ""
        return f"editNotebook: {nb} {new}cell {idx}"

    if name in ("todoWrite", "updateTodos"):
        n = len(args.get("todos", []))
        merge = args.get("merge", False)
        return f"todos: {'merge' if merge else 'replace'} {n} item(s)"

    if name == "readLints":
        paths = args.get("paths", [])
        if paths:
            shown = ", ".join(_shorten_path(p) for p in paths[:3])
            extra = f" +{len(paths) - 3}" if len(paths) > 3 else ""
            return f"lints: {shown}{extra}"
        return "lints: (workspace)"

    if name in ("semanticSearch", "codebaseSearch"):
        q = _truncate(args.get("query", ""), 60)
        dirs = args.get("targetDirectories", args.get("target_directories", []))
        where = ", ".join(_shorten_path(d) for d in dirs[:2]) if dirs else "all"
        return f'search: "{q}" in {where}'

    if name == "webSearch":
        term = _truncate(args.get("searchTerm", args.get("search_term", "")), 60)
        return f'webSearch: "{term}"'

    if name in ("webFetch", "urlFetch"):
        url = _truncate(args.get("url", ""), 80)
        return f"fetch: {url}"

    if name == "generateImage":
        desc = _truncate(args.get("description", ""), 60)
        return f'image: "{desc}"'

    if name == "askQuestion":
        qs = args.get("questions", [])
        title = args.get("title", "")
        label = title or f"{len(qs)} question(s)"
        return f"ask: {label}"

    if name == "task":
        desc = args.get("description", "?")
        model = args.get("model", "")
        sub = args.get("subagentType", args.get("subagent_type", ""))
        parts = [desc]
        if sub:
            parts.append(f"[{sub}]")
        if model:
            parts.append(f"({model})")
        return f"task: {' '.join(parts)}"

    if name == "await":
        tid = args.get("taskId", args.get("task_id", ""))
        ms = args.get("blockUntilMs", args.get("block_until_ms", ""))
        pat = args.get("pattern", "")
        parts = []
        if tid:
            parts.append(f"id={tid}")
        if ms:
            parts.append(f"{ms}ms")
        if pat:
            parts.append(f"/{_truncate(pat, 30)}/")
        return f"await: {' '.join(parts)}" if parts else "await"

    if name == "fetchMcpResource":
        srv = args.get("server", "")
        uri = _truncate(args.get("uri", ""), 60)
        return f"mcpResource: {srv} {uri}"

    if name in ("callMcpTool", "mcpTool"):
        srv = args.get("server", "")
        tn = args.get("toolName", "")
        return f"mcp: {srv}/{tn}"

    if name == "switchMode":
        mode = args.get("targetModeId", args.get("target_mode_id", ""))
        expl = args.get("explanation", "")
        return f"switchMode → {mode}" + (f" ({_truncate(expl, 40)})" if expl else "")

    if name == "listMcpResources":
        return "listMcpResources"

    s = json.dumps(args, separators=(",", ":"))
    return f"{name}({_truncate(s, 200)})" if s != "{}" else name


def _fmt_result(name: str, args: dict, result: object) -> str:  # noqa: C901
    """Human-readable one-liner for a tool result."""
    if isinstance(result, dict):
        if "rejected" in result:
            reason = result["rejected"].get("reason", "")
            return f"shell ✗ rejected" + (f" ({reason})" if reason else "")

        s = result.get("success")
        if isinstance(s, dict):

            if name == "shell":
                ec = s.get("exitCode", "?")
                sym = "✓" if ec == 0 else "✗"
                out = (s.get("stdout") or "").strip()
                first = out.split("\n")[0][:80] if out else ""
                return f"shell {sym} exit {ec}" + (f": {first}" if first else "")

            if name == "glob":
                n = s.get("totalFiles", 0)
                files = s.get("files", [])
                shown = ", ".join(files[:3])
                extra = f" +{n - 3}" if n > 3 else ""
                return f"glob ✓ {n} file(s)" + (f": {shown}{extra}" if shown else "")

            if name == "grep":
                pat = _truncate(args.get("pattern", ""), 30)
                total = s.get("totalMatchedLines", s.get("totalLines", None))
                if total is None:
                    for ws in (s.get("workspaceResults") or {}).values():
                        c = ws.get("content", {})
                        total = c.get("totalMatchedLines", c.get("totalLines"))
                        if total is not None:
                            break
                return f'grep ✓ {total if total is not None else "?"} match(es) for "{pat}"'

            if name == "read":
                p = _shorten_path(s.get("path", args.get("path", "")))
                if s.get("isEmpty"):
                    return f"read ✓ {p} (empty)"
                return f"read ✓ {p} ({s.get('totalLines', '?')} lines)"

            if name == "delete":
                p = _shorten_path(args.get("path", ""))
                return f"delete ✓ {p}"

            if name in ("strReplace", "edit"):
                p = _shorten_path(s.get("path", args.get("path", "")))
                added = s.get("linesAdded", 0)
                removed = s.get("linesRemoved", 0)
                return f"edit ✓ {p} (+{added} −{removed})"

            if name in ("write", "createFile"):
                p = _shorten_path(s.get("path", args.get("path", "")))
                lines = s.get("totalLines", "?")
                return f"write ✓ {p} ({lines} lines)"

            if name == "editNotebook":
                nb = _shorten_path(args.get("targetNotebook", args.get("target_notebook", "")))
                return f"editNotebook ✓ {nb}"

            if name in ("todoWrite", "updateTodos"):
                return "todos ✓ updated"

            if name == "readLints":
                n = len(s.get("diagnostics", s.get("lints", [])))
                return f"lints ✓ {n} diagnostic(s)"

            if name in ("semanticSearch", "codebaseSearch"):
                n = len(s.get("results", s.get("chunks", [])))
                return f"search ✓ {n} result(s)"

            if name == "webSearch":
                return "webSearch ✓"

            if name in ("webFetch", "urlFetch"):
                return "fetch ✓"

            if name == "generateImage":
                return "image ✓"

            if name == "askQuestion":
                return "ask ✓ answered"

            if name == "task":
                return "task ✓ completed"

            if name == "await":
                return "await ✓"

            if name == "fetchMcpResource":
                return "mcpResource ✓"

            if name in ("callMcpTool", "mcpTool"):
                tn = args.get("toolName", "mcp")
                return f"mcp ✓ {tn}"

            if name == "switchMode":
                mode = args.get("targetModeId", args.get("target_mode_id", ""))
                return f"switchMode ✓ → {mode}"

            if name == "listMcpResources":
                return f"listMcpResources ✓ {len(s.get('resources', []))} resource(s)"

    short = json.dumps(result, separators=(",", ":"))
    return f"{name} ⇒ {_truncate(short, 200)}"
