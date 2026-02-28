#!/usr/bin/env python3
"""chat.py — Interactive CLI for vLLM models with tool calling.

Single-file CLI tool for chatting with persona_kappa (or any model) running
on vLLM. Supports multi-turn conversation, bash tool calling, streaming,
and persona conditioning.

Usage:
    python3 chat.py                          # auto-detect model
    python3 chat.py --persona lawful_evil    # with persona
    python3 chat.py --model kappa_20b_131k   # explicit model
"""

import argparse
import ast
import concurrent.futures
import json
import multiprocessing
import os
import re
import readline
import sqlite3
import subprocess
import sys
import uuid
import ipaddress
import socket
import tempfile
import textwrap
import threading
import traceback
from html import unescape as _html_unescape
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openai import OpenAI

# ─── ANSI helpers ──────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"


def red(s):
    return f"{RED}{s}{RESET}"


def yellow(s):
    return f"{YELLOW}{s}{RESET}"


def dim(s):
    return f"{DIM}{s}{RESET}"


def bold(s):
    return f"{BOLD}{s}{RESET}"


def cyan(s):
    return f"{CYAN}{s}{RESET}"


def green(s):
    return f"{GREEN}{s}{RESET}"


# ─── Markdown → ANSI renderer ─────────────────────────────────────────────


class MarkdownRenderer:
    """Line-buffered markdown → ANSI converter for streaming output.

    Buffers content until a newline arrives, then renders the complete line
    with regex-based markdown → ANSI conversion. Multi-line constructs
    (fenced code blocks) track state across lines.
    """

    def __init__(self):
        self.in_code_block = False
        self._buf = ""

    def feed(self, text: str) -> str:
        """Feed text, return ANSI-rendered output for complete lines."""
        self._buf += text
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            out.append(self._render_line(line))
        return "\n".join(out) + "\n" if out else ""

    def flush(self) -> str:
        """Flush remaining buffer (end of stream)."""
        if self._buf:
            rendered = self._render_line(self._buf)
            self._buf = ""
            return rendered
        return ""

    def _render_line(self, line: str) -> str:
        # Code block fence toggle
        if line.strip().startswith("```"):
            self.in_code_block = not self.in_code_block
            return f"{DIM}{line}{RESET}"

        # Inside code block — cyan, no further markdown processing
        if self.in_code_block:
            return f"{CYAN}{line}{RESET}"

        # Headers (# H1, ## H2, ### H3, #### H4, ##### H5, ###### H6)
        m = re.match(r"^(#{1,6}) (.+)", line)
        if m:
            return f"{BOLD}{MAGENTA}{m.group(2)}{RESET}"

        # Inline formatting (order matters: bold before italic)
        line = re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", line)
        line = re.sub(r"__(.+?)__", f"{BOLD}\\1{RESET}", line)
        line = re.sub(
            r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", f"{ITALIC}\\1{RESET}", line
        )
        line = re.sub(r"`(.+?)`", f"{CYAN}\\1{RESET}", line)

        # Bullet lists — cyan bullet
        line = re.sub(r"^(\s*)([-*]) ", f"\\1{CYAN}\\2{RESET} ", line)

        # Numbered lists — cyan number
        line = re.sub(r"^(\s*)(\d+)\. ", f"\\1{CYAN}\\2.{RESET} ", line)

        return line


# ─── Tool definition ──────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command and return stdout + stderr. "
                "Use this tool freely for any task: checking time, "
                "reading files, running programs, system info, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if needed. Overwrites existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns numbered lines. Must be called before edit_file on the same path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based, default: 1).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Omit to read entire file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search file contents for a regex pattern. "
                "Returns matching lines with file paths and line numbers. "
                "Searches recursively when path is a directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Regex pattern to search for (extended regex).",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default: current directory).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file with new content. Fails if old_string is not found or matches multiple locations (use near_line to disambiguate). Requires read_file on the same path first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                    "near_line": {
                        "type": "integer",
                        "description": "When old_string matches multiple locations, pick the one nearest this line number.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "math",
            "description": (
                "Execute Python code for math/computation. "
                "Code MUST use print() to produce output. "
                "Available: sympy, numpy, scipy (with scipy.special, "
                "scipy.optimize, scipy.integrate, scipy.linalg), math, "
                "fractions, itertools, functools, collections, decimal, "
                "operator, random, re, string. "
                "Common sympy names (symbols, solve, simplify, expand, factor, "
                "sqrt, Rational, Matrix, integrate, diff, etc.) are pre-imported. "
                "Example: x = symbols('x'); print(solve(x**2 - 4, x))"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Must use print() for output.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "man",
            "description": (
                "Look up a man page or info page for a command, syscall, or "
                "library function. Returns the formatted manual entry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "description": "The man page name (e.g. 'grep', 'socket', 'printf').",
                    },
                    "section": {
                        "type": "string",
                        "description": "Manual section (e.g. '1' commands, '2' syscalls, '3' library). Optional.",
                    },
                },
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and extract specific information from it. "
                "You must provide a question or extraction guidance — "
                "the page is fetched, analyzed, and only relevant "
                "information is returned (not raw page content)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (must start with http:// or https://).",
                    },
                    "question": {
                        "type": "string",
                        "description": "What to extract or answer from the page content.",
                    },
                },
                "required": ["url", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using a text query. Returns ranked results "
                "with titles, URLs, and content snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (default 5, max 20).",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Search topic: general, news, or finance (default general).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "Delegate a general-purpose task to an autonomous sub-agent. "
                "The agent inherits all tools and can read, write, edit, search, "
                "and run commands. Use task for work that requires file modifications "
                "or command execution. Provide a clear, self-contained prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Complete task description for the sub-agent.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan",
            "description": (
                "Plan before implementing. An autonomous agent explores the "
                "codebase and writes a structured plan to .plan.md. "
                "Use plan BEFORE writing code — when the user asks to build, "
                "add, refactor, or change something that touches multiple files "
                "or has unclear scope. The plan identifies files to modify, "
                "existing patterns to reuse, and risks to consider."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What to plan — the goal, constraints, and scope.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a persistent memory. Memories persist across sessions. "
                "Use to remember IPs, paths, commands, conventions, or any "
                "fact worth recalling later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short identifier (e.g. 'user_name').",
                    },
                    "value": {
                        "type": "string",
                        "description": "Content to remember.",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search memories and past conversations. "
                "With no query, lists all saved memories. "
                "With a query, searches both memories and conversation history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term or phrase. Omit to list all memories.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max conversation results to return (default 20).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": (
                "Remove a persistent memory by key. Use when the user asks "
                "to forget, remove, or delete a stored memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The memory key to remove (e.g. 'user_name').",
                    },
                },
                "required": ["key"],
            },
        },
    },
]

# Tools available to read-only sub-agents (plan).
# Excludes agent tools (no recursion) and write tools (security boundary).
AGENT_TOOLS = [
    t
    for t in TOOLS
    if t["function"]["name"]
    in ("read_file", "search", "math", "man", "web_fetch", "web_search")
]

# Tools available to the task sub-agent (read + write + execute).
# Still excludes agent tools (no recursion).
TASK_AGENT_TOOLS = [
    t
    for t in TOOLS
    if t["function"]["name"]
    in (
        "read_file",
        "search",
        "math",
        "man",
        "web_fetch",
        "web_search",
        "bash",
        "write_file",
        "edit_file",
    )
]

# ─── Edit helpers ─────────────────────────────────────────────────────────


def _find_occurrences(content: str, old_string: str) -> list[int]:
    """Return 1-based line numbers where each occurrence of old_string starts."""
    if not old_string:
        return []
    # Build a prefix-sum of line starts for O(1) line-number lookup.
    line_starts = [0]
    for i, ch in enumerate(content):
        if ch == "\n":
            line_starts.append(i + 1)
    results = []
    start = 0
    while True:
        idx = content.find(old_string, start)
        if idx == -1:
            break
        # bisect: find the line containing idx
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        results.append(lo + 1)  # 1-based
        start = idx + 1
    return results


def _pick_nearest(content: str, old_string: str, near_line: int) -> int:
    """Return the char index of the occurrence of old_string nearest to near_line."""
    line_starts = [0]
    for i, ch in enumerate(content):
        if ch == "\n":
            line_starts.append(i + 1)

    best_idx = -1
    best_dist = float("inf")
    start = 0
    while True:
        idx = content.find(old_string, start)
        if idx == -1:
            break
        # Find line number for this occurrence
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        line_num = lo + 1
        dist = abs(line_num - near_line)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
        start = idx + 1
    return best_idx


# ─── Sandboxed Python executor ────────────────────────────────────────────

_MATH_BLOCKED_BUILTINS = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "breakpoint",
    "memoryview",
    "globals",
    "locals",
    "vars",
}

_MATH_BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "http",
    "urllib",
    "requests",
    "pickle",
    "marshal",
    "shelve",
    "dbm",
    "sqlite3",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
    "concurrent",
    "signal",
    "pty",
    "tty",
    "termios",
    "fcntl",
    "resource",
    "syslog",
    "tempfile",
    "io",
    "builtins",
    "__builtin__",
    "importlib",
}


class _ASTValidator(ast.NodeVisitor):
    """Validates AST for dangerous constructs."""

    def __init__(self):
        self.errors: list[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split(".")[0] in _MATH_BLOCKED_MODULES:
                self.errors.append(f"Import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split(".")[0] in _MATH_BLOCKED_MODULES:
            self.errors.append(f"Import from '{node.module}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in _MATH_BLOCKED_BUILTINS:
            self.errors.append(f"Call to '{node.func.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in {"__name__", "__doc__", "__class__"}:
                self.errors.append(f"Access to '{node.attr}' is not allowed")
        self.generic_visit(node)


def _validate_math_code(code: str) -> list[str]:
    """Validate code for dangerous constructs. Returns list of errors."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        lines = code.split("\n")
        msg = f"Syntax error on line {e.lineno}: {e.msg}"
        if e.lineno and e.lineno <= len(lines):
            msg += f"\n  {e.lineno}: {lines[e.lineno - 1]}"
            if e.offset:
                msg += f"\n      {' ' * (e.offset - 1)}^"
        return [msg]
    except (ValueError, UnicodeError) as e:
        return [f"Code contains invalid characters: {e}"]
    v = _ASTValidator()
    v.visit(tree)
    return v.errors


def _math_exec_in_process(code: str, result_queue: multiprocessing.Queue):
    """Execute code in a subprocess, put (status, output) in queue."""
    import signal as _signal
    import sys as _sys
    from io import StringIO

    _signal.signal(_signal.SIGTERM, _signal.SIG_DFL)
    _signal.signal(_signal.SIGINT, _signal.SIG_DFL)
    _sys.set_int_max_str_digits(100_000)

    try:
        captured = StringIO()
        _sys.stdout = captured

        def _safe_import(name, *args, **kwargs):
            if name.split(".")[0] in _MATH_BLOCKED_MODULES:
                raise ImportError(f"Import of '{name}' is blocked")
            return original_import(name, *args, **kwargs)

        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )
        safe_builtins = (
            {k: v for k, v in __builtins__.items() if k not in _MATH_BLOCKED_BUILTINS}
            if isinstance(__builtins__, dict)
            else {
                k: getattr(__builtins__, k)
                for k in dir(__builtins__)
                if k not in _MATH_BLOCKED_BUILTINS and not k.startswith("_")
            }
        )
        safe_builtins["__import__"] = _safe_import

        # Pre-import safe modules
        import math, fractions, itertools, functools, operator
        import collections, decimal, random, re, string

        ns: dict = {
            "__builtins__": safe_builtins,
            "math": math,
            "fractions": fractions,
            "Fraction": fractions.Fraction,
            "itertools": itertools,
            "functools": functools,
            "operator": operator,
            "collections": collections,
            "decimal": decimal,
            "Decimal": decimal.Decimal,
            "random": random,
            "re": re,
            "string": string,
        }

        try:
            import sympy

            ns["sympy"] = sympy
            for name in (
                "symbols",
                "Symbol",
                "solve",
                "simplify",
                "expand",
                "factor",
                "Eq",
                "sqrt",
                "Rational",
                "pi",
                "E",
                "I",
                "oo",
                "sin",
                "cos",
                "tan",
                "exp",
                "log",
                "factorial",
                "binomial",
                "gcd",
                "lcm",
                "prime",
                "isprime",
                "factorint",
                "divisors",
                "totient",
                "mod_inverse",
                "Matrix",
                "integrate",
                "diff",
                "limit",
                "series",
                "Sum",
                "Product",
                "floor",
                "ceiling",
                "Abs",
            ):
                ns[name] = getattr(sympy, name)
        except ImportError:
            pass

        try:
            import numpy as _np

            ns["np"] = ns["numpy"] = _np
        except ImportError:
            pass

        try:
            import scipy, scipy.special, scipy.optimize, scipy.integrate, scipy.linalg

            ns["scipy"] = scipy
            ns["special"] = scipy.special
            ns["optimize"] = scipy.optimize
            ns["comb"] = scipy.special.comb
            ns["perm"] = scipy.special.perm
            ns["gamma"] = scipy.special.gamma
            ns["beta"] = scipy.special.beta
        except ImportError:
            pass

        exec(code, ns)

        _sys.stdout = _sys.__stdout__
        printed = captured.getvalue()
        result_var = ns.get("result")
        if result_var is not None:
            out = (
                f"{printed.rstrip()}\nresult = {result_var}"
                if printed
                else str(result_var)
            )
        elif printed:
            out = printed.rstrip()
        else:
            out = "No output. Add print() to see results."
        result_queue.put(("success", out))

    except Exception as e:
        _sys.stdout = _sys.__stdout__
        result_queue.put(
            ("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        )


def _auto_print_wrap(code: str) -> str:
    """If code has no print/result and the last statement is an expression, wrap it in print()."""
    # Skip if code already has print() or assigns to 'result'
    if "print(" in code or re.search(r"\bresult\s*=", code):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        # Get the source of the last expression and wrap in print()
        lines = code.split("\n")
        last_line_start = last.lineno - 1  # 0-based
        last_line_end = last.end_lineno  # 1-based, exclusive after slicing
        expr_lines = lines[last_line_start:last_line_end]
        expr_text = "\n".join(expr_lines)
        prefix = lines[:last_line_start]
        wrapped = prefix + [f"print({expr_text})"]
        return "\n".join(wrapped)
    return code


def _execute_math_sandboxed(code: str, timeout: float = 30.0) -> tuple[str, bool]:
    """Execute Python code in a sandboxed subprocess. Returns (output, is_error)."""
    code = _auto_print_wrap(code)
    errors = _validate_math_code(code)
    if errors:
        return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors), True

    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_math_exec_in_process, args=(code, result_queue)
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        result_queue.close()
        result_queue.join_thread()
        return f"Execution timed out after {timeout}s", True

    if result_queue.empty():
        result_queue.close()
        result_queue.join_thread()
        return "Execution failed with no output", True

    status, output = result_queue.get()
    result_queue.close()
    result_queue.join_thread()
    return output, status == "error"


# ─── HTML stripper (for web_fetch) ─────────────────────────────────────────

_RE_TAGS = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"[ \t]+")
_RE_BLANKLINES = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    """Convert HTML to plain text: strip tags, decode entities, collapse whitespace."""
    text = _RE_TAGS.sub("", html)
    text = _html_unescape(text)
    text = _RE_WS.sub(" ", text)
    text = _RE_BLANKLINES.sub("\n\n", text)
    return text.strip()


# ─── Safety ────────────────────────────────────────────────────────────────

# Soft guardrail — catches common accidental destructive commands but is
# trivially bypassable (e.g. extra spaces, shell variable expansion).
# The user approval prompt is the primary security boundary.
BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
    "> /dev/sda",
    "mv / ",
    "chmod -R 777 /",
    "chown -R ",
]


def _sanitize_command(cmd: str) -> str:
    """Replace common unicode look-alikes that break the shell."""
    return (
        cmd.replace("\u2018", "'")  # left single curly quote
        .replace("\u2019", "'")  # right single curly quote
        .replace("\u201c", '"')  # left double curly quote
        .replace("\u201d", '"')  # right double curly quote
        .replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
    )


def is_command_blocked(cmd: str) -> str | None:
    """Return reason string if command is blocked, None otherwise."""
    cmd_stripped = cmd.strip()
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_stripped:
            return f"Blocked: command matches dangerous pattern '{pattern}'"
    return None


# ─── Readline setup ───────────────────────────────────────────────────────

PCODE_DB = os.path.join(os.getcwd(), ".pcode_memories.db")
_db_override: str | None = None
_db_initialized: set[str] = set()
_fts5_available: bool = False

_tavily_key: str | None = None
_tavily_key_loaded: bool = False


def _get_tavily_key() -> str | None:
    """Load Tavily API key from file or env var (cached after first call)."""
    global _tavily_key, _tavily_key_loaded
    if _tavily_key_loaded:
        return _tavily_key
    _tavily_key_loaded = True
    key_path = os.path.expanduser("~/.config/pcode/tavily_key")
    if os.path.isfile(key_path):
        try:
            with open(key_path) as f:
                key = f.read().strip()
            if key:
                _tavily_key = key
                return _tavily_key
        except OSError:
            pass
    env_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if env_key:
        _tavily_key = env_key
    return _tavily_key


def _open_db() -> sqlite3.Connection:
    """Open the pcode database, creating tables on first use per path."""
    path = _db_override or PCODE_DB
    conn = sqlite3.connect(path)
    if path not in _db_initialized:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "created TEXT NOT NULL, updated TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, timestamp TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT, "
            "tool_name TEXT, tool_args TEXT)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id)"
        )
        global _fts5_available
        try:
            # Check if FTS table already exists
            fts_exists = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='conversations_fts'"
            ).fetchone()
            if not fts_exists:
                conn.execute(
                    "CREATE VIRTUAL TABLE conversations_fts "
                    "USING fts5(content, content=conversations, content_rowid=id)"
                )
                conn.execute(
                    "INSERT INTO conversations_fts(conversations_fts) VALUES('rebuild')"
                )
                conn.commit()
            _fts5_available = True
        except Exception:
            _fts5_available = False
        _db_initialized.add(path)
    return conn


def _normalize_key(key: str) -> str:
    """Normalize a memory key for consistent lookup."""
    return key.lower().replace("-", "_").replace(" ", "_")


def _load_memories() -> list[tuple[str, str]]:
    """Return all (key, value) pairs sorted by key."""
    try:
        conn = _open_db()
        try:
            return conn.execute(
                "SELECT key, value FROM memories ORDER BY key"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []


def _save_message(
    session_id: str,
    role: str,
    content: str | None,
    tool_name: str | None = None,
    tool_args: str | None = None,
) -> None:
    """Log a message to the conversations table."""
    global _fts5_available
    try:
        conn = _open_db()
        try:
            conn.execute(
                "INSERT INTO conversations (session_id, timestamp, role, content, "
                "tool_name, tool_args) VALUES (?, datetime('now'), ?, ?, ?, ?)",
                (session_id, role, content, tool_name, tool_args),
            )
            if _fts5_available and content:
                try:
                    rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    conn.execute(
                        "INSERT INTO conversations_fts(rowid, content) VALUES (?, ?)",
                        (rowid, content),
                    )
                except Exception:
                    _fts5_available = False  # degrade to LIKE for rest of session
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # Don't let logging failures break the session


def _escape_like(s: str) -> str:
    """Escape LIKE metacharacters for use with ESCAPE '\\'."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _fts5_query(query: str) -> str:
    """Convert a plain search string into a safe FTS5 query.

    Quotes each term so FTS5 special characters (*, -, etc.) are treated
    as literals, then joins with implicit AND.  Embedded double quotes
    are doubled per FTS5 quoting convention.
    """
    terms = query.split()
    safe = []
    for t in terms:
        if t:
            safe.append(f'"{t.replace(chr(34), chr(34) + chr(34))}"')
    return " ".join(safe)


def _search_history(query: str, limit: int = 20) -> list[tuple]:
    """Search conversation history. Returns (timestamp, session_id, role, content, tool_name)."""
    if not query or not query.strip():
        return []
    try:
        conn = _open_db()
        try:
            if _fts5_available:
                return conn.execute(
                    "SELECT c.timestamp, c.session_id, c.role, c.content, c.tool_name "
                    "FROM conversations_fts f "
                    "JOIN conversations c ON c.id = f.rowid "
                    "WHERE conversations_fts MATCH ? "
                    "ORDER BY f.rank ASC LIMIT ?",
                    (_fts5_query(query), min(limit, 100)),
                ).fetchall()
            return conn.execute(
                "SELECT timestamp, session_id, role, content, tool_name "
                "FROM conversations WHERE content LIKE ? ESCAPE '\\' "
                "ORDER BY timestamp DESC LIMIT ?",
                (f"%{_escape_like(query)}%", min(limit, 100)),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []


def _search_history_recent(limit: int = 20) -> list[tuple]:
    """Return most recent conversation messages."""
    try:
        conn = _open_db()
        try:
            return conn.execute(
                "SELECT timestamp, session_id, role, content, tool_name "
                "FROM conversations ORDER BY timestamp DESC LIMIT ?",
                (min(limit, 100),),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []


SLASH_COMMANDS = [
    "/persona",
    "/instructions",
    "/clear",
    "/new",
    "/history",
    "/model",
    "/raw",
    "/reason",
    "/compact",
    "/creative",
    "/debug",
    "/help",
    "/exit",
    "/quit",
    "/q",
]


def _completer(text, state):
    """Tab-complete slash commands."""
    if text.startswith("/"):
        matches = [c for c in SLASH_COMMANDS if c.startswith(text)]
    else:
        matches = []
    if state < len(matches):
        return matches[state] + " "
    return None


def setup_readline():
    """Set up readline with tab completion."""
    readline.set_history_length(1000)
    readline.set_completer(_completer)
    readline.set_completer_delims("")  # treat entire line as completion input
    readline.parse_and_bind("tab: complete")


# ─── Spinner ───────────────────────────────────────────────────────────────


class Spinner:
    """Simple terminal spinner for waiting periods.

    Supports both context manager and explicit start/stop usage.
    Call stop() early (e.g., on first streaming token) to clear the spinner
    before printing content.
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message="Thinking"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
        self._stopped = True

    def start(self):
        self._stop_event.clear()
        self._stopped = False
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the spinner and clear its line. Safe to call multiple times."""
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def _spin(self):
        i = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r{DIM}{frame} {self.message}...{RESET}")
            sys.stdout.flush()
            i += 1
            self._stop_event.wait(0.08)


# ─── Chat Session ─────────────────────────────────────────────────────────


class ChatSession:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        persona: str | None,
        instructions: str | None,
        temperature: float,
        max_tokens: int,
        tool_timeout: int,
        reasoning_effort: str = "medium",
        context_window: int = 131072,
    ):
        self.client = client
        self.model = model
        self.persona = persona
        self.instructions = instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_timeout = tool_timeout
        self.reasoning_effort = reasoning_effort
        self.context_window = context_window
        self.show_reasoning = True
        self.debug = False
        self.auto_approve = False
        self._session_id = uuid.uuid4().hex[:12]
        self._read_files: set[str] = set()
        self.md = MarkdownRenderer()
        self.messages: list[dict] = []
        self._last_usage: dict[str, int] | None = None
        self._chars_per_token = 4.0  # calibrated from API usage
        self._msg_tokens: list[int] = []  # parallel to self.messages
        self._system_tokens = 0  # tokens for system_messages
        self._assistant_pending_tokens = 0
        self.creative_mode = False
        self._init_system_messages()

    def _init_system_messages(self):
        """Build the system/developer prefix messages.

        System message format matches training distribution:
          Persona: X           (optional)
          Knowledge cutoff: X  (from base model pretraining)
          Current date: X      (from base model pretraining)
          Reasoning: X         (low/medium/high)
          # Valid channels: ... (dynamic based on tools/reasoning)
          Calls to these tools... (only when tools present)

        Developer message uses # Instructions header when combined
        with tool definitions (tool defs appended by chat template).
        """
        from datetime import date

        self.system_messages = []
        today = date.today().strftime("%Y-%m-%d")
        has_tools = not self.creative_mode

        # -- Chat template kwargs --
        # The chat template builds the system message (persona, knowledge
        # cutoff, date, reasoning, channels) from kwargs.  We pass persona
        # via model_identity; reasoning_effort is already passed.
        self._chat_template_kwargs_base = {
            "reasoning_effort": self.reasoning_effort,
        }
        self._chat_template_kwargs = dict(self._chat_template_kwargs_base)
        if self.persona:
            self._chat_template_kwargs["model_identity"] = f"Persona: {self.persona}"

        # -- Developer message --
        if self.creative_mode:
            dev_parts = [
                "# Instructions",
                "",
                "You are a creative writing partner. Use the analysis channel to "
                "think through structure, voice, and intent before drafting.",
                "",
                "Craft principles:",
                "- Ground scenes in concrete sensory detail — what is seen, heard, felt.",
                "- Vary rhythm. Short sentences hit hard. Longer ones carry the reader "
                "through texture and nuance, building toward something.",
                "- Dialogue should do at least two things: reveal character AND advance "
                "plot or tension. Cut anything that's just exchanging information.",
                "- Earn your abstractions. Don't say 'she felt sad' — show the thing "
                "that makes the reader feel it.",
                "- Trust subtext. Leave room for the reader.",
                "",
                "Match the user's genre and tone. If they want literary fiction, write "
                "literary fiction. If they want pulp, write pulp with conviction. "
                "Never condescend to the form.",
            ]
        else:
            dev_parts = [
                "Always respond with tool calls, not just text.\n\n"
                "TOOL PATTERNS:\n\n"
                "Modify existing file → read_file then edit_file:\n"
                "   read_file(path='config.py') → "
                "edit_file(path='config.py')\n\n"
                "Create new file → write_file:\n"
                "   write_file(path='hello.py', content='...')\n\n"
                "Find something across files → search:\n"
                "   search(query='test_')\n\n"
                "Complex or multi-step task → plan first:\n"
                "   plan(prompt='refactor database from API')\n\n"
                "Run a command, git, or tests → bash:\n"
                "   bash(command='git log -5')\n"
                "   bash(command='pytest')\n\n"
                "Retrieve a URL → web_fetch:\n"
                "   web_fetch(url='https://example.com')\n\n"
                "Look up documentation → man:\n"
                "   man(page='tar')",
            ]
        if self.instructions:
            dev_parts.append("")
            dev_parts.append(self.instructions)
        memories = _load_memories()
        if memories:
            dev_parts.append("")
            dev_parts.append(
                f"REMINDER: You currently have {len(memories)} memories stored. "
                "Use recall to see them."
            )
        self.system_messages.append(
            {"role": "developer", "content": "\n".join(dev_parts)}
        )
        # Agent prefix: system + developer only (no memories)
        self._agent_system_messages = list(self.system_messages)

    def _full_messages(self) -> list[dict]:
        """System messages + conversation history."""
        return self.system_messages + self.messages

    def send(self, user_input: str):
        """Send user input and handle the response loop (including tool calls)."""
        self.messages.append({"role": "user", "content": user_input})
        self._msg_tokens.append(max(1, int(len(user_input) / self._chars_per_token)))
        _save_message(self._session_id, "user", user_input)

        try:
            while True:
                msgs = self._full_messages()

                if self.debug:
                    self._debug_print_request(msgs)

                spinner = Spinner("Thinking")
                spinner.start()
                try:
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=msgs,
                        **({"tools": TOOLS} if not self.creative_mode else {}),
                        max_completion_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stream=True,
                        stream_options={"include_usage": True},
                        extra_body={
                            "chat_template_kwargs": self._chat_template_kwargs,
                        },
                    )
                    assistant_msg = self._stream_response(stream, spinner)
                finally:
                    spinner.stop()

                self._update_token_table(assistant_msg)
                self.messages.append(assistant_msg)
                self._msg_tokens.append(
                    self._assistant_pending_tokens
                    or max(
                        1,
                        int(
                            self._msg_char_count(assistant_msg) / self._chars_per_token
                        ),
                    )
                )

                # Log assistant message to conversation history
                content = assistant_msg.get("content", "")
                tc = assistant_msg.get("tool_calls")
                if content:
                    _save_message(self._session_id, "assistant", content)
                if tc:
                    for call in tc:
                        fn = call.get("function", {})
                        name = fn.get("name", "")
                        if name not in (
                            "remember",
                            "forget",
                            "recall",
                        ):
                            _save_message(
                                self._session_id,
                                "tool_call",
                                None,
                                name,
                                fn.get("arguments", ""),
                            )

                tool_calls = assistant_msg.get("tool_calls")
                if not tool_calls:
                    self._print_status_line()
                    # Auto-compact when prompt exceeds 80% of context window
                    if (
                        self._last_usage
                        and self._last_usage["prompt_tokens"]
                        > self.context_window * 0.8
                    ):
                        print(
                            dim(
                                "\n[Auto-compacting: prompt exceeds "
                                "80% of context window]"
                            )
                        )
                        self._compact_messages(auto=True)
                    break

                # Execute tool calls (potentially in parallel)
                results, user_feedback = self._execute_tools(tool_calls)
                # Map tool_call_id → tool name for logging
                _tc_names = {
                    c["id"]: c.get("function", {}).get("name", "") for c in tool_calls
                }
                for tc_id, output in results:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": output,
                    }
                    self.messages.append(tool_msg)
                    self._msg_tokens.append(
                        max(1, int(len(output) / self._chars_per_token))
                    )
                    # Log tool result (skip memory tools to avoid noise)
                    _tname = _tc_names.get(tc_id, "")
                    if _tname not in (
                        "remember",
                        "forget",
                        "recall",
                    ):
                        _save_message(
                            self._session_id,
                            "tool_result",
                            output[:2000],
                            _tname,
                        )
                # Inject user feedback from approval prompt (e.g. "y, use full path")
                if user_feedback:
                    self.messages.append({"role": "user", "content": user_feedback})
                    self._msg_tokens.append(
                        max(1, int(len(user_feedback) / self._chars_per_token))
                    )
        except KeyboardInterrupt:
            # Remove any partial tool results, then the originating assistant
            # message with unanswered tool_calls — keep _msg_tokens in sync
            while self.messages and self.messages[-1]["role"] == "tool":
                self.messages.pop()
                if self._msg_tokens:
                    self._msg_tokens.pop()
            while (
                self.messages
                and self.messages[-1]["role"] == "assistant"
                and self.messages[-1].get("tool_calls")
            ):
                self.messages.pop()
                if self._msg_tokens:
                    self._msg_tokens.pop()
            raise

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove <think>/<reasoning> tags and their content."""
        for open_t, close_t in [
            ("<think>", "</think>"),
            ("<reasoning>", "</reasoning>"),
        ]:
            while open_t in text:
                start = text.find(open_t)
                end = text.find(close_t, start)
                if end != -1:
                    text = text[:start] + text[end + len(close_t) :]
                else:
                    text = text[:start]
        return text.strip()

    # Tags that delimit reasoning blocks in content stream.
    # Checked in order; first match wins.
    _THINK_OPEN_TAGS = ("<think>", "<reasoning>")
    _THINK_CLOSE_TAGS = ("</think>", "</reasoning>")
    _MAX_TAG_LEN = max(len(t) for t in _THINK_OPEN_TAGS + _THINK_CLOSE_TAGS)

    def _stream_response(self, stream, spinner: Spinner | None = None) -> dict:
        """Stream response, printing tokens as they arrive.

        Handles two reasoning delivery mechanisms:
        1. vLLM's `reasoning_content` field (when --reasoning-parser is set)
        2. <think>...</think> tags in regular content (common default)

        Stops the spinner on the first received delta.

        Returns the complete assistant message as a dict suitable for
        appending to self.messages.
        """
        content_parts = []
        reasoning_parts = []
        tool_calls_acc: dict[int, dict] = {}
        first_token = True
        in_think = False  # inside a <think>...</think> block
        path1_reasoning = False  # last reasoning came via reasoning_content field
        pending = ""  # buffer for partial tag detection
        dim_active = False  # tracks whether DIM ANSI code is active on terminal

        def _set_dim(active: bool):
            """Ensure terminal DIM state matches desired state."""
            nonlocal dim_active
            if active and not dim_active:
                if self.show_reasoning:
                    sys.stdout.write(f"{DIM}")
                    sys.stdout.flush()
                dim_active = True
            elif not active and dim_active:
                sys.stdout.write(f"{RESET}")
                sys.stdout.flush()
                dim_active = False

        def _flush_text(text: str, is_reasoning: bool):
            """Print text with appropriate styling."""
            if not text:
                return
            if is_reasoning:
                reasoning_parts.append(text)
                if self.show_reasoning:
                    _set_dim(True)
                    sys.stdout.write(text)
                    sys.stdout.flush()
            else:
                _set_dim(False)
                content_parts.append(text)
                rendered = self.md.feed(text)
                if rendered:
                    sys.stdout.write(rendered)
                    sys.stdout.flush()

        def _drain_pending():
            """Process the pending buffer, flushing content and detecting tags."""
            nonlocal pending, in_think

            while pending:
                if in_think:
                    # Look for any close tag
                    best_idx, best_tag = None, None
                    for tag in self._THINK_CLOSE_TAGS:
                        idx = pending.find(tag)
                        if idx != -1 and (best_idx is None or idx < best_idx):
                            best_idx, best_tag = idx, tag

                    if best_idx is not None:
                        _flush_text(pending[:best_idx], True)
                        pending = pending[best_idx + len(best_tag) :]
                        in_think = False
                        _set_dim(False)
                        if self.show_reasoning:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        continue

                    # No close tag found — check if tail could be a partial tag
                    safe = len(pending) - self._MAX_TAG_LEN
                    if safe > 0:
                        _flush_text(pending[:safe], True)
                        pending = pending[safe:]
                    break
                else:
                    # Look for any open tag
                    best_idx, best_tag = None, None
                    for tag in self._THINK_OPEN_TAGS:
                        idx = pending.find(tag)
                        if idx != -1 and (best_idx is None or idx < best_idx):
                            best_idx, best_tag = idx, tag

                    if best_idx is not None:
                        _flush_text(pending[:best_idx], False)
                        pending = pending[best_idx + len(best_tag) :]
                        in_think = True
                        _set_dim(True)
                        continue

                    # No open tag found — flush all but potential partial tag
                    safe = len(pending) - self._MAX_TAG_LEN
                    if safe > 0:
                        _flush_text(pending[:safe], False)
                        pending = pending[safe:]
                    break

        def _stop_spinner_once():
            """Stop the spinner on first real content. Call is idempotent."""
            nonlocal first_token
            if first_token and spinner:
                spinner.stop()
                first_token = False

        for chunk in stream:
            # Capture usage from final chunk (stream_options.include_usage)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                u = chunk.usage
                pt = getattr(u, "prompt_tokens", None)
                ct = getattr(u, "completion_tokens", None)
                tt = getattr(u, "total_tokens", None)
                if pt is not None and ct is not None:
                    self._last_usage = {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": tt or (pt + ct),
                    }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if self.debug:
                extras = dict(delta.model_extra) if delta.model_extra else {}
                parts = []
                if delta.role:
                    parts.append(f"role={delta.role}")
                if delta.content:
                    parts.append(f"content={delta.content!r}")
                if delta.tool_calls:
                    parts.append(f"tool_calls=...")
                for k, v in extras.items():
                    if v is not None:
                        parts.append(f"{k}={v!r}")
                if parts:
                    sys.stdout.write(f"{GRAY}[delta: {', '.join(parts)}]{RESET}\n")
                    sys.stdout.flush()

            # Path 1: reasoning field (vLLM sends as "reasoning" or "reasoning_content")
            rc = getattr(delta, "reasoning", None) or getattr(
                delta, "reasoning_content", None
            )
            if rc:
                _stop_spinner_once()
                reasoning_parts.append(rc)
                in_think = True
                path1_reasoning = True
                if self.show_reasoning:
                    _set_dim(True)
                    sys.stdout.write(rc)
                    sys.stdout.flush()
                continue

            # Path 2: regular content (may contain <think> tags)
            if delta.content:
                _stop_spinner_once()
                # Close reasoning dim if transitioning from Path 1 reasoning
                if path1_reasoning:
                    path1_reasoning = False
                    in_think = False
                    _set_dim(False)
                    if self.show_reasoning:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                pending += delta.content
                _drain_pending()

            # Handle tool call deltas
            if delta.tool_calls:
                _stop_spinner_once()
                # Close reasoning dim if transitioning from reasoning
                if in_think:
                    in_think = False
                    _set_dim(False)
                    if self.show_reasoning:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_acc[idx]
                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

        # Flush any remaining buffered text
        if pending:
            _flush_text(pending, in_think)
            pending = ""

        # Flush markdown renderer's remaining partial line and reset state
        remainder = self.md.flush()
        if remainder:
            sys.stdout.write(remainder)
            sys.stdout.flush()
        self.md.in_code_block = False

        # Close reasoning styling if still open
        _set_dim(False)
        if in_think and self.show_reasoning:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Ensure newline after content
        if content_parts:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Build assistant message dict
        msg: dict = {"role": "assistant"}

        content = "".join(content_parts)
        if content:
            msg["content"] = content
        else:
            msg["content"] = None

        if tool_calls_acc:
            msg["tool_calls"] = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]

        return msg

    _print_lock = threading.Lock()

    # ── Debug ──────────────────────────────────────────────────────────

    def _debug_print_request(self, msgs: list[dict]):
        """Print the full API request payload when debug mode is on."""
        w = sys.stdout.write
        w(f"\n{GRAY}{'─' * 60}{RESET}\n")
        w(
            f"{GRAY}[request] model={self.model}  "
            f"max_tokens={self.max_tokens}  temp={self.temperature}  "
            f"reasoning={self.reasoning_effort}  "
            f"tools={0 if self.creative_mode else len(TOOLS)}{RESET}\n"
        )
        w(f"{GRAY}[request] {len(msgs)} messages:{RESET}\n")
        for i, m in enumerate(msgs):
            role = m["role"]
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls")
            tc_id = m.get("tool_call_id")

            # Truncate long content for readability
            if len(content) > 300:
                display = (
                    content[:200] + f"...({len(content)} chars)..." + content[-50:]
                )
            else:
                display = content
            # Escape newlines for compact display
            display = display.replace("\n", "\\n")

            header = f"  [{i}] {role}"
            if tc_id:
                header += f" (tool_call_id={tc_id})"

            w(f"{GRAY}{header}: {display}{RESET}\n")

            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "?")
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 200:
                        args = args[:150] + f"...({len(args)} chars)"
                    w(f"{GRAY}    → {name}({args}){RESET}\n")

        w(f"{GRAY}{'─' * 60}{RESET}\n\n")
        sys.stdout.flush()

    # ── Token tracking & status ────────────────────────────────────────

    def _msg_char_count(self, msg: dict) -> int:
        """Count characters in a message, including tool call arguments."""
        n = len(msg.get("content") or "")
        for tc in msg.get("tool_calls", []):
            n += len(tc.get("function", {}).get("name", ""))
            n += len(tc.get("function", {}).get("arguments", ""))
        return n

    def _update_token_table(self, assistant_msg: dict):
        """Update per-message token estimates using API usage data."""
        if not self._last_usage:
            return

        prompt_tok = self._last_usage["prompt_tokens"]
        compl_tok = self._last_usage["completion_tokens"]

        # Calibrate chars_per_token ratio from actual usage.
        # Include tool definition chars (prompt_tok includes tool schema tokens).
        all_msgs = self._full_messages()  # system + self.messages (before append)
        tool_def_chars = sum(len(json.dumps(t)) for t in TOOLS)
        total_chars = sum(self._msg_char_count(m) for m in all_msgs) + tool_def_chars
        if total_chars > 0 and prompt_tok > 0:
            self._chars_per_token = total_chars / prompt_tok

        # Compute system_tokens (stable after first call)
        sys_chars = sum(self._msg_char_count(m) for m in self.system_messages)
        self._system_tokens = max(1, int(sys_chars / self._chars_per_token))

        # Re-estimate all message token counts with calibrated ratio
        self._msg_tokens = [
            max(1, int(self._msg_char_count(m) / self._chars_per_token))
            for m in self.messages
        ]

        # Stash completion_tokens for the assistant message about to be appended
        self._assistant_pending_tokens = compl_tok

    def _print_status_line(self):
        """Print a dim inline status line with token usage and settings."""
        if not self._last_usage:
            return
        total_tok = (
            self._last_usage["prompt_tokens"] + self._last_usage["completion_tokens"]
        )
        ctx = self.context_window
        pct = total_tok / ctx * 100 if ctx > 0 else 0
        parts = [f"{total_tok:,} / {ctx:,} tokens ({pct:.0f}%)"]
        if self.reasoning_effort != "medium":
            parts.append(f"reasoning: {self.reasoning_effort}")
        sys.stdout.write(f"\n  {DIM}[{' · '.join(parts)}]{RESET}\n")
        sys.stdout.flush()

    # ── Conversation compaction ──────────────────────────────────────────

    def _format_messages_for_summary(self, messages: list[dict]) -> str:
        """Format messages into a readable string for the summarization prompt."""
        # Build tool_call_id → tool_name lookup for labeling tool results
        tc_names: dict[str, str] = {}
        for m in messages:
            for tc in m.get("tool_calls", []):
                tc_id = tc.get("id", "")
                tc_name = tc.get("function", {}).get("name", "unknown")
                if tc_id:
                    tc_names[tc_id] = tc_name

        parts = []
        for m in messages:
            role = m["role"].upper()
            content = m.get("content") or ""

            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    name = tc.get("function", {}).get("name", "?")
                    args = tc.get("function", {}).get("arguments", "")
                    calls.append(f"{name}({args})")
                content += "\n[Called: " + ", ".join(calls) + "]"

            # Label tool results with the tool name
            if role == "TOOL":
                tc_id = m.get("tool_call_id", "")
                name = tc_names.get(tc_id, "tool")
                role = f"TOOL[{name}]"

            if content:
                if len(content) > 2000:
                    content = content[:1000] + "\n...[truncated]...\n" + content[-500:]
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _compact_messages(self, auto: bool = False):
        """Compact conversation history by summarizing all messages.

        Summarizes the entire conversation via a separate model call,
        budget-fitted to 80% of the context window.

        When auto=True (triggered by context limit), appends a continuation
        hint with the last user message so the model can resume seamlessly.
        """
        if len(self.messages) < 2:
            print(dim("Not enough messages to compact."))
            return

        # Find the last user message for the continuation hint
        last_user_content = None
        if auto:
            for m in reversed(self.messages):
                if m["role"] == "user":
                    last_user_content = m.get("content") or ""
                    break

        to_summarize = self.messages

        # Budget: fit as many messages as possible into summary request
        summary_max_tokens = 4096
        prompt_budget = (
            int(self.context_window * 0.8) - summary_max_tokens - self._system_tokens
        )
        selected = []
        running = 0
        for i, msg in enumerate(to_summarize):
            msg_tok = (
                self._msg_tokens[i]
                if i < len(self._msg_tokens)
                else max(1, int(self._msg_char_count(msg) / self._chars_per_token))
            )
            if running + msg_tok > prompt_budget:
                break
            selected.append(msg)
            running += msg_tok

        if not selected:
            print(dim("Messages too large to fit in summary context."))
            return

        # Build summary prompt and call model
        formatted = self._format_messages_for_summary(selected)
        summary_msgs = [
            {
                "role": "developer",
                "content": (
                    "# Conversation Compactor\n\n"
                    "Your output REPLACES the conversation history — the assistant "
                    "will continue from your summary with no access to the original messages.\n\n"
                    "1. **Output format** — use these exact sections, omit any that are empty:\n"
                    "   - **## Decisions**: Choices made (architecture, libraries, approaches).\n"
                    "   - **## Files**: Files read, created, or modified, with brief notes.\n"
                    "   - **## Key code**: Exact function names, class names, variable names, "
                    "and short code snippets the assistant will need. "
                    "Preserve identifiers verbatim — do NOT paraphrase.\n"
                    "   - **## Tool results**: Important tool outputs (errors, search matches, "
                    "file contents) that inform ongoing work.\n"
                    "   - **## Open tasks**: What the user asked for that is not yet done, "
                    "with enough context to continue.\n"
                    "   - **## User preferences**: Workflow preferences, constraints, or "
                    "instructions the user stated.\n\n"
                    "2. **Density rules:**\n"
                    "   - Every token should carry information.\n"
                    "   - Preserve exact paths, identifiers, and numbers — never paraphrase these.\n"
                    "   - Omit pleasantries, acknowledgments, and reasoning that led to dead ends.\n"
                    "   - If a tool call's result was an error that was later resolved, "
                    "keep only the resolution.\n\n"
                    "3. **Common mistakes to avoid:**\n"
                    "   - Paraphrasing file paths, function names, or variable names\n"
                    "   - Including dead-end explorations or superseded decisions\n"
                    "   - Omitting the open tasks section when work remains\n"
                    "   - Being verbose — this is a summary, not a transcript"
                ),
            },
            {
                "role": "user",
                "content": ("Compact the following conversation:\n\n" + formatted),
            },
        ]

        spinner = Spinner("Compacting")
        spinner.start()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_msgs,
                max_completion_tokens=summary_max_tokens,
                temperature=0.3,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {
                        **self._chat_template_kwargs_base,
                        "reasoning_effort": "low",
                    }
                },
            )
            summary = response.choices[0].message.content or ""
            # Strip any <think>/<reasoning> tags the summarizer may emit
            summary = self._strip_reasoning(summary)
        except Exception as e:
            spinner.stop()
            print(red(f"Compaction failed: {e}"))
            return
        finally:
            spinner.stop()

        # Append continuation hint for auto-compact
        if last_user_content:
            # Truncate very long user messages
            if len(last_user_content) > 500:
                last_user_content = last_user_content[:400] + "..."
            summary += (
                f"\n\n## Continue\n"
                f"The user's last message was: {last_user_content}\n"
                f"Continue assisting from where we left off."
            )

        # Replace messages
        before_tokens = self._system_tokens + sum(self._msg_tokens)
        summary_user = {"role": "user", "content": "[Conversation summary]"}
        summary_asst = {"role": "assistant", "content": summary}
        self.messages = [summary_user, summary_asst]
        # File contents are gone after compaction — force re-read before edit_file
        self._read_files.clear()

        # Rebuild token table
        su_tok = max(1, int(self._msg_char_count(summary_user) / self._chars_per_token))
        sa_tok = max(1, int(self._msg_char_count(summary_asst) / self._chars_per_token))
        self._msg_tokens = [su_tok, sa_tok]
        after_tokens = self._system_tokens + sum(self._msg_tokens)

        print(dim(f"[compacted: ~{before_tokens:,} \u2192 ~{after_tokens:,} tokens]"))
        print(dim("─" * 60))
        for line in summary.splitlines():
            print(dim(f"  {line}"))
        print(dim("─" * 60))

    # ── Two-phase tool execution ────────────────────────────────────────
    #
    # Phase 1 — prepare: parse args, validate, build preview text (serial)
    # Phase 2 — approve: display all previews, single prompt (serial)
    # Phase 3 — execute: run approved tools (parallel if multiple)

    def _execute_tools(
        self, tool_calls: list[dict]
    ) -> tuple[list[tuple[str, str]], str | None]:
        """Execute tool calls with batch preview and approval.

        Returns (results, user_feedback) where user_feedback is an optional
        message the user typed alongside their approval (e.g. "y, use full path").
        """
        # Phase 1: prepare all tool calls
        items = [self._prepare_tool(tc) for tc in tool_calls]

        # Phase 2: display previews and prompt
        user_feedback = self._display_and_approve(items)

        # Phase 3: execute
        def run_one(item: dict) -> tuple[str, str]:
            if item.get("error"):
                return item["call_id"], item["error"]
            if item.get("denied"):
                return item["call_id"], item.get("denial_msg", "Denied by user")
            return item["execute"](item)

        if len(items) == 1:
            results = [run_one(items[0])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(run_one, items))

        # Post-plan gate: prompt user on main thread after plan completes
        for i, item in enumerate(items):
            if (
                item.get("func_name") == "plan"
                and not item.get("error")
                and not item.get("denied")
                and not self.auto_approve
            ):
                cid, output = results[i]
                # Show the plan content so user can review before deciding
                sys.stdout.write(f"\n{DIM}{'─' * 60}{RESET}\n")
                for line in output.splitlines():
                    sys.stdout.write(f"  {line}\n")
                sys.stdout.write(f"{DIM}{'─' * 60}{RESET}\n")
                sys.stdout.flush()
                try:
                    prompt_text = (
                        f"    \001{BOLD}\002Plan ready.\001{RESET}\002 "
                        f"\001{DIM}\002[enter to approve, or give feedback]"
                        f"\001{RESET}\002 "
                    )
                    resp = input(prompt_text).strip()
                except EOFError:
                    resp = ""
                except KeyboardInterrupt:
                    resp = "reject"
                if resp.lower() in ("n", "no", "reject"):
                    output += (
                        "\n\n---\nUser REJECTED this plan. Do not proceed "
                        "with implementation. Ask the user what they want instead."
                    )
                elif resp:
                    output += f"\n\n---\nUser feedback on this plan: {resp}"
                results[i] = (cid, output)

        return results, user_feedback

    def _prepare_tool(self, tc: dict) -> dict:
        """Parse a tool call and prepare preview info for display."""
        call_id = tc["id"]
        func_name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]

        # Map tool name → primary argument key for bare-string fallback
        _primary_key = {
            "bash": "command",
            "math": "code",
            "read_file": "path",
            "search": "query",
            "write_file": "content",
            "edit_file": "old_string",
            "man": "page",
            "web_fetch": "url",
            "web_search": "query",
            "task": "prompt",
            "plan": "prompt",
            "remember": "key",
            "recall": "query",
            "forget": "key",
        }

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            args = None
            # Fallback 1: regex-extract a known key from malformed JSON
            for key in (
                "command",
                "code",
                "content",
                "page",
                "path",
                "pattern",
                "prompt",
                "query",
                "url",
            ):
                m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_args)
                if m:
                    try:
                        val = json.loads('"' + m.group(1) + '"')
                    except (json.JSONDecodeError, Exception):
                        val = m.group(1)
                    args = {key: val}
                    break
            # Fallback 2: bare string (no JSON wrapper at all)
            if (
                args is None
                and raw_args.strip()
                and not raw_args.strip().startswith("{")
            ):
                pk = _primary_key.get(func_name)
                if pk:
                    args = {pk: raw_args}
            if args is None:
                preview = raw_args[:300] + ("..." if len(raw_args) > 300 else "")
                return {
                    "call_id": call_id,
                    "func_name": func_name,
                    "header": f"✗ {func_name}: {exc}",
                    "preview": f"    {RED}{preview}{RESET}",
                    "needs_approval": False,
                    "error": f"JSON parse error: {exc}\nRaw arguments: {raw_args[:500]}",
                }

        preparers = {
            "bash": self._prepare_bash,
            "read_file": self._prepare_read_file,
            "search": self._prepare_search,
            "write_file": self._prepare_write_file,
            "edit_file": self._prepare_edit_file,
            "math": self._prepare_math,
            "man": self._prepare_man,
            "web_fetch": self._prepare_web_fetch,
            "web_search": self._prepare_web_search,
            "task": self._prepare_task,
            "plan": self._prepare_plan,
            "remember": self._prepare_remember,
            "recall": self._prepare_recall,
            "forget": self._prepare_forget,
        }
        preparer = preparers.get(func_name)
        if not preparer:
            return {
                "call_id": call_id,
                "func_name": func_name,
                "header": f"✗ Unknown tool: {func_name}",
                "preview": "",
                "needs_approval": False,
                "error": f"Unknown tool: {func_name}",
            }
        return preparer(call_id, args)

    def _display_and_approve(self, items: list[dict]) -> str | None:
        """Display all tool previews and prompt for batch approval.

        Returns an optional user feedback message (text after y/n),
        or None if no feedback was given.
        """
        pending = [
            it for it in items if it.get("needs_approval") and not it.get("error")
        ]

        with self._print_lock:
            # Print all headers and previews
            for item in items:
                if item.get("error"):
                    sys.stdout.write(f"  {red(item['header'])}\n")
                else:
                    sys.stdout.write(f"  {yellow(item['header'])}\n")
                if item.get("preview"):
                    sys.stdout.write(item["preview"] + "\n")
            sys.stdout.flush()

            if not pending or self.auto_approve:
                return None

            # Prompt
            try:
                if len(pending) == 1:
                    label = pending[0].get("approval_label", pending[0]["func_name"])
                    prompt_text = (
                        f"    \001{BOLD}\002Allow {label}?\001{RESET}\002 "
                        f"\001{DIM}\002[y/n/a(lways), optional message]\001{RESET}\002 "
                    )
                else:
                    labels = ", ".join(
                        it.get("approval_label", it["func_name"]) for it in pending
                    )
                    prompt_text = (
                        f"    \001{BOLD}\002Allow {len(pending)} tools ({labels})?\001{RESET}\002 "
                        f"\001{DIM}\002[y/n/a(lways), optional message]\001{RESET}\002 "
                    )
                resp = input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                sys.stdout.write("\n")
                resp = "n"

            # Parse decision and optional feedback: "y, use absolute path"
            decision = resp.lower()
            feedback = None
            for sep in (",", " "):
                if sep in resp:
                    decision = resp[: resp.index(sep)].strip().lower()
                    feedback = resp[resp.index(sep) + 1 :].strip() or None
                    break

            if decision in ("a", "always"):
                self.auto_approve = True
            elif decision not in ("y", "yes"):
                denial_msg = "Denied by user"
                if feedback:
                    denial_msg += f": {feedback}"
                for item in pending:
                    item["denied"] = True
                    item["denial_msg"] = denial_msg
                return None  # feedback already in denial_msg

            return feedback

    # ── Prepare methods (build preview, validate, no side effects) ────

    def _prepare_bash(self, call_id: str, args: dict) -> dict:
        command = _sanitize_command(args.get("command", ""))
        if not command:
            return {
                "call_id": call_id,
                "func_name": "bash",
                "header": "✗ bash: empty command",
                "preview": "",
                "needs_approval": False,
                "error": "Error: empty command",
            }
        blocked = is_command_blocked(command)
        if blocked:
            return {
                "call_id": call_id,
                "func_name": "bash",
                "header": f"✗ {blocked}",
                "preview": "",
                "needs_approval": False,
                "error": blocked,
            }
        display_cmd = command.split("\n")[0]
        if "\n" in command:
            display_cmd += f" ... ({command.count(chr(10))} more lines)"
        return {
            "call_id": call_id,
            "func_name": "bash",
            "header": f"⚙ bash: {display_cmd}",
            "preview": "",
            "needs_approval": True,
            "approval_label": "bash",
            "execute": self._exec_bash,
            "command": command,
        }

    def _prepare_read_file(self, call_id: str, args: dict) -> dict:
        path = args.get("path", "")
        if not path:
            return {
                "call_id": call_id,
                "func_name": "read_file",
                "header": "✗ read_file: missing path",
                "preview": "",
                "needs_approval": False,
                "error": "Error: missing path",
            }
        path = os.path.expanduser(path)
        resolved = os.path.realpath(path)
        offset = args.get("offset")  # 1-based line number, or None
        limit = args.get("limit")  # max lines, or None
        # Coerce to int safely (model may send strings or floats)
        try:
            if offset is not None:
                offset = int(offset)
            if limit is not None:
                limit = int(limit)
        except (ValueError, TypeError):
            return {
                "call_id": call_id,
                "func_name": "read_file",
                "header": "✗ read_file: invalid offset/limit",
                "preview": "",
                "needs_approval": False,
                "error": (
                    f"Error: offset/limit must be integers "
                    f"(got offset={args.get('offset')!r}, "
                    f"limit={args.get('limit')!r})"
                ),
            }
        if offset is not None and offset < 1:
            return {
                "call_id": call_id,
                "func_name": "read_file",
                "header": "✗ read_file: offset must be >= 1",
                "preview": "",
                "needs_approval": False,
                "error": f"Error: offset must be >= 1 (got {offset})",
            }
        if limit is not None and limit < 1:
            return {
                "call_id": call_id,
                "func_name": "read_file",
                "header": "✗ read_file: limit must be >= 1",
                "preview": "",
                "needs_approval": False,
                "error": f"Error: limit must be >= 1 (got {limit})",
            }
        # Register early so a same-batch edit_file can pass the read guard.
        # If the file doesn't exist, _exec_read_file returns an error and
        # _exec_edit_file's re-read will also fail naturally.
        self._read_files.add(resolved)
        # Build header showing range if specified
        header = f"⚙ read_file: {path}"
        if offset is not None or limit is not None:
            start = offset or 1
            if limit is not None:
                header += f" (lines {start}-{start + limit - 1})"
            else:
                header += f" (from line {start})"
        return {
            "call_id": call_id,
            "func_name": "read_file",
            "header": header,
            "preview": "",
            "needs_approval": False,
            "execute": self._exec_read_file,
            "path": path,
            "offset": offset,
            "limit": limit,
        }

    def _prepare_search(self, call_id: str, args: dict) -> dict:
        pattern = args.get("query", "")
        if not pattern:
            return {
                "call_id": call_id,
                "func_name": "search",
                "header": "✗ search: missing query",
                "preview": "",
                "needs_approval": False,
                "error": "Error: missing query",
            }
        path = os.path.expanduser(args.get("path", "") or ".")
        preview = f"    {DIM}/{pattern}/ in {path}{RESET}"
        return {
            "call_id": call_id,
            "func_name": "search",
            "header": f"⚙ search: /{pattern}/ in {path}",
            "preview": preview,
            "needs_approval": False,
            "execute": self._exec_search,
            "pattern": pattern,
            "path": path,
        }

    def _prepare_write_file(self, call_id: str, args: dict) -> dict:
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return {
                "call_id": call_id,
                "func_name": "write_file",
                "header": "✗ write_file: missing path",
                "preview": "",
                "needs_approval": False,
                "error": "Error: missing path",
            }
        path = os.path.expanduser(path)
        resolved = os.path.realpath(path)
        exists = os.path.exists(resolved)
        is_overwrite = exists and resolved not in self._read_files

        # Build preview
        preview_parts = []
        if is_overwrite:
            preview_parts.append(
                f"    {YELLOW}Warning: overwriting existing file not previously read{RESET}"
            )
        text = content[:500]
        if len(content) > 500:
            text += f"\n... ({len(content)} chars total)"
        preview_parts.append(f"{DIM}{textwrap.indent(text, '    ')}{RESET}")

        return {
            "call_id": call_id,
            "func_name": "write_file",
            "header": f"⚙ write_file: {path} ({len(content)} chars)",
            "preview": "\n".join(preview_parts),
            "needs_approval": True,
            "approval_label": "overwrite_file" if is_overwrite else "write_file",
            "execute": self._exec_write_file,
            "path": path,
            "resolved": resolved,
            "content": content,
        }

    def _prepare_edit_file(self, call_id: str, args: dict) -> dict:
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        near_line = args.get("near_line")
        if isinstance(near_line, str):
            try:
                near_line = int(near_line)
            except ValueError:
                near_line = None
        if not path:
            return {
                "call_id": call_id,
                "func_name": "edit_file",
                "header": "✗ edit_file: missing path",
                "preview": "",
                "needs_approval": False,
                "error": "Error: missing path",
            }
        if not old_string:
            return {
                "call_id": call_id,
                "func_name": "edit_file",
                "header": "✗ edit_file: missing old_string",
                "preview": "",
                "needs_approval": False,
                "error": "Error: missing old_string",
            }
        path = os.path.expanduser(path)
        resolved = os.path.realpath(path)

        if resolved not in self._read_files:
            return {
                "call_id": call_id,
                "func_name": "edit_file",
                "header": f"✗ edit_file: {path}",
                "preview": "",
                "needs_approval": False,
                "error": f"Error: must read_file {path} before editing it",
            }

        # Pre-read to validate and build diff preview
        try:
            with open(path, "r") as f:
                content = f.read()
            occurrences = _find_occurrences(content, old_string)
            if len(occurrences) == 0:
                return {
                    "call_id": call_id,
                    "func_name": "edit_file",
                    "header": f"✗ edit_file: {path}",
                    "preview": "",
                    "needs_approval": False,
                    "error": f"Error: old_string not found in {path}",
                }
            if len(occurrences) > 1 and near_line is None:
                line_list = ", ".join(str(ln) for ln in occurrences)
                return {
                    "call_id": call_id,
                    "func_name": "edit_file",
                    "header": f"✗ edit_file: {path}",
                    "preview": "",
                    "needs_approval": False,
                    "error": (
                        f"Error: old_string found {len(occurrences)} times "
                        f"at lines {line_list} — use near_line to pick one"
                    ),
                }
        except FileNotFoundError:
            return {
                "call_id": call_id,
                "func_name": "edit_file",
                "header": f"✗ edit_file: {path}",
                "preview": "",
                "needs_approval": False,
                "error": f"Error: {path} not found",
            }
        except Exception as e:
            return {
                "call_id": call_id,
                "func_name": "edit_file",
                "header": f"✗ edit_file: {path}",
                "preview": "",
                "needs_approval": False,
                "error": f"Error editing {path}: {e}",
            }

        # Build diff preview
        preview_parts = []
        old_preview = old_string[:200] + ("..." if len(old_string) > 200 else "")
        new_preview = new_string[:200] + ("..." if len(new_string) > 200 else "")
        for line in old_preview.splitlines():
            preview_parts.append(f"    {RED}- {line}{RESET}")
        if new_string:
            for line in new_preview.splitlines():
                preview_parts.append(f"    {GREEN}+ {line}{RESET}")
        else:
            preview_parts.append(
                f"    {YELLOW}(deletion — {len(old_string)} chars removed){RESET}"
            )

        return {
            "call_id": call_id,
            "func_name": "edit_file",
            "header": f"⚙ edit_file: {path}",
            "preview": "\n".join(preview_parts),
            "needs_approval": True,
            "approval_label": "edit_file",
            "execute": self._exec_edit_file,
            "path": path,
            "resolved": resolved,
            "old_string": old_string,
            "new_string": new_string,
            "near_line": near_line,
        }

    def _prepare_math(self, call_id: str, args: dict) -> dict:
        code = args.get("code", "")
        if isinstance(code, list):
            code = "\n".join(code)
        if not code:
            return {
                "call_id": call_id,
                "func_name": "math",
                "header": "✗ math: empty code",
                "preview": "",
                "needs_approval": False,
                "error": "Error: no code provided",
            }
        # Show code preview
        display = code[:300]
        if len(code) > 300:
            display += f"\n... ({len(code)} chars total)"
        preview = f"{DIM}{textwrap.indent(display, '    ')}{RESET}"
        return {
            "call_id": call_id,
            "func_name": "math",
            "header": f"⚙ math: ({len(code)} chars)",
            "preview": preview,
            "needs_approval": True,
            "approval_label": "math",
            "execute": self._exec_math,
            "code": code,
        }

    def _prepare_man(self, call_id: str, args: dict) -> dict:
        """Prepare a man/info page lookup."""
        page = (args.get("page") or "").strip()
        if not page:
            return {
                "call_id": call_id,
                "func_name": "man",
                "header": "✗ man: empty page",
                "preview": "",
                "needs_approval": False,
                "error": "Error: no page name provided",
            }
        # Sanitize: only allow alphanumeric, dash, underscore, dot
        if not re.match(r"^[a-zA-Z0-9._-]+$", page):
            return {
                "call_id": call_id,
                "func_name": "man",
                "header": "✗ man: invalid page name",
                "preview": f"    {RED}{page}{RESET}",
                "needs_approval": False,
                "error": f"Error: invalid page name {page!r}",
            }
        section = (args.get("section") or "").strip()
        if section and not re.match(r"^[1-9][a-z]?$", section):
            section = ""
        label = f"{page}({section})" if section else page
        preview = f"    {DIM}{label}{RESET}"
        return {
            "call_id": call_id,
            "func_name": "man",
            "header": f"⚙ man: {label}",
            "preview": preview,
            "needs_approval": False,
            "execute": self._exec_man,
            "page": page,
            "section": section,
        }

    def _prepare_web_fetch(self, call_id: str, args: dict) -> dict:
        url = args.get("url", "").strip()
        question = args.get("question", "").strip()
        if not url:
            return {
                "call_id": call_id,
                "func_name": "web_fetch",
                "header": "✗ web_fetch: empty url",
                "preview": "",
                "needs_approval": False,
                "error": "Error: no URL provided",
            }
        if not question:
            return {
                "call_id": call_id,
                "func_name": "web_fetch",
                "header": "✗ web_fetch: empty question",
                "preview": "",
                "needs_approval": False,
                "error": "Error: no question provided",
            }
        if not url.startswith(("http://", "https://")):
            return {
                "call_id": call_id,
                "func_name": "web_fetch",
                "header": "✗ web_fetch: invalid url",
                "preview": f"    {RED}{url}{RESET}",
                "needs_approval": False,
                "error": f"Error: URL must start with http:// or https:// (got {url!r})",
            }
        # SSRF protection: reject private/link-local/metadata IPs
        try:
            hostname = urlparse(url).hostname
            if hostname:
                addr = ipaddress.ip_address(socket.gethostbyname(hostname))
                if addr.is_private or addr.is_loopback or addr.is_link_local:
                    return {
                        "call_id": call_id,
                        "func_name": "web_fetch",
                        "header": "✗ web_fetch: blocked (private network)",
                        "preview": f"    {RED}{url}{RESET}",
                        "needs_approval": False,
                        "error": f"Error: URL resolves to private/internal address ({addr})",
                    }
        except (socket.gaierror, ValueError):
            pass  # DNS failure handled later during actual fetch
        q_preview = question[:200] + ("..." if len(question) > 200 else "")
        preview = f"    {DIM}{url}\n    Q: {q_preview}{RESET}"
        return {
            "call_id": call_id,
            "func_name": "web_fetch",
            "header": f"⚙ web_fetch: {url[:80]}",
            "preview": preview,
            "needs_approval": True,
            "approval_label": "web_fetch",
            "execute": self._exec_web_fetch,
            "url": url,
            "question": question,
        }

    def _prepare_web_search(self, call_id: str, args: dict) -> dict:
        """Prepare a web search via Tavily for approval."""
        query = (args.get("query") or "").strip()
        if not query:
            return {
                "call_id": call_id,
                "func_name": "web_search",
                "header": "✗ web_search: empty query",
                "preview": "",
                "needs_approval": False,
                "error": "Error: no query provided",
            }
        if not _get_tavily_key():
            return {
                "call_id": call_id,
                "func_name": "web_search",
                "header": "✗ web_search: no API key",
                "preview": "",
                "needs_approval": False,
                "error": (
                    "Error: Tavily API key not configured. "
                    "Set it in ~/.config/pcode/tavily_key or $TAVILY_API_KEY. "
                    "Use web_fetch with a direct URL as an alternative."
                ),
            }
        try:
            max_results = min(max(int(args.get("max_results") or 5), 1), 20)
        except (ValueError, TypeError):
            max_results = 5
        topic = args.get("topic", "general") or "general"
        if topic not in ("general", "news", "finance"):
            topic = "general"
        q_preview = query[:200] + ("..." if len(query) > 200 else "")
        preview = f"    {DIM}{q_preview}{RESET}"
        return {
            "call_id": call_id,
            "func_name": "web_search",
            "header": f"⚙ web_search: {query[:80]}",
            "preview": preview,
            "needs_approval": True,
            "approval_label": "web_search",
            "execute": self._exec_web_search,
            "query": query,
            "max_results": max_results,
            "topic": topic,
        }

    def _prepare_task(self, call_id: str, args: dict) -> dict:
        """Prepare a general-purpose sub-agent task for approval."""
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return {
                "call_id": call_id,
                "func_name": "task",
                "header": "✗ task: empty prompt",
                "preview": "",
                "needs_approval": False,
                "error": "Error: empty prompt",
            }
        preview_text = prompt[:300] + ("..." if len(prompt) > 300 else "")
        return {
            "call_id": call_id,
            "func_name": "task",
            "header": "⚙ task (autonomous agent)",
            "preview": f"    {DIM}{preview_text}{RESET}",
            "needs_approval": True,
            "approval_label": "task",
            "execute": self._exec_task,
            "prompt": prompt,
        }

    def _prepare_plan(self, call_id: str, args: dict) -> dict:
        """Prepare a planning agent for approval."""
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return {
                "call_id": call_id,
                "func_name": "plan",
                "header": "✗ plan: empty prompt",
                "preview": "",
                "needs_approval": False,
                "error": "Error: empty prompt",
            }
        preview_text = prompt[:300] + ("..." if len(prompt) > 300 else "")
        return {
            "call_id": call_id,
            "func_name": "plan",
            "header": "⚙ plan (planning agent)",
            "preview": f"    {DIM}{preview_text}{RESET}",
            "needs_approval": True,
            "approval_label": "plan",
            "execute": self._exec_plan,
            "prompt": prompt,
        }

    def _prepare_remember(self, call_id: str, args: dict) -> dict:
        """Prepare a remember (save memory) action."""
        key = _normalize_key((args.get("key") or "").strip())
        value = (args.get("value") or "").strip()
        if not key or not value:
            return {
                "call_id": call_id,
                "func_name": "remember",
                "header": "✗ remember: requires key and value",
                "preview": "",
                "needs_approval": False,
                "error": "Error: both 'key' and 'value' are required",
            }
        return {
            "call_id": call_id,
            "func_name": "remember",
            "header": f"⚙ remember: {key}",
            "preview": "",
            "needs_approval": False,
            "execute": self._exec_remember,
            "key": key,
            "value": value,
        }

    def _prepare_forget(self, call_id: str, args: dict) -> dict:
        """Prepare a forget (delete memory) action."""
        key = _normalize_key((args.get("key") or "").strip())
        if not key:
            return {
                "call_id": call_id,
                "func_name": "forget",
                "header": "✗ forget: empty key",
                "preview": "",
                "needs_approval": False,
                "error": "Error: key is required",
            }
        return {
            "call_id": call_id,
            "func_name": "forget",
            "header": f"⚙ forget: {key}",
            "preview": "",
            "needs_approval": False,
            "execute": self._exec_forget,
            "key": key,
        }

    def _prepare_recall(self, call_id: str, args: dict) -> dict:
        """Prepare a recall action."""
        query = (args.get("query") or "").strip()
        limit = args.get("limit", 20)
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                limit = 20
        return {
            "call_id": call_id,
            "func_name": "recall",
            "header": f"⚙ recall{': ' + query[:80] if query else ''}",
            "preview": "",
            "needs_approval": False,
            "execute": self._exec_recall,
            "query": query,
            "limit": min(limit, 50),
        }

    # ── Execute methods (do the work, display output) ─────────────────

    def _exec_bash(self, item: dict) -> tuple[str, str]:
        """Execute a bash command via temp script."""
        call_id, command = item["call_id"], item["command"]
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write(command)
                script_path = f.name
            try:
                result = subprocess.run(
                    ["bash", script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.tool_timeout,
                )
            finally:
                os.unlink(script_path)
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            output = output.strip()
            original_len = len(output)

            if original_len > 10_000:
                output = (
                    output[:5_000]
                    + f"\n\n... [{original_len - 10_000} chars truncated] ...\n\n"
                    + output[-5_000:]
                )

            with self._print_lock:
                preview = output[:500]
                if original_len > 500:
                    preview += f"\n  ... ({original_len} chars total)"
                if preview:
                    indented = textwrap.indent(preview, "    ")
                    sys.stdout.write(f"{DIM}{indented}{RESET}\n")
                    sys.stdout.flush()

            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            return call_id, output if output else "(no output)"

        except subprocess.TimeoutExpired:
            msg = f"Command timed out after {self.tool_timeout}s"
            with self._print_lock:
                sys.stdout.write(f"    {red(msg)}\n")
                sys.stdout.flush()
            return call_id, msg
        except Exception as e:
            msg = f"Error executing command: {e}"
            with self._print_lock:
                sys.stdout.write(f"    {red(msg)}\n")
                sys.stdout.flush()
            return call_id, msg

    def _exec_read_file(self, item: dict) -> tuple[str, str]:
        """Read a file and return numbered lines, optionally sliced."""
        call_id, path = item["call_id"], item["path"]
        offset = item.get("offset")  # 1-based, or None
        limit = item.get("limit")  # max lines, or None
        resolved = os.path.realpath(path)

        try:
            with open(path, "r") as f:
                all_lines = f.readlines()
        except FileNotFoundError:
            self._read_files.discard(resolved)
            return call_id, f"Error: {path} not found"
        except Exception as e:
            self._read_files.discard(resolved)
            return call_id, f"Error reading {path}: {e}"

        self._read_files.add(resolved)
        total_lines = len(all_lines)

        # Slice if offset/limit specified
        start = max(1, offset or 1)
        if limit is not None:
            lines = all_lines[start - 1 : start - 1 + limit]
        else:
            lines = all_lines[start - 1 :]

        numbered = []
        for i, line in enumerate(lines, start=start):
            numbered.append(f"{i:>4}\t{line.rstrip()}")
        output = "\n".join(numbered)
        original_len = len(output)

        if original_len > 10_000:
            output = (
                output[:5_000]
                + f"\n\n... [{original_len - 10_000} chars truncated] ...\n\n"
                + output[-5_000:]
            )

        with self._print_lock:
            desc = f"{len(lines)} lines"
            if offset is not None or limit is not None:
                end = start + len(lines) - 1
                desc += f" (lines {start}-{end} of {total_lines})"
            sys.stdout.write(f"    {DIM}{desc}{RESET}\n")
            sys.stdout.flush()

        return call_id, output if output else "(empty file)"

    def _exec_search(self, item: dict) -> tuple[str, str]:
        """Search file contents for a regex pattern using grep."""
        call_id = item["call_id"]
        pattern, path = item["pattern"], item["path"]
        try:
            result = subprocess.run(
                [
                    "grep",
                    "-rn",
                    "-I",
                    "-E",
                    "-m",
                    "200",  # max matches per file
                    "--color=never",  # no ANSI codes in output
                    "--",
                    pattern,
                    path,  # -- prevents pattern as flag
                ],
                capture_output=True,
                text=True,
                timeout=self.tool_timeout,
            )
            output = result.stdout.strip()
            if result.returncode == 1:
                output = "(no matches)"
            elif result.returncode > 1:
                output = (
                    result.stderr.strip() or f"grep error (exit {result.returncode})"
                )

            # Count matches BEFORE truncation
            match_count = (
                output.count("\n") + 1 if result.returncode == 0 and output else 0
            )

            original_len = len(output)
            if original_len > 10_000:
                output = (
                    output[:5_000]
                    + f"\n\n... [{original_len - 10_000} chars truncated] ...\n\n"
                    + output[-5_000:]
                )

            with self._print_lock:
                desc = f"{match_count} matches" if match_count else "no matches"
                if original_len > 500:
                    desc += f" ({original_len} chars)"
                sys.stdout.write(f"    {DIM}{desc}{RESET}\n")
                sys.stdout.flush()

            return call_id, output

        except subprocess.TimeoutExpired:
            msg = f"Search timed out after {self.tool_timeout}s"
            with self._print_lock:
                sys.stdout.write(f"    {red(msg)}\n")
                sys.stdout.flush()
            return call_id, msg
        except Exception as e:
            msg = f"Search error: {e}"
            with self._print_lock:
                sys.stdout.write(f"    {red(msg)}\n")
                sys.stdout.flush()
            return call_id, msg

    # Tools the agent can auto-execute without user approval (read-only).
    # bash, write_file, edit_file are excluded — the user approval prompt
    # is the primary security boundary against prompt injection.
    _AGENT_AUTO_TOOLS = {
        "read_file",
        "search",
        "math",
        "man",
        "web_fetch",
        "web_search",
    }

    def _run_agent(
        self,
        agent_messages: list[dict],
        label: str = "agent",
        tools: list[dict] | None = None,
        auto_tools: set[str] | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Run an autonomous agent loop.

        Args:
            agent_messages: Pre-built message list (system + developer + user).
            label: Display prefix for progress lines ("agent" or "plan").
            tools: Tool definitions to send to the API. Defaults to AGENT_TOOLS (read-only).
            auto_tools: Set of tool names the agent may execute. Defaults to _AGENT_AUTO_TOOLS.
            reasoning_effort: Override reasoning effort for this agent.

        Returns:
            Final content string from the agent.
        """
        if tools is None:
            tools = AGENT_TOOLS
        if auto_tools is None:
            auto_tools = self._AGENT_AUTO_TOOLS
        max_tool_turns = 20

        kwargs = dict(self._chat_template_kwargs_base)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        def _api_call(messages, _tools=tools):
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=_tools,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body={
                    "chat_template_kwargs": kwargs,
                },
            )

        for turn in range(max_tool_turns):
            response = _api_call(agent_messages)
            choice = response.choices[0]
            assistant_msg = choice.message

            # Build message dict for agent history
            msg_dict = {
                "role": "assistant",
                "content": assistant_msg.content or "",
            }
            if assistant_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
            agent_messages.append(msg_dict)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content or "(no output)"
                with self._print_lock:
                    sys.stdout.write(
                        f"  {DIM}[{label} done] {len(content)} chars{RESET}\n"
                    )
                    sys.stdout.flush()
                return content

            # Execute tools sequentially (not parallel) to avoid
            # concurrent _read_files mutation from worker threads.
            tool_names = {t["function"]["name"] for t in tools}
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name

                # Guard 1: block recursive agent calls.
                if tool_name in ("task", "plan"):
                    output = "Error: agents cannot spawn further agents"
                # Guard 2: tool not in this agent's API tool list.
                elif tool_name not in tool_names:
                    output = (
                        f"Error: tool '{tool_name}' is not available in "
                        f"agent mode. "
                        f"Available: {', '.join(sorted(tool_names))}"
                    )
                else:
                    tc_dict = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    prepared = self._prepare_tool(tc_dict)

                    with self._print_lock:
                        lbl = prepared.get("header", tool_name)
                        sys.stdout.write(
                            f"  {DIM}[{label} turn {turn + 1}] {lbl}{RESET}\n"
                        )
                        sys.stdout.flush()

                    if prepared.get("error"):
                        output = prepared["error"]
                    # Auto-execute tools in the auto_tools set.
                    elif tool_name in auto_tools:
                        _, output = prepared["execute"](prepared)
                    # Tools not in auto_tools require user approval.
                    elif "execute" in prepared:
                        self._display_and_approve([prepared])
                        if prepared.get("denied"):
                            output = prepared.get("denial_msg", "Denied by user")
                        else:
                            _, output = prepared["execute"](prepared)
                    else:
                        output = f"Unknown tool: {tool_name}"

                agent_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

        # Exhausted tool turns — force a final synthesis response.
        with self._print_lock:
            sys.stdout.write(
                f"  {DIM}[{label}] turn limit reached, requesting synthesis...{RESET}\n"
            )
            sys.stdout.flush()
        agent_messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the tool call limit. "
                    "Provide your complete response now using "
                    "the information you have gathered so far."
                ),
            }
        )
        response = _api_call(agent_messages, _tools=[])
        content = response.choices[0].message.content or "(no output)"
        with self._print_lock:
            sys.stdout.write(f"  {DIM}[{label} done] {len(content)} chars{RESET}\n")
            sys.stdout.flush()
        return content

    # Task agent read-only tools auto-execute without approval.
    # bash, write_file, edit_file are in TASK_AGENT_TOOLS (the API tool list)
    # but NOT here — they go through _display_and_approve for user approval.
    _TASK_AUTO_TOOLS = {
        "read_file",
        "search",
        "math",
        "man",
        "web_fetch",
        "web_search",
    }

    def _exec_task(self, item: dict) -> tuple[str, str]:
        """Delegate to a general-purpose autonomous sub-agent."""
        call_id, prompt = item["call_id"], item["prompt"]
        task_instruction = {
            "role": "developer",
            "content": (
                "# Task Agent\n\n"
                "You are an autonomous task agent with full tool access. "
                "You can use bash, read_file, write_file, edit_file, search, "
                "math, web_fetch, and web_search.\n\n"
                "1. **Follow through on actions:** Do not describe changes — "
                "use the tools to make them. After read_file, call edit_file "
                "or write_file.\n\n"
                "2. **Tool selection:**\n"
                "   - Use read_file before edit_file on existing files.\n"
                "   - Use write_file for new files (not bash).\n"
                "   - Use bash for shell commands (git, python, tests).\n"
                "   - Use search to find code across files.\n\n"
                "3. **Complete the task fully.** Do not ask follow-up "
                "questions — execute the work as described in the prompt."
            ),
        }
        agent_messages = list(self._agent_system_messages) + [
            task_instruction,
            {"role": "user", "content": prompt},
        ]
        try:
            return call_id, self._run_agent(
                agent_messages,
                label="task",
                tools=TASK_AGENT_TOOLS,
                auto_tools=self._TASK_AUTO_TOOLS,
            )
        except KeyboardInterrupt:
            return call_id, "(task interrupted by user)"
        except Exception as e:
            with self._print_lock:
                sys.stdout.write(f"  {DIM}[task error] {e}{RESET}\n")
                sys.stdout.flush()
            return call_id, f"Task error: {e}"

    def _exec_plan(self, item: dict) -> tuple[str, str]:
        """Run a planning agent and write the result to .plan.md."""
        call_id, prompt = item["call_id"], item["prompt"]
        plan_path = ".plan.md"

        plan_instruction = {
            "role": "developer",
            "content": (
                "# Planning Agent\n\n"
                "1. **Exploration first:** Use read_file and search to understand the "
                "codebase before writing any plan. Do not guess at file structure or "
                "assume patterns — verify them.\n\n"
                "2. **Output format** — use these exact sections:\n"
                "   - **## Goal**: What we're building and why (1-2 sentences).\n"
                "   - **## Current State**: Relevant existing code and patterns found "
                "during exploration. Reference specific file paths and line numbers.\n"
                "   - **## Plan**: Numbered steps with specific files and functions "
                "to create or modify. Each step should be actionable.\n"
                "   - **## Risks**: Edge cases, breaking changes, or unknowns.\n\n"
                "3. **Specificity rules:**\n"
                "   - Reference file paths, line numbers, and function names — not vague areas.\n"
                "   - Each plan step should name the exact file(s) and describe the change.\n"
                "   - Identify dependencies between steps (what must happen first).\n\n"
                "4. **Common mistakes to avoid:**\n"
                "   - Writing a plan without reading the codebase first\n"
                '   - Vague steps like "update the code" without naming files or functions\n'
                "   - Ignoring existing patterns or conventions in the codebase\n"
                "   - Planning changes to files that don't exist"
            ),
        }
        agent_messages = list(self._agent_system_messages) + [
            plan_instruction,
            {"role": "user", "content": prompt},
        ]

        try:
            content = self._run_agent(
                agent_messages, label="plan", reasoning_effort="high"
            )
        except KeyboardInterrupt:
            return call_id, "(plan interrupted by user)"
        except Exception as e:
            with self._print_lock:
                sys.stdout.write(f"  {DIM}[plan error] {e}{RESET}\n")
                sys.stdout.flush()
            return call_id, f"Plan error: {e}"

        # Write to file separately — always return content even if write fails
        try:
            with open(plan_path, "w") as f:
                f.write(content)
            with self._print_lock:
                sys.stdout.write(f"  {DIM}Plan written to {plan_path}{RESET}\n")
                sys.stdout.flush()
        except OSError as e:
            with self._print_lock:
                sys.stdout.write(
                    f"  {DIM}[plan] could not write {plan_path}: {e}{RESET}\n"
                )
                sys.stdout.flush()

        return call_id, content

    def _exec_remember(self, item: dict) -> tuple[str, str]:
        """Save a persistent memory."""
        call_id, key, value = item["call_id"], item["key"], item["value"]
        try:
            conn = _open_db()
            try:
                existing = conn.execute(
                    "SELECT value FROM memories WHERE key = ?", (key,)
                ).fetchone()
                conn.execute(
                    "INSERT OR REPLACE INTO memories (key, value, created, updated) "
                    "VALUES (?, ?, COALESCE("
                    "  (SELECT created FROM memories WHERE key = ?), "
                    "  datetime('now')"
                    "), datetime('now'))",
                    (key, value, key),
                )
                conn.commit()
                self._init_system_messages()
                if existing:
                    msg = f"Updated memory: {key} = {value} (was: {existing[0]})"
                else:
                    msg = f"Saved memory: {key} = {value}"
                with self._print_lock:
                    sys.stdout.write(f"    {DIM}{msg}{RESET}\n")
                    sys.stdout.flush()
                return call_id, msg
            finally:
                conn.close()
        except Exception as e:
            return call_id, f"Error: {e}"

    def _exec_forget(self, item: dict) -> tuple[str, str]:
        """Remove a persistent memory by key."""
        call_id, key = item["call_id"], item["key"]
        try:
            conn = _open_db()
            try:
                cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
                conn.commit()
                if cursor.rowcount == 0:
                    msg = f"Error: memory '{key}' not found"
                else:
                    self._init_system_messages()
                    msg = f"Forgot: {key}"
                with self._print_lock:
                    sys.stdout.write(f"    {DIM}{msg}{RESET}\n")
                    sys.stdout.flush()
                return call_id, msg
            finally:
                conn.close()
        except Exception as e:
            return call_id, f"Error: {e}"

    def _exec_recall(self, item: dict) -> tuple[str, str]:
        """Search memories and conversation history."""
        call_id = item["call_id"]
        query, limit = item["query"], item["limit"]
        parts: list[str] = []

        # Memories: list all (no query) or search (with query)
        try:
            conn = _open_db()
            try:
                if not query:
                    rows = conn.execute(
                        "SELECT key, value FROM memories ORDER BY key"
                    ).fetchall()
                else:
                    terms = query.split()
                    clauses = []
                    params: list[str] = []
                    for t in terms:
                        escaped = _escape_like(t)
                        clauses.append(
                            "(key LIKE ? ESCAPE '\\' OR value LIKE ? ESCAPE '\\')"
                        )
                        params.extend([f"%{escaped}%", f"%{escaped}%"])
                    rows = conn.execute(
                        "SELECT key, value FROM memories WHERE "
                        + " AND ".join(clauses)
                        + " ORDER BY key",
                        params,
                    ).fetchall()
                if rows:
                    parts.append(
                        "Memories:\n" + "\n".join(f"  {k}={v}" for k, v in rows)
                    )
                elif not query:
                    parts.append("No memories stored.")
            finally:
                conn.close()
        except Exception:
            pass

        # Conversations: only when a query is provided
        if query:
            conv_rows = _search_history(query, limit)
            if conv_rows:
                lines = []
                for ts, sid, role, content, tool_name in conv_rows:
                    label = f"{role}({tool_name})" if tool_name else role
                    text = (content or "")[:500]
                    if content and len(content) > 500:
                        text += "..."
                    lines.append(f"[{ts} {sid}] {label}: {text}")
                parts.append(
                    f"Conversations ({len(conv_rows)} matches):\n" + "\n".join(lines)
                )

        output = "\n\n".join(parts) if parts else f"No results for '{query}'."
        with self._print_lock:
            sys.stdout.write(f"    {DIM}{output}{RESET}\n")
            sys.stdout.flush()
        return call_id, output

    def _exec_write_file(self, item: dict) -> tuple[str, str]:
        """Write content to a file, creating parent directories as needed."""
        call_id = item["call_id"]
        path, content, resolved = item["path"], item["content"], item["resolved"]
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            self._read_files.add(resolved)
            return call_id, f"Wrote {len(content)} chars to {path}"
        except Exception as e:
            return call_id, f"Error writing {path}: {e}"

    def _exec_edit_file(self, item: dict) -> tuple[str, str]:
        """Replace an exact string in a file (re-reads to avoid TOCTOU).

        When near_line is set, picks the occurrence nearest that line
        instead of requiring uniqueness.
        """
        call_id = item["call_id"]
        path, old_string, new_string = (
            item["path"],
            item["old_string"],
            item["new_string"],
        )
        near_line = item.get("near_line")
        try:
            with open(path, "r") as f:
                content = f.read()
            occurrences = _find_occurrences(content, old_string)
            if len(occurrences) == 0:
                return (
                    call_id,
                    f"Error: old_string no longer found in {path} (file changed)",
                )
            if len(occurrences) > 1 and near_line is None:
                line_list = ", ".join(str(ln) for ln in occurrences)
                return (
                    call_id,
                    f"Error: old_string found {len(occurrences)} times "
                    f"at lines {line_list} (file changed)",
                )
            if near_line is not None and len(occurrences) > 1:
                # Replace only the occurrence nearest to near_line
                idx = _pick_nearest(content, old_string, near_line)
                content = content[:idx] + new_string + content[idx + len(old_string) :]
            else:
                content = content.replace(old_string, new_string, 1)
            with open(path, "w") as f:
                f.write(content)
            return call_id, f"Edited {path}: replaced 1 occurrence"
        except Exception as e:
            return call_id, f"Error writing {path}: {e}"

    def _exec_math(self, item: dict) -> tuple[str, str]:
        """Execute Python code in sandboxed subprocess."""
        call_id, code = item["call_id"], item["code"]
        output, is_error = _execute_math_sandboxed(code, timeout=self.tool_timeout)
        original_len = len(output)

        if original_len > 10_000:
            output = (
                output[:5_000]
                + f"\n\n... [{original_len - 10_000} chars truncated] ...\n\n"
                + output[-5_000:]
            )

        with self._print_lock:
            preview = output[:500]
            if original_len > 500:
                preview += f"\n  ... ({original_len} chars total)"
            if preview:
                indented = textwrap.indent(preview, "    ")
                color = RED if is_error else DIM
                sys.stdout.write(f"{color}{indented}{RESET}\n")
                sys.stdout.flush()

        if is_error:
            return call_id, f"Error:\n{output}"
        return call_id, output if output else "(no output)"

    def _exec_man(self, item: dict) -> tuple[str, str]:
        """Look up a man or info page."""
        call_id = item["call_id"]
        page = item["page"]
        section = item.get("section", "")

        # Try man first, fall back to info
        cmd = ["man"]
        if section:
            cmd.append(section)
        cmd.append(page)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "MANWIDTH": "80", "MAN_KEEP_FORMATTING": "0"},
            )
            if result.returncode == 0 and result.stdout.strip():
                # Strip formatting: backspace overstrikes and ANSI escapes
                text = re.sub(r".\x08", "", result.stdout)
                text = re.sub(r"\x1b\[[0-9;]*m", "", text)
            else:
                # Fall back to info
                result = subprocess.run(
                    ["info", page],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    text = result.stdout
                else:
                    msg = f"No man or info page found for '{page}'"
                    with self._print_lock:
                        sys.stdout.write(f"    {DIM}{msg}{RESET}\n")
                        sys.stdout.flush()
                    return call_id, msg
        except FileNotFoundError:
            return call_id, "Error: man command not available"
        except subprocess.TimeoutExpired:
            return call_id, "Error: man page lookup timed out"

        # Truncate very long pages
        if len(text) > 30_000:
            total = len(text)
            text = (
                text[:30_000]
                + f"\n\n... [truncated: showing first 30000 of {total} chars. "
                f"Use a specific section number to narrow results.]"
            )

        with self._print_lock:
            sys.stdout.write(f"    {DIM}{len(text)} chars{RESET}\n")
            sys.stdout.flush()

        return call_id, text

    def _exec_web_fetch(self, item: dict) -> tuple[str, str]:
        """Fetch a URL, then summarize/extract using an API call."""
        call_id, url = item["call_id"], item["url"]
        question = item.get("question", "Summarize the key content of this page.")

        # Phase 1: fetch the URL
        try:
            req = Request(url, headers={"User-Agent": "chat.py/1.0"})
            with urlopen(req, timeout=self.tool_timeout) as resp:
                ct = resp.headers.get_content_type() or ""
                charset = resp.headers.get_content_charset() or "utf-8"
                raw = resp.read(10 * 1024 * 1024)  # 10 MB cap
                text = raw.decode(charset, errors="replace")

            if "html" in ct:
                text = _strip_html(text)

        except (URLError, TimeoutError) as e:
            msg = f"Fetch failed: {e}"
            with self._print_lock:
                sys.stdout.write(f"    {red(msg)}\n")
                sys.stdout.flush()
            return call_id, msg
        except ValueError as e:
            msg = f"Invalid URL: {e}"
            with self._print_lock:
                sys.stdout.write(f"    {red(msg)}\n")
                sys.stdout.flush()
            return call_id, msg
        except Exception as e:
            msg = f"Error fetching URL: {e}"
            with self._print_lock:
                sys.stdout.write(f"    {red(msg)}\n")
                sys.stdout.flush()
            return call_id, msg

        if not text.strip():
            return call_id, "(empty response from URL)"

        original_len = len(text)
        with self._print_lock:
            sys.stdout.write(
                f"    {DIM}fetched {original_len} chars, extracting...{RESET}\n"
            )
            sys.stdout.flush()

        # Phase 2: truncate for summarization context
        max_content = 50_000
        if len(text) > max_content:
            text = (
                text[: max_content // 2]
                + f"\n\n... [{len(text) - max_content} chars omitted] ...\n\n"
                + text[-(max_content // 2) :]
            )

        # Phase 3: summarization API call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a web content extraction assistant. "
                            "Answer the user's question using ONLY the "
                            "provided page content. Be concise and factual. "
                            "If the content doesn't contain the answer, say so."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Page URL: {url}\n"
                            f"Page content ({original_len} chars):\n\n"
                            f"{text}\n\n---\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                max_completion_tokens=2000,
                temperature=0.2,
            )
            answer = response.choices[0].message.content or "(no answer)"
        except Exception as e:
            answer = (
                f"Extraction failed (page was fetched but summarization errored): {e}"
            )

        with self._print_lock:
            preview = answer[:300]
            if len(answer) > 300:
                preview += "..."
            indented = textwrap.indent(preview, "    ")
            sys.stdout.write(f"{DIM}{indented}{RESET}\n")
            sys.stdout.flush()

        return call_id, answer

    def _exec_web_search(self, item: dict) -> tuple[str, str]:
        """Search the web via Tavily API."""
        call_id = item["call_id"]
        query = item["query"]
        max_results = item.get("max_results", 5)
        topic = item.get("topic", "general")
        api_key = _get_tavily_key()

        payload = json.dumps(
            {
                "query": query,
                "max_results": max_results,
                "topic": topic,
                "include_answer": True,
            }
        ).encode()
        req = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.tool_timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            msg = f"Tavily search failed: {e}"
            with self._print_lock:
                sys.stdout.write(f"    {red(msg)}\n")
                sys.stdout.flush()
            return call_id, msg

        parts: list[str] = []
        answer = (data.get("answer") or "").strip()
        if answer:
            parts.append(f"Answer: {answer}")

        results = data.get("results") or []
        if results:
            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                content = (r.get("content") or "")[:500]
                lines.append(f"{i}. [{title}]({url})\n   {content}")
            parts.append("\n".join(lines))

        output = "\n\n".join(parts) if parts else f"No results for '{query}'."

        with self._print_lock:
            preview = output[:400]
            if len(output) > 400:
                preview += "..."
            indented = textwrap.indent(preview, "    ")
            sys.stdout.write(f"{DIM}{indented}{RESET}\n")
            sys.stdout.flush()

        return call_id, output

    def handle_command(self, cmd_line: str) -> bool:
        """Handle slash commands. Returns True if should exit."""
        parts = cmd_line.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            return True

        elif cmd == "/persona":
            if not arg:
                if self.persona:
                    print(f"Current persona: {cyan(self.persona)}")
                else:
                    print("No persona set. Usage: /persona <name>")
            else:
                self.persona = arg.strip()
                self._init_system_messages()
                print(f"Switched persona to {cyan(self.persona)}")

        elif cmd == "/instructions":
            if not arg:
                if self.instructions:
                    print(f"Current instructions: {self.instructions[:100]}...")
                else:
                    print("No instructions set. Usage: /instructions <text>")
            else:
                self.instructions = arg.strip()
                self._init_system_messages()
                print(f"Instructions updated.")

        elif cmd in ("/clear", "/new"):
            self.messages.clear()
            self._read_files.clear()
            self.md = MarkdownRenderer()
            self._last_usage = None
            self._msg_tokens = []
            self._chars_per_token = 4.0
            try:
                db = _open_db()
                db.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (self._session_id,),
                )
                db.commit()
                db.close()
            except Exception:
                pass
            print("History cleared.")

        elif cmd == "/history":
            query = arg.strip() if arg else None
            if query:
                rows = _search_history(query, limit=20)
                if not rows:
                    print(f"No results for {query!r}")
                else:
                    print(f"Found {len(rows)} result(s) for {query!r}:\n")
                    for ts, sid, role, content, tool_name in rows:
                        label = tool_name if tool_name else role
                        text = (content or "")[:200]
                        print(f"  {dim(ts)} {dim(sid)} {bold(label)}: {text}")
            else:
                # Show recent conversations (last 20 messages)
                rows = _search_history_recent(limit=20)
                if not rows:
                    print("No conversation history yet.")
                else:
                    print("Recent history:\n")
                    for ts, sid, role, content, tool_name in rows:
                        label = tool_name if tool_name else role
                        text = (content or "")[:200]
                        print(f"  {dim(ts)} {dim(sid)} {bold(label)}: {text}")

        elif cmd == "/model":
            print(f"Model: {cyan(self.model)}")

        elif cmd == "/raw":
            self.show_reasoning = not self.show_reasoning
            state = "on" if self.show_reasoning else "off"
            print(f"Reasoning display: {bold(state)}")

        elif cmd == "/reason":
            valid = ("low", "medium", "high")
            aliases = {"med": "medium", "lo": "low", "hi": "high"}
            if not arg:
                print(f"Reasoning effort: {cyan(self.reasoning_effort)}")
            else:
                value = aliases.get(arg.lower(), arg.lower())
                if value in valid:
                    self.reasoning_effort = value
                    self._init_system_messages()
                    print(f"Reasoning effort set to {cyan(self.reasoning_effort)}")
                else:
                    print(f"Invalid. Choose from: {', '.join(valid)}")

        elif cmd == "/compact":
            self._compact_messages()

        elif cmd == "/creative":
            self.creative_mode = not self.creative_mode
            self._init_system_messages()
            # Clear history when toggling ON if it contains tool messages,
            # because the API rejects tool-call history without tool definitions
            if self.creative_mode and any(
                m.get("tool_calls") or m.get("role") == "tool" for m in self.messages
            ):
                self.messages.clear()
                self._read_files.clear()
                self._msg_tokens.clear()
                print(
                    dim(
                        "[history cleared — creative mode is incompatible with tool history]"
                    )
                )
            state = "on" if self.creative_mode else "off"
            print(
                f"Creative mode: {bold(state)} (tools {'disabled' if self.creative_mode else 'enabled'})"
            )

        elif cmd == "/debug":
            self.debug = not self.debug
            state = "on" if self.debug else "off"
            print(f"Debug mode: {bold(state)} (prints raw SSE deltas)")

        elif cmd == "/help":
            print(
                f"""
{bold("Slash Commands:")}
  /persona <name>       Set persona (system message)
  /instructions <text>  Set developer instructions
  /clear, /new          Clear conversation history
  /history [query]      Search conversation history (or show recent)
  /model                Show current model
  /raw                  Toggle reasoning content display
  /reason [low|med|high] Set/show reasoning effort
  /compact              Compact conversation (summarize old messages)
  /creative             Toggle creative writing mode (no tools)
  /debug                Toggle raw SSE delta logging
  /help                 Show this help
  /exit                 Exit (also: Ctrl+D)
""".strip()
            )

        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")

        return False


# ─── Model auto-detection ─────────────────────────────────────────────────


def detect_model(client: OpenAI) -> str:
    """Auto-detect the model from vLLM's /v1/models endpoint."""
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if not model_ids:
            print(red("No models found at server. Use --model to specify."))
            sys.exit(1)
        if len(model_ids) == 1:
            return model_ids[0]
        # Multiple models — pick first, but inform user
        print(f"Available models: {', '.join(model_ids)}")
        print(f"Using: {bold(model_ids[0])} (override with --model)")
        return model_ids[0]
    except Exception as e:
        print(red(f"Could not connect to server: {e}"))
        print("Is vLLM running? Start it or use --base-url to point elsewhere.")
        sys.exit(1)


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI for vLLM models with tool calling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 chat.py                          # auto-detect model
              python3 chat.py --persona lawful_evil     # with persona
              python3 chat.py --model kappa_20b_131k    # explicit model
              python3 chat.py --temperature 0.7         # lower temperature
        """),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: auto-detect from server)",
    )
    parser.add_argument(
        "--persona",
        default=None,
        help="Persona name injected as system message",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Developer instructions injected as developer message",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Max completion tokens (default: 32768)",
    )
    parser.add_argument(
        "--tool-timeout",
        type=int,
        default=30,
        help="Bash command timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level (default: medium)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=131072,
        help="Context window size in tokens (default: 131072)",
    )
    parser.add_argument(
        "--skip-permissions",
        action="store_true",
        help="Auto-approve all tool calls (no confirmation prompts)",
    )
    args = parser.parse_args()

    # Set up readline
    setup_readline()

    # Create client
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
    )

    # Detect or use provided model
    if args.model:
        model = args.model
    else:
        model = detect_model(client)

    # Create session
    session = ChatSession(
        client=client,
        model=model,
        persona=args.persona,
        instructions=args.instructions,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tool_timeout=args.tool_timeout,
        reasoning_effort=args.reasoning_effort,
        context_window=args.context_window,
    )
    if args.skip_permissions:
        session.auto_approve = True

    # Print banner
    print(f"\n{bold('Chat')} with {cyan(model)}")
    if args.persona:
        print(f"Persona: {cyan(args.persona)}")
    print(f"Type /help for commands, /exit or Ctrl+D to quit.\n")

    # Prompt string — use a short display name
    display_name = model.split("/")[-1]  # strip path prefixes if any
    if len(display_name) > 30:
        display_name = display_name[:27] + "..."

    # Main loop
    while True:
        try:
            user_input = input(f"\001{BOLD}\002[{display_name}]\001{RESET}\002 > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            should_exit = session.handle_command(user_input)
            if should_exit:
                break
        else:
            try:
                session.send(user_input)
            except KeyboardInterrupt:
                print(f"\n{yellow('Interrupted.')}")
            except Exception as e:
                print(f"\n{red(f'Error: {e}')}")

    print("Goodbye.")


if __name__ == "__main__":
    main()
