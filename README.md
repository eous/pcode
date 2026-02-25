# pcode

Single-file AI chat client with tool use, streaming, and conversation management.

<p align="center">
  <img src="demo.svg" alt="pcode demo — plan agent exploring codebase" width="680">
</p>

## Features

- **Streaming responses** with reasoning display (think tags, vLLM reasoning fields)
- **7 built-in tools**: bash, read_file, write_file, edit_file, search, math, web_fetch
- **3 agent tools**: task (autonomous work), plan (writes `.plan.md`), review (returned inline)
- **Tool approval workflow** with user feedback (`y, use absolute path`)
- **Conversation compaction** for long sessions (auto-triggers at 80% context)
- **Markdown rendering** with ANSI colors in terminal

## Quickstart

```bash
pip install pcode
pcode http://localhost:8000/v1 --model your-model-name
```

Or run directly:

```bash
python -m pcode.chat http://localhost:8000/v1 --model your-model-name
```

## Usage

Chat naturally — the model decides when to use tools.

```
User: What does the main function in server.py do?
⚙ read_file: server.py
    142 lines
The main function in server.py sets up an async HTTP server...
```

### Tool approval

Every tool call shows a preview and asks for approval:

- `y` — approve
- `n` — deny (model sees denial message)
- `a` — always approve this tool type
- `y, use the full path` — approve with feedback injected as a user message
- `n, try math instead` — deny with guidance

### Slash commands

| Command | Description |
|---------|-------------|
| `/compact` | Summarize conversation to free context |
| `/creative` | Toggle creative writing mode (disables tools) |
| `/reason low\|medium\|high` | Adjust reasoning effort |
| `/debug` | Toggle debug mode (shows full API requests) |
| `/usage` | Show token usage breakdown |

### Agent tools

Three agent tools delegate work to autonomous sub-agents with read-only tool access:

| Tool | Purpose | Output |
|------|---------|--------|
| `task` | General autonomous work | Returned inline |
| `plan` | Explore codebase, design implementation | Written to `.plan.md` |
| `review` | Code review for correctness, security, style | Returned inline |

```
User: Review the auth module for security issues
⚙ review (code review agent)
    Review auth.py for security vulnerabilities
    Allow review? [y/n/a(lways), optional message] y
  [review turn 1] ⚙ read_file: auth.py
  [review turn 2] ⚙ search: /password|secret|token/ in auth.py
  [review done] 2340 chars
```

Agents have read-only tool access (read_file, search, math, web_fetch). Write operations require the main conversation's approval flow.

## Requirements

- Python 3.10+
- An OpenAI-compatible API endpoint (e.g., [vLLM](https://github.com/vllm-project/vllm))

## License

MIT
