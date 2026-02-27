# pcode

Single-file AI chat client with tool use, agent tools, and persistent memory.

<p align="center">
  <img src="demo.svg" alt="pcode demo — plan agent exploring codebase" width="680">
</p>

## Features

- **Streaming responses** with reasoning display (think tags, vLLM reasoning fields)
- **13 tools**: bash, read_file, write_file, edit_file, search, math, web_fetch, web_search, task, plan, remember, recall, forget
- **2 agent tools**: task (autonomous work with full tool access), plan (explores codebase, writes `.plan.md`)
- **Persistent memory** across sessions — remember facts, recall memories and past conversations (SQLite + FTS5 full-text search)
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
| `/history [query]` | Show recent history, or search past conversations |
| `/reason low\|medium\|high` | Adjust reasoning effort |
| `/debug` | Toggle debug mode (shows full API requests) |
| `/usage` | Show token usage breakdown |

### Agent tools

Two agent tools delegate work to autonomous sub-agents:

| Tool | Purpose | Output |
|------|---------|--------|
| `task` | General autonomous work (full tool access) | Returned inline |
| `plan` | Explore codebase, design implementation | Written to `.plan.md`, shown for approval |

The plan agent has access to read_file, search, math, web_fetch, and web_search. The task agent also has bash, write_file, and edit_file — write operations go through the main conversation's approval flow.

### Persistent memory

All conversations are stored in `.pcode_memories.db` (SQLite) and persist across sessions.

- **remember** — save key-value facts (IPs, paths, conventions) that are always visible in the model's instructions
- **recall** — with no query, lists all memories; with a query, searches both memories and conversation history (FTS5 BM25-ranked)
- **forget** — remove a saved memory
- **`/history`** — CLI command to browse or search your conversation history

### Web search (optional)

The `web_search` tool uses the [Tavily API](https://tavily.com). To enable it, set your API key:

```bash
mkdir -p ~/.config/pcode
echo "tvly-YOUR_API_KEY" > ~/.config/pcode/tavily_key
```

Or set the `TAVILY_API_KEY` environment variable. Without a key, the model is told to use `web_fetch` with a direct URL instead.

## Requirements

- Python 3.10+
- An OpenAI-compatible API endpoint (e.g., [vLLM](https://github.com/vllm-project/vllm))

The demo above uses [kappa-20b-131k](https://huggingface.co/eousphoros/kappa-20b-131k), a 20B MoE model fine-tuned for tool use and long context.

## License

MIT
