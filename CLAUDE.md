# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanocode is a minimal Claude Code alternative: a single-file (~280 lines) Python agentic coding assistant with zero external dependencies. It implements a full agentic loop with tool use, conversation history, and colored terminal output. The entire application lives in `nanocode.py`.

## Running

```bash
# Anthropic API (default model: claude-opus-4-5)
export ANTHROPIC_API_KEY="your-key"
python nanocode.py

# OpenRouter (default model: anthropic/claude-opus-4.5)
export OPENROUTER_API_KEY="your-key"
python nanocode.py

# Custom model via OpenRouter
export OPENROUTER_API_KEY="your-key"
export MODEL="openai/gpt-5.2"
python nanocode.py
```

No build step, no package installation, no virtual environment needed. Requires only Python 3 with standard library.

## Testing and Linting

No test suite, linting, or CI/CD exists. Validate changes by running `python nanocode.py` and exercising the affected tool or flow interactively.

## Architecture

Everything is in `nanocode.py`. The code follows this structure:

1. **Config** (lines 6-18): API URL, model selection, ANSI color constants. Auto-detects Anthropic vs OpenRouter based on which env var is set.
2. **Tool implementations** (lines 24-98): Six standalone functions (`read`, `write`, `edit`, `glob`, `grep`, `bash`) that operate on files and shell commands.
3. **Tool registry** (lines 103-134): `TOOLS` dict maps tool names to `(description, param_schema, function)` tuples. The schema uses a custom shorthand: `"string"` for required string, `"number?"` for optional number.
4. **Schema generation** (lines 146-171): `make_schema()` converts the shorthand param definitions into proper Anthropic API JSON schema format.
5. **API client** (lines 174-195): `call_api()` sends messages to the Anthropic Messages API (or OpenRouter's compatible endpoint) via `urllib.request`.
6. **Main loop** (lines 208-276): `main()` drives the REPL and the agentic loop — sends user input to the API, processes content blocks (text output + tool calls), feeds tool results back to the API, and repeats until the model returns no tool calls.

### Agentic Loop

The core pattern: user message → API call → process response blocks → if tool calls exist, execute them and feed results back → repeat until no tool calls. This is the standard Anthropic Messages API tool-use loop.

### Key Design Details

- Tool parameters use a custom schema shorthand (`"string"`, `"number?"`) converted to JSON Schema at API call time
- The `edit` tool requires unique match strings by default; pass `all=true` to replace all occurrences
- The `bash` tool streams output in real-time to the terminal and has a 30-second timeout
- The `grep` tool limits results to 50 matches
- API authentication: OpenRouter uses `Authorization: Bearer`, Anthropic uses `x-api-key` header
