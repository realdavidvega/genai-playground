# Ollama agents

Experiments building small LLM agents against a **local Ollama** server, aimed at
7B-class models (default `qwen2.5-coder:7b`). Explores a few approaches: raw JSON
tool-loops, parallel fan-out, and the [Pi](https://github.com/) agent framework.

## Scripts

| Script | Approach | Third-party deps |
| --- | --- | --- |
| `parallel_agent.py` | Fan-out: N direct Ollama calls in parallel, no tool loop (most reliable with 7B). | `httpx` |
| `multi_turn_agent.py` | Multi-turn agent with a hand-rolled JSON tool-loop parsed from text — works with models lacking native function calling. | `httpx` |
| `pi_agent_demo.py` | Multi-turn tool loop via the Pi framework, with a `git_diff` tool. | `pi_llm`, `pi_llm_agent` |
| `micro_agent.py` | Minimal Pi + Ollama harness for parallel agents with simple tools. | `pi_llm`, `pi_llm_agent` |

## Requirements

- A running **Ollama** server at `http://localhost:11434` with the model pulled:
  ```bash
  ollama pull qwen2.5-coder:7b
  ```
- Python deps for the `httpx`-only scripts:
  ```bash
  pip install -r requirements.txt
  ```
- `pi_agent_demo.py` and `micro_agent.py` additionally require the **Pi** framework
  (`pi_llm`, `pi_llm_agent`), which is not on PyPI — install it separately before
  running those two.

## Usage

```bash
python parallel_agent.py
python multi_turn_agent.py
```

## Security

These are local experiments meant to run against a **trusted local model**.
`micro_agent.py`'s `BashTool` intentionally executes arbitrary shell commands
the model emits — do not point these agents at untrusted input. The file-read
tool is confined to the working-directory tree and the git tool runs without a
shell, but the bash tool is deliberately unrestricted.
