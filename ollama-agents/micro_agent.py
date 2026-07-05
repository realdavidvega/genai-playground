#!/usr/bin/env python3
"""
Micro-agent harness for Ollama + Pi (pi-llm-agent)
Minimal token overhead, parallel execution, simple tools.

SECURITY: this is a local experiment. `BashTool` intentionally executes
arbitrary shell commands the model produces, so only run it against a
trusted, local model — never expose it to untrusted input.
"""

import asyncio
import json
import os
import shlex
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Pi imports
from pi_llm import (
    get_model, complete_simple, Context, UserMessage, 
    TextContent, AssistantMessage
)
from pi_llm.providers import register_builtin_providers
from pi_llm_agent import Agent, AgentOptions, InitialAgentState, AgentTool, AgentToolResult

# Register providers (OpenAI, Anthropic, etc.)
register_builtin_providers()


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    system_prompt: str
    model: str = "ollama/qwen2.5-coder:7b"
    tools: Optional[List[AgentTool]] = None
    max_turns: int = 3


class OllamaProvider:
    """Lightweight Ollama provider wrapper for Pi."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat request to Ollama."""
        payload = {
            "model": model.replace("ollama/", ""),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_ctx": kwargs.get("num_ctx", 8192),
            }
        }
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]


class BashTool(AgentTool):
    """Execute bash commands.

    WARNING: runs model-supplied commands via a shell with the user's
    permissions. Intended only for local experiments with trusted models.
    """

    def __init__(self):
        super().__init__(
            name="bash",
            label="Bash",
            description="Execute a bash command and return its output",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["command"],
            },
        )
    
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        try:
            result = subprocess.run(
                params["command"],
                shell=True,
                capture_output=True,
                text=True,
                cwd=params.get("cwd", "."),
                timeout=30
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            return AgentToolResult(content=[TextContent(text=output[:2000])])
        except Exception as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {str(e)}")])


class ReadFileTool(AgentTool):
    """Read file contents."""
    
    def __init__(self):
        super().__init__(
            name="read_file",
            label="Read File",
            description="Read the contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "limit": {"type": "integer", "description": "Max lines to read (optional)", "default": 100},
                },
                "required": ["path"],
            },
        )
    
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        try:
            # Confine reads to the working-directory tree so a prompt-injected
            # tool call can't exfiltrate arbitrary files (SSH keys, env, etc.).
            base = os.path.realpath(os.getcwd())
            target = os.path.realpath(params["path"])
            if os.path.commonpath([base, target]) != base:
                return AgentToolResult(content=[TextContent(
                    text=f"Error: path outside working directory: {params['path']}")])
            with open(target, "r") as f:
                lines = f.readlines()
                limit = params.get("limit", 100)
                content = "".join(lines[:limit])
                if len(lines) > limit:
                    content += f"\n... ({len(lines) - limit} more lines)"
                return AgentToolResult(content=[TextContent(text=content)])
        except Exception as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {str(e)}")])


class GitTool(AgentTool):
    """Git operations."""
    
    def __init__(self):
        super().__init__(
            name="git",
            label="Git",
            description="Execute git commands",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Git subcommand (e.g., 'diff --staged')"},
                },
                "required": ["command"],
            },
        )
    
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        try:
            # No shell: split into argv so git args can't inject shell commands.
            result = subprocess.run(
                ["git", *shlex.split(params["command"])],
                capture_output=True,
                text=True,
                timeout=10
            )
            output = result.stdout or result.stderr
            return AgentToolResult(content=[TextContent(text=output[:3000])])
        except Exception as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {str(e)}")])


class MicroAgentRunner:
    """Run multiple lightweight agents in parallel with minimal overhead."""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama = OllamaProvider(ollama_base_url)
        self.results: Dict[str, Any] = {}
    
    async def run_single(self, config: AgentConfig, prompt: str) -> str:
        """Run a single agent."""
        print(f"🚀 [{config.name}] Starting with model {config.model}")
        
        # Use Ollama directly for minimal overhead
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.ollama.chat(
            config.model.replace("ollama/", ""),
            messages
        )
        
        print(f"✅ [{config.name}] Complete")
        return response
    
    async def run_parallel(self, tasks: List[tuple]) -> Dict[str, str]:
        """Run multiple agents in parallel.
        
        tasks: List of (AgentConfig, prompt) tuples
        """
        coroutines = [
            self.run_single(config, prompt)
            for config, prompt in tasks
        ]
        
        # Execute all in parallel
        responses = await asyncio.gather(*coroutines, return_exceptions=True)
        
        results = {}
        for i, (config, _) in enumerate(tasks):
            if isinstance(responses[i], Exception):
                results[config.name] = f"Error: {str(responses[i])}"
                print(f"❌ [{config.name}] Failed: {responses[i]}")
            else:
                results[config.name] = responses[i]
        
        return results


# Pre-defined agent configurations
COMMIT_AGENT = AgentConfig(
    name="commit_generator",
    system_prompt="""You are a git commit message generator.
Rules:
- Use conventional commits format (feat:, fix:, docs:, refactor:, etc.)
- First line: max 72 characters
- Be specific about what changed
- Do not explain why, only what""",
    model="ollama/qwen2.5-coder:7b",
)

REVIEW_AGENT = AgentConfig(
    name="code_reviewer",
    system_prompt="""You are a code reviewer. Focus on:
- Bugs and logic errors
- Security issues  
- Performance problems
- Code clarity

Format: Brief bullet points. Max 5 issues. Be concise.""",
    model="ollama/qwen2.5-coder:7b",
)

SUMMARY_AGENT = AgentConfig(
    name="summarizer",
    system_prompt="""Summarize the given content in 2-3 sentences.
Focus on the key changes and their purpose. Be concise.""",
    model="ollama/qwen2.5-coder:7b",
)


async def demo():
    """Demo: Run multiple agents in parallel."""
    runner = MicroAgentRunner()
    
    # Get git diff (unstaged changes)
    diff_result = subprocess.run(
        ["git", "diff", "--stat"],
        capture_output=True,
        text=True
    )
    diff_stat = diff_result.stdout
    
    # Also get full diff for better context
    full_diff = subprocess.run(
        ["git", "diff"],
        capture_output=True,
        text=True
    ).stdout[:3000]  # Limit length
    
    # Prepare parallel tasks
    tasks = [
        (COMMIT_AGENT, f"Files changed:\n{diff_stat}\n\nGenerate a concise conventional commit message."),
        (REVIEW_AGENT, f"Review this code diff for issues:\n{full_diff}"),
        (SUMMARY_AGENT, f"Summarize these changes in 2 sentences:\n{diff_stat}"),
    ]
    
    print("=" * 60)
    print("Running 3 agents in parallel...")
    print("=" * 60)
    
    results = await runner.run_parallel(tasks)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"\n📋 {name}:")
        print("-" * 40)
        print(result[:500])  # Truncate long outputs
        if len(result) > 500:
            print("...")


if __name__ == "__main__":
    asyncio.run(demo())
