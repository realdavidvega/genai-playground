#!/usr/bin/env python3
"""
Multi-turn agent with Pi framework + Ollama.
Demonstrates tool loops (agent calls tool → sees result → decides next action).
"""

import asyncio
import json
import subprocess
from typing import Optional

from pi_llm import (
    get_model, complete_simple, Context, UserMessage, 
    TextContent, AssistantMessage, ToolCall
)
from pi_llm.providers import register_builtin_providers
from pi_llm_agent import (
    Agent, AgentOptions, InitialAgentState, AgentTool, 
    AgentToolResult, ToolExecutionMode
)

# Register all providers
register_builtin_providers()


class GitDiffTool(AgentTool):
    """Get git diff for staged changes."""
    
    def __init__(self):
        super().__init__(
            name="git_diff",
            label="Git Diff",
            description="Get the diff of staged changes",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
    
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        result = subprocess.run(
            ["git", "diff", "--staged"],
            capture_output=True,
            text=True
        )
        return AgentToolResult(content=[TextContent(text=result.stdout[:4000])])


class GitStatusTool(AgentTool):
    """Get git status."""
    
    def __init__(self):
        super().__init__(
            name="git_status",
            label="Git Status",
            description="Get current git status",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
    
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True
        )
        return AgentToolResult(content=[TextContent(text=result.stdout)])


class ReadFileTool(AgentTool):
    """Read a file's contents."""
    
    def __init__(self):
        super().__init__(
            name="read_file",
            label="Read File",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                },
                "required": ["path"],
            },
        )
    
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        try:
            with open(params["path"], "r") as f:
                content = f.read()
            return AgentToolResult(content=[TextContent(text=content[:3000])])
        except Exception as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")])


async def run_commit_agent():
    """
    Multi-turn agent that:
    1. Checks git status
    2. Reads the diff
    3. Generates commit message
    4. Can be asked to refine it
    """
    
    # Use Ollama through OpenAI-compatible API
    # The key is using OpenAIModel pointing to Ollama
    from pi_llm import OpenAIModel
    
    model = OpenAIModel(
        model_id="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't need real key
    )
    
    # Create agent with tools
    agent = Agent(AgentOptions(
        initial_state=InitialAgentState(
            model=model,
            system_prompt="""You are a git commit message generator.
You have access to git tools. Follow this workflow:
1. Check git status to see what files changed
2. Read the diff to understand the changes  
3. Write a concise conventional commit message

Rules:
- Use format: type(scope): description
- Types: feat, fix, docs, style, refactor, test, chore
- Max 72 chars for first line
- Be specific, not vague""",
            tools=[GitStatusTool(), GitDiffTool(), ReadFileTool()],
        ),
        stream_fn=complete_simple,  # Non-streaming for simplicity
        tool_execution=ToolExecutionMode.parallel,
    ))
    
    print("🚀 Starting commit agent with tool loop...")
    print("=" * 60)
    
    # Run the agent - it will automatically loop through tools
    result = await agent.prompt("Generate a commit message for the current changes")
    
    print("\n" + "=" * 60)
    print("AGENT RESULT:")
    print("=" * 60)
    
    # The agent returns the final message after all tool calls
    if hasattr(result, 'content'):
        for item in result.content:
            if hasattr(item, 'text'):
                print(item.text)
    else:
        print(result)
    
    # Show what tools were called
    print("\n📋 Tools used:")
    # Access agent state to see history
    state = agent.state
    for msg in state.messages:
        if hasattr(msg, 'tool_calls'):
            for tc in msg.tool_calls:
                print(f"  - {tc.name}")


async def run_simple_direct():
    """
    Direct Ollama call for comparison (single turn, no tool loop).
    Same prompt but no tool capabilities.
    """
    import httpx
    
    # Get diff ourselves
    diff = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True).stdout[:2000]
    
    prompt = f"""Generate a conventional commit message for these changes.
Rules: type(scope): description, max 72 chars, be specific.

Changes:
{diff}

Commit message:"""
    
    print("\n🚀 Direct Ollama call (no tool loop)...")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5-coder:7b",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
            },
            timeout=60.0
        )
        data = response.json()
        print("\nRESULT:")
        print(data["message"]["content"])


async def main():
    """Run both approaches for comparison."""
    
    print("\n" + "=" * 70)
    print("APPROACH 1: Pi Agent with Multi-Turn Tool Loop")
    print("=" * 70)
    print("Agent decides which tools to call and in what order")
    
    try:
        await run_commit_agent()
    except Exception as e:
        print(f"Agent approach error: {e}")
        print("Falling back to direct approach...")
    
    print("\n" + "=" * 70)
    print("APPROACH 2: Direct Ollama Call (Single Turn)")
    print("=" * 70)
    print("We manually provide the diff, no tool loop")
    
    await run_simple_direct()
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("""
Pi Agent (Multi-turn):
- ✅ Agent decides what info it needs
- ✅ Can explore multiple files
- ✅ More flexible for complex tasks
- ⚠️ Slightly more overhead (~500 tokens for tool schemas)

Direct Call (Single turn):
- ✅ Minimal overhead
- ✅ Faster for simple tasks
- ⚠️ Must provide all context upfront
- ⚠️ No exploration capability

For 7B models: Both work! Use direct for simple tasks,
Pi Agent for tasks requiring exploration.
""")


if __name__ == "__main__":
    asyncio.run(main())
