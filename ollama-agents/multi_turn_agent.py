#!/usr/bin/env python3
"""
Multi-turn agent with JSON-based tool loop.
Model outputs tool requests as JSON in text, we parse and execute.
Works reliably with 7B models that don't support native function calling.
"""

import asyncio
import json
import subprocess
import re
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

import httpx


@dataclass
class Tool:
    """Simple tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class LightweightAgent:
    """
    Multi-turn agent with manual JSON parsing for tool calls.
    Minimal overhead, works with any Ollama model.
    """
    
    def __init__(self, model: str = "qwen2.5-coder:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.tools: Dict[str, Tool] = {}
        self.conversation: List[Dict[str, str]] = []
    
    def register_tool(self, tool: Tool):
        """Add a tool the agent can use."""
        self.tools[tool.name] = tool
    
    def _build_system_prompt(self, task_description: str) -> str:
        """Build minimal system prompt with tool descriptions."""
        tool_list = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools.values()
        ])
        
        return f"""You are an AI assistant with access to tools.

Workflow:
1. Gather information using tools (1-2 tool calls max)
2. Once you have enough info, provide the FINAL ANSWER immediately
3. Do NOT call the same tool twice with the same parameters

Available tools:
{tool_list}

When you need to use a tool, respond with EXACTLY this JSON:
{{"tool": "tool_name", "params": {{}}}}

After I send you the tool result, you MUST provide your final answer.
Do not call another tool unless absolutely necessary.

Task: {task_description}"""
    
    async def _call_ollama(self, messages: List[Dict]) -> str:
        """Call Ollama API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temp for predictable JSON
                "num_ctx": 8192,
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
    
    def _extract_tool_call(self, text: str) -> tuple:
        """
        Extract tool call from model response.
        Returns: (tool_name, params) or (None, None) if no tool call.
        """
        # Scan for the first syntactically valid JSON object. raw_decode
        # correctly handles braces that appear inside JSON strings, which
        # naive brace-counting would miscount.
        decoder = json.JSONDecoder()
        idx = text.find('{')
        while idx != -1:
            try:
                data, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                idx = text.find('{', idx + 1)
                continue
            if isinstance(data, dict) and "tool" in data and "params" in data:
                return data["tool"], data["params"]
            idx = text.find('{', idx + 1)

        return None, None
    
    async def run(self, task: str, max_turns: int = 5) -> str:
        """
        Run agent with multi-turn tool loop.
        """
        # Initialize conversation
        system_prompt = self._build_system_prompt(task)
        self.conversation = [
            {"role": "system", "content": system_prompt},
        ]
        
        current_message = task
        
        for turn in range(max_turns):
            print(f"\n  Turn {turn + 1}:")
            
            # Add user message
            self.conversation.append({"role": "user", "content": current_message})
            
            # Call model
            print(f"    → Calling model...")
            response = await self._call_ollama(self.conversation)
            print(f"    ← Model: {response[:150]}...")
            
            # Add assistant response
            self.conversation.append({"role": "assistant", "content": response})
            
            # Check if model wants to use a tool
            tool_name, tool_params = self._extract_tool_call(response)
            
            if tool_name and tool_name in self.tools:
                print(f"    🔧 Tool call: {tool_name}({json.dumps(tool_params)})")
                
                # Execute tool
                try:
                    result = await self.tools[tool_name].function(**tool_params)
                except Exception as e:
                    result = f"Error: {e}"
                
                print(f"    📊 Result: {str(result)[:100]}...")
                
                # Set up next turn with tool result
                current_message = f"Tool '{tool_name}' returned:\n{result}\n\nContinue with your task."
            else:
                # Model provided final answer
                print(f"    ✓ Final answer received")
                return response
        
        return "Max turns reached"


# Define tools
async def get_git_status(**kwargs) -> str:
    """Get git status. Shows modified, staged, and untracked files."""
    result = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True
    )
    output = result.stdout.strip()
    return output or "No changes"

async def get_git_diff(**kwargs) -> str:
    """Get diff of staged changes. Shows code changes ready to commit."""
    result = subprocess.run(
        ["git", "diff", "--staged"],
        capture_output=True,
        text=True
    )
    output = result.stdout.strip()
    if not output:
        return "No staged changes. Try git add <files> first."
    return output[:4000]

async def read_file(path: str = "", **kwargs) -> str:
    """Read file contents. Provide the file path."""
    if not path:
        return "Error: No file path provided"
    try:
        with open(path, "r") as f:
            return f.read()[:3000]
    except Exception as e:
        return f"Error reading {path}: {e}"


async def main():
    """Demo: Multi-turn agent that generates commit messages."""
    
    # Create agent
    agent = LightweightAgent(model="qwen2.5-coder:7b")
    
    # Register tools
    agent.register_tool(Tool(
        name="git_status",
        description="Get list of changed files",
        parameters={"type": "object", "properties": {}},
        function=get_git_status
    ))
    
    agent.register_tool(Tool(
        name="git_diff",
        description="Get detailed code changes",
        parameters={"type": "object", "properties": {}},
        function=get_git_diff
    ))
    
    agent.register_tool(Tool(
        name="read_file",
        description="Read contents of a specific file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        },
        function=read_file
    ))
    
    print("=" * 70)
    print("MULTI-TURN AGENT WITH TOOL LOOP")
    print("=" * 70)
    print("Agent will gather info using tools, then generate commit message\n")
    
    # Run agent
    result = await agent.run(
        task="Generate a conventional commit message for the current changes. "
             "First check git_status to see what files changed, "
             "then use git_diff to see the actual changes, "
             "then write the commit message."
    )
    
    print("\n" + "=" * 70)
    print("FINAL COMMIT MESSAGE:")
    print("=" * 70)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
