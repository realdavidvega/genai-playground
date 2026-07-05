#!/usr/bin/env python3
"""
Simple parallel agent runner - works reliably with 7B models.
No multi-turn loops (7B models struggle with those).
Instead: parallel direct calls with pre-gathered context.
"""

import asyncio
import subprocess
import httpx
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class AgentTask:
    """Single agent task with all context pre-loaded."""
    name: str
    system_prompt: str
    user_prompt: str
    model: str = "qwen2.5-coder:7b"


class ParallelAgentRunner:
    """Run multiple agents in parallel against Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def run_task(self, task: AgentTask) -> str:
        """Run a single agent task."""
        print(f"🚀 [{task.name}] Running...")
        
        payload = {
            "model": task.model,
            "messages": [
                {"role": "system", "content": task.system_prompt},
                {"role": "user", "content": task.user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
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
            result = data["message"]["content"]
        
        print(f"✅ [{task.name}] Done")
        return result
    
    async def run_parallel(self, tasks: List[AgentTask]) -> Dict[str, str]:
        """Run multiple tasks in parallel."""
        coroutines = [self.run_task(t) for t in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        output = {}
        for i, task in enumerate(tasks):
            if isinstance(results[i], Exception):
                output[task.name] = f"Error: {results[i]}"
                print(f"❌ [{task.name}] Failed: {results[i]}")
            else:
                output[task.name] = results[i]
        
        return output


def get_git_context() -> Dict[str, str]:
    """Pre-gather all git context."""
    # Get status
    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True
    ).stdout
    
    # Get staged diff
    diff = subprocess.run(
        ["git", "diff", "--staged"],
        capture_output=True, text=True
    ).stdout[:3000]
    
    # Get diff stat
    diff_stat = subprocess.run(
        ["git", "diff", "--staged", "--stat"],
        capture_output=True, text=True
    ).stdout
    
    return {
        "status": status,
        "diff": diff,
        "diff_stat": diff_stat,
    }


async def main():
    """Demo: Run multiple analysis agents in parallel."""
    
    print("=" * 70)
    print("PARALLEL AGENT RUNNER (7B Model Friendly)")
    print("=" * 70)
    
    # Pre-gather all context
    ctx = get_git_context()
    
    # Define parallel tasks - each gets full context upfront
    tasks = [
        AgentTask(
            name="commit_generator",
            system_prompt="""You are a git commit message generator.
Rules:
- Use conventional commits: type(scope): description
- Types: feat, fix, docs, style, refactor, test, chore
- First line max 72 chars
- Be specific about what changed""",
            user_prompt=f"Generate a commit message for these changes:\n\nFiles changed:\n{ctx['diff_stat']}\n\nDetailed diff:\n{ctx['diff']}",
        ),
        AgentTask(
            name="impact_analyzer",
            system_prompt="""Analyze the impact of code changes.
Focus on: breaking changes, API changes, dependencies, risks.
Format: Bullet points. Be concise.""",
            user_prompt=f"Analyze the impact of these changes:\n\n{ctx['diff']}",
        ),
        AgentTask(
            name="security_scanner",
            system_prompt="""Security-focused code review.
Look for: exposed secrets, SQL injection, XSS, unsafe eval, path traversal.
Format: List issues or say "No security concerns found." Be concise.""",
            user_prompt=f"Review for security issues:\n\n{ctx['diff']}",
        ),
    ]
    
    print(f"Running {len(tasks)} agents in parallel...\n")
    
    # Run all agents
    runner = ParallelAgentRunner()
    results = await runner.run_parallel(tasks)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for name, result in results.items():
        print(f"\n📋 {name}:")
        print("-" * 40)
        print(result[:600])
        if len(result) > 600:
            print("...")


if __name__ == "__main__":
    asyncio.run(main())
