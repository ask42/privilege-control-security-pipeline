from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class PrivilegeContext:
    """Initial privileges granted to the agent for a task."""
    enabled_actions: set[str]
    task: str  # original user task for reference
    timestamp: float = 0.0


_SYSTEM_PROMPT = """\
You are a privilege scoping assistant for an AI agent.

Given a user task and a list of available actions, determine which actions the agent needs to complete the task.

Be conservative: only enable actions that are clearly required for the task.

Respond ONLY with valid JSON in this format:
{"enabled_actions": ["action1", "action2", ...], "reasoning": "brief explanation"}"""


class PrivilegeControlLLM:
    """
    Runs once at initialization to scope initial privileges.
    Sees the full user task, decides which actions to enable.
    """

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-3B-Instruct",
    ):
        self.model = model
        self._client = OpenAI(base_url=vllm_base_url, api_key="EMPTY")

    def scope_privileges(
        self, task: str, available_actions: list[str]
    ) -> PrivilegeContext:
        """
        Determine which actions the agent should be allowed to use.
        Runs once at task start.
        """
        user_msg = json.dumps({
            "user_task": task,
            "available_actions": available_actions,
        })

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            enabled = set(parsed.get("enabled_actions", []))
            return PrivilegeContext(
                enabled_actions=enabled,
                task=task,
            )
        except Exception as exc:
            # Fail safe, enables nothing, requires explicit escalation
            print(f"PrivilegeControlLLM error: {exc}")
            return PrivilegeContext(
                enabled_actions=set(),
                task=task,
            )