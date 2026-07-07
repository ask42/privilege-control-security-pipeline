from __future__ import annotations

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import LLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptive_policy.core.action_descriptor import _serialize_action_descriptor
@dataclass
class PrivilegeContext:
    """Initial privileges granted to the agent for a task."""
    enabled_actions: set[str]
    task: str  # original user task for reference
    timestamp: float = 0.0


_SYSTEM_PROMPT = """
You are a security privilege controller. 

Follow these examples:
Task: "Delete all draft emails" -> Enabled: ["get_emails", "search_emails", "delete_email"]
Task: "Look up [name]'s contact info and add them to tomorrow's event" -> Enabled: ["get_calendar_events", "add_attendees", "search_contacts"]

INSTRUCTIONS:
Based on keywords in the user request, identify which actions to enable. Limit to only necessary actions for the task defined.

RESPOND ONLY WITH VALID JSON:
{
  "reasoning": "Step-by-step logic",
  "enabled_actions": ["action_name"]
}
"""


class PrivilegeControlLLM:
    """
    Runs once at initialization to scope initial privileges.
    Sees the full user task, decides which actions to enable.
    """

    def __init__(
        self,
        llm: LLM,   # shared vLLM instance
        model: str = "Qwen/Qwen3-8B",
    ):
        self.model_name = model
        self.llm = llm
        self.tokenizer = self.llm.get_tokenizer()
        
        from vllm.sampling_params import SamplingParams     # prevents runtime imports if not running this part of the pipeline
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
        )

    def scope_privileges(
        self, task: str, available_actions: Sequence[Any] | None
    ) -> PrivilegeContext:
        """
        Determine which actions the agent should be allowed to use.
        Runs once at task start.
        """
        available_action_descriptors = [
            _serialize_action_descriptor(action) for action in (available_actions or [])
        ]

        user_msg = json.dumps({
            "user_task": task,
            "available_actions": available_action_descriptors,
        })

        try:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    enable_thinking=False,
                )
            else:
                prompt = messages[-1]["content"]
            
            outputs = self.llm.generate([prompt], self.sampling_params)
            raw = outputs[0].outputs[0].text.strip()
            parsed = json.loads(raw)
            enabled = set(parsed.get("enabled_actions", []))
            
            return PrivilegeContext(
                enabled_actions=enabled,
                task=task,
            )
        except Exception as exc:
            print(f"PrivilegeControlLLM error: {exc}")
            return PrivilegeContext(
                enabled_actions=set(),
                task=task,
            )