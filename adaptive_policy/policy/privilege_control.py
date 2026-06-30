from __future__ import annotations
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vllm import LLM

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
        llm: LLM,   # shared vLLM instance
        model: str = "Qwen/Qwen2.5-3B-Instruct",
    ):
        self.model_name = model
        self.llm = llm
        self.tokenizer = self.llm.get_tokenizer()
        
        from vllm.sampling_params import SamplingParams     # prevents runtime imports if not running this part of the pipeline
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

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
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
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