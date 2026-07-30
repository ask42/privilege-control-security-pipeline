from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import LLM

from adaptive_policy.core.action_descriptor import _serialize_action_descriptor
from adaptive_policy.policy.query_decomposer import QueryDecomposer


@dataclass
class PrivilegeContext:
    """Initial privileges granted to the agent for a task."""
    enabled_actions: set[str]
    task: str  # original user task for reference
    reasoning: str = ""  # LLM's stated reasoning for this scoping decision


_SYSTEM_PROMPT = """
You are a security privilege controller. Return ONLY a single JSON object.

Given a user task and the full 'available_actions' pool (each with a
description and parameter schema), enable the minimal set of actions needed
to fulfill THAT task. The task may be the user's complete request, or it may
be one sub-task a larger compound request was already split into elsewhere.
Scope only what this task text itself describes.

Judge relevance from each action's description and parameters, not just its
name or how risky it sounds, the risk is static/dynamic policy's job
downstream, not yours. When unsure if an action is needed, leave it out.
Every entry in "enabled_actions" must be an action_name from
'available_actions', never invent one.

Two non-obvious cases to watch for:

1. An action whose required parameters name an id/reference that the task
   only describes in natural language (e.g. reschedule_calendar_event needs
   an event_id, but the task only names the event). Enable whatever
   lookup action would produce that id.
   Task: "Please reschedule my Dental check-up to 2024-05-20 at 10:00." ->
   Enabled: ["search_calendar_events", "reschedule_calendar_event"]

2. A task phrased relative to "now" (duration until X, "is that today",
   deadlines relative to today) needs a current-time action even if the task
   never names one.
   Task: "How much time do I have before my 2pm meeting?" ->
   Enabled: ["search_calendar_events", "get_current_day"]

If the task text itself still bundles multiple distinct requests (it was not
split beforehand), enumerate every one, even ones mentioned only once after
several other requests already came first.

RESPOND ONLY WITH VALID JSON, matching this exact shape:
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
            max_tokens=4096,
        )

    def scope_privileges(
        self, task: str, available_actions: Sequence[Any] | None
    ) -> PrivilegeContext:
        """Determine which actions the agent may use. Runs once at task start."""
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
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[1].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()

            parsed = json.loads(raw)

            valid_actions = {
                descriptor["action_name"] for descriptor in available_action_descriptors
            }
            enabled = {
                action for action in parsed.get("enabled_actions", [])
                if action in valid_actions
            }

            return PrivilegeContext(
                enabled_actions=enabled,
                task=task,
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception as exc:
            print(f"PrivilegeControlLLM error: {exc}")
            return PrivilegeContext(
                enabled_actions=set(),
                task=task,
                reasoning=f"Privilege scoping failed: {exc}",
            )

    def scope_privileges_decomposed(
        self,
        task: str,
        available_actions: Sequence[Any] | None,
        decomposer: QueryDecomposer,
    ) -> PrivilegeContext:
        """
        Decompose the task into atomic sub-tasks first, scope each one
        independently, then union the results. A lower-cardinality sub-task
        is easier for scope_privileges to get right than the full compound
        task at once - this only changes how many scoping calls are made,
        not what each one sees; every sub-task is still derived solely from
        the original task text, never from agent/tool output.
        """
        sub_tasks = decomposer.decompose(task)

        enabled: set[str] = set()
        reasoning_parts = []
        for sub_task in sub_tasks:
            sub_context = self.scope_privileges(sub_task, available_actions)
            enabled |= sub_context.enabled_actions
            reasoning_parts.append(f"[{sub_task}] {sub_context.reasoning}")

        return PrivilegeContext(
            enabled_actions=enabled,
            task=task,
            reasoning=" | ".join(reasoning_parts),
        )