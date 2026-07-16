from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import LLM

from adaptive_policy.core.action_descriptor import _serialize_action_descriptor


@dataclass
class PrivilegeContext:
    """Initial privileges granted to the agent for a task."""
    enabled_actions: set[str]
    task: str  # original user task for reference
    reasoning: str = ""  # LLM's stated reasoning for this scoping decision


_SYSTEM_PROMPT = """
You are a security privilege controller. Return ONLY a single JSON object.

You are given the user's task and the full 'available_actions' pool, each with
a description and parameter schema. This is a least-privilege scoping step,
not a judgment on whether any individual action is safe, static and dynamic
policy gates handle that afterward for each action you enable here.

Follow these examples:
Task: "Delete all draft emails" -> Enabled: ["get_emails", "search_emails", "delete_email"]
Task: "Look up john's contact info and add him to tomorrow's event" -> Enabled: ["search_contacts", "get_calendar_events", "add_attendees"]
Task: "Send a confirmation email to bob@company.com about the meeting tomorrow" -> Enabled: ["send_email", "get_calendar_events"]
Task: "Reschedule my 2pm meeting and notify attendees" -> Enabled: ["get_calendar_events", "reschedule_event", "send_invite"]
Task: "Create a new file with project notes and share it with the team" -> Enabled: ["create_file", "share_file"]
Task: "Generate an export of all my data for backup" -> Enabled: ["export_data", "create_file", "download_file"]
  (export_data is enabled here even though it sounds sensitive - the task
  genuinely calls for it. Relevance decides inclusion, not how risky an
  action sounds; risk is static/dynamic policy's job downstream.)
Task: "Read my calendar for next week" -> Enabled: ["get_calendar_events"]
  (read-only task -> read-only action, even if the pool also contains
  delete_event, change_password, export_data, etc. - none of those are
  relevant here, so none are enabled.)

INSTRUCTIONS:
- Judge relevance from each action's description and parameters, not just its name.
- Enable the minimal set of actions necessary to fulfill the task. When unsure
  whether an action is needed, leave it out - a missing action can be caught
  and corrected through user verification later; an unnecessarily enabled one
  cannot.
- Every entry in "enabled_actions" MUST be an "action_name" from the provided
  'available_actions'. Never invent an action name.

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