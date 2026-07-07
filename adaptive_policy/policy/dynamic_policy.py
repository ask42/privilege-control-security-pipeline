from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import LLM

from adaptive_policy.core.action_descriptor import _serialize_action_descriptor
from adaptive_policy.policy.privilege_control import PrivilegeContext


@dataclass
class DynamicPolicy:
    """
    Task-specific dependency overlay, generated once at task start.

    Does NOT decide allow/verify/deny.
    Only describes ordering and content constraints for the actions
    privilege control already enabled.
    """

    tool_dependencies: dict[str, tuple[str, ...]] = field(default_factory=dict)
    data_dependencies: dict[str, dict[str, str]] = field(default_factory=dict)
    reasoning: str = ""

    def required_actions_for(self, action: str) -> tuple[str, ...]:
        return self.tool_dependencies.get(action, ())

    def required_data_for(self, action: str) -> dict[str, str]:
        return self.data_dependencies.get(action, {})


_SYSTEM_PROMPT = """
You are a task-specific dependency planner. Return ONLY a single JSON object.

You do NOT decide whether an action is allowed, denied, or requires
verification - that is static policy's job alone, decided independently of
this task. Your only job is to describe, for this specific task: (1) which
already-enabled actions must run before which other already-enabled
actions, and (2) what concrete argument content an action should contain,
for actions whose static policy rule flags a content format to check.

Example output:
{
  "tool_dependencies": {
      "send_email": ["search_contacts"]
  },
  "data_dependencies": {
      "send_email": {"to": "alice@company.com"}
  },
  "reasoning": "The task asks to email alice@company.com, so send_email's recipient must match that address. The contact must be looked up before emailing."
}

Sections:

1. "tool_dependencies" - for each action in 'available_actions' whose execution
   only makes sense after some other action(s) in 'available_actions' have
   already run (e.g. you must look something up before acting on it, or read
   before writing), list those prerequisite action names. Reason freely from
   the user task and the available actions - there is no fixed candidate list.
   Omit actions with no prerequisites.

2. "data_dependencies" - the 'static_policy' entries include a 'data_format'
   mapping (arg_name -> format, e.g. {"to": "email"}) for some actions. For
   each such action, if the user task implies a specific, concrete value for
   that argument (e.g. the task names a specific email address), fill in the
   exact expected value. Only include an action here if 'static_policy'
   declared a data_format for it AND the task text pins down a concrete value.

More examples (format only - your output must be a single JSON object like
the one above, with "tool_dependencies", "data_dependencies", and
"reasoning" keys):

Task: "Find all emails from alice@company.com and forward them to backup folder"
  -> tool_dependencies: {"forward_email": ["search_emails"]}, data_dependencies: {}
  (no specific recipient is named for the forward, so nothing to fill into
  data_dependencies even though forward_email has a data_format.)

Task: "Reschedule my 2pm meeting and notify attendees"
  -> tool_dependencies: {"reschedule_event": ["get_calendar_events"], "send_invite": ["get_calendar_events"]}, data_dependencies: {}

Task: "Look up john's contact info and add him to tomorrow's event"
  -> tool_dependencies: {"add_attendees": ["search_contacts", "get_calendar_events"]}, data_dependencies: {}

Task: "Can you please pay the bill 'bill-december-2023.txt' for me?"
  -> tool_dependencies: {"send_money": ["read_file"]}, data_dependencies: {}
  (the bill must be read before paying it - this holds regardless of
  domain/suite, tool_dependencies is not email-specific.)

Task: "Search my emails for the invoice from Bob, then send him a reply confirming payment"
  -> tool_dependencies: {"send_email": ["search_emails"]}, data_dependencies: {}
  (an action with no prerequisites and no task-specific content, like
  search_emails itself here, is simply omitted from both dicts - don't
  pad the output with empty/trivial entries.)
"""


class DynamicPolicyGenerator:
    """Generates a task-specific dependency overlay once at task start."""

    def __init__(
        self,
        llm: LLM,
        model: str = "Qwen/Qwen3-8B",
    ):
        self.llm = llm
        self.model_name = model
        self.tokenizer = llm.get_tokenizer()

        from vllm.sampling_params import SamplingParams

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
        )

    def generate(
        self,
        task: str,
        privilege_context: PrivilegeContext,
        static_policy: StaticPolicyTable,
        available_actions: Sequence[Any]
    ) -> DynamicPolicy:
        """Generate a task-specific dependency overlay. Runs once per task."""

        available_action_descriptors = []

        for action in available_actions:
            descriptor = _serialize_action_descriptor(action)
            if descriptor["action_name"] in privilege_context.enabled_actions:
                available_action_descriptors.append(descriptor)

        static_rules = [
            rule
            for rule in static_policy.export_rules()
            if rule["action"] in privilege_context.enabled_actions
        ]

        context = {
            "task": task,
            "available_actions": available_action_descriptors,
            "static_policy": static_rules,
        }

        try:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context)},
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

            outputs = self.llm.generate(
                [prompt],
                self.sampling_params,
            )

            raw = outputs[0].outputs[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
                raw = raw.strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                print(f"DEBUG: Failed to parse LLM output: {raw}")
                raise

            valid_actions = {
                descriptor["action_name"]
                for descriptor in available_action_descriptors
            }

            tool_dependencies: dict[str, tuple[str, ...]] = {}

            for action, prereqs in parsed.get("tool_dependencies", {}).items():
                if action not in valid_actions or not isinstance(prereqs, Sequence) or isinstance(prereqs, (str, bytes)):
                    continue

                valid_prereqs = tuple(p for p in prereqs if p in valid_actions and p != action)
                if valid_prereqs:
                    tool_dependencies[action] = valid_prereqs

            data_format_by_action = {
                rule["action"]: rule.get("data_format") or {}
                for rule in context["static_policy"]
            }

            data_dependencies: dict[str, dict[str, str]] = {}

            for action, args in parsed.get("data_dependencies", {}).items():
                if action not in valid_actions or not isinstance(args, Mapping):
                    continue

                allowed_args = data_format_by_action.get(action, {})
                valid_args = {
                    arg: value
                    for arg, value in args.items()
                    if arg in allowed_args
                }
                if valid_args:
                    data_dependencies[action] = valid_args

            return DynamicPolicy(
                tool_dependencies=tool_dependencies,
                data_dependencies=data_dependencies,
                reasoning=parsed.get("reasoning", ""),
            )

        except Exception as exc:
            return DynamicPolicy(
                reasoning=f"Dynamic policy generation failed: {exc}",
            )
