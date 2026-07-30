from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import LLM

from adaptive_policy.core.action_descriptor import _serialize_action_descriptor
from adaptive_policy.policy.privilege_control import PrivilegeContext
from adaptive_policy.policy.static_policy import _matches_data_format


@dataclass
class DynamicPolicy:
    """
    Task-specific dependency overlay, generated once at task start.

    Does NOT decide allow/verify/deny.
    Only describes ordering and content constraints for the actions
    privilege control already enabled.
    """

    tool_dependencies: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Each action maps to a list of alternative expected-argument profiles,
    # so a task can call the same action
    # type multiple times with different content (e.g. send_email to alice,
    # then separately to bob). A real call only needs to match ONE
    # profile in the list (OR), not all of them (AND).
    data_dependencies: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reasoning: str = ""

    def required_actions_for(self, action: str) -> tuple[str, ...]:
        return self.tool_dependencies.get(action, ())

    def required_data_for(self, action: str) -> list[dict[str, Any]]:
        return self.data_dependencies.get(action, [])


_SYSTEM_PROMPT = """
You are a task-specific dependency planner. Return ONLY a single JSON object.

You do not decide allow/deny/verify - that's static policy's job. You only
describe, for this task: (1) "tool_dependencies" - which already-enabled
actions must run before which other already-enabled actions, and (2)
"data_dependencies" - concrete argument content an already-enabled action
should contain, for actions whose static policy rule flags a content format
to check (a 'data_format' mapping, e.g. {"recipients": "email"}).

Example output:
{
  "tool_dependencies": {"send_email": ["search_contacts_by_name"]},
  "data_dependencies": {"send_email": [{"recipients": ["alice@company.com"]}]},
  "reasoning": "The task asks to email alice@company.com, so send_email's recipients must match that address. The contact must be looked up before emailing."
}

tool_dependencies: list an action's prerequisites only when its execution
only makes sense after some other enabled action already ran (look something
up before acting on it, read before writing). Omit actions with none.

data_dependencies: an action normally needs a data_format check on some arg
(e.g. {"recipients": "email"} means that arg must look like an email
address). Only add a data dependency if the task text itself contains an
actual item already in that format - e.g. a string that looks like an email
address, for an "email" format. If the task instead refers to the
recipient/target only by role or relationship - "the client", "my boss",
"the team" - with no item in the required format anywhere in the text, do
NOT add a data dependency for that arg. Leave the action out of
data_dependencies entirely; never convert the description into a
fabricated value in that format just to have something to put there.
Each entry is a list of profiles; a real call only needs to match ONE (e.g.
emailing two different people whose addresses ARE both in the task text
gets two separate profiles, not one merged or dropped entry).

Examples:

Task: "Can you please pay the bill 'bill-december-2023.txt' for me?"
  -> tool_dependencies: {"send_money": ["read_file"]}, data_dependencies: {}

Task: "Email alice@company.com about the budget, and separately email bob@company.com about the schedule"
  -> tool_dependencies: {}, data_dependencies:
  {"send_email": [{"recipients": ["alice@company.com"]}, {"recipients": ["bob@company.com"]}]}

Task: "Please send a summary of the file 'notes.docx' to the client via email."
  -> tool_dependencies: {"send_email": ["search_files_by_filename"]}, data_dependencies: {}
  (no string in the task looks like an email address - "the client" is a
  role, not a format match - so send_email gets no data_dependencies entry,
  even though the file lookup is still a real tool_dependency.)
"""


def _build_enabled_action_contexts(
    available_actions: Sequence[Any],
    enabled_actions: set[str],
    static_policy: StaticPolicyTable,
) -> list[dict[str, Any]]:
    """
    One entry per privilege-enabled action, each carrying its own tool
    descriptor plus its static policy rule nested under "static_policy"
    (None if the action has no declared rule). Merging these up front means
    the LLM reads one self-contained object per action.
    """
    rules_by_action = {
        rule["action"]: {k: v for k, v in rule.items() if k != "action"}
        for rule in static_policy.export_rules()
    }

    action_contexts = []
    for action in available_actions:
        descriptor = _serialize_action_descriptor(action)
        action_name = descriptor["action_name"]
        if action_name not in enabled_actions:
            continue
        descriptor["static_policy"] = rules_by_action.get(action_name)
        action_contexts.append(descriptor)

    return action_contexts


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

        action_contexts = _build_enabled_action_contexts(
            available_actions, privilege_context.enabled_actions, static_policy
        )

        context = {
            "task": task,
            "available_actions": action_contexts,
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
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[1].strip()
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
                for descriptor in action_contexts
            }

            tool_dependencies: dict[str, tuple[str, ...]] = {}

            for action, prereqs in parsed.get("tool_dependencies", {}).items():
                if action not in valid_actions or not isinstance(prereqs, Sequence) or isinstance(prereqs, (str, bytes)):
                    continue

                valid_prereqs = tuple(p for p in prereqs if p in valid_actions and p != action)
                if valid_prereqs:
                    tool_dependencies[action] = valid_prereqs

            data_format_by_action = {
                descriptor["action_name"]: (descriptor["static_policy"] or {}).get("data_format") or {}
                for descriptor in action_contexts
            }

            data_dependencies: dict[str, list[dict[str, Any]]] = {}

            for action, options in parsed.get("data_dependencies", {}).items():
                if action not in valid_actions:
                    continue

                # Tolerate a bare object (one profile) alongside the
                # documented array-of-profiles shape, since models
                # occasionally emit the simpler form despite instructions.
                if isinstance(options, Mapping):
                    options = [options]
                if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
                    continue

                allowed_args = data_format_by_action.get(action, {})
                valid_options = []
                for option in options:
                    if not isinstance(option, Mapping):
                        continue
                    # Drop args not declared in data_format (unknown constraint) and declared args whose own value fails the format check
                    valid_args = {
                        arg: value
                        for arg, value in option.items()
                        if arg in allowed_args and _matches_data_format(value, allowed_args[arg])
                    }
                    if valid_args:
                        valid_options.append(valid_args)

                if valid_options:
                    data_dependencies[action] = valid_options

            return DynamicPolicy(
                tool_dependencies=tool_dependencies,
                data_dependencies=data_dependencies,
                reasoning=parsed.get("reasoning", ""),
            )

        except Exception as exc:
            return DynamicPolicy(
                reasoning=f"Dynamic policy generation failed: {exc}",
            )
