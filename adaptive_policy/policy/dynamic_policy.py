from __future__ import annotations

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from enum import Enum

if TYPE_CHECKING:
    from vllm import LLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptive_policy.core.action_descriptor import _serialize_action_descriptor
from adaptive_policy.policy.privilege_control import PrivilegeContext


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    VERIFY = "verify"
    DENY = "deny"


@dataclass
class DynamicPolicy:
    """
    Task-specific policy overlay generated once during task initialization.

    Only contains modifications relative to the static policy.
    """

    overrides: dict[str, PolicyDecision] = field(default_factory=dict)
    reasoning: str = ""

    def get(self, action: str) -> PolicyDecision | None:
        return self.overrides.get(action)

    def allows(self, action: str) -> bool:
        return self.overrides.get(action) == PolicyDecision.ALLOW

    def requires_verification(self, action: str) -> bool:
        return self.overrides.get(action) == PolicyDecision.VERIFY

    def denies(self, action: str) -> bool:
        return self.overrides.get(action) == PolicyDecision.DENY


_SYSTEM_PROMPT = """
You are a security policy engine. Return ONLY a single JSON object.

Example output:
{
  "overrides": {
      "create_draft": "allow"
  },
  "reasoning": "Creating a draft is required to fulfill the user request."
}

Rules:
- Keys in overrides MUST be from the provided 'available_actions'.
- Values MUST be exactly: "allow", "verify", or "deny".
- You are provided with a 'static_policy' list of rules, it may not cover all available actions.
- DO NOT generate an override if the static policy already covers the action with an ALLOW rule, unless the specific task context requires a more restrictive decision (VERIFY or DENY).
- ALWAYS generate overrides for actions that are:
    1. Undefined in the static policy.
    2. Require a higher security level (VERIFY or DENY) than the static policy allows.
- Use the static policy rules as the baseline safety floor.
"""


class DynamicPolicyGenerator:
    """
    Generates a task-specific policy overlay once during task initialization.

    The resulting DynamicPolicy is cached and used for the remainder of the task.
    """

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
            max_tokens=2048,
        )

    def generate(
        self,
        task: str,
        privilege_context: PrivilegeContext,
        static_policy: StaticPolicyTable,
        available_actions: Sequence[Any]
    ) -> DynamicPolicy:
        """
        Generate a task-specific policy overlay.

        This runs exactly once per task.
        """

        available_action_descriptors = []

        for action in available_actions:
            descriptor = _serialize_action_descriptor(action)
            if descriptor["action_name"] in privilege_context.enabled_actions:
                available_action_descriptors.append(descriptor)

        context = {
            "task": task,
            "available_actions": available_action_descriptors,
            "static_policy": static_policy.export_rules(),
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

            overrides: dict[str, PolicyDecision] = {}

            for action, decision in parsed.get("overrides", {}).items():
                if action not in valid_actions:
                    continue

                try:
                    overrides[action] = PolicyDecision(decision)
                except ValueError:
                    continue

            return DynamicPolicy(
                overrides=overrides,
                reasoning=parsed.get("reasoning", ""),
            )

        except Exception as exc:
            return DynamicPolicy(
                overrides={},
                reasoning=f"Dynamic policy generation failed: {exc}",
            )