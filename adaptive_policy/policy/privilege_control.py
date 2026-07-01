from __future__ import annotations
import json
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
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

Given a user task and a list of structured available actions, determine which actions the agent needs to complete the task.

Each action may include metadata such as descriptions, parameter names, required parameters, and dependencies.
Use that metadata to distinguish clearly required actions from merely related ones.

Be conservative: only enable actions that are clearly required for the task.
Prefer the narrowest action set that satisfies the task.

Respond ONLY with valid JSON in this format:
{"enabled_actions": ["action1", "action2", ...], "reasoning": "brief explanation"}"""


def _serialize_action_descriptor(action: Any) -> dict[str, Any]:
    """Normalize action inputs into a JSON-safe descriptor for the scoping prompt."""
    if isinstance(action, str):
        return {"action_name": action, "metadata": {}}

    if isinstance(action, Mapping):
        action_name = action.get("action_name") or action.get("name") or action.get("tool_name")
        if action_name is None:
            raise ValueError("Action mapping is missing an action name")

        metadata: dict[str, Any] = {}
        nested_metadata = action.get("metadata")
        if isinstance(nested_metadata, Mapping):
            metadata.update(nested_metadata)

        for key in ("description", "parameter_names", "required_parameters", "dependencies", "docstring", "schema"):
            value = action.get(key)
            if value is not None:
                metadata[key] = value

        return {"action_name": action_name, "metadata": metadata}

    action_name = getattr(action, "action_name", None) or getattr(action, "name", None)
    if action_name is None:
        raise ValueError(f"Unsupported action descriptor: {action!r}")

    metadata: dict[str, Any] = {}

    description = getattr(action, "description", None)
    if description is not None:
        metadata["description"] = description

    parameters = getattr(action, "parameters", None)
    if parameters is not None:
        if hasattr(parameters, "model_json_schema"):
            schema = parameters.model_json_schema()
            metadata["parameter_names"] = sorted(schema.get("properties", {}).keys())
            metadata["required_parameters"] = list(schema.get("required", []))
        elif hasattr(parameters, "schema"):
            schema = parameters.schema()
            metadata["parameter_names"] = sorted(schema.get("properties", {}).keys())
            metadata["required_parameters"] = list(schema.get("required", []))
        else:
            metadata["parameter_names"] = []
            metadata["required_parameters"] = []

    dependencies = getattr(action, "dependencies", None)
    if dependencies is not None:
        if isinstance(dependencies, Mapping):
            metadata["dependencies"] = sorted(dependencies.keys())
        elif isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes)):
            metadata["dependencies"] = list(dependencies)
        else:
            metadata["dependencies"] = str(dependencies)

    extra_metadata = getattr(action, "metadata", None)
    if isinstance(extra_metadata, Mapping):
        for key, value in extra_metadata.items():
            metadata.setdefault(key, value)

    return {"action_name": action_name, "metadata": metadata}


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