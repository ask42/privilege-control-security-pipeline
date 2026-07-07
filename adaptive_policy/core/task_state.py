from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any
from adaptive_policy.core.value import Value

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

@dataclass(frozen=True)
class ActionDependencyRule:
    """Declarative dependency constraints for an action."""

    required_actions: tuple[str, ...] = ()
    required_input_args: tuple[str, ...] = ()
    data_sources: tuple[str, ...] = ()
    verification_note: str = ""


@dataclass
class TaskExecutionState:
    """Runtime state for a single task execution."""

    task: str
    completed_actions: list[str] = field(default_factory=list)
    action_outputs: dict[str, Value] = field(default_factory=dict)
    verified_actions: set[str] = field(default_factory=set) # can be used later to track user verification

    def mark_completed(self, action_name: str, output: Value | None = None) -> None:
        if action_name not in self.completed_actions:
            self.completed_actions.append(action_name)
        if output is not None:
            self.action_outputs[action_name] = output

    def has_completed(self, action_name: str) -> bool:
        return action_name in self.completed_actions


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _extract_descriptor(action: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(action, str):
        return action, {}

    if isinstance(action, Mapping):
        action_name = action.get("action_name") or action.get("name") or action.get("tool_name")
        if action_name is None:
            raise ValueError("Action mapping is missing an action name")
        metadata = action.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        return str(action_name), metadata

    action_name = getattr(action, "action_name", None) or getattr(action, "name", None)
    if action_name is None:
        raise ValueError(f"Unsupported action descriptor: {action!r}")
    metadata = getattr(action, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return str(action_name), metadata


def build_dependency_rules(available_actions: Sequence[Any] | None) -> dict[str, ActionDependencyRule]:
    """Build dependency rules from action descriptors."""

    dependency_rules: dict[str, ActionDependencyRule] = {}
    for action in available_actions or []:
        action_name, metadata = _extract_descriptor(action)
        dependency_rules[action_name] = ActionDependencyRule(
            required_actions=_as_tuple(metadata.get("dependencies")),
            required_input_args=_as_tuple(metadata.get("required_input_args", metadata.get("required_parameters"))),
            data_sources=_as_tuple(metadata.get("data_sources", metadata.get("input_sources"))),
            verification_note=str(metadata.get("verification_note", "")),
        )
    return dependency_rules