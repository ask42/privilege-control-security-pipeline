from __future__ import annotations

from dataclasses import dataclass, field
from adaptive_policy.core.value import Value


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