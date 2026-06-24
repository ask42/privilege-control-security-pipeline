from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from adaptive_policy.core.value import Value


class ActionSource(str, Enum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"


@dataclass
class ActionRequest:
    """
    A request to perform an action, flowing through the gate chain.
    Contains provenance info to prevent data-flow injection attacks.
    """
    action_name: str
    args: dict[str, Value]
    user_request: str  # original user query/task (not the action's data)
    source: ActionSource  # did this come from user, agent, or tool?
    metadata: dict = field(default_factory=dict)  # CaMeL-style w/ taint info
    timestamp: float = 0.0

    def get_arg(self, key: str) -> Value | None:
        return self.args.get(key)

    def all_args_user_sourced(self) -> bool:
        """True if all args came directly from the user."""
        return all(arg.is_user_sourced() for arg in self.args.values())

    def untrusted_args(self) -> dict[str, Value]:
        """Return only args that came from tools (untrusted)."""
        return {k: v for k, v in self.args.items() if v.is_tool_sourced()}

    def trusted_args(self) -> dict[str, Value]:
        """Return only args that came from user (trusted)."""
        return {k: v for k, v in self.args.items() if v.is_user_sourced()}
