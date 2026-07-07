from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest


class FinalDecision(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    VERIFICATION_REQUIRED = "verification_required"


@dataclass
class AuditEntry:
    """Record of a single action passing through the gate chain."""
    action_request_name: str
    user_request: str
    static_decision: str  # "allowed" or "denied"
    static_reason: str
    final_decision: str = FinalDecision.ALLOWED
    timestamp: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "action_name": self.action_request_name,
            "user_request": self.user_request,
            "static_decision": self.static_decision,
            "static_reason": self.static_reason,
            "final_decision": self.final_decision,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


class AuditLog:
    """
    Append-only log of all action requests and gate decisions.
    """

    def __init__(self):
        self.entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        """Add an entry to the log."""
        self.entries.append(entry)

    def get_entries(self) -> list[AuditEntry]:
        """Get all log entries."""
        return self.entries.copy()

    def to_dict(self) -> list[dict]:
        """Serialize log to list of dicts."""
        return [e.to_dict() for e in self.entries]

    def summary(self) -> dict:
        """Summary statistics."""
        total = len(self.entries)
        allowed = sum(1 for e in self.entries if e.final_decision == FinalDecision.ALLOWED)
        denied = sum(1 for e in self.entries if e.final_decision == FinalDecision.DENIED)
        verification_required = sum(
            1 for e in self.entries if e.final_decision == FinalDecision.VERIFICATION_REQUIRED
        )

        return {
            "total_actions": total,
            "allowed": allowed,
            "denied": denied,
            "verification_required": verification_required,
            "allow_rate": allowed / total if total > 0 else 0.0,
            "deny_rate": denied / total if total > 0 else 0.0,
            "verification_rate": verification_required / total if total > 0 else 0.0,
        }
