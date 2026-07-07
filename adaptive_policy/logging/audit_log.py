from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from adaptive_policy.policy.security_policy import Allowed, Denied

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest
    from adaptive_policy.policy.policy_modification import PolicyModification


class FinalDecision(str):
    ALLOWED = "allowed"
    DENIED = "denied"
    ESCALATED = "escalated"
    VERIFICATION_REQUIRED = "verification_required"


@dataclass
class AuditEntry:
    """
    Record of a single action passing through the gate chain.
    Tracks all decisions and escalations.
    """
    action_request_name: str
    user_request: str
    static_decision: str  # "allowed" or "denied"
    static_reason: str
    llm_modification: dict | None = None  # PolicyModification.to_dict()
    final_decision: str = FinalDecision.ALLOWED
    escalation_approved: bool | None = None  # None = not escalated, True/False = human decision
    privilege_added: set[str] = field(default_factory=set)  # actions enabled by escalation
    timestamp: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "action_name": self.action_request_name,
            "user_request": self.user_request,
            "static_decision": self.static_decision,
            "static_reason": self.static_reason,
            "llm_modification": self.llm_modification,
            "final_decision": self.final_decision,
            "escalation_approved": self.escalation_approved,
            "privilege_added": list(self.privilege_added),
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

    def get_action_history(self, action_name: str | None = None) -> list[AuditEntry]:
        """Filter log by action name."""
        if action_name is None:
            return self.entries.copy()
        return [e for e in self.entries if e.action_request_name == action_name]

    def get_escalations(self) -> list[AuditEntry]:
        """Get only escalated actions."""
        return [e for e in self.entries if e.final_decision == FinalDecision.ESCALATED]

    def get_denials(self) -> list[AuditEntry]:
        """Get only denied actions."""
        return [e for e in self.entries if e.final_decision == FinalDecision.DENIED]

    def to_dict(self) -> list[dict]:
        """Serialize log to list of dicts."""
        return [e.to_dict() for e in self.entries]

    def to_json(self) -> str:
        """Serialize log to JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def summary(self) -> dict:
        """Summary statistics."""
        total = len(self.entries)
        allowed = sum(1 for e in self.entries if e.final_decision == FinalDecision.ALLOWED)
        denied = sum(1 for e in self.entries if e.final_decision == FinalDecision.DENIED)
        escalated = sum(1 for e in self.entries if e.final_decision == FinalDecision.ESCALATED)
        escalated_approved = sum(1 for e in self.entries if e.escalation_approved is True)
        escalated_denied = sum(1 for e in self.entries if e.escalation_approved is False)

        return {
            "total_actions": total,
            "allowed": allowed,
            "denied": denied,
            "escalated": escalated,
            "escalated_approved": escalated_approved,
            "escalated_denied": escalated_denied,
            "allow_rate": allowed / total if total > 0 else 0.0,
            "deny_rate": denied / total if total > 0 else 0.0,
            "escalation_rate": escalated / total if total > 0 else 0.0,
        }
