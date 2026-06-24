from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PolicyDecision(str, Enum):
    """What the LLM policy engine decides to do with a denied action."""
    TIGHTEN = "tighten"  # Deny or upgrade Denied reason
    ESCALATE = "escalate"  # Ask human for permission
    MAINTAIN = "maintain"  # Keep the base decision as-is


@dataclass
class PolicyModification:
    """
    A modification proposed by the LLM policy engine.
    Applied on top of the static policy decision.
    """
    decision: PolicyDecision
    reason: str
    override_field: str | None = None  # e.g., "max_recipients"
    override_value: Any | None = None  # e.g., 3
    action_to_enable: str | None = None  # if ESCALATE, which action to request
    confidence: float = 1.0  # 0.0-1.0, for future learning

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "override_field": self.override_field,
            "override_value": self.override_value,
            "action_to_enable": self.action_to_enable,
            "confidence": self.confidence,
        }
