from __future__ import annotations

from dataclasses import dataclass

@dataclass
class PolicyResult:
    """Base class for all policy evaluation results."""
    pass

@dataclass
class Allowed(PolicyResult):
    reason: str = "Action allowed"

@dataclass
class Denied(PolicyResult):
    reason: str = "Action denied"

@dataclass
class VerificationRequired(PolicyResult):
    reason: str = "Action requires user verification"