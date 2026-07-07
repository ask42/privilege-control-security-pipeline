from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, asdict
from enum import Enum

from adaptive_policy.policy.security_policy import Allowed, Denied, PolicyResult

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest


class StaticDecision(str, Enum):
    ALLOW = "allow"
    VERIFY = "verify"
    DENY = "deny"


@dataclass(frozen=True)
class StaticPolicyRule:
    action: str
    default_decision: StaticDecision
    description: str
    conditions: tuple[str, ...]


class StaticPolicy(ABC):
    """
    Base class for environment-specific static policies.
    Evaluates action requests against hardcoded rules.
    
    TODO: Convert to YAML/JSON config-driven policies.
    """

    @abstractmethod
    def evaluate(self, action_request: ActionRequest) -> PolicyResult:
        raise NotImplementedError

    @abstractmethod
    def export_rules(self) -> list[StaticPolicyRule]:
        """Machine-readable representation of this policy."""
        raise NotImplementedError


class EmailStaticPolicy(StaticPolicy):
    """
    Hardcoded security policy for email actions.
    Rules match common email security patterns.
    """

    def __init__(self, trusted_domains: set[str] | None = None):
        self.trusted_domains = trusted_domains or {"company.com", "internal.org"}

    def evaluate(self, action_request: ActionRequest) -> PolicyResult:
        method_name = f"_{action_request.action_name}_policy"
        if hasattr(self, method_name):
            return getattr(self, method_name)(action_request)
        return Allowed()

    def export_rules(self) -> list[StaticPolicyRule]:
        return [
            StaticPolicyRule(
                action="send_email",
                default_decision=StaticDecision.ALLOW,
                description="Send email",
                conditions=(
                    "Recipient domain must be trusted.",
                ),
            ),
            StaticPolicyRule(
                action="delete_email",
                default_decision=StaticDecision.VERIFY,
                description="Delete email",
                conditions=(
                    "Requires explicit user authorization.",
                ),
            ),
            StaticPolicyRule(
                action="forward_email",
                default_decision=StaticDecision.ALLOW,
                description="Forward email",
                conditions=(
                    "Recipient domain must be trusted.",
                ),
            ),
            StaticPolicyRule(
                action="archive_email",
                default_decision=StaticDecision.ALLOW,
                description="Archive email",
                conditions=(),
            ),
            StaticPolicyRule(
                action="mark_as_spam",
                default_decision=StaticDecision.ALLOW,
                description="Mark email as spam",
                conditions=(),
            ),
            StaticPolicyRule(
                action="search_emails",
                default_decision=StaticDecision.ALLOW,
                description="Search emails",
                conditions=(),
            ),
            StaticPolicyRule(
                action="get_emails",
                default_decision=StaticDecision.ALLOW,
                description="Read emails",
                conditions=(),
            ),
        ]


class StaticPolicyTable:
    """
    Registry of static policies per action, scoped to an environment.
    """

    def __init__(self, env_type: str = "email"):
        self.env_type = env_type
        self.policies: dict[str, StaticPolicy] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Initialize default policies for the environment."""
        if self.env_type == "email":
            policy = EmailStaticPolicy()
            # Register all email actions to use EmailStaticPolicy
            for action in [
                "send_email", "delete_email", "forward_email",
                "archive_email", "mark_as_spam", "search_emails", "get_emails"
            ]:
                self.policies[action] = policy

    def evaluate(self, action_request: ActionRequest) -> PolicyResult:
        """Evaluate action against the static policy."""
        policy = self.policies.get(action_request.action_name)
        if policy is None:
            return Allowed()
        return policy.evaluate(action_request)

    def register(self, action_name: str, policy: StaticPolicy) -> None:
        """Register a custom policy for an action."""
        self.policies[action_name] = policy

    def export_rules(self) -> list[dict[str, Any]]:
        rules = []

        seen = set()

        for policy in self.policies.values():
            if id(policy) in seen:
                continue
            seen.add(id(policy))

            for rule in policy.export_rules():
                rule_dict = asdict(rule)
                rule_dict["default_decision"] = rule.default_decision.value
                rules.append(rule_dict)

        return rules    


