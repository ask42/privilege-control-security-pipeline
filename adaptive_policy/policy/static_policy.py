from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, asdict
from enum import Enum

from adaptive_policy.policy.security_policy import Allowed, Denied, PolicyResult, VerificationRequired

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest


class StaticDecision(str, Enum):
    ALLOW = "allow"
    VERIFY = "verify"
    DENY = "deny"


_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _is_valid_email(value: Any) -> bool:
    return bool(_EMAIL_RE.fullmatch(str(value)))


# Maps a data_format label (see StaticPolicyRule.data_format) to a validator.
# Can extend when a new format label is introduced (e.g. "iban" or "date").
FORMAT_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "email": _is_valid_email,
}


def _matches_data_format(value: Any, format_name: str) -> bool:
    """
    True if value (a single item or a list of items) matches format_name.
    An unrecognized format_name has no registered validator, so it's not
    checked (fails open rather than blocking on formats not yet supported).
    A missing/absent value (None - the arg wasn't supplied at all) also
    passes, since there's nothing to validate; this only matters for
    optional args like cc/bcc. An explicitly empty list still fails, since
    that's a provided-but-empty value, not an absent one.
    """
    if value is None:
        return True

    validator = FORMAT_VALIDATORS.get(format_name)
    if validator is None:
        return True

    entries = value if isinstance(value, (list, tuple, set)) else [value]
    if not entries:
        return False
    return all(validator(entry) for entry in entries)


# A data_format key is a canonical/logical field name (e.g. "recipients" for
# a destination address list). Different tool schemas name the same field
# differently - AgentDojo's real email tools use "recipients", hand-written
# fixtures use "to". Check every known alias, not just the canonical key.
_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "recipients": ("recipients", "to", "recipient"),
}


def _resolve_arg_value(action_request: ActionRequest, canonical_name: str) -> Any:
    for alias in _ARG_ALIASES.get(canonical_name, (canonical_name,)):
        value = action_request.get_arg(alias)
        if value is not None:
            return value.raw
    return None


def _values_match(actual: Any, expected: Any) -> bool:
    """
    True if actual (a real action arg's raw value) satisfies expected (a
    dynamic-policy-generated target for it). If actual is a list/tuple/set,
    comparison is order-independent, and expected is treated as the exact
    set of entries actual must contain - no more, no fewer. A real value
    with an extra or different entry than what the task specified is a
    mismatch, not silently allowed, since that divergence from the task's
    intent is exactly what this gate exists to catch.
    """
    if isinstance(actual, (list, tuple, set)):
        expected_entries = expected if isinstance(expected, (list, tuple, set)) else [expected]
        return set(actual) == set(expected_entries)
    return actual == expected


@dataclass(frozen=True)
class StaticPolicyRule:
    action: str
    default_decision: StaticDecision
    description: str
    conditions: tuple[str, ...]
    data_format: Mapping[str, str] = None  # arg_name -> format hint (e.g. {"to": "email"})

    def __post_init__(self):
        if self.data_format is None:
            object.__setattr__(self, "data_format", {})


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

    def default_evaluate(self, action_request: ActionRequest) -> PolicyResult:
        """Check the rule's data_format first, then use `_{action_name}_policy`
        if defined, else the rule's default_decision. No rule at all means
        VerificationRequired."""
        rule = next(
            (r for r in self.export_rules() if r.action == action_request.action_name),
            None,
        )
        if rule is None:
            return VerificationRequired(reason="No static policy rule defined for this action")

        format_failures = [
            arg
            for arg, format_name in rule.data_format.items()
            if not _matches_data_format(_resolve_arg_value(action_request, arg), format_name)
        ]
        if format_failures:
            return VerificationRequired(
                reason=f"{rule.description}: argument(s) do not match required format: "
                + ", ".join(format_failures)
            )

        method_name = f"_{action_request.action_name}_policy"
        if hasattr(self, method_name):
            return getattr(self, method_name)(action_request)

        if rule.default_decision == StaticDecision.ALLOW:
            return Allowed(reason=rule.description)
        if rule.default_decision == StaticDecision.DENY:
            return Denied(reason=rule.description)
        return VerificationRequired(reason=rule.description)


class EmailStaticPolicy(StaticPolicy):
    """
    Hardcoded security policy for the AgentDojo workspace suite's 9 email
    actions (send_email, delete_email, get_unread_emails, get_sent_emails,
    get_received_emails, get_draft_emails, search_emails,
    search_contacts_by_name, search_contacts_by_email).
    """

    def __init__(self, trusted_domains: set[str] | None = None):
        self.trusted_domains = trusted_domains or {"company.com", "internal.org"}

    def evaluate(self, action_request: ActionRequest) -> PolicyResult:
        return self.default_evaluate(action_request)

    def export_rules(self) -> list[StaticPolicyRule]:
        return [
            StaticPolicyRule(
                action="send_email",
                default_decision=StaticDecision.ALLOW,
                description="Send email",
                conditions=(
                    "Recipient domain must be trusted.",
                ),
                data_format={"recipients": "email", "cc": "email", "bcc": "email"},
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
                action="get_unread_emails",
                default_decision=StaticDecision.ALLOW,
                description="Read unread emails",
                conditions=(),
            ),
            StaticPolicyRule(
                action="get_sent_emails",
                default_decision=StaticDecision.ALLOW,
                description="Read sent emails",
                conditions=(),
            ),
            StaticPolicyRule(
                action="get_received_emails",
                default_decision=StaticDecision.ALLOW,
                description="Read received emails",
                conditions=(),
            ),
            StaticPolicyRule(
                action="get_draft_emails",
                default_decision=StaticDecision.ALLOW,
                description="Read draft emails",
                conditions=(),
            ),
            StaticPolicyRule(
                action="search_emails",
                default_decision=StaticDecision.ALLOW,
                description="Search emails",
                conditions=(),
            ),
            StaticPolicyRule(
                action="search_contacts_by_name",
                default_decision=StaticDecision.ALLOW,
                description="Search contacts by name",
                conditions=(),
            ),
            StaticPolicyRule(
                action="search_contacts_by_email",
                default_decision=StaticDecision.ALLOW,
                description="Search contacts by email",
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
            # Register all 9 AgentDojo workspace email actions to use EmailStaticPolicy
            for action in [
                "send_email", "delete_email", "get_unread_emails",
                "get_sent_emails", "get_received_emails", "get_draft_emails",
                "search_emails", "search_contacts_by_name", "search_contacts_by_email",
            ]:
                self.policies[action] = policy

    def evaluate(self, action_request: ActionRequest) -> PolicyResult:
        """Evaluate action against the static policy."""
        policy = self.policies.get(action_request.action_name)
        if policy is None:
            return VerificationRequired(reason="No static policy registered for this action")
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


