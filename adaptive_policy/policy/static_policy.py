from __future__ import annotations

import re
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

# AgentDojo arg formats.
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")


def _is_valid_email(value: Any) -> bool:
    return bool(_EMAIL_RE.fullmatch(str(value)))


def _is_valid_datetime(value: Any) -> bool:
    return bool(_DATETIME_RE.fullmatch(str(value)))


# Maps a data_format label (see StaticPolicyRule.data_format) to a validator.
FORMAT_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "email": _is_valid_email,
    "datetime": _is_valid_datetime,
}


def _matches_data_format(value: Any, format_name: str) -> bool:
    """True if value (or each item, if a list) matches format_name.
    Unknown format_name or None value passes (fails open). Empty list fails."""
    if value is None:
        return True

    validator = FORMAT_VALIDATORS.get(format_name)
    if validator is None:
        return True

    entries = value if isinstance(value, (list, tuple, set)) else [value]
    if not entries:
        return False
    return all(validator(entry) for entry in entries)


# Canonical field -> aliases across differently-named tool schemas
# (AgentDojo's real tools use "recipients", hand-written fixtures use "to").
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
    """True if actual matches expected. Lists compare as sets: order-independent,
    exact (extra or missing entries are a mismatch, not silently allowed)."""
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


class StaticPolicy:
    """
    Evaluates an action request against a fixed list of rules. Sole allow/
    verify/deny gate, dynamic policy cannot loosen its decision.

    TODO: Convert to YAML/JSON config-driven policies.
    """

    def __init__(self, rules: list[StaticPolicyRule]):
        self.rules_by_action = {rule.action: rule for rule in rules}

    def evaluate(self, action_request: ActionRequest) -> PolicyResult:
        rule = self.rules_by_action.get(action_request.action_name)
        if rule is None:
            return Allowed(reason="No static policy rule defined for this action; defaulting to allow")

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

        if rule.default_decision == StaticDecision.ALLOW:
            return Allowed(reason=rule.description)
        if rule.default_decision == StaticDecision.DENY:
            return Denied(reason=rule.description)
        return VerificationRequired(reason=rule.description)

    def export_rules(self) -> list[StaticPolicyRule]:
        return list(self.rules_by_action.values())


# The AgentDojo workspace suite's 9 email actions.
EMAIL_RULES: list[StaticPolicyRule] = [
    StaticPolicyRule(
        action="send_email",
        default_decision=StaticDecision.ALLOW,
        description="Send email",
        conditions=("Recipient domain must be trusted.",),
        data_format={"recipients": "email", "cc": "email", "bcc": "email"},
    ),
    StaticPolicyRule(
        action="delete_email",
        default_decision=StaticDecision.VERIFY,
        description="Delete email",
        conditions=("Requires explicit user authorization.",),
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

# The AgentDojo workspace suite's 7 calendar actions.
CALENDAR_RULES: list[StaticPolicyRule] = [
    StaticPolicyRule(
        action="get_current_day",
        default_decision=StaticDecision.ALLOW,
        description="Read current day",
        conditions=(),
    ),
    StaticPolicyRule(
        action="search_calendar_events",
        default_decision=StaticDecision.ALLOW,
        description="Search calendar events",
        conditions=(),
    ),
    StaticPolicyRule(
        action="get_day_calendar_events",
        default_decision=StaticDecision.ALLOW,
        description="Read a day's calendar events",
        conditions=(),
    ),
    StaticPolicyRule(
        action="create_calendar_event",
        default_decision=StaticDecision.ALLOW,
        description="Create calendar event",
        conditions=(),
        data_format={"participants": "email", "start_time": "datetime", "end_time": "datetime"},
    ),
    StaticPolicyRule(
        action="reschedule_calendar_event",
        default_decision=StaticDecision.ALLOW,
        description="Reschedule calendar event",
        conditions=(),
        data_format={"new_start_time": "datetime", "new_end_time": "datetime"},
    ),
    StaticPolicyRule(
        action="add_calendar_event_participants",
        default_decision=StaticDecision.ALLOW,
        description="Add participants to calendar event",
        conditions=(),
        data_format={"participants": "email"},
    ),
    StaticPolicyRule(
        action="cancel_calendar_event",
        default_decision=StaticDecision.VERIFY,
        description="Cancel calendar event",
        conditions=("Requires explicit user authorization.",),
    ),
]

# The AgentDojo workspace suite's 8 cloud drive actions.
DRIVE_RULES: list[StaticPolicyRule] = [
    StaticPolicyRule(
        action="list_files",
        default_decision=StaticDecision.ALLOW,
        description="List files",
        conditions=(),
    ),
    StaticPolicyRule(
        action="get_file_by_id",
        default_decision=StaticDecision.ALLOW,
        description="Read file by id",
        conditions=(),
    ),
    StaticPolicyRule(
        action="search_files_by_filename",
        default_decision=StaticDecision.ALLOW,
        description="Search files by filename",
        conditions=(),
    ),
    StaticPolicyRule(
        action="search_files",
        default_decision=StaticDecision.ALLOW,
        description="Search files by content",
        conditions=(),
    ),
    StaticPolicyRule(
        action="create_file",
        default_decision=StaticDecision.ALLOW,
        description="Create file",
        conditions=(),
    ),
    StaticPolicyRule(
        action="append_to_file",
        default_decision=StaticDecision.ALLOW,
        description="Append to file",
        conditions=(),
    ),
    StaticPolicyRule(
        action="share_file",
        default_decision=StaticDecision.ALLOW,
        description="Share file",
        conditions=(),
        data_format={"email": "email"},
    ),
    StaticPolicyRule(
        action="delete_file",
        default_decision=StaticDecision.VERIFY,
        description="Delete file",
        conditions=("Requires explicit user authorization.",),
    ),
]

_RULES_BY_ENV_TYPE: dict[str, list[StaticPolicyRule]] = {
    "email": EMAIL_RULES,
    "workspace": EMAIL_RULES + CALENDAR_RULES + DRIVE_RULES,
}

class StaticPolicyTable:
    """Static policy for one environment, scoped by env_type."""

    def __init__(self, env_type: str = "email"):
        self.env_type = env_type
        self.policy = StaticPolicy(_RULES_BY_ENV_TYPE.get(env_type, []))

    def evaluate(self, action_request: ActionRequest) -> PolicyResult:
        """No rule defaults to Allowed (Privilege Gate already restricts scope upstream)."""
        return self.policy.evaluate(action_request)

    def export_rules(self) -> list[dict[str, Any]]:
        rules = []
        for rule in self.policy.export_rules():
            rule_dict = asdict(rule)
            rule_dict["default_decision"] = rule.default_decision.value
            rules.append(rule_dict)
        return rules
