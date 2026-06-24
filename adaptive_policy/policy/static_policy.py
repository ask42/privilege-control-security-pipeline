from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from adaptive_policy.policy.security_policy import Allowed, Denied, PolicyResult

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest


class StaticPolicy(ABC):
    """
    Base class for environment-specific static policies.
    Evaluates action requests against hardcoded rules.
    
    TODO: Convert to YAML/JSON config-driven policies.
    """

    @abstractmethod
    def evaluate(self, action_request: ActionRequest) -> PolicyResult:
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

    def _send_email_policy(self, action_request: ActionRequest) -> PolicyResult:
        """Allow if recipients are in trusted domains."""
        recipients = action_request.get_arg("to")
        if recipients:
            email = recipients.raw
            if "@" in email:
                domain = email.split("@")[-1]
                if domain not in self.trusted_domains:
                    return Denied(f"Recipient domain {domain} is not in trusted domains")
        return Allowed()

    def _delete_email_policy(self, action_request: ActionRequest) -> PolicyResult:
        """Deny delete without explicit authorization."""
        return Denied("Deleting emails requires explicit user authorization")

    def _forward_email_policy(self, action_request: ActionRequest) -> PolicyResult:
        """Allow forwarding to trusted domains; deny otherwise."""
        to_addr = action_request.get_arg("to")
        if to_addr:
            email = to_addr.raw
            if "@" in email:
                domain = email.split("@")[-1]
                if domain not in self.trusted_domains:
                    return Denied(f"Cannot forward to untrusted domain {domain}")
        return Allowed()

    def _archive_email_policy(self, action_request: ActionRequest) -> PolicyResult:
        """Allow archiving."""
        return Allowed()

    def _mark_as_spam_policy(self, action_request: ActionRequest) -> PolicyResult:
        """Allow marking as spam."""
        return Allowed()

    def _search_emails_policy(self, action_request: ActionRequest) -> PolicyResult:
        """Allow email search."""
        return Allowed()

    def _get_emails_policy(self, action_request: ActionRequest) -> PolicyResult:
        """Allow reading emails."""
        return Allowed()


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
