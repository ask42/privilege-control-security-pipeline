import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2])) # quick fix to access adaptive_policy
from adaptive_policy import (
    ActionRequest,
    ActionSource,
    AuditEntry,
    AuditLog,
    Allowed,
    Denied,
    EmailStaticPolicy,
    PolicyDecision,
    PolicyModification,
    user_literal,
    tool_result,
)
from adaptive_policy.policy.static_policy import StaticPolicyTable
from adaptive_policy.core.data_flow import DataFlowTracker


class TestValue:
    def test_user_literal(self):
        v = user_literal("hello")
        assert v.is_user_sourced()
        assert v.is_trusted()
        assert not v.is_tool_sourced()

    def test_tool_result(self):
        v = tool_result("data", tool_name="get_emails")
        assert v.is_tool_sourced()
        assert not v.is_trusted()
        assert v.provenance.tool_name == "get_emails"


class TestActionRequest:
    def test_action_request_creation(self):
        ar = ActionRequest(
            action_name="send_email",
            args={
                "to": user_literal("alice@company.com"),
                "body": user_literal("hello"),
            },
            user_request="Send email to alice",
            source=ActionSource.AGENT,
        )
        assert ar.action_name == "send_email"
        assert ar.all_args_user_sourced()

    def test_untrusted_args(self):
        ar = ActionRequest(
            action_name="forward_email",
            args={
                "email_id": tool_result("msg_123", "get_emails"),
                "to": user_literal("bob@external.com"),
            },
            user_request="Forward the email",
            source=ActionSource.AGENT,
        )
        assert not ar.all_args_user_sourced()
        untrusted = ar.untrusted_args()
        assert "email_id" in untrusted
        assert "to" not in untrusted


class TestStaticPolicy:
    def test_send_email_trusted_domain(self):
        policy = EmailStaticPolicy(trusted_domains={"company.com"})
        ar = ActionRequest(
            action_name="send_email",
            args={"to": user_literal("alice@company.com")},
            user_request="Send",
            source=ActionSource.AGENT,
        )
        result = policy.evaluate(ar)
        assert isinstance(result, Allowed)

    def test_send_email_untrusted_domain(self):
        policy = EmailStaticPolicy(trusted_domains={"company.com"})
        ar = ActionRequest(
            action_name="send_email",
            args={"to": user_literal("attacker@evil.com")},
            user_request="Send",
            source=ActionSource.AGENT,
        )
        result = policy.evaluate(ar)
        assert isinstance(result, Denied)
        assert "evil.com" in result.reason

    def test_delete_email_denied(self):
        policy = EmailStaticPolicy()
        ar = ActionRequest(
            action_name="delete_email",
            args={"email_id": user_literal("msg_123")},
            user_request="Delete",
            source=ActionSource.AGENT,
        )
        result = policy.evaluate(ar)
        assert isinstance(result, Denied)

    def test_forward_email_trusted_domain(self):
        policy = EmailStaticPolicy(trusted_domains={"company.com"})
        ar = ActionRequest(
            action_name="forward_email",
            args={"to": user_literal("bob@company.com")},
            user_request="Forward",
            source=ActionSource.AGENT,
        )
        result = policy.evaluate(ar)
        assert isinstance(result, Allowed)

    def test_search_emails_allowed(self):
        policy = EmailStaticPolicy()
        ar = ActionRequest(
            action_name="search_emails",
            args={"query": user_literal("invoice")},
            user_request="Search",
            source=ActionSource.AGENT,
        )
        result = policy.evaluate(ar)
        assert isinstance(result, Allowed)


class TestStaticPolicyTable:
    def test_policy_table_init(self):
        table = StaticPolicyTable(env_type="email")
        assert "send_email" in table.policies
        assert "delete_email" in table.policies

    def test_policy_table_evaluate(self):
        table = StaticPolicyTable(env_type="email")
        ar = ActionRequest(
            action_name="send_email",
            args={"to": user_literal("alice@company.com")},
            user_request="Send",
            source=ActionSource.AGENT,
        )
        result = table.evaluate(ar)
        assert isinstance(result, Allowed)


class TestDataFlowTracker:
    def test_all_user_sourced(self):
        ar = ActionRequest(
            action_name="send_email",
            args={
                "to": user_literal("alice@company.com"),
                "body": user_literal("hi"),
            },
            user_request="Send",
            source=ActionSource.AGENT,
        )
        assert DataFlowTracker.is_all_user_sourced(ar)

    def test_mixed_provenance(self):
        ar = ActionRequest(
            action_name="forward_email",
            args={
                "email_id": tool_result("msg_123", "get_emails"),
                "to": user_literal("bob@company.com"),
            },
            user_request="Forward",
            source=ActionSource.AGENT,
        )
        assert not DataFlowTracker.is_all_user_sourced(ar)
        assert "email_id" in DataFlowTracker.get_untrusted_args(ar)


class TestPolicyModification:
    def test_modification_maintain(self):
        mod = PolicyModification(
            decision=PolicyDecision.MAINTAIN,
            reason="Static policy sufficient",
        )
        assert mod.decision == PolicyDecision.MAINTAIN
        d = mod.to_dict()
        assert d["decision"] == "maintain"

    def test_modification_escalate(self):
        mod = PolicyModification(
            decision=PolicyDecision.ESCALATE,
            reason="Delete action requires approval",
            action_to_enable="delete_email",
        )
        assert mod.action_to_enable == "delete_email"
        d = mod.to_dict()
        assert d["action_to_enable"] == "delete_email"


class TestAuditLog:
    def test_audit_entry_creation(self):
        entry = AuditEntry(
            action_request_name="send_email",
            user_request="Send email",
            static_decision="allowed",
            static_reason="Trusted domain",
            timestamp=1234.5,
        )
        assert entry.action_request_name == "send_email"
        assert entry.static_decision == "allowed"

    def test_audit_log_record(self):
        log = AuditLog()
        entry = AuditEntry(
            action_request_name="send_email",
            user_request="Send email",
            static_decision="allowed",
            static_reason="Trusted domain",
        )
        log.record(entry)
        assert len(log.get_entries()) == 1

    def test_audit_log_summary(self):
        log = AuditLog()
        log.record(
            AuditEntry(
                action_request_name="send_email",
                user_request="Send",
                static_decision="allowed",
                static_reason="OK",
                final_decision="allowed",
            )
        )
        log.record(
            AuditEntry(
                action_request_name="delete_email",
                user_request="Delete",
                static_decision="denied",
                static_reason="Not allowed",
                final_decision="denied",
            )
        )
        summary = log.summary()
        assert summary["total_actions"] == 2
        assert summary["allowed"] == 1
        assert summary["denied"] == 1

    def test_audit_log_escalation_tracking(self):
        log = AuditLog()
        log.record(
            AuditEntry(
                action_request_name="delete_email",
                user_request="Clean up",
                static_decision="denied",
                static_reason="Requires auth",
                final_decision="escalated",
                escalation_approved=True,
                privilege_added={"delete_email"},
            )
        )
        summary = log.summary()
        assert summary["escalated"] == 1
        assert summary["escalated_approved"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
