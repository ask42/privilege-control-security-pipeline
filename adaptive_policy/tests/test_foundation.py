import pytest
from adaptive_policy import (
    ActionRequest,
    ActionSource,
    AuditEntry,
    AuditLog,
    Allowed,
    StaticPolicy,
    user_literal,
    tool_result,
)
from adaptive_policy.policy.security_policy import VerificationRequired
from adaptive_policy.policy.static_policy import EMAIL_RULES, StaticPolicyTable
from adaptive_policy.core.data_flow import DataFlowTracker


class TestValue:
    def test_user_literal(self):
        v = user_literal("hello")
        assert v.is_user_sourced()
        assert v.is_trusted()
        assert not v.is_tool_sourced()

    def test_tool_result(self):
        v = tool_result("data", tool_name="get_unread_emails")
        assert v.is_tool_sourced()
        assert not v.is_trusted()
        assert v.provenance.tool_name == "get_unread_emails"


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
            action_name="send_email",
            args={
                "body": tool_result("Reply text", "search_emails"),
                "recipients": user_literal("bob@external.com"),
            },
            user_request="Reply to the email",
            source=ActionSource.AGENT,
        )
        assert not ar.all_args_user_sourced()
        untrusted = ar.untrusted_args()
        assert "body" in untrusted
        assert "recipients" not in untrusted


class TestStaticPolicy:
    def test_send_email_allowed_by_declared_rule(self):
        policy = StaticPolicy(EMAIL_RULES)
        ar = ActionRequest(
            action_name="send_email",
            args={"to": user_literal("alice@company.com")},
            user_request="Send",
            source=ActionSource.AGENT,
        )
        result = policy.evaluate(ar)
        assert isinstance(result, Allowed)

    def test_delete_email_requires_verification(self):
        """Deletion asks for verification instead of a hardcoded refusal."""
        policy = StaticPolicy(EMAIL_RULES)
        ar = ActionRequest(
            action_name="delete_email",
            args={"email_id": user_literal("msg_123")},
            user_request="Delete",
            source=ActionSource.AGENT,
        )
        result = policy.evaluate(ar)
        assert isinstance(result, VerificationRequired)

    def test_get_unread_emails_allowed_by_declared_rule(self):
        policy = StaticPolicy(EMAIL_RULES)
        ar = ActionRequest(
            action_name="get_unread_emails",
            args={},
            user_request="Show unread",
            source=ActionSource.AGENT,
        )
        result = policy.evaluate(ar)
        assert isinstance(result, Allowed)

    def test_send_email_cc_bcc_optional_and_format_checked(self):
        policy = StaticPolicy(EMAIL_RULES)
        omitted = ActionRequest(
            action_name="send_email",
            args={"recipients": user_literal("alice@company.com")},
            user_request="Send",
            source=ActionSource.AGENT,
        )
        assert isinstance(policy.evaluate(omitted), Allowed)

        invalid_bcc = ActionRequest(
            action_name="send_email",
            args={
                "recipients": user_literal("alice@company.com"),
                "bcc": user_literal("not-an-email"),
            },
            user_request="Send",
            source=ActionSource.AGENT,
        )
        assert isinstance(policy.evaluate(invalid_bcc), VerificationRequired)

    def test_search_emails_allowed(self):
        policy = StaticPolicy(EMAIL_RULES)
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
        assert "send_email" in table.policy.rules_by_action
        assert "delete_email" in table.policy.rules_by_action

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
            action_name="send_email",
            args={
                "body": tool_result("Reply text", "search_emails"),
                "recipients": user_literal("bob@company.com"),
            },
            user_request="Reply",
            source=ActionSource.AGENT,
        )
        assert not DataFlowTracker.is_all_user_sourced(ar)
        assert "body" in DataFlowTracker.get_untrusted_args(ar)


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

    def test_audit_log_verification_required_tracking(self):
        log = AuditLog()
        log.record(
            AuditEntry(
                action_request_name="delete_email",
                user_request="Clean up",
                static_decision="denied",
                static_reason="Requires verification",
                final_decision="verification_required",
            )
        )
        summary = log.summary()
        assert summary["verification_required"] == 1
        assert summary["verification_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
