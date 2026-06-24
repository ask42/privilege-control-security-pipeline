import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2])) # quick fix to access adaptive_policy
from adaptive_policy.core.action_request import (
    ActionRequest,
    ActionSource,
)
from adaptive_policy.core.value import user_literal
from adaptive_policy.logging.audit_log import (
    FinalDecision,
    AuditLog
)
from adaptive_policy.integrations.benchmark_adapter import AgentDojoToBenchmarkAdapter
from adaptive_policy.policy.gate_chain import GateChain
from adaptive_policy.policy.llm_policy_engine import LLMPolicyEngine
from adaptive_policy.policy.policy_modification import PolicyDecision, PolicyModification
from adaptive_policy.policy.privilege_control import PrivilegeContext, PrivilegeControlLLM
from adaptive_policy.policy.pipeline import AdaptiveSecurityPipeline
from adaptive_policy.policy.static_policy import StaticPolicyTable


class TestPrivilegeControlLLM:
    def test_scope_privileges_with_mock(self):
        with patch("adaptive_policy.policy.privilege_control.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_choice = MagicMock()
            mock_choice.message.content = '{"enabled_actions": ["send_email", "get_emails"], "reasoning": "task is about email"}'
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )

            llm = PrivilegeControlLLM()
            context = llm.scope_privileges(
                "Send emails to the team",
                ["send_email", "delete_email", "get_emails"],
            )

        assert "send_email" in context.enabled_actions
        assert "get_emails" in context.enabled_actions
        assert "delete_email" not in context.enabled_actions
        assert context.task == "Send emails to the team"

    def test_privilege_control_llm_error_handling(self):
        with patch("adaptive_policy.policy.privilege_control.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = RuntimeError("timeout")

            llm = PrivilegeControlLLM()
            context = llm.scope_privileges(
                "Send emails",
                ["send_email", "delete_email"],
            )

        # Fails safely, with no actions enabled
        assert len(context.enabled_actions) == 0


class TestLLMPolicyEngine:
    def test_escalate_decision(self):
        with patch("adaptive_policy.policy.llm_policy_engine.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_choice = MagicMock()
            mock_choice.message.content = '{"decision": "escalate", "reason": "User might want to delete", "action_to_enable": "delete_email"}'
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )

            engine = LLMPolicyEngine()
            ar = ActionRequest(
                action_name="delete_email",
                args={"email_id": user_literal("msg_123")},
                user_request="Clean up old emails",
                source=ActionSource.AGENT,
            )
            audit_log = AuditLog()

            mod = engine.evaluate(ar, "Requires explicit authorization", audit_log)

        assert mod.decision == PolicyDecision.ESCALATE
        assert mod.action_to_enable == "delete_email"

    def test_tighten_decision(self):
        with patch("adaptive_policy.policy.llm_policy_engine.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_choice = MagicMock()
            mock_choice.message.content = '{"decision": "tighten", "reason": "Suspicious pattern detected"}'
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )

            engine = LLMPolicyEngine()
            ar = ActionRequest(
                action_name="forward_email",
                args={"to": user_literal("attacker@evil.com")},
                user_request="Forward",
                source=ActionSource.AGENT,
            )
            audit_log = AuditLog()

            mod = engine.evaluate(ar, "Untrusted domain", audit_log)

        assert mod.decision == PolicyDecision.TIGHTEN


class TestGateChain:
    def test_privilege_gate_blocks_disabled_action(self):
        static_table = StaticPolicyTable(env_type="email")
        with patch("adaptive_policy.policy.llm_policy_engine.OpenAI"):
            llm_engine = LLMPolicyEngine()

        gate_chain = GateChain(static_table, llm_engine)
        privilege_context = PrivilegeContext(
            enabled_actions={"send_email", "get_emails"},
            task="Send emails",
        )

        ar = ActionRequest(
            action_name="delete_email",
            args={"email_id": user_literal("msg_123")},
            user_request="Delete",
            source=ActionSource.AGENT,
        )

        decision, entry = gate_chain.process_action(ar, privilege_context)
        assert decision == FinalDecision.DENIED
        assert "Privilege" in entry.reason or "not enabled" in entry.reason

    def test_static_policy_gate_allows_send_trusted(self):
        static_table = StaticPolicyTable(env_type="email")
        with patch("adaptive_policy.policy.llm_policy_engine.OpenAI"):
            llm_engine = LLMPolicyEngine()

        gate_chain = GateChain(static_table, llm_engine)
        privilege_context = PrivilegeContext(
            enabled_actions={"send_email"},
            task="Send emails",
        )

        ar = ActionRequest(
            action_name="send_email",
            args={"to": user_literal("alice@company.com")},
            user_request="Send",
            source=ActionSource.AGENT,
        )

        decision, entry = gate_chain.process_action(ar, privilege_context)
        assert decision == FinalDecision.ALLOWED

    def test_escalation_workflow_approved(self):
        static_table = StaticPolicyTable(env_type="email")
        with patch("adaptive_policy.policy.llm_policy_engine.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_choice = MagicMock()
            mock_choice.message.content = '{"decision": "escalate", "reason": "User might want to delete", "action_to_enable": "delete_email"}'
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            llm_engine = LLMPolicyEngine()

            gate_chain = GateChain(static_table, llm_engine)
            # Enable delete_email so privilege gate passes, but static policy will deny it
            privilege_context = PrivilegeContext(
                enabled_actions={"get_emails", "delete_email"},
                task="Clean up",
            )

            ar = ActionRequest(
                action_name="delete_email",
                args={"email_id": user_literal("msg_123")},
                user_request="Clean up",
                source=ActionSource.AGENT,
            )

            # Mock escalation callback that approves
            def mock_escalation(reason, action_to_enable):
                return True

            decision, entry = gate_chain.process_action(ar, privilege_context, mock_escalation)
            assert decision == FinalDecision.ESCALATED
            assert entry.escalation_approved is True
            # Mocks the LLM requesting to enable it (even though it's already enabled)
            assert entry.llm_modification is not None


class TestBenchmarkAdapter:
    def test_adapt_function_call(self):
        adapter = AgentDojoToBenchmarkAdapter(task_description="Send emails to team")

        # Mock FunctionCall
        func_call = MagicMock()
        func_call.function = "send_email"
        func_call.args = {"to": "alice@company.com", "body": "hello"}
        func_call.id = "call_123"

        ar = adapter.adapt(func_call)

        assert ar.action_name == "send_email"
        assert ar.user_request == "Send emails to team"
        assert ar.source == ActionSource.AGENT
        assert ar.get_arg("to").raw == "alice@company.com"
        assert ar.get_arg("to").is_user_sourced()


class TestAdaptiveSecurityPipeline:
    def test_pipeline_initialization(self):
        static_table = StaticPolicyTable(env_type="email")
        with patch("adaptive_policy.policy.llm_policy_engine.OpenAI"):
            llm_engine = LLMPolicyEngine()
        with patch("adaptive_policy.policy.privilege_control.OpenAI") as mock_priv:
            mock_client = MagicMock()
            mock_priv.return_value = mock_client
            mock_choice = MagicMock()
            mock_choice.message.content = '{"enabled_actions": ["send_email"], "reasoning": "needed"}'
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            priv_llm = PrivilegeControlLLM()

        pipeline = AdaptiveSecurityPipeline(
            static_table,
            llm_engine,
            priv_llm,
            ["send_email", "delete_email", "get_emails"],
        )
        pipeline.initialize_task("Send emails to team")

        context = pipeline.get_privilege_context()
        assert "send_email" in context["enabled_actions"]

    def test_pipeline_process_action(self):
        static_table = StaticPolicyTable(env_type="email")
        with patch("adaptive_policy.policy.llm_policy_engine.OpenAI"):
            llm_engine = LLMPolicyEngine()
        with patch("adaptive_policy.policy.privilege_control.OpenAI") as mock_priv:
            mock_client = MagicMock()
            mock_priv.return_value = mock_client
            mock_choice = MagicMock()
            mock_choice.message.content = '{"enabled_actions": ["send_email", "get_emails"], "reasoning": ""}'
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            priv_llm = PrivilegeControlLLM()

        pipeline = AdaptiveSecurityPipeline(
            static_table,
            llm_engine,
            priv_llm,
            ["send_email", "delete_email", "get_emails"],
        )
        pipeline.initialize_task("Send email")

        ar = ActionRequest(
            action_name="send_email",
            args={"to": user_literal("alice@company.com")},
            user_request="Send email",
            source=ActionSource.AGENT,
        )

        decision, metadata = pipeline.process_action(ar)
        assert decision == FinalDecision.ALLOWED
        assert metadata["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])