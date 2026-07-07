import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptive_policy.core.action_request import ActionRequest, ActionSource
from adaptive_policy.core.value import tool_result, user_literal
from adaptive_policy.core.task_state import ActionDependencyRule, TaskExecutionState
from adaptive_policy.integrations.benchmark_adapter import AgentDojoToBenchmarkAdapter
from adaptive_policy.logging.audit_log import FinalDecision, AuditLog
from adaptive_policy.policy.gate_chain import GateChain
from adaptive_policy.policy.llm_policy_engine import LLMPolicyEngine
from adaptive_policy.policy.policy_modification import PolicyDecision
from adaptive_policy.policy.privilege_control import PrivilegeContext, PrivilegeControlLLM
from adaptive_policy.policy.pipeline import AdaptiveSecurityPipeline
from adaptive_policy.policy.static_policy import StaticPolicyTable

VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return messages[-1]["content"]


class _StaticLLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.tokenizer = _RecordingTokenizer()
        self.prompts: list[str] = []

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params):
        self.prompts.extend(prompts)
        return [type("Response", (), {"outputs": [type("Output", (), {"text": self.response_text})()]})()]


class TestPrivilegeControlLLM:
    def test_scope_privileges_with_mock(self):
        llm = PrivilegeControlLLM(
            llm=_StaticLLM('{"enabled_actions": ["send_email", "get_emails"], "reasoning": "task is about email"}'),
            model=VLLM_MODEL,
        )
        context = llm.scope_privileges(
            "Send emails to the team",
            ["send_email", "delete_email", "get_emails"],
        )

        assert "send_email" in context.enabled_actions
        assert "get_emails" in context.enabled_actions
        assert "delete_email" not in context.enabled_actions
        assert context.task == "Send emails to the team"

    def test_privilege_control_llm_error_handling(self):
        llm = PrivilegeControlLLM(
            llm=_StaticLLM('{"enabled_actions": [], "reasoning": "timeout"}'),
            model=VLLM_MODEL,
        )
        context = llm.scope_privileges(
            "Send emails",
            ["send_email", "delete_email"],
        )

        assert len(context.enabled_actions) == 0


class TestLLMPolicyEngine:
    def test_escalate_decision(self):
        engine = LLMPolicyEngine(
            llm=_StaticLLM('{"decision": "escalate", "reason": "User might want to delete", "action_to_enable": "delete_email"}'),
            model=VLLM_MODEL,
        )
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
        engine = LLMPolicyEngine(
            llm=_StaticLLM('{"decision": "tighten", "reason": "Suspicious pattern detected"}'),
            model=VLLM_MODEL,
        )
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
        gate_chain = GateChain(static_table)
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
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "verification required" in entry.reason.lower()

    def test_static_policy_gate_allows_send_trusted(self):
        static_table = StaticPolicyTable(env_type="email")
        gate_chain = GateChain(static_table)
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
        assert entry.reason == "All gates passed"

    def test_static_policy_denied_requires_verification(self):
        static_table = StaticPolicyTable(env_type="email")
        gate_chain = GateChain(static_table)
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

        decision, entry = gate_chain.process_action(ar, privilege_context)
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "verification required" in entry.reason.lower()

    def test_dependency_gate_requires_completed_read_before_write(self):
        static_table = StaticPolicyTable(env_type="email")
        gate_chain = GateChain(static_table)
        privilege_context = PrivilegeContext(
            enabled_actions={"get_emails", "forward_email"},
            task="Read inbox and forward relevant message",
        )
        task_state = TaskExecutionState(task="Read inbox and forward relevant message")
        dependency_rules = {
            "forward_email": ActionDependencyRule(
                required_actions=("get_emails",),
                required_input_args=("email_id", "to"),
                data_sources=("get_emails",),
            )
        }

        forward_request = ActionRequest(
            action_name="forward_email",
            args={
                "email_id": user_literal("msg_123"),
                "to": user_literal("bob@company.com"),
            },
            user_request="Read inbox and forward relevant message",
            source=ActionSource.AGENT,
        )

        decision, entry = gate_chain.process_action(
            forward_request,
            privilege_context,
            task_state=task_state,
            dependency_rules=dependency_rules,
        )

        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "prerequisite actions" in entry.reason.lower()

        task_state.mark_completed("get_emails", output={"email_id": "msg_123"})
        decision, entry = gate_chain.process_action(
            forward_request,
            privilege_context,
            task_state=task_state,
            dependency_rules=dependency_rules,
        )
        assert decision == FinalDecision.ALLOWED

    def test_dependency_gate_rejects_untrusted_data_source(self):
        static_table = StaticPolicyTable(env_type="email")
        gate_chain = GateChain(static_table)
        privilege_context = PrivilegeContext(
            enabled_actions={"search_emails", "forward_email"},
            task="Search inbox and forward a result",
        )
        task_state = TaskExecutionState(task="Search inbox and forward a result")
        task_state.mark_completed("search_emails", output={"email_id": "msg_123"})
        dependency_rules = {
            "forward_email": ActionDependencyRule(
                required_actions=("search_emails",),
                required_input_args=("email_id", "to"),
                data_sources=("search_emails",),
            )
        }

        forward_request = ActionRequest(
            action_name="forward_email",
            args={
                "email_id": user_literal("msg_123"),
                "to": user_literal("bob@company.com"),
            },
            user_request="Search inbox and forward a result",
            source=ActionSource.AGENT,
        )

        decision, entry = gate_chain.process_action(
            forward_request,
            privilege_context,
            task_state=task_state,
            dependency_rules=dependency_rules,
        )
        assert decision == FinalDecision.ALLOWED

        tainted_request = ActionRequest(
            action_name="forward_email",
            args={
                "email_id": tool_result("msg_123", tool_name="search_emails"),
                "to": user_literal("bob@company.com"),
            },
            user_request="Search inbox and forward a result",
            source=ActionSource.AGENT,
        )

        dependency_rules = {
            "forward_email": ActionDependencyRule(
                required_actions=("search_emails",),
                required_input_args=("email_id", "to"),
                data_sources=("forward_email",),
            )
        }

        decision, entry = gate_chain.process_action(
            tainted_request,
            privilege_context,
            task_state=task_state,
            dependency_rules=dependency_rules,
        )
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "data sources" in entry.reason.lower()


class TestBenchmarkAdapter:
    def test_adapt_function_call(self):
        from unittest.mock import MagicMock

        adapter = AgentDojoToBenchmarkAdapter(task_description="Send emails to team")
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
        assert ar.metadata["adapter"] == "agentdojo"
        assert ar.metadata["function_call_id"] == "call_123"


class TestAdaptiveSecurityPipeline:
    def test_pipeline_initialization(self):
        static_table = StaticPolicyTable(env_type="email")
        llm_engine = LLMPolicyEngine(
            llm=_StaticLLM('{"decision": "maintain", "reason": "no change"}'),
            model=VLLM_MODEL,
        )
        priv_llm = PrivilegeControlLLM(
            llm=_StaticLLM('{"enabled_actions": ["send_email"], "reasoning": "needed"}'),
            model=VLLM_MODEL,
        )

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
        llm_engine = LLMPolicyEngine(
            llm=_StaticLLM('{"decision": "maintain", "reason": "no change"}'),
            model=VLLM_MODEL,
        )
        priv_llm = PrivilegeControlLLM(
            llm=_StaticLLM('{"enabled_actions": ["send_email", "get_emails"], "reasoning": "needed"}'),
            model=VLLM_MODEL,
        )

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
        assert metadata["verification_required"] is False

    def test_pipeline_requires_verification_for_disabled_action(self):
        static_table = StaticPolicyTable(env_type="email")
        llm_engine = LLMPolicyEngine(
            llm=_StaticLLM('{"decision": "maintain", "reason": "no change"}'),
            model=VLLM_MODEL,
        )
        priv_llm = PrivilegeControlLLM(
            llm=_StaticLLM('{"enabled_actions": ["get_emails"], "reasoning": "read only"}'),
            model=VLLM_MODEL,
        )

        pipeline = AdaptiveSecurityPipeline(
            static_table,
            llm_engine,
            priv_llm,
            ["send_email", "delete_email", "get_emails"],
        )
        pipeline.initialize_task("Read inbox and clean up")

        ar = ActionRequest(
            action_name="delete_email",
            args={"email_id": user_literal("msg_123")},
            user_request="Read inbox and clean up",
            source=ActionSource.AGENT,
        )

        decision, metadata = pipeline.process_action(ar)
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert metadata["verification_required"] is True
        assert "verification required" in metadata["reason"].lower()

    def test_pipeline_enforces_sequence_and_task_state(self):
        static_table = StaticPolicyTable(env_type="email")
        priv_llm = PrivilegeControlLLM(
            llm=_StaticLLM('{"enabled_actions": ["get_emails", "forward_email"], "reasoning": "read then write"}'),
            model=VLLM_MODEL,
        )
        action_pool = [
            {
                "action_name": "get_emails",
                "metadata": {
                    "description": "Read inbox messages",
                    "parameter_names": ["folder"],
                    "required_parameters": ["folder"],
                    "dependencies": [],
                },
            },
            {
                "action_name": "forward_email",
                "metadata": {
                    "description": "Forward an email",
                    "parameter_names": ["email_id", "to"],
                    "required_parameters": ["email_id", "to"],
                    "dependencies": ["get_emails"],
                    "data_sources": ["get_emails"],
                },
            },
        ]

        pipeline = AdaptiveSecurityPipeline(
            static_table,
            None,
            priv_llm,
            action_pool,
        )
        pipeline.initialize_task("Read inbox and forward the relevant message")

        forward_request = ActionRequest(
            action_name="forward_email",
            args={
                "email_id": tool_result("msg_123", tool_name="get_emails"),
                "to": user_literal("bob@company.com"),
            },
            user_request="Read inbox and forward the relevant message",
            source=ActionSource.AGENT,
        )

        decision, metadata = pipeline.process_action(forward_request)
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "prerequisite actions" in metadata["reason"].lower()

        pipeline.record_action_result("get_emails", output={"email_id": "msg_123"})

        decision, metadata = pipeline.process_action(forward_request)
        assert decision == FinalDecision.ALLOWED
        assert metadata["verification_required"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
