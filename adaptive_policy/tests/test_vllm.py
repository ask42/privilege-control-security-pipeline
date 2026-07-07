import pytest
import sys
from pathlib import Path
from vllm import LLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptive_policy.core.action_request import ActionRequest, ActionSource
from adaptive_policy.core.value import tool_result, user_literal
from adaptive_policy.integrations.benchmark_adapter import AgentDojoToBenchmarkAdapter
from adaptive_policy.logging.audit_log import AuditLog, FinalDecision
from adaptive_policy.policy.gate_chain import GateChain
from adaptive_policy.policy.privilege_control import PrivilegeControlLLM, PrivilegeContext
from adaptive_policy.policy.pipeline import AdaptiveSecurityPipeline
from adaptive_policy.policy.static_policy import StaticPolicyTable

VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return messages[-1]["content"]


class _RecordingLLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.tokenizer = _RecordingTokenizer()
        self.prompts = []

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params):
        self.prompts.extend(prompts)
        return [type("Response", (), {"outputs": [type("Output", (), {"text": self.response_text})()]})()]


@pytest.fixture(scope="session")
def shared_llm():
    return LLM(
        model=VLLM_MODEL,
        dtype="half",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=True,
    )


class TestPrivilegeControlLLM:
    def test_scope_privileges_with_metadata(self):
        llm = PrivilegeControlLLM(
            llm=_RecordingLLM('{"enabled_actions": ["send_email", "get_emails"], "reasoning": "task is about email"}'),
            model=VLLM_MODEL,
        )
        actions = [
            {
                "action_name": "send_email",
                "metadata": {
                    "description": "Send an email",
                    "parameter_names": ["to", "body"],
                    "required_parameters": ["to", "body"],
                    "dependencies": [],
                },
            },
            {
                "action_name": "delete_email",
                "metadata": {
                    "description": "Delete an email",
                    "parameter_names": ["email_id"],
                    "required_parameters": ["email_id"],
                    "dependencies": [],
                },
            },
            {
                "action_name": "get_emails",
                "metadata": {
                    "description": "Read inbox messages",
                    "parameter_names": ["folder"],
                    "required_parameters": ["folder"],
                    "dependencies": [],
                },
            },
        ]

        llm.scope_privileges("Send emails to the team", actions)

        prompt = llm.llm.prompts[0]
        assert '"available_actions"' in prompt
        assert '"description"' in prompt
        assert '"parameter_names"' in prompt
        assert '"required_parameters"' in prompt
        assert '"dependencies"' in prompt

    def test_scope_privileges_returns_subset(self):
        llm = PrivilegeControlLLM(
            llm=_RecordingLLM('{"enabled_actions": ["get_emails"], "reasoning": "read only"}'),
            model=VLLM_MODEL,
        )
        context = llm.scope_privileges(
            "Read the inbox",
            ["send_email", "delete_email", "get_emails"],
        )

        assert isinstance(context.enabled_actions, set)
        assert context.enabled_actions == {"get_emails"}
        assert context.task == "Read the inbox"

    def test_privilege_control_llm_invalid_input_error_handling(self):
        llm = PrivilegeControlLLM(
            llm=_RecordingLLM('{"enabled_actions": [], "reasoning": "invalid input"}'),
            model=VLLM_MODEL,
        )
        context = llm.scope_privileges(task=None, available_actions=None)
        assert len(context.enabled_actions) == 0


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
            task="Clean up history",
        )

        ar = ActionRequest(
            action_name="delete_email",
            args={"email_id": user_literal("msg_123")},
            user_request="Clean up history",
            source=ActionSource.AGENT,
        )

        decision, entry = gate_chain.process_action(ar, privilege_context)
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert entry.escalation_approved is None


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
        priv_llm = PrivilegeControlLLM(
            llm=_RecordingLLM('{"enabled_actions": ["send_email"], "reasoning": "needed"}'),
            model=VLLM_MODEL,
        )

        pipeline = AdaptiveSecurityPipeline(
            static_table,
            None,
            priv_llm,
            ["send_email", "delete_email", "get_emails"],
        )
        pipeline.initialize_task("Send emails to team")

        context = pipeline.get_privilege_context()
        assert "send_email" in context["enabled_actions"]

    def test_pipeline_process_action(self):
        static_table = StaticPolicyTable(env_type="email")
        priv_llm = PrivilegeControlLLM(
            llm=_RecordingLLM('{"enabled_actions": ["send_email", "get_emails"], "reasoning": "needed"}'),
            model=VLLM_MODEL,
        )

        pipeline = AdaptiveSecurityPipeline(
            static_table,
            None,
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
        priv_llm = PrivilegeControlLLM(
            llm=_RecordingLLM('{"enabled_actions": ["get_emails"], "reasoning": "read only"}'),
            model=VLLM_MODEL,
        )

        pipeline = AdaptiveSecurityPipeline(
            static_table,
            None,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
