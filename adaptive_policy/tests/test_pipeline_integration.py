import pytest
import sys
from pathlib import Path
from vllm import LLM
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_policy.core.action_request import ActionRequest, ActionSource
from adaptive_policy.core.value import user_literal
from adaptive_policy.logging.audit_log import AuditLog, FinalDecision
from adaptive_policy.policy.llm_policy_engine import LLMPolicyEngine
from adaptive_policy.policy.privilege_control import PrivilegeControlLLM
from adaptive_policy.policy.pipeline import AdaptiveSecurityPipeline
from adaptive_policy.policy.static_policy import StaticPolicyTable

VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"


@pytest.fixture(scope="session")
def shared_llm():
    return LLM(
        model=VLLM_MODEL,
        dtype="half",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=True,
    )


class TestPrivilegeControlIntegration:
    def test_scope_privileges(self, shared_llm):
        llm = PrivilegeControlLLM(
            llm=shared_llm,
            model=VLLM_MODEL,
        )

        context = llm.scope_privileges(
            "Send emails to the engineering team",
            [
                "send_email",
                "delete_email",
                "get_emails",
                "forward_email",
            ],
        )

        assert context.task == "Send emails to the engineering team"
        assert isinstance(context.enabled_actions, set)
        assert len(context.enabled_actions) > 0
        assert context.enabled_actions.issubset(
            {
                "send_email",
                "delete_email",
                "get_emails",
                "forward_email",
            }
        )


class TestLLMPolicyEngineIntegration:
    def test_policy_engine_returns_valid_decision(self, shared_llm):
        engine = LLMPolicyEngine(
            llm=shared_llm,
            model=VLLM_MODEL,
        )

        action = ActionRequest(
            action_name="delete_email",
            args={"email_id": user_literal("msg_123")},
            user_request="Delete an old email.",
            source=ActionSource.AGENT,
        )

        audit_log = AuditLog()
        decision = engine.evaluate(
            action,
            "Requires explicit authorization.",
            audit_log,
        )

        assert decision.decision is not None
        assert decision.reason


class TestPipelineIntegration:
    def test_pipeline_runs_end_to_end(self, shared_llm):
        static_table = StaticPolicyTable(env_type="email")

        privilege_llm = PrivilegeControlLLM(
            llm=shared_llm,
            model=VLLM_MODEL,
        )

        policy_llm = LLMPolicyEngine(
            llm=shared_llm,
            model=VLLM_MODEL,
        )

        pipeline = AdaptiveSecurityPipeline(
            static_table,
            policy_llm,
            privilege_llm,
            [
                "send_email",
                "delete_email",
                "get_emails",
                "forward_email",
            ],
        )

        pipeline.initialize_task("Send an email to Alice.")

        action = ActionRequest(
            action_name="send_email",
            args={"to": user_literal("alice@company.com")},
            user_request="Send an email to Alice.",
            source=ActionSource.AGENT,
        )

        decision, metadata = pipeline.process_action(action)

        assert decision in (
            FinalDecision.ALLOWED,
            FinalDecision.VERIFICATION_REQUIRED,
        )

        assert "reason" in metadata
        assert isinstance(metadata["verification_required"], bool)