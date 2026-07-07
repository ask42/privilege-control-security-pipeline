from __future__ import annotations

import json

import pytest

from adaptive_policy.core.action_request import ActionRequest, ActionSource
from adaptive_policy.core.task_state import TaskExecutionState
from adaptive_policy.core.value import user_literal
from adaptive_policy.integrations.benchmark_adapter import AgentDojoToBenchmarkAdapter
from adaptive_policy.logging.audit_log import FinalDecision
from adaptive_policy.policy.dynamic_policy import DynamicPolicy, DynamicPolicyGenerator
from adaptive_policy.policy.gate_chain import GateChain
from adaptive_policy.policy.pipeline import AdaptiveSecurityPipeline
from adaptive_policy.policy.privilege_control import PrivilegeContext, PrivilegeControlLLM
from adaptive_policy.policy.static_policy import StaticPolicyTable

"""
Deterministic, LLM-free tests for gate mechanics (static gate, tool/data
dependency gates) and generator JSON parsing, using stub LLMs. No GPU
needed. Live-model quality tests are in test_dynamic_policy.py and
test_agentdojo_integration.py.
"""


class _RecordingTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return messages[-1]["content"]


class _StaticLLM:
    """Deterministic stand-in for a vLLM engine: always returns the same text."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    def get_tokenizer(self):
        return _RecordingTokenizer()

    def generate(self, prompts, sampling_params):
        self.prompts.extend(prompts)
        return [type("Response", (), {"outputs": [type("Output", (), {"text": self.response_text})()]})()]


def _action(name: str, args: dict | None = None, task: str = "task") -> ActionRequest:
    return ActionRequest(
        action_name=name,
        args={k: user_literal(v) for k, v in (args or {}).items()},
        user_request=task,
        source=ActionSource.AGENT,
    )


class TestStaticPolicyGate:
    def test_send_email_allowed_by_declared_rule(self):
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table)
        ar = _action("send_email", {"to": "alice@company.com", "body": "hi"})
        priv = PrivilegeContext({"send_email"}, "task")
        decision, _ = gc.process_action(ar, priv)
        assert decision == FinalDecision.ALLOWED

    def test_delete_email_requires_verification_by_declared_rule(self):
        """Regression test: evaluate() used to always return Allowed()
        regardless of the declared rule. Proves the VERIFY rule is enforced."""
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table)
        ar = _action("delete_email", {"email_id": "m1"})
        priv = PrivilegeContext({"delete_email"}, "task")
        decision, _ = gc.process_action(ar, priv)
        assert decision == FinalDecision.VERIFICATION_REQUIRED

    def test_unregistered_action_defaults_to_verification_required(self):
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table)
        ar = _action("delete_universe")
        priv = PrivilegeContext({"delete_universe"}, "task")
        decision, _ = gc.process_action(ar, priv)
        assert decision == FinalDecision.VERIFICATION_REQUIRED

    def test_dynamic_policy_cannot_influence_allow_verify_deny(self):
        """Dynamic policy has no allow/deny mechanism, only dependencies.
        It must not change a static decision either way."""
        table = StaticPolicyTable(env_type="email")

        gc = GateChain(table, dynamic_policy=DynamicPolicy(tool_dependencies={"send_email": ("get_emails",)}))
        ar = _action("send_email", {"to": "alice@company.com", "body": "hi"})
        priv = PrivilegeContext({"send_email"}, "task")
        # No task_state, so the tool dependency gate can't block. Static ALLOW decides.
        decision, _ = gc.process_action(ar, priv)
        assert decision == FinalDecision.ALLOWED

        # email_id matches what's expected, so the data dependency gate would
        # pass if reached - any VERIFICATION_REQUIRED below is from the static gate.
        gc2 = GateChain(table, dynamic_policy=DynamicPolicy(data_dependencies={"delete_email": {"email_id": "m1"}}))
        ar2 = _action("delete_email", {"email_id": "m1"})
        priv2 = PrivilegeContext({"delete_email"}, "task")
        decision2, entry2 = gc2.process_action(ar2, priv2)
        assert decision2 == FinalDecision.VERIFICATION_REQUIRED
        assert "Static Policy Gate" in entry2.reason

    def test_undefined_static_action_stays_verify_regardless_of_dynamic_policy(self):
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table, dynamic_policy=DynamicPolicy(tool_dependencies={"create_draft": ("get_emails",)}))
        ar = _action("create_draft")
        priv = PrivilegeContext({"create_draft"}, "task")
        decision, _ = gc.process_action(ar, priv)
        assert decision == FinalDecision.VERIFICATION_REQUIRED


class TestToolDependencyGate:
    def test_blocks_when_prerequisite_incomplete(self):
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table, dynamic_policy=DynamicPolicy(tool_dependencies={"forward_email": ("search_emails",)}))
        state = TaskExecutionState(task="t")
        ar = _action("forward_email", {"to": "alice@company.com"})
        priv = PrivilegeContext({"forward_email"}, "task")
        decision, entry = gc.process_action(ar, priv, task_state=state)
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "Tool Dependency Gate" in entry.reason

    def test_allows_once_prerequisite_completed(self):
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table, dynamic_policy=DynamicPolicy(tool_dependencies={"forward_email": ("search_emails",)}))
        state = TaskExecutionState(task="t")
        state.mark_completed("search_emails")
        ar = _action("forward_email", {"to": "alice@company.com"})
        priv = PrivilegeContext({"forward_email"}, "task")
        decision, _ = gc.process_action(ar, priv, task_state=state)
        assert decision == FinalDecision.ALLOWED

    def test_no_task_state_skips_tool_dependency_check(self):
        """No TaskExecutionState means nothing to check completion against,
        so the gate must not block or crash."""
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table, dynamic_policy=DynamicPolicy(tool_dependencies={"forward_email": ("search_emails",)}))
        ar = _action("forward_email", {"to": "alice@company.com"})
        priv = PrivilegeContext({"forward_email"}, "task")
        decision, _ = gc.process_action(ar, priv)
        assert decision == FinalDecision.ALLOWED


class TestDataDependencyGate:
    def test_blocks_on_content_mismatch(self):
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table, dynamic_policy=DynamicPolicy(data_dependencies={"send_email": {"to": "alice@company.com"}}))
        ar = _action("send_email", {"to": "mallory@attacker.com", "body": "hi"})
        priv = PrivilegeContext({"send_email"}, "task")
        decision, entry = gc.process_action(ar, priv)
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "Data Dependency Gate" in entry.reason

    def test_allows_on_content_match(self):
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table, dynamic_policy=DynamicPolicy(data_dependencies={"send_email": {"to": "alice@company.com"}}))
        ar = _action("send_email", {"to": "alice@company.com", "body": "hi"})
        priv = PrivilegeContext({"send_email"}, "task")
        decision, _ = gc.process_action(ar, priv)
        assert decision == FinalDecision.ALLOWED

    def test_blocks_when_expected_arg_missing_entirely(self):
        """Dynamic policy expects a 'subject' value not present in the request."""
        table = StaticPolicyTable(env_type="email")
        gc = GateChain(table, dynamic_policy=DynamicPolicy(data_dependencies={"send_email": {"subject": "Re: invoice"}}))
        ar = _action("send_email", {"to": "alice@company.com", "body": "hi"})
        priv = PrivilegeContext({"send_email"}, "task")
        decision, entry = gc.process_action(ar, priv)
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "Data Dependency Gate" in entry.reason


class TestDynamicPolicyGeneratorParsing:
    """Tests generate()'s JSON parsing/validation with a stub LLM, so the
    filtering logic is covered without depending on live model output."""

    def _generator(self, response_text: str) -> DynamicPolicyGenerator:
        return DynamicPolicyGenerator(llm=_StaticLLM(response_text), model="stub")

    def test_filters_hallucinated_tool_dependency_actions(self):
        response = json.dumps({
            "tool_dependencies": {"send_email": ["search_emails", "delete_universe"]},
            "data_dependencies": {},
            "reasoning": "test",
        })
        gen = self._generator(response)
        table = StaticPolicyTable(env_type="email")
        priv = PrivilegeContext({"send_email", "search_emails"}, "task")
        policy = gen.generate(
            task="task", privilege_context=priv, static_policy=table,
            available_actions=["send_email", "search_emails"],
        )
        assert policy.tool_dependencies["send_email"] == ("search_emails",)

    def test_filters_data_dependency_args_not_declared_in_data_format(self):
        response = json.dumps({
            "tool_dependencies": {},
            "data_dependencies": {"send_email": {"to": "alice@company.com", "body": "malicious injected text"}},
            "reasoning": "test",
        })
        gen = self._generator(response)
        table = StaticPolicyTable(env_type="email")
        priv = PrivilegeContext({"send_email"}, "task")
        policy = gen.generate(
            task="task", privilege_context=priv, static_policy=table,
            available_actions=["send_email"],
        )
        # send_email's static rule only declares data_format for "to", not "body".
        assert policy.data_dependencies["send_email"] == {"to": "alice@company.com"}

    def test_prompt_context_excludes_static_rules_for_non_enabled_actions(self):
        """delete_email is registered but not enabled for this task, so its
        rule should not be sent to the LLM."""
        response = json.dumps({"tool_dependencies": {}, "data_dependencies": {}, "reasoning": "test"})
        stub_llm = _StaticLLM(response)
        gen = DynamicPolicyGenerator(llm=stub_llm, model="stub")
        table = StaticPolicyTable(env_type="email")
        priv = PrivilegeContext({"send_email"}, "task")

        gen.generate(
            task="task", privilege_context=priv, static_policy=table,
            available_actions=["send_email", "delete_email"],
        )

        assert len(stub_llm.prompts) == 1
        assert '"send_email"' in stub_llm.prompts[0]
        assert '"delete_email"' not in stub_llm.prompts[0]

    def test_strips_markdown_fences_before_parsing(self):
        response = "```json\n" + json.dumps({
            "tool_dependencies": {"send_email": ["get_emails"]},
            "reasoning": "fenced",
        }) + "\n```"
        gen = self._generator(response)
        table = StaticPolicyTable(env_type="email")
        priv = PrivilegeContext({"send_email", "get_emails"}, "task")
        policy = gen.generate(
            task="task", privilege_context=priv, static_policy=table,
            available_actions=["send_email", "get_emails"],
        )
        assert policy.tool_dependencies.get("send_email") == ("get_emails",)
        assert policy.reasoning == "fenced"

    def test_malformed_json_fails_closed_with_empty_policy(self):
        gen = self._generator("not json at all")
        table = StaticPolicyTable(env_type="email")
        priv = PrivilegeContext({"send_email"}, "task")
        policy = gen.generate(
            task="task", privilege_context=priv, static_policy=table,
            available_actions=["send_email"],
        )
        assert policy.tool_dependencies == {}
        assert policy.data_dependencies == {}


class TestPrivilegeControlParsing:
    def test_filters_hallucinated_actions_and_strips_fences(self):
        response = "```json\n" + json.dumps({
            "reasoning": "x",
            "enabled_actions": ["send_email", "delete_universe"],
        }) + "\n```"
        llm = PrivilegeControlLLM(llm=_StaticLLM(response), model="stub")
        ctx = llm.scope_privileges("send an email", [
            {"action_name": "send_email"}, {"action_name": "get_emails"},
        ])
        assert ctx.enabled_actions == {"send_email"}

    def test_malformed_json_fails_closed_with_empty_privileges(self):
        llm = PrivilegeControlLLM(llm=_StaticLLM("not json"), model="stub")
        ctx = llm.scope_privileges("do something", [{"action_name": "send_email"}])
        assert ctx.enabled_actions == set()


_GENERIC_PRIVILEGE_SCENARIOS = [
    ("Send an email to Alice and then archive the sent copy.", ["send_email", "archive_email"]),
    ("Read the inbox and then search old email threads.", ["get_emails", "search_emails"]),
    ("Delete the spam message after confirming it is spam.", ["mark_as_spam", "delete_email"]),
    ("Forward the message to Bob after checking the sender.", ["forward_email", "get_emails"]),
    ("Please show me my unread emails.", ["get_emails"]),
]


class TestPrivilegeControlScoping:
    """Stub-LLM checks that scope_privileges() threads a response through to
    enabled_actions. Scoping quality against a real model is tested live in
    test_dynamic_policy.py - a stub can only test plumbing, not judgment."""

    @pytest.mark.parametrize("task,expected_actions", _GENERIC_PRIVILEGE_SCENARIOS)
    def test_scope_privileges_threads_stubbed_response(self, task, expected_actions):
        action_pool = [
            {"action_name": name} for name in
            ["send_email", "archive_email", "get_emails", "search_emails",
             "delete_email", "mark_as_spam", "forward_email"]
        ]
        llm = PrivilegeControlLLM(
            llm=_StaticLLM(json.dumps({"enabled_actions": expected_actions, "reasoning": "stub"})),
            model="stub",
        )
        context = llm.scope_privileges(task, action_pool)
        assert context.enabled_actions == set(expected_actions)


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


class TestFullPipelineDependencySequencing:
    """Tests the full pipeline (not just GateChain) for tool dependencies:
    record_action_result() and task_state plumbing end to end. DynamicPolicy
    is injected directly to stay deterministic."""

    def test_enforces_sequence_and_unblocks_after_record_action_result(self):
        static_table = StaticPolicyTable(env_type="email")
        stub_response = json.dumps({
            "enabled_actions": ["get_emails", "forward_email"],
            "reasoning": "read then write",
        })
        priv_llm = PrivilegeControlLLM(llm=_StaticLLM(stub_response), model="stub")

        pipeline = AdaptiveSecurityPipeline(
            static_policy_table=static_table,
            dynamic_policy_generator=None,
            privilege_control_llm=priv_llm,
            available_actions=["get_emails", "forward_email"],
        )
        pipeline.initialize_task("Read inbox and forward the relevant message")
        pipeline.gate_chain.set_dynamic_policy(
            DynamicPolicy(tool_dependencies={"forward_email": ("get_emails",)})
        )

        forward_request = ActionRequest(
            action_name="forward_email",
            args={"to": user_literal("bob@company.com")},
            user_request="Read inbox and forward the relevant message",
            source=ActionSource.AGENT,
        )

        decision, metadata = pipeline.process_action(forward_request)
        assert decision == FinalDecision.VERIFICATION_REQUIRED
        assert "prerequisite actions" in metadata["reason"].lower()

        pipeline.record_action_result("get_emails", output="msg_123")

        decision, metadata = pipeline.process_action(forward_request)
        assert decision == FinalDecision.ALLOWED
        assert metadata["verification_required"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
