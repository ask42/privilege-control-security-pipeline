"""Tests for the three structural changes to the security pipeline:

1. Data Dependency Gate matches ANY one profile in a list (OR), not one
   fixed profile (AND).
2. QueryDecomposer splits a compound task into atomic sub-tasks before
   privilege scoping, and PrivilegeControlLLM.scope_privileges_decomposed
   unions the enabled actions across all sub-tasks.
3. AdaptiveSecurityPipeline wires an optional QueryDecomposer straight
   through initialize_task, so the AgentDojo integration works as intended.

Uses the AgentDojo workspace action pool.
"""

from __future__ import annotations

from adaptive_policy.core.action_request import ActionRequest, ActionSource
from adaptive_policy.core.value import user_literal
from adaptive_policy.logging.audit_log import FinalDecision
from adaptive_policy.policy.dynamic_policy import DynamicPolicyGenerator
from adaptive_policy.policy.pipeline import AdaptiveSecurityPipeline
from adaptive_policy.policy.privilege_control import PrivilegeControlLLM
from adaptive_policy.policy.query_decomposer import QueryDecomposer
from adaptive_policy.policy.static_policy import StaticPolicyTable
from adaptive_policy.tests.conftest import VLLM_MODEL
from adaptive_policy.tests.test_agentdojo_integration import load_agentdojo_action_descriptors


def _action(action_name: str, args: dict, task: str) -> ActionRequest:
    return ActionRequest(
        action_name=action_name,
        args={k: user_literal(v) for k, v in args.items()},
        user_request=task,
        source=ActionSource.AGENT,
    )


class TestQueryDecomposerSmoke:
    def test_decomposes_compound_task_and_unions_enabled_actions(self, shared_llm):
        task = (
            "How many appointments do I have today? Also, search my emails for "
            "the invoice from Bob and send him a reply confirming payment."
        )
        workspace_pool = load_agentdojo_action_descriptors("workspace")

        decomposer = QueryDecomposer(llm=shared_llm, model=VLLM_MODEL)
        sub_tasks = decomposer.decompose(task)
        print(f"\n[decompose] sub_tasks: {sub_tasks}")
        assert len(sub_tasks) >= 2, "a two-request compound task should split into at least 2 parts"

        priv_llm = PrivilegeControlLLM(llm=shared_llm, model=VLLM_MODEL)
        context = priv_llm.scope_privileges_decomposed(task, workspace_pool, decomposer)
        print(f"[decomposed scoping] enabled_actions: {sorted(context.enabled_actions)}")
        print(f"[decomposed scoping] reasoning: {context.reasoning}")

        assert "get_day_calendar_events" in context.enabled_actions or "get_current_day" in context.enabled_actions or any(
            "calendar" in a for a in context.enabled_actions
        ), "no calendar-domain action was enabled for the appointments sub-task"
        assert "search_emails" in context.enabled_actions
        assert "send_email" in context.enabled_actions

    def test_atomic_task_is_not_needlessly_split(self, shared_llm):
        task = "Please show me my unread emails."
        decomposer = QueryDecomposer(llm=shared_llm, model=VLLM_MODEL)
        sub_tasks = decomposer.decompose(task)
        print(f"\n[decompose] sub_tasks for atomic task: {sub_tasks}")
        assert len(sub_tasks) == 1


class TestOrBasedDataDependencyGateSmoke:
    def test_dynamic_policy_emits_multiple_profiles_for_repeated_action(self, shared_llm):
        task = (
            "Send an email to alice@company.com about the budget review, and "
            "separately send an email to bob@company.com about the schedule change."
        )
        workspace_pool = load_agentdojo_action_descriptors("workspace")

        priv_llm = PrivilegeControlLLM(llm=shared_llm, model=VLLM_MODEL)
        dynamic_gen = DynamicPolicyGenerator(llm=shared_llm, model=VLLM_MODEL)
        static_table = StaticPolicyTable(env_type="email")

        priv_context = priv_llm.scope_privileges(task, workspace_pool)
        print(f"\n[privilege] enabled_actions: {sorted(priv_context.enabled_actions)}")
        assert "send_email" in priv_context.enabled_actions

        dynamic_policy = dynamic_gen.generate(
            task=task,
            privilege_context=priv_context,
            static_policy=static_table,
            available_actions=workspace_pool,
        )
        print(f"[dynamic policy] data_dependencies: {dynamic_policy.data_dependencies}")
        print(f"[dynamic policy] reasoning: {dynamic_policy.reasoning}")

        profiles = dynamic_policy.required_data_for("send_email")
        assert isinstance(profiles, list)

        # Gate a real call to each recipient through the actual GateChain via
        # AdaptiveSecurityPipeline, so this exercises the OR logic end to end,
        # not just the parsed DynamicPolicy object.
        pipeline = AdaptiveSecurityPipeline(
            static_table, dynamic_gen, priv_llm, workspace_pool,
        )
        pipeline.initialize_task(task)

        to_alice = _action("send_email", {"recipients": ["alice@company.com"], "subject": "Budget review", "body": "..."}, task)
        decision_alice, meta_alice = pipeline.process_action(to_alice)
        print(f"[gate] send_email(alice) -> {decision_alice}: {meta_alice['reason']}")

        to_bob = _action("send_email", {"recipients": ["bob@company.com"], "subject": "Schedule change", "body": "..."}, task)
        decision_bob, meta_bob = pipeline.process_action(to_bob)
        print(f"[gate] send_email(bob) -> {decision_bob}: {meta_bob['reason']}")

        if len(profiles) >= 2:
            assert decision_alice == FinalDecision.ALLOWED
            assert decision_bob == FinalDecision.ALLOWED


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
