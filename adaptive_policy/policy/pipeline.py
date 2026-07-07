from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adaptive_policy.logging.audit_log import AuditLog, FinalDecision
from adaptive_policy.policy.gate_chain import GateChain
from adaptive_policy.core.task_state import TaskExecutionState
from adaptive_policy.policy.dynamic_policy import (
    DynamicPolicy,
    DynamicPolicyGenerator,
)
from adaptive_policy.policy.privilege_control import (
    PrivilegeControlLLM,
    PrivilegeContext,
)

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest
    from adaptive_policy.policy.static_policy import StaticPolicyTable


class AdaptiveSecurityPipeline:
    """
    Main orchestrator for the security policy framework.

    1. Scope initial privileges
    2. Generate the dynamic policy
    3. Run each action through the gate chain
    4. Return results + audit log
    """

    def __init__(
        self,
        static_policy_table: StaticPolicyTable,
        dynamic_policy_generator: DynamicPolicyGenerator | None = None,
        privilege_control_llm: PrivilegeControlLLM | None = None,
        available_actions: list[Any] | None = None,
    ):
        self.static_policy = static_policy_table
        if privilege_control_llm is None:
            raise ValueError("privilege_control_llm is required")
        self.privilege_control_llm = privilege_control_llm
        self.available_actions = available_actions or []
        self.dynamic_policy_generator = dynamic_policy_generator
        self.dynamic_policy: DynamicPolicy | None = None
        self.audit_log = AuditLog()
        self.gate_chain = GateChain(
            static_policy_table=static_policy_table,
            audit_log=self.audit_log,
            dynamic_policy=None,
        )
        self.privilege_context = None
        self.task_state = TaskExecutionState(task="")

    def initialize_task(self, task_description: str) -> None:
        """Runs once at task start to scope initial privileges."""
        self.task_state = TaskExecutionState(task=task_description)
        self.privilege_context = self.privilege_control_llm.scope_privileges(
            task_description,
            self.available_actions,
        )
        if self.dynamic_policy_generator is not None:
            self.dynamic_policy = self.dynamic_policy_generator.generate(
                task=task_description,
                privilege_context=self.privilege_context,
                static_policy=self.static_policy,
                available_actions=self.available_actions,
            )

        self.gate_chain.set_dynamic_policy(self.dynamic_policy)

    def process_action(
        self,
        action_request: ActionRequest,
    ) -> tuple[str, dict]:
        """
        Process a single action through the gate chain.

        Returns (decision, metadata). decision is one of
        "allowed" | "denied" | "verification_required".
        """
        if self.privilege_context is None:
            raise RuntimeError("Must call initialize_task() first")

        decision, audit_entry = self.gate_chain.process_action(
            action_request,
            self.privilege_context,
            task_state=self.task_state,
        )

        metadata = {
            "decision": decision,
            "action": action_request.action_name,
            "reason": audit_entry.reason,
            "audit_entry": audit_entry.to_dict(),
            "dynamic_policy": (
                {
                    "tool_dependencies": list(
                        self.dynamic_policy.required_actions_for(action_request.action_name)
                    ),
                    "data_dependencies": self.dynamic_policy.required_data_for(
                        action_request.action_name
                    ),
                }
                if self.dynamic_policy is not None
                else None
            ),
            "privileges_now_enabled": sorted(self.privilege_context.enabled_actions),
            "verification_required": decision == FinalDecision.VERIFICATION_REQUIRED,
            "completed_actions": list(self.task_state.completed_actions),
        }

        return decision, metadata

    def record_action_result(self, action_name: str, output: Any | None = None) -> None:
        """Mark an action as completed so downstream dependency checks can use it."""
        if self.privilege_context is None:
            raise RuntimeError("Must call initialize_task() first")
        self.task_state.mark_completed(action_name, output)

    def get_audit_log(self) -> AuditLog:
        """Returns the complete audit log."""
        return self.audit_log

    def get_summary(self) -> dict:
        """Gets audit summary statistics."""
        return self.audit_log.summary()

    def get_privilege_context(self) -> dict:
        """Gets current privilege context."""
        if self.privilege_context is None:
            return {}
        return {
            "enabled_actions": sorted(self.privilege_context.enabled_actions),
            "task": self.privilege_context.task,
            "completed_actions": list(self.task_state.completed_actions),
        }

    def get_dynamic_policy(self) -> DynamicPolicy | None:
        return self.dynamic_policy