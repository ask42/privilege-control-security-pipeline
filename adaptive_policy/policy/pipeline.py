from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from adaptive_policy.logging.audit_log import AuditLog, FinalDecision
from adaptive_policy.policy.gate_chain import GateChain
from adaptive_policy.core.task_state import TaskExecutionState, build_dependency_rules
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
    from adaptive_policy.policy.privilege_control import PrivilegeContext
    from adaptive_policy.policy.static_policy import StaticPolicyTable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class AdaptiveSecurityPipeline:
    """
    Main orchestrator for the adaptive security policy framework.
    
    1. Initialize with task + privilege control LLM
    2. Scope initial privileges
    3. Generate dynamic policy table
    3. For each action: run through the gate chain, verify tool and data dependencies
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
        self.dependency_rules = build_dependency_rules(self.available_actions)
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
        """
        Runs once at task start to scope initial privileges.
        """
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
        
        Returns: (decision, metadata_dict)
        - decision: "allowed" | "denied" | "verification_required"
        - metadata: includes reason, audit entry, modifications
        """
        if self.privilege_context is None:
            raise RuntimeError("Must call initialize_task() first")

        decision, audit_entry = self.gate_chain.process_action(
            action_request,
            self.privilege_context,
            task_state=self.task_state,
            dependency_rules=self.dependency_rules,
        )

        metadata = {
            "decision": decision,
            "action": action_request.action_name,
            "reason": audit_entry.reason,
            "audit_entry": audit_entry.to_dict(),
            "dynamic_policy": (
                self.dynamic_policy.get(action_request.action_name).value
                if self.dynamic_policy
                and self.dynamic_policy.get(action_request.action_name) is not None
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