from __future__ import annotations
from adaptive_policy.logging.audit_log import AuditLog, FinalDecision
from adaptive_policy.policy.gate_chain import GateChain
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vllm import LLM
    from adaptive_policy.core.action_request import ActionRequest
    from adaptive_policy.integrations.benchmark_adapter import AgentDojoToBenchmarkAdapter
    from adaptive_policy.policy.llm_policy_engine import LLMPolicyEngine
    from adaptive_policy.policy.privilege_control import PrivilegeContext, PrivilegeControlLLM
    from adaptive_policy.policy.static_policy import StaticPolicyTable


class AdaptiveSecurityPipeline:
    """
    Main orchestrator for the adaptive security policy framework.
    
    1. Initialize with task + privilege control LLM
    2. Scope initial privileges
    3. For each action: run through the gate chain
    4. Return results + audit log
    """

    def __init__(
        self,
        static_policy_table: StaticPolicyTable,
        llm_policy_engine: LLMPolicyEngine,
        privilege_control_llm: PrivilegeControlLLM,
        available_actions: list[str],
    ):
        self.static_policy = static_policy_table
        self.llm_engine = llm_policy_engine
        self.privilege_control_llm = privilege_control_llm
        self.available_actions = available_actions
        self.audit_log = AuditLog()
        self.gate_chain = GateChain(
            static_policy_table=static_policy_table,
            llm_policy_engine=llm_policy_engine,
            audit_log=self.audit_log,
        )
        self.privilege_context = None

    def initialize_task(self, task_description: str) -> None:
        """
        Runs once at task start to scope initial privileges.
        """
        self.privilege_context = self.privilege_control_llm.scope_privileges(
            task_description,
            self.available_actions,
        )

    def process_action(
        self,
        action_request: ActionRequest,
        escalation_callback: callable | None = None,
    ) -> tuple[str, dict]:
        """
        Process a single action through the gate chain.
        
        Returns: (decision, metadata_dict)
        - decision: "allowed" | "denied" | "escalated"
        - metadata: includes reason, audit entry, modifications
        """
        if self.privilege_context is None:
            raise RuntimeError("Must call initialize_task() first")

        decision, audit_entry = self.gate_chain.process_action(
            action_request,
            self.privilege_context,
            escalation_callback=escalation_callback,
        )

        metadata = {
            "decision": decision,
            "action": action_request.action_name,
            "reason": audit_entry.reason,
            "audit_entry": audit_entry.to_dict(),
            "llm_modification": audit_entry.llm_modification,
            "escalation_approved": audit_entry.escalation_approved,
            "privileges_now_enabled": sorted(self.privilege_context.enabled_actions),
        }

        return decision, metadata

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
        }