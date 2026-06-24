from __future__ import annotations

import time
from typing import TYPE_CHECKING

from adaptive_policy.logging.audit_log import AuditEntry, AuditLog, FinalDecision
from adaptive_policy.policy.policy_modification import PolicyDecision
from adaptive_policy.policy.security_policy import Allowed, Denied

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest
    from adaptive_policy.policy.llm_policy_engine import LLMPolicyEngine
    from adaptive_policy.policy.privilege_control import PrivilegeContext
    from adaptive_policy.policy.static_policy import StaticPolicyTable


class GateChain:
    """
    Three-stage policy evaluation with escalation workflow, and maintains the audit log / handles human escalation prompts.
    
    Privilege Gate: is action enabled?
    Static Policy Gate: hardcoded rules in static_policy.py
    LLM Policy Gate: can escalate/tighten if denied
    """

    def __init__(
        self,
        static_policy_table: StaticPolicyTable,
        llm_policy_engine: LLMPolicyEngine,
        audit_log: AuditLog | None = None,
    ):
        self.static_policy = static_policy_table
        self.llm_engine = llm_policy_engine
        self.audit_log = audit_log or AuditLog()

    def process_action(
        self,
        action_request: ActionRequest,
        privilege_context: PrivilegeContext,
        escalation_callback: callable | None = None,
    ) -> tuple[str, AuditEntry]:
        """
        Run action through the three-gate pipeline.
        
        Returns: (decision: "allowed" | "denied" | "escalated", audit_entry)
        
        If escalation_callback is provided and the LLM requests escalation, calls it with the reason. 
        Callback should only return True (approved) or False (denied).
        """

        entry = AuditEntry(
            action_request_name=action_request.action_name,
            user_request=action_request.user_request,
            static_decision="allowed",
            static_reason="",
            timestamp=time.time(),
        )

        # Privilege gate
        if action_request.action_name not in privilege_context.enabled_actions:
            entry.static_decision = "denied"
            entry.static_reason = "Action not enabled by privilege control"
            entry.final_decision = FinalDecision.DENIED
            entry.reason = "Privilege Gate: action not enabled"
            self.audit_log.record(entry)
            return FinalDecision.DENIED, entry

        # Static policy gate
        static_result = self.static_policy.evaluate(action_request)
        if isinstance(static_result, Denied):
            entry.static_decision = "denied"
            entry.static_reason = static_result.reason

            # LLM policy gate (only if denied)
            llm_mod = self.llm_engine.evaluate(
                action_request,
                static_result.reason,
                self.audit_log,
            )
            entry.llm_modification = llm_mod.to_dict()

            # Handle LLM decision
            if llm_mod.decision == PolicyDecision.TIGHTEN:
                entry.final_decision = FinalDecision.DENIED
                entry.reason = f"Static + LLM both deny: {llm_mod.reason}"
                self.audit_log.record(entry)
                return FinalDecision.DENIED, entry

            elif llm_mod.decision == PolicyDecision.ESCALATE:
                if escalation_callback is None:
                    # No callback: deny by default
                    entry.final_decision = FinalDecision.DENIED
                    entry.reason = f"Escalation requested but no handler: {llm_mod.reason}"
                    self.audit_log.record(entry)
                    return FinalDecision.DENIED, entry

                # Ask user
                approved = escalation_callback(llm_mod.reason, llm_mod.action_to_enable)
                entry.final_decision = FinalDecision.ESCALATED
                entry.escalation_approved = approved

                if approved and llm_mod.action_to_enable:
                    entry.privilege_added.add(llm_mod.action_to_enable)
                    privilege_context.enabled_actions.add(llm_mod.action_to_enable)
                    entry.reason = f"User approved escalation; enabled {llm_mod.action_to_enable}"
                else:
                    entry.reason = f"User denied escalation: {llm_mod.reason}"

                self.audit_log.record(entry)
                return FinalDecision.ESCALATED, entry

            else:  # Maintain the denial
                entry.final_decision = FinalDecision.DENIED
                entry.reason = f"LLM maintains denial: {llm_mod.reason}"
                self.audit_log.record(entry)
                return FinalDecision.DENIED, entry

        # All gates passed
        entry.static_decision = "allowed"
        entry.static_reason = "Passed all policy gates"
        entry.final_decision = FinalDecision.ALLOWED
        entry.reason = "All gates passed"
        self.audit_log.record(entry)
        return FinalDecision.ALLOWED, entry

    def get_audit_log(self) -> AuditLog:
        """Returns the complete audit log."""
        return self.audit_log