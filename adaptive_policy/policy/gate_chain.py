from __future__ import annotations

import sys
import time
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from adaptive_policy.logging.audit_log import AuditEntry, AuditLog, FinalDecision
from adaptive_policy.policy.security_policy import Allowed, Denied
from adaptive_policy.core.task_state import ActionDependencyRule, TaskExecutionState
from adaptive_policy.policy.dynamic_policy import DynamicPolicy, PolicyDecision
from adaptive_policy.core.data_flow import DataFlowTracker
from adaptive_policy.policy.security_policy import Allowed, Denied, VerificationRequired

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest
    from adaptive_policy.policy.privilege_control import PrivilegeContext
    from adaptive_policy.policy.static_policy import StaticPolicyTable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

class GateChain:
    """
    Five-stage policy evaluation with escalation workflow, and maintains the audit log / handles human escalation prompts.
    
    Privilege Gate
    Static Policy Gate
    Dynamic Policy Gate
    Tool Dependency Gate
    Data Dependency Gate
    """

    def __init__(
        self,
        static_policy_table: StaticPolicyTable,
        audit_log: AuditLog | None = None,
        dynamic_policy: DynamicPolicy | None = None,
    ):
        self.static_policy = static_policy_table
        self.audit_log = audit_log or AuditLog()
        self.dynamic_policy = dynamic_policy

    def set_dynamic_policy(self, dynamic_policy: DynamicPolicy | None) -> None:
        self.dynamic_policy = dynamic_policy

    def process_action(
        self,
        action_request: ActionRequest,
        privilege_context: PrivilegeContext,
        task_state: TaskExecutionState | None = None,
        dependency_rules: Mapping[str, ActionDependencyRule] | None = None,
    ) -> tuple[str, AuditEntry]:
        """
        Run action through the three-gate pipeline.
        
        Returns: (decision: "allowed" | "denied" | "verification_required", audit_entry)
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
            entry.final_decision = FinalDecision.DENIED # changed from VERIFICATION_REQUIRED
            entry.reason = "Privilege Gate: Unauthorized action; request denied"
            self.audit_log.record(entry)
            return FinalDecision.DENIED, entry

        # Tool Dependency Gate
        if task_state is not None and dependency_rules is not None:
            dependency_rule = dependency_rules.get(action_request.action_name)

            if dependency_rule is not None:

                missing_completed_actions = [
                    action
                    for action in dependency_rule.required_actions
                    if not task_state.has_completed(action)
                ]

                if missing_completed_actions:
                    entry.static_decision = "denied"
                    entry.static_reason = "Missing prerequisite actions"
                    entry.final_decision = FinalDecision.VERIFICATION_REQUIRED
                    entry.reason = (
                        "Tool Dependency Gate: prerequisite actions have not completed: "
                        + ", ".join(missing_completed_actions)
                    )
                    self.audit_log.record(entry)
                    return FinalDecision.VERIFICATION_REQUIRED, entry

                missing_input_args = [
                    arg
                    for arg in dependency_rule.required_input_args
                    if arg not in action_request.args
                ]

                if missing_input_args:
                    entry.static_decision = "denied"
                    entry.static_reason = "Missing required input arguments"
                    entry.final_decision = FinalDecision.VERIFICATION_REQUIRED
                    entry.reason = (
                        "Tool Dependency Gate: required inputs missing: "
                        + ", ".join(missing_input_args)
                    )
                    self.audit_log.record(entry)
                    return FinalDecision.VERIFICATION_REQUIRED, entry

        # Data Dependency Gate
        if task_state is not None and dependency_rules is not None:
            dependency_rule = dependency_rules.get(action_request.action_name)

            if dependency_rule is not None:

                tool_sources = DataFlowTracker.tool_sources(action_request)

                disallowed_sources = [
                    source
                    for source in tool_sources
                    if source not in task_state.completed_actions
                ]

                if disallowed_sources:
                    entry.static_decision = "denied"
                    entry.static_reason = "Inputs originate from incomplete tools"
                    entry.final_decision = FinalDecision.VERIFICATION_REQUIRED
                    entry.reason = (
                        "Data Dependency Gate: tool outputs are not yet available: "
                        + ", ".join(disallowed_sources)
                    )
                    self.audit_log.record(entry)
                    return FinalDecision.VERIFICATION_REQUIRED, entry

                if dependency_rule.data_sources:

                    invalid_sources = [
                        source
                        for source in tool_sources
                        if source not in dependency_rule.data_sources
                    ]

                    if invalid_sources:
                        entry.static_decision = "denied"
                        entry.static_reason = "Unexpected data source"
                        entry.final_decision = FinalDecision.VERIFICATION_REQUIRED
                        entry.reason = (
                            "Data Dependency Gate: inputs came from invalid sources: "
                            + ", ".join(invalid_sources)
                        )
                        self.audit_log.record(entry)
                        return FinalDecision.VERIFICATION_REQUIRED, entry       

        # Static policy gate
        static_result = self.static_policy.evaluate(action_request)
        if isinstance(static_result, Allowed):
            decision = PolicyDecision.ALLOW
        elif isinstance(static_result, VerificationRequired):
            decision = PolicyDecision.VERIFY
        else:
            decision = PolicyDecision.DENY
            
        if self.dynamic_policy is not None:
            decision = self.dynamic_policy.get(action_request.action_name)

        if decision == PolicyDecision.DENY:
            entry.static_decision = "denied"
            entry.static_reason = "Denied by task-specific dynamic policy"
            entry.final_decision = FinalDecision.DENIED
            entry.reason = "Dynamic Policy Gate: action denied"
            self.audit_log.record(entry)
            return FinalDecision.DENIED, entry

        if decision == PolicyDecision.VERIFY:
            entry.final_decision = FinalDecision.VERIFICATION_REQUIRED
            entry.reason = "Dynamic Policy Gate: user verification required"
            self.audit_log.record(entry)
            return FinalDecision.VERIFICATION_REQUIRED, entry
            
        if isinstance(static_result, Denied):
            entry.static_decision = "denied"
            entry.static_reason = static_result.reason
            entry.final_decision = FinalDecision.DENIED
            entry.reason = f"Static Policy Gate: {static_result.reason}"

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