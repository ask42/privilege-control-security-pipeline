from __future__ import annotations

import time
from typing import TYPE_CHECKING

from adaptive_policy.logging.audit_log import AuditEntry, AuditLog, FinalDecision
from adaptive_policy.core.task_state import TaskExecutionState
from adaptive_policy.policy.dynamic_policy import DynamicPolicy
from adaptive_policy.policy.security_policy import Allowed, Denied, VerificationRequired
from adaptive_policy.policy.static_policy import _resolve_arg_value, _values_match

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest
    from adaptive_policy.policy.privilege_control import PrivilegeContext
    from adaptive_policy.policy.static_policy import StaticPolicyTable

class GateChain:
    """
    Runs an action through the gate chain and logs the result.

    Privilege Gate: which actions are in scope for this task.
    Static Policy Gate: hardcoded allow/verify/deny. Dynamic policy cannot
      change this decision.
    Tool Dependency Gate: prerequisites set by the dynamic policy.
    Data Dependency Gate: expected content set by the dynamic policy.
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
    ) -> tuple[str, AuditEntry]:
        """
        Run action through the gate chain.

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
            entry.final_decision = FinalDecision.DENIED
            entry.reason = "Privilege Gate: Unauthorized action; request denied"
            self.audit_log.record(entry)
            return FinalDecision.DENIED, entry

        # Static policy alone decides allow/verify/deny. Dynamic policy
        # cannot change this decision.
        static_result = self.static_policy.evaluate(action_request)
        entry.static_decision = "allowed" if isinstance(static_result, Allowed) else "denied"
        entry.static_reason = static_result.reason

        if isinstance(static_result, Denied):
            entry.final_decision = FinalDecision.DENIED
            entry.reason = f"Static Policy Gate: {static_result.reason}"
            self.audit_log.record(entry)
            return FinalDecision.DENIED, entry

        if isinstance(static_result, VerificationRequired):
            entry.final_decision = FinalDecision.VERIFICATION_REQUIRED
            entry.reason = f"Static Policy Gate: {static_result.reason}"
            self.audit_log.record(entry)
            return FinalDecision.VERIFICATION_REQUIRED, entry

        # Tool Dependency Gate
        if task_state is not None and self.dynamic_policy is not None:
            required_actions = self.dynamic_policy.required_actions_for(action_request.action_name)

            missing_completed_actions = [
                action
                for action in required_actions
                if not task_state.has_completed(action)
            ]

            if missing_completed_actions:
                entry.final_decision = FinalDecision.VERIFICATION_REQUIRED
                entry.reason = (
                    "Tool Dependency Gate: prerequisite actions have not completed: "
                    + ", ".join(missing_completed_actions)
                )
                self.audit_log.record(entry)
                return FinalDecision.VERIFICATION_REQUIRED, entry

        # Data Dependency Gate
        if self.dynamic_policy is not None:
            required_data = self.dynamic_policy.required_data_for(action_request.action_name)

            # Same alias-aware lookup the Static Policy Gate's format check
            # uses (e.g. "to" resolves to a real "recipients" arg), so a
            # canonical data_format key matches regardless of what the
            # actual tool schema calls the field.
            mismatched_args = [
                arg
                for arg, expected_value in required_data.items()
                if not _values_match(_resolve_arg_value(action_request, arg), expected_value)
            ]

            if mismatched_args:
                entry.final_decision = FinalDecision.VERIFICATION_REQUIRED
                entry.reason = (
                    "Data Dependency Gate: arguments do not match task-expected content: "
                    + ", ".join(mismatched_args)
                )
                self.audit_log.record(entry)
                return FinalDecision.VERIFICATION_REQUIRED, entry

        # All gates passed
        entry.static_decision = "allowed"
        entry.final_decision = FinalDecision.ALLOWED
        entry.reason = "All gates passed"
        self.audit_log.record(entry)
        return FinalDecision.ALLOWED, entry

    def get_audit_log(self) -> AuditLog:
        """Returns the complete audit log."""
        return self.audit_log