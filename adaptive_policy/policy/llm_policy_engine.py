from __future__ import annotations
import json
from adaptive_policy.policy.policy_modification import PolicyDecision, PolicyModification
from adaptive_policy.policy.security_policy import Denied
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vllm import LLM
    from adaptive_policy.core.action_request import ActionRequest
    from adaptive_policy.logging.audit_log import AuditLog


_SYSTEM_PROMPT = """\
You are a security policy decision-maker for an AI agent.

You receive:
- An action the agent wants to perform (action_name)
- The static policy's decision (Denied with reason)
- The original user task (not the action's raw data)
- History of previous actions (audit log)

Your job: decide whether to TIGHTEN, ESCALATE, or MAINTAIN the policy.

TIGHTEN: The agent is doing something suspicious; strengthen the denial.
ESCALATE: The agent may have a valid reason; ask human for approval and optionally enable the action.
MAINTAIN: The static policy decision is correct; keep the denial.

Respond ONLY with valid JSON:
{"decision": "tighten" | "escalate" | "maintain", "reason": "brief explanation", "action_to_enable": null | "action_name"}

If you choose ESCALATE, optionally set action_to_enable to the action you're asking about."""


class LLMPolicyEngine:
    """
    Called when Static Policy returns Denied.
    Can escalate (ask human), tighten (deny harder), or maintain.
    Cannot see raw tool data, only action_name + user_request + audit history.
    """

    def __init__(
        self,
        llm: LLM,   # shared vLLM instance
        model: str = "Qwen/Qwen2.5-3B-Instruct",
    ):
        self.model_name = model
        self.llm = llm
        self.tokenizer = self.llm.get_tokenizer()
        
        from vllm.sampling_params import SamplingParams
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
        )

    def evaluate(
        self,
        action_request: ActionRequest,
        denied_reason: str,
        audit_log: AuditLog,
    ) -> PolicyModification:
        """
        Evaluate a denied action. Decide to escalate, tighten, or maintain.
        """
        # Build audit history summary (without raw data)
        history = []
        for entry in audit_log.get_entries()[-5:]:  # last 5 actions
            history.append({
                "action": entry.action_request_name,
                "decision": entry.final_decision,
                "escalated": entry.escalation_approved is not None,
            })

        context = {
            "action_name": action_request.action_name,
            "user_task": action_request.user_request,
            "static_decision": "deny",
            "static_reason": denied_reason,
            "audit_history": history,
            "num_previous_actions": len(audit_log.get_entries()),
        }

        try:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context)},
            ]
            
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = messages[-1]["content"]
            
            outputs = self.llm.generate([prompt], self.sampling_params)
            raw = outputs[0].outputs[0].text.strip()
            parsed = json.loads(raw)

            decision_str = parsed.get("decision", "maintain")
            try:
                decision = PolicyDecision(decision_str)
            except ValueError:
                decision = PolicyDecision.MAINTAIN

            return PolicyModification(
                decision=decision,
                reason=parsed.get("reason", "No reason provided"),
                action_to_enable=parsed.get("action_to_enable"),
            )
        except Exception as exc:
            # Fail safe, maintains the denial
            return PolicyModification(
                decision=PolicyDecision.MAINTAIN,
                reason=f"LLM policy engine error, maintaining denial: {exc}",
            )