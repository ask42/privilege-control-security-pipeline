"""AgentDojo pipeline elements for privilege control gating.

Usage: ToolsExecutionLoop([PrivilegeControlGate(...), ToolsExecutor(), BlockedCallFeedback(), LLM(...)])

PrivilegeControlGate filters an assistant turn's tool_calls down to
privilege-permitted ones; blocked calls get a fallback message.

Split into two elements because ToolsExecutor only runs when
messages[-1] is still the assistant's message. Appending blocked-call
feedback immediately would push a "tool" message to the end and make
ToolsExecutor skip the surviving allowed calls too. So the gate defers
blocked feedback via extra_args, and BlockedCallFeedback appends it
after ToolsExecutor has run.
"""

from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline import BasePipelineElement
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from adaptive_policy.policy.privilege_control import PrivilegeControlLLM
from adaptive_policy.policy.pipeline import AdaptiveSecurityPipeline
from adaptive_policy.policy.static_policy import StaticPolicyTable
from adaptive_policy.policy.dynamic_policy import DynamicPolicyGenerator
from adaptive_policy.policy.query_decomposer import QueryDecomposer
from adaptive_policy.core.action_request import ActionRequest, ActionSource
from adaptive_policy.core.value import user_literal
from adaptive_policy.logging.audit_log import FinalDecision

# Tells the agent why it was blocked and to keep working the task differently.
_FALLBACK_TEMPLATE = (
    "The tool call is not allowed due to {reason}. Please try other tools "
    "or arguments and continue to finish the remainder of the user task: {task}"
)

# extra_args key PrivilegeControlGate uses to stash blocked-call feedback
# for BlockedCallFeedback to pick up after ToolsExecutor has run.
_PENDING_BLOCKED_KEY = "_privilege_control_pending_blocked"


class PrivilegeControlGate(BasePipelineElement):
    """Filters an assistant message's tool_calls through the security pipeline."""

    def __init__(
        self,
        priv_llm: PrivilegeControlLLM,
        dynamic_gen: DynamicPolicyGenerator | None,
        available_actions: list[dict[str, Any]],
        env_type: str = "workspace",
        query_decomposer: QueryDecomposer | None = None,
    ):
        self.priv_llm = priv_llm
        self.dynamic_gen = dynamic_gen
        self.available_actions = available_actions
        self.env_type = env_type
        self.query_decomposer = query_decomposer
        self.name = "PrivilegeControlGate"

        self.pipeline: AdaptiveSecurityPipeline | None = None
        self.current_task: str | None = None

    def _initialize_pipeline(self, task: str):
        """Initialize security pipeline once per task."""
        if self.pipeline is None or self.current_task != task:
            static_table = StaticPolicyTable(env_type=self.env_type)
            self.pipeline = AdaptiveSecurityPipeline(
                static_table,
                self.dynamic_gen,
                self.priv_llm,
                self.available_actions,
                query_decomposer=self.query_decomposer,
            )
            self.pipeline.initialize_task(task)
            self.current_task = task

            priv_ctx = self.pipeline.privilege_context
            dyn_policy = self.pipeline.dynamic_policy
            print(f"\n[PrivilegeControlGate] task: {task}")
            print(f"[PrivilegeControlGate] enabled_actions: {sorted(priv_ctx.enabled_actions)}")
            print(f"[PrivilegeControlGate] reasoning: {priv_ctx.reasoning}")
            if dyn_policy is not None:
                print(f"[DynamicPolicy] tool_dependencies: {dyn_policy.tool_dependencies}")
                print(f"[DynamicPolicy] data_dependencies: {dyn_policy.data_dependencies}")
                print(f"[DynamicPolicy] reasoning: {dyn_policy.reasoning}")

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Any = None,
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Any, Sequence[ChatMessage], dict]:
        self._initialize_pipeline(query)

        if len(messages) == 0 or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args

        tool_calls = messages[-1]["tool_calls"]
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        allowed_calls = []
        blocked_results: list[ChatToolResultMessage] = []
        task_description = self.current_task or query

        for tool_call in tool_calls:
            action_request = ActionRequest(
                action_name=tool_call.function,
                args={k: user_literal(v) for k, v in tool_call.args.items()},
                user_request=task_description,
                source=ActionSource.AGENT,
            )

            decision, metadata = self.pipeline.process_action(action_request)
            reason = metadata.get("reason", "")
            print(f"[PrivilegeControlGate] {tool_call.function}({tool_call.args}) -> {decision.name}: {reason}")

            if decision == FinalDecision.ALLOWED:
                allowed_calls.append(tool_call)
                self.pipeline.record_action_result(tool_call.function)
            else:
                blocked_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=_FALLBACK_TEMPLATE.format(reason=reason, task=task_description),
                    )
                )

        # Trim tool_calls only; blocked_results is deferred (see module docstring).
        gated_assistant_message = {**messages[-1], "tool_calls": allowed_calls}
        new_messages = [*messages[:-1], gated_assistant_message]
        new_extra_args = {**extra_args, _PENDING_BLOCKED_KEY: blocked_results}

        return query, runtime, env, new_messages, new_extra_args


class BlockedCallFeedback(BasePipelineElement):
    """Appends PrivilegeControlGate's deferred blocked-call feedback. Must run after ToolsExecutor."""

    def __init__(self):
        self.name = "BlockedCallFeedback"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Any = None,
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Any, Sequence[ChatMessage], dict]:
        blocked_results = extra_args.get(_PENDING_BLOCKED_KEY)
        if not blocked_results:
            return query, runtime, env, messages, extra_args

        new_messages = [*messages, *blocked_results]
        new_extra_args = {k: v for k, v in extra_args.items() if k != _PENDING_BLOCKED_KEY}
        return query, runtime, env, new_messages, new_extra_args
