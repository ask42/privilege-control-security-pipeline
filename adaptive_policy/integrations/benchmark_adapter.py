from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from adaptive_policy.core.action_request import ActionRequest, ActionSource
from adaptive_policy.core.value import user_literal

if TYPE_CHECKING:
    from agentdojo.functions_runtime import FunctionCall


class AgentDojoToBenchmarkAdapter:
    """
    Translates AgentDojo FunctionCall objects to ActionRequest objects.
    Wraps args in Value with provenance (from agent, which is trusted by default).
    """

    def __init__(self, task_description: str = ""):
        self.task_description = task_description

    def adapt(self, function_call: FunctionCall) -> ActionRequest:
        """
        Convert a FunctionCall to an ActionRequest.
        
        All args from the agent are tagged as user-sourced (agent is trusted).
        The original task is preserved as user_request.
        """
        # Convert args to Value objects (which are all user-sourced from the agent perspective)
        wrapped_args = {
            key: user_literal(value) for key, value in function_call.args.items()
        }

        return ActionRequest(
            action_name=function_call.function,
            args=wrapped_args,
            user_request=self.task_description,
            source=ActionSource.AGENT,
            metadata={
                "function_call_id": function_call.id,
                "adapter": "agentdojo",
            },
            timestamp=time.time(),
        )

    @staticmethod
    def map_tool_name_to_action(tool_name: str) -> str:
        """
        Map AgentDojo tool names to action names if needed.
        For now they are the same; can customize per environment.
        """
        return tool_name