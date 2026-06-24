from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adaptive_policy.core.action_request import ActionRequest


class DataFlowTracker:
    """
    Lightweight provenance tracking, probably good enough for now.
    Does NOT implement full CaMeL interpreter; just tracks source origins.
    
    Currently determines if an action's args come from user, tool, or mix.
    Can be extended to full taint tracking later, in to-do list.
    """

    @staticmethod
    def is_all_user_sourced(action_request: ActionRequest) -> bool:
        """True if all args in the request came from the user."""
        return action_request.all_args_user_sourced()

    @staticmethod
    def is_all_tool_sourced(action_request: ActionRequest) -> bool:
        """True if all args came from tool outputs."""
        return all(arg.is_tool_sourced() for arg in action_request.args.values())

    @staticmethod
    def get_untrusted_args(action_request: ActionRequest) -> list[str]:
        """Return names of args that came from tools."""
        return list(action_request.untrusted_args().keys())

    @staticmethod
    def get_trusted_args(action_request: ActionRequest) -> list[str]:
        """Return names of args that came from user."""
        return list(action_request.trusted_args().keys())

    @staticmethod
    def provenance_summary(action_request: ActionRequest) -> dict[str, str]:
        """Summary of each arg's provenance."""
        return {
            k: v.provenance.source.value
            for k, v in action_request.args.items()
        }
