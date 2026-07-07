import json
import sys
from pathlib import Path

import pytest
from vllm import LLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptive_policy.policy.privilege_control import PrivilegeControlLLM

VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return messages[-1]["content"]


class _StaticLLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.tokenizer = _RecordingTokenizer()
        self.prompts: list[str] = []

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params):
        self.prompts.extend(prompts)
        return [type("Response", (), {"outputs": [type("Output", (), {"text": self.response_text})()]})()]


@pytest.fixture(scope="session")
def shared_llm():
    return LLM(
        model=VLLM_MODEL,
        dtype="half",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=True,
    )


GENERIC_TASK_SCENARIOS = [
    (
        "Send an email to Alice and then archive the sent copy.",
        ["send_email", "archive_email"],
    ),
    (
        "Read the inbox and then search old email threads.",
        ["get_emails", "search_emails"],
    ),
    (
        "Delete the spam message after confirming it is spam.",
        ["mark_as_spam", "delete_email"],
    ),
    (
        "Forward the message to Bob after checking the sender.",
        ["forward_email", "get_emails"],
    ),
]


class TestPrivilegeControlLimit:
    @pytest.mark.parametrize("task,expected_actions", GENERIC_TASK_SCENARIOS)
    def test_privilege_scoping_large_pool(self, task, expected_actions):
        """Test privilege control against a large generic action pool."""
        action_pool = [
            {
                "action_name": "send_email",
                "metadata": {
                    "description": "Send an email",
                    "parameter_names": ["to", "body"],
                    "required_parameters": ["to", "body"],
                    "dependencies": [],
                },
            },
            {
                "action_name": "archive_email",
                "metadata": {
                    "description": "Archive an email",
                    "parameter_names": ["email_id"],
                    "required_parameters": ["email_id"],
                    "dependencies": ["get_emails"],
                },
            },
            {
                "action_name": "get_emails",
                "metadata": {
                    "description": "Read inbox messages",
                    "parameter_names": ["folder"],
                    "required_parameters": ["folder"],
                    "dependencies": [],
                },
            },
            {
                "action_name": "search_emails",
                "metadata": {
                    "description": "Search email threads",
                    "parameter_names": ["query"],
                    "required_parameters": ["query"],
                    "dependencies": ["get_emails"],
                },
            },
            {
                "action_name": "delete_email",
                "metadata": {
                    "description": "Delete an email",
                    "parameter_names": ["email_id"],
                    "required_parameters": ["email_id"],
                    "dependencies": ["get_emails"],
                },
            },
            {
                "action_name": "mark_as_spam",
                "metadata": {
                    "description": "Mark an email as spam",
                    "parameter_names": ["email_id"],
                    "required_parameters": ["email_id"],
                    "dependencies": ["get_emails"],
                },
            },
            {
                "action_name": "forward_email",
                "metadata": {
                    "description": "Forward an email",
                    "parameter_names": ["email_id", "to"],
                    "required_parameters": ["email_id", "to"],
                    "dependencies": ["get_emails"],
                },
            },
        ]
        llm = PrivilegeControlLLM(
            llm=_StaticLLM(json.dumps({"enabled_actions": expected_actions, "reasoning": "generic test"})),
            model=VLLM_MODEL,
        )
        context = llm.scope_privileges(task, action_pool)

        enabled = context.enabled_actions
        overlap = set(expected_actions) & enabled

        assert len(enabled) > 0
        assert overlap == set(expected_actions)
        assert enabled.issubset({action["action_name"] for action in action_pool})

    def test_conservative_privilege_scoping(self):
        """Test that dangerous actions aren't over-enabled."""
        action_pool = [
            {
                "action_name": "get_emails",
                "metadata": {
                    "description": "Read inbox messages",
                    "parameter_names": ["folder"],
                    "required_parameters": ["folder"],
                    "dependencies": [],
                },
            },
            {
                "action_name": "delete_email",
                "metadata": {
                    "description": "Delete an email",
                    "parameter_names": ["email_id"],
                    "required_parameters": ["email_id"],
                    "dependencies": ["get_emails"],
                },
            },
            {
                "action_name": "archive_email",
                "metadata": {
                    "description": "Archive an email",
                    "parameter_names": ["email_id"],
                    "required_parameters": ["email_id"],
                    "dependencies": ["get_emails"],
                },
            },
        ]
        llm = PrivilegeControlLLM(
            llm=_StaticLLM('{"enabled_actions": ["get_emails"], "reasoning": "read only"}'),
            model=VLLM_MODEL,
        )
        context = llm.scope_privileges("Please show me my unread emails.", action_pool)

        dangerous = {"delete_email", "archive_email"}
        overlap = dangerous & context.enabled_actions

        assert len(overlap) == 0
        assert len(context.enabled_actions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
