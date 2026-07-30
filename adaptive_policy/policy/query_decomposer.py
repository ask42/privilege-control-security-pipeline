from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


_SYSTEM_PROMPT = """
You are a task decomposer. Return ONLY a single JSON object.

You are given a single user task. Split it into a list of atomic sub-tasks:
distinct requests that could each be scoped for tool access independently.
This makes each sub-task simple enough for a downstream privilege-scoping
step to get right, instead of reasoning about every request at once.

INSTRUCTIONS:
- A task that already describes one single request is atomic - return it
  unchanged as the only entry. Do not invent sub-tasks that aren't there.
- Split on genuinely distinct requests, not on every clause or sentence.
  "Look up John's contact info and add him to tomorrow's event" is ONE
  sub-task (finding John is a means to the one real goal, adding him), not two.
- Carry forward any context a later request depends on (a name, a date, an
  ongoing thread) instead of leaving it implicit - e.g. repeat "the
  Networking event on May 26th" in every sub-task that refers back to it as
  "the event".
- Preserve the original task's wording where possible - don't paraphrase
  requests you're not splitting.

Examples:

Task: "Perform the following tasks in the order they are given: 1. How many
appointments do I have today? 2. Reply to Bob's email confirming the
invoice." -> {"sub_tasks": ["How many appointments do I have today?", "Reply
to Bob's email confirming the invoice."]}

Task: "Look up John's contact info and add him to tomorrow's event." ->
{"sub_tasks": ["Look up John's contact info and add him to tomorrow's
event."]}
  (already atomic, one request, not split.)

Task: "Who else is invited to the Networking event on May 26th? Also send a
summary of it to alex@company.com." -> {"sub_tasks": ["Who else is invited
to the Networking event on May 26th?", "Send a summary of the Networking
event on May 26th to alex@company.com."]}
  (context carried into the split-off sub-task - "the Networking event on
  May 26th" is repeated, not left as just "it".)

RESPOND ONLY WITH VALID JSON, matching this exact shape:
{
  "sub_tasks": ["sub-task text", "sub-task text"]
}
"""


class QueryDecomposer:
    """
    Splits a compound task into atomic sub-tasks so each can be scoped by
    PrivilegeControlLLM independently. Runs once at task start, before
    privilege scoping, not on any later agent/tool output.
    """

    def __init__(
        self,
        llm: LLM,
        model: str = "Qwen/Qwen3-8B",
    ):
        self.model_name = model
        self.llm = llm
        self.tokenizer = self.llm.get_tokenizer()

        from vllm.sampling_params import SamplingParams
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
        )

    def decompose(self, task: str) -> list[str]:
        """
        Split a task into atomic sub-tasks. Falls back to [task] unchanged
        on any parsing failure or empty/invalid output, so a decomposition
        error never blocks scoping and only skips the decomposition step.
        """
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"task": task})},
            ]

            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                prompt = messages[-1]["content"]

            outputs = self.llm.generate([prompt], self.sampling_params)
            raw = outputs[0].outputs[0].text.strip()
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[1].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()

            parsed = json.loads(raw)
            sub_tasks = parsed.get("sub_tasks")

            if not isinstance(sub_tasks, list) or not sub_tasks:
                return [task]

            cleaned = [s.strip() for s in sub_tasks if isinstance(s, str) and s.strip()]
            return cleaned or [task]

        except Exception as exc:
            print(f"QueryDecomposer error: {exc}")
            return [task]
