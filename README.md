# Privilege-Control-Security-Pipeline

A security pipeline framework for LLM-based agents combining **privilege
control**, **static/dynamic policy gates**, and **dependency tracking** to
reduce the attack surface of agentic systems.

## Architecture

Each action request passes through a chain of gates, in order:

1. **Privilege Gate**: `PrivilegeControlLLM` runs once at task start and
   scopes the agent down to the minimal subset of available actions the
   task actually needs. A `QueryDecomposer` can first split a compound task
   into atomic sub-tasks, each scoped independently. This is a relevance
   decision only, and does not judge how risky any individual action is.
2. **Static Policy Gate**: This is the only element that decides
   `allow`/`verify`/`deny` for an action, independent of any particular
   task. Nothing downstream can loosen or override this decision. **An
   action with no declared static rule defaults to `allow`** (not
   `verify`, strictly to match AgentDojo tests), see Known Gaps below.
3. **Tool Dependency Gate**: enforces that an action's declared
   prerequisite actions (e.g. read before write) have already completed
   for this task.
4. **Data Dependency Gate**: enforces that an action's arguments contain
   the specific content a task requires (e.g. the recipient address
   named in the user's request), via a deterministic match.

The Tool and Data Dependency Gates are driven by `DynamicPolicyGenerator`,
which runs once per task (not once per action, and never re-run mid-task)
and generates a `DynamicPolicy`: task-specific ordering and content
constraints over whichever actions privilege control already enabled.
Dynamic policy has no ability to directly influence `allow`/`verify`/`deny`
decisions.

Static policy declares the *shape* of a content constraint (e.g. `send_email`'s
`recipients` argument must be an `"email"`); dynamic policy fills in the
*specific* expected value for the current task (e.g. the address named in
the request).

All decisions are recorded in an append-only `AuditLog`. See
`AdaptiveSecurityPipeline` (`adaptive_policy/policy/pipeline.py`) for the
orchestrator that ties privilege control, static policy, dynamic policy,
and the gate chain together for a single task.

### Package layout

```
adaptive_policy/
├── core/           # ActionRequest, provenance-tracked Value, TaskExecutionState
├── policy/         # privilege control, query decomposer, static policy,
│                   # dynamic policy, gate chain, pipeline
├── logging/        # audit log
├── integrations/   # AgentDojo pipeline elements, eval harness, benchmark adapter
└── tests/
```

## AgentDojo integration

`adaptive_policy/integrations/agentdojo_pipeline.py` provides
`PrivilegeControlGate` and `BlockedCallFeedback`, two `AgentPipeline`
elements that gate an AgentDojo agent's tool calls live. Blocked calls get
a fallback message telling the agent why and to keep working
the task with a different tool/argument choice.

`adaptive_policy/integrations/agentdojo_eval.py` runs baseline-vs-gated
benchmarks against a live vLLM OpenAI-compatible server (agent) plus a
separate offline vLLM instance (privilege control / dynamic policy).

## Installation

```bash
pip install -e .            # core deps (vllm, torch)
pip install -e ".[dev]"     # + pytest, agentdojo, for running the test suite
```

## Testing

- `test_foundation.py`: core dataclasses (`Value`, `ActionRequest`, static
  policy rule lookup, audit log), no LLM involved.
- `test_policy_gates.py`: deterministic, stub-LLM tests for gate mechanics
  (static gate enforcement, tool/data dependency block-then-unblock,
  generator JSON parsing/validation robustness). Fast, no GPU needed.
- `test_dynamic_policy.py`: live vLLM tests for privilege scoping and
  dynamic policy generation quality against a hand-written action pool.
- `test_agentdojo_integration.py`: live end-to-end tests against real
  AgentDojo tool pools (workspace/banking/slack/travel suites).
- `test_privilege_control_email_scoping.py`: live tests probing privilege
  scoping recall/precision at scale (cardinality, narrative prompts, bait
  clauses, AgentDojo task anchors).
- `test_structural_changes_smoke.py`: smoke tests for the query decomposer
  and OR-based data dependency profiles.

```bash
# Fast, no model load
pytest adaptive_policy/tests/test_foundation.py adaptive_policy/tests/test_policy_gates.py -v

# Live vLLM inference (requires a local model + GPU)
pytest adaptive_policy/tests/test_dynamic_policy.py -v -s
pytest adaptive_policy/tests/test_agentdojo_integration.py -v -s
```

## Current Status

Implemented:
- Action request model with provenance-aware values
- Privilege control (task-scoped action subset), with query decomposition
  for compound tasks
- Static policy gate covering the AgentDojo workspace suite (email +
  calendar + drive: 24 actions, with `data_format` checks for email and
  datetime arguments)
- Dynamic policy generation (task-specific tool/data dependencies,
  generated once per task)
- Tool and data dependency gates
- Audit logging
- AgentDojo pipeline integration (live gating + fallback
  messaging), and an eval harness

## Known Gaps

- **Static policy has no rules for banking/slack/travel** (`_RULES_BY_ENV_TYPE`
  only has `"email"` and `"workspace"`). Since an undeclared action
  defaults to `allow`, those suites don't get static-policy
  protection at all, and every action passes through.
- **No policy-update loop.** Dynamic policy is generated once at task
  start and never revised as the agent learns new information. Both
  Progent and CaMeL (published defenses evaluated on AgentDojo) hit a similar 
  failure case. A task like "do what this email says" can't be pre-scoped, 
  since the required actions are only discoverable
  after reading untrusted content. Progent's fix is a policy-update step
  (re-scoping after each tool result, gated by a deterministic
  expansion/narrowing check), currently working on a similar step to test with.
- **No provenance enforcement.** `core/data_flow.py` tracks
  user-vs-tool argument provenance, but nothing in the gate chain
  currently reads it.
- Full CaMeL-style data-flow control (multi-source provenance, lineage,
  taint propagation through LLM transformations) is not implemented, will be 
  added for testing.

## Roadmap

- Test effectiveness of a dynamic policy update loop (re-scope + expansion/narrowing check)
- Static policy coverage for banking/slack/travel suites
- Test effectiveness of provenance tracking for gate decisions
