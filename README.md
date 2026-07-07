# Privilege-Control-Security-Pipeline

A security pipeline framework for LLM-based agents combining **privilege
control**, **static/dynamic policy gates**, and **dependency tracking** to
reduce the attack surface of agentic systems.

## Architecture

Each action request passes through a chain of gates, in order:

1. **Privilege Gate**: `PrivilegeControlLLM` runs once at task start and
   scopes the agent down to the minimal subset of available actions the
   task actually needs. This is a relevance decision only; it does not
   judge how risky any individual action is.
2. **Static Policy Gate**: the only thing that decides
   `allow`/`verify`/`deny` for an action, independent of any particular
   task. Nothing downstream can loosen or override this decision. An
   action with no declared static rule defaults to `verify`.
3. **Tool Dependency Gate**: enforces that an action's declared
   prerequisite actions (e.g. read before write) have already completed
   for this task.
4. **Data Dependency Gate**: enforces that an action's arguments contain
   the specific content a task requires (e.g. the exact recipient address
   named in the user's request), via a deterministic exact match.

The Tool and Data Dependency Gates are driven by `DynamicPolicyGenerator`,
which runs once per task (not once per action) and generates a
`DynamicPolicy`: a set of task-specific ordering and content constraints
over whichever actions privilege control already enabled. Dynamic policy
has no ability to influence `allow`/`verify`/`deny` decisions; that
separation is deliberate and covered by regression tests (see
`adaptive_policy/tests/test_policy_gates.py`).

Static policy declares the *shape* of a content constraint (e.g. `send_email`'s
`to` argument must be an `"email"`); dynamic policy fills in the *specific*
expected value for the current task (e.g. the address named in the request).

All decisions are recorded in an append-only `AuditLog`. See
`AdaptiveSecurityPipeline` (`adaptive_policy/policy/pipeline.py`) for the
orchestrator that ties privilege control, static policy, dynamic policy,
and the gate chain together for a single task.

### Package layout

```
adaptive_policy/
├── core/           # ActionRequest, provenance-tracked Value, TaskExecutionState
├── policy/         # privilege control, static policy, dynamic policy, gate chain, pipeline
├── logging/        # audit log
├── integrations/   # AgentDojo benchmark adapter
└── tests/
```

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
- Privilege control (task-scoped action subset)
- Static policy gate (hardcoded allow/verify/deny per action)
- Dynamic policy generation (task-specific tool/data dependencies, generated once per task)
- Tool and data dependency gates
- Audit logging
- AgentDojo integration and live end-to-end tests

Not yet implemented:
- Static policy coverage for non-email AgentDojo suites (banking/slack/travel
  currently fall back to the default `verify` for every action, since no
  static rules are declared for them yet)
- Full CaMeL-style data flow / taint propagation (currently lightweight
  provenance tracking only, in `core/data_flow.py`)
- End-to-end eval harness / benchmark scoring

Roadmap:
- Data Flow
    - Multi-source provenance labels
    - Provenance joins and lineage tracking
    - Taint propagation through LLM transformations
    - Provenance-aware policy rules
    - Full CaMeL-style data flow control
- Evaluation
    - End-to-end eval harness
