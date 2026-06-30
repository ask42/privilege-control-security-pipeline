# Privilege-Control-Security-Pipeline
A security pipeline framework for LLM-based agents combining **dynamic privilege control** with **data flow tracking** to reduce the attack surface of agentic systems.

# Architecture
There are 4 main components, **Privilege Control**, **Data Flow Tracking**, **Dynamic Gate Chains**, and **Audit Logging**.

# Current Status
Implemented:
- Action request model
- Provenance-aware values
- Data flow tracking (Camel-style light provenance tracking)
- Static policy enforcement
- LLM policy engine interface with structured XML prompt safeguards
- Dynamic privilege scoping
- Escalation workflow
- Audit logging
- Unit tests for core functionality
- Verified live vLLM integration and automated prompt injection resilience tests

Not yet implemented:
- Full CaMeL provenance propagation
- End-to-end eval harness

Roadmap: 
- Data Flow
    - Multi-source provenance labels
    - Provenance joins and lineage tracking
    - Taint propagation through LLM transformations
    - Provenance-aware policy rules
    - Full CaMeL-style data flow control
- Evaluation
    - End-to-end eval harness

# Testing
Codebase supports both isolated unit tests and live infrastructure integration verification.

### Run Core Unit Tests
To run standard unit tests with mocked LLM responses:
```bash
pytest Control-Security-Pipeline/adaptive_policy/tests/ -v
```
### Run Live vLLM Tests
To verify live inference, token parsing, and prompt injection defense layers against your active local model engine:
```bash
pytest Control-Security-Pipeline/adaptive_policy/tests/test_vllm.py -v -s
```