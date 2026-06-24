# Privilege-Control-Security-Pipeline
A security pipeline framework for LLM-based agents combining **dynamic privilege control** with **data flow tracking** to reduce the attack surface of agentic systems.

# Architecture
There are 4 main components, **Privilege Control**, **Data Flow Tracking**, **Dynamic Gate Chains**, and **Audit Logging**.

# Current Status
Implemented: 
- Action request model
- Provenance-aware values
- Data flow tracking
- Static policy enforcement
- LLM policy engine interface
- Dynamic privilege scoping
- Escalation workflow
- Audit logging
- Unit tests for core functionality

Not yet implemented:
- Full CaMeL provenance propagation
- End-to-end eval harness
- vLLM integration tests

All current tests (through pytest) **currently use mocked LLM responses**, am currently adding tests with a live server serving vLLM.

Roadmap: 
- Data Flow
    - Multi-source provenance labels
    - Provenance joins and lineage tracking
    - Taint propagation through LLM transformations
    - Provenance-aware policy rules
    - Full CaMeL-style data flow control
- Evaluation
    - Real vLLM integration tests
    - End-to-end eval harness

