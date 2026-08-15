"""Evaluate Qwen3-8B on AgentDojo with and without privilege control.

Agent LLM: OpenAILLM against vLLM's OpenAI-compatible server (hermes
tool-call parser). AgentDojo's PromptingLLM was tried first but expects a
Llama-3-style text format Qwen3 doesn't produce, silently dropping calls.

Privilege control / dynamic policy: offline vLLM that runs on a separate GPU, so
that it runs concurrently with the served agent model.

Pipeline: AgentDojo's SystemMessage/InitQuery/ToolsExecutionLoop/ToolsExecutor,
with PrivilegeControlGate + BlockedCallFeedback in the tool loop.
"""

import os
from pathlib import Path
from dataclasses import dataclass

from openai import OpenAI

from agentdojo.benchmark import benchmark_suite_without_injections, benchmark_suite_with_injections, get_suite
from agentdojo.attacks.baseline_attacks import DirectAttack
from agentdojo.agent_pipeline import (
    AgentPipeline,
    SystemMessage,
    InitQuery,
    ToolsExecutor,
    ToolsExecutionLoop,
    OpenAILLM,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.logging import OutputLogger

from adaptive_policy.policy.privilege_control import PrivilegeControlLLM
from adaptive_policy.policy.dynamic_policy import DynamicPolicyGenerator
from adaptive_policy.policy.query_decomposer import QueryDecomposer
from adaptive_policy.integrations.agentdojo_pipeline import PrivilegeControlGate, BlockedCallFeedback
from adaptive_policy.tests.test_agentdojo_integration import load_agentdojo_action_descriptors

BENCHMARK_VERSION = "v1"
DEFAULT_SYSTEM_MESSAGE = load_system_message(None)

# Box is shared with other users; pick a free GPU rather than hardcoding one.
def _pick_free_gpu(exclude: set[str], min_free_gib: float = 20.0) -> str:
    import subprocess
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in output.strip().splitlines():
        index, free_mib = (part.strip() for part in line.split(","))
        if index not in exclude and float(free_mib) / 1024 >= min_free_gib:
            return index
    raise RuntimeError(f"No GPU with >= {min_free_gib} GiB free (excluding {exclude})")


PRIVILEGE_CONTROL_CUDA_DEVICE = _pick_free_gpu(exclude={"0"})


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""
    agent_type: str
    suite_name: str
    total_tasks: int
    completed_tasks: int
    success_rate: float
    attack_success_rate: float | None = None


def _utility_rate(suite_results: dict) -> tuple[int, int, float]:
    """Extract (completed, total, rate) from SuiteResults['utility_results']."""
    utility_results = suite_results["utility_results"]
    total = len(utility_results)
    completed = sum(1 for passed in utility_results.values() if passed)
    rate = completed / total if total else 0.0
    return completed, total, rate


def _security_rate(suite_results: dict) -> float:
    """Extract attack success rate from SuiteResults['security_results'].

    security=True means the agent was NOT compromised (AgentDojo convention).
    Attack success rate is therefore 1 - avg(security_results).
    """
    security_results = suite_results["security_results"]
    if not security_results:
        return 0.0
    avg_security = sum(1 for passed in security_results.values() if passed) / len(security_results)
    return 1.0 - avg_security


def _make_offline_privilege_llm(
    model_name: str,
) -> tuple[PrivilegeControlLLM, DynamicPolicyGenerator, QueryDecomposer]:
    """Build the privilege-control/dynamic-policy/decomposer set on a dedicated GPU."""
    prior = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = PRIVILEGE_CONTROL_CUDA_DEVICE
    try:
        from vllm import LLM as OfflineLLM
        offline_llm = OfflineLLM(
            model=model_name,
            dtype="half",
            max_model_len=32768,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            block_size=16,
            enforce_eager=True,
            enable_chunked_prefill=False,
        )
    finally:
        if prior is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior

    priv_llm = PrivilegeControlLLM(llm=offline_llm, model=model_name)
    dynamic_gen = DynamicPolicyGenerator(llm=offline_llm, model=model_name)
    decomposer = QueryDecomposer(llm=offline_llm, model=model_name)
    return priv_llm, dynamic_gen, decomposer


def create_baseline_pipeline(client: OpenAI, model: str) -> AgentPipeline:
    """SystemMessage -> InitQuery -> LLM -> loop([ToolsExecutor, LLM]). No gating."""
    llm = OpenAILLM(client=client, model=model)
    pipeline = AgentPipeline(
        [
            SystemMessage(DEFAULT_SYSTEM_MESSAGE),
            InitQuery(),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), llm]),
        ]
    )
    pipeline.name = f"{model}-baseline"
    return pipeline


def create_gated_pipeline(
    client: OpenAI,
    model: str,
    priv_llm: PrivilegeControlLLM,
    dynamic_gen: DynamicPolicyGenerator,
    available_actions: list,
    env_type: str = "workspace",
    query_decomposer: QueryDecomposer | None = None,
) -> AgentPipeline:
    """Same structure as baseline, with PrivilegeControlGate ahead of ToolsExecutor
    in the loop so every round of tool_calls is filtered before execution."""
    llm = OpenAILLM(client=client, model=model)
    gate = PrivilegeControlGate(
        priv_llm=priv_llm,
        dynamic_gen=dynamic_gen,
        available_actions=available_actions,
        env_type=env_type,
        query_decomposer=query_decomposer,
    )
    pipeline = AgentPipeline(
        [
            SystemMessage(DEFAULT_SYSTEM_MESSAGE),
            InitQuery(),
            llm,
            ToolsExecutionLoop([gate, ToolsExecutor(), BlockedCallFeedback(), llm]),
        ]
    )
    pipeline.name = f"{model}-privilege-control"
    return pipeline


def run_agent_dojo_eval(
    suite_names: list[str] | None = None,
    llm_model: str = "Qwen/Qwen3-8B",
    server_base_url: str = "http://localhost:8123/v1",
    logdir: Path | None = None,
    include_injections: bool = False,
) -> list[BenchmarkResult]:
    """
    Evaluate baseline vs gated agent on AgentDojo suites.

    Requires a vLLM OpenAI-compatible server already running at
    server_base_url (e.g. `python -m vllm.entrypoints.openai.api_server
    --model Qwen/Qwen3-8B ...`).

    Args:
        suite_names: Suites to run. Default: ["workspace"] for quick iteration.
        llm_model: Model name (must match what the server was started with).
        server_base_url: vLLM OpenAI-compatible server URL.
        logdir: Log directory. Default: /tmp/agentdojo_eval.
        include_injections: Run injection attack tests.

    Returns:
        List of BenchmarkResult objects.
    """

    if suite_names is None:
        suite_names = ["workspace"]

    if logdir is None:
        logdir = Path("/tmp/agentdojo_eval")
    logdir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=server_base_url, api_key="EMPTY")
    priv_llm, dynamic_gen, query_decomposer = _make_offline_privilege_llm(llm_model)

    results = []

    for suite_name in suite_names:
        print(f"\n{'='*60}")
        print(f"Suite: {suite_name}")
        print(f"{'='*60}")

        suite = get_suite(BENCHMARK_VERSION, suite_name)
        action_pool = load_agentdojo_action_descriptors(suite_name)

        # Baseline
        print(f"\n[BASELINE] Running...")
        baseline_pipeline = create_baseline_pipeline(client, llm_model)
        baseline_logdir = logdir / "baseline" / suite_name
        with OutputLogger(str(baseline_logdir)):
            baseline_results = benchmark_suite_without_injections(
                agent_pipeline=baseline_pipeline,
                suite=suite,
                logdir=baseline_logdir,
                force_rerun=False,
                benchmark_version=BENCHMARK_VERSION,
            )

        baseline_completed, baseline_total, baseline_sr = _utility_rate(baseline_results)
        baseline_result = BenchmarkResult(
            agent_type="baseline",
            suite_name=suite_name,
            total_tasks=baseline_total,
            completed_tasks=baseline_completed,
            success_rate=baseline_sr,
        )
        results.append(baseline_result)

        print(f"  Completed: {baseline_completed}/{baseline_total}")
        print(f"  Success rate: {baseline_sr:.2%}")

        # Gated
        print(f"\n[GATED] Running...")
        gated_pipeline = create_gated_pipeline(
            client,
            llm_model,
            priv_llm=priv_llm,
            dynamic_gen=dynamic_gen,
            available_actions=action_pool,
            env_type="workspace" if suite_name == "workspace" else "general",
            query_decomposer=query_decomposer,
        )
        gated_logdir = logdir / "gated" / suite_name
        with OutputLogger(str(gated_logdir)):
            gated_results = benchmark_suite_without_injections(
                agent_pipeline=gated_pipeline,
                suite=suite,
                logdir=gated_logdir,
                force_rerun=False,
                benchmark_version=BENCHMARK_VERSION,
            )

        gated_completed, gated_total, gated_sr = _utility_rate(gated_results)
        gated_result = BenchmarkResult(
            agent_type="gated",
            suite_name=suite_name,
            total_tasks=gated_total,
            completed_tasks=gated_completed,
            success_rate=gated_sr,
        )
        results.append(gated_result)

        print(f"  Completed: {gated_completed}/{gated_total}")
        print(f"  Success rate: {gated_sr:.2%}")

        # Compare
        print(f"\n[COMPARISON]")
        print(f"  Baseline utility:  {baseline_sr:.2%}")
        print(f"  Gated utility:     {gated_sr:.2%}")
        print(f"  Delta Utility:         {gated_sr - baseline_sr:+.2%}")

        # Injections (optional)
        if include_injections:
            print(f"\n[INJECTIONS] Running attack tests...")

            baseline_attack = DirectAttack(task_suite=suite, target_pipeline=baseline_pipeline)
            baseline_inj_logdir = logdir / "baseline_injection" / suite_name
            with OutputLogger(str(baseline_inj_logdir)):
                baseline_inj = benchmark_suite_with_injections(
                    agent_pipeline=baseline_pipeline,
                    suite=suite,
                    attack=baseline_attack,
                    logdir=baseline_inj_logdir,
                    force_rerun=False,
                    benchmark_version=BENCHMARK_VERSION,
                )

            gated_attack = DirectAttack(task_suite=suite, target_pipeline=gated_pipeline)
            gated_inj_logdir = logdir / "gated_injection" / suite_name
            with OutputLogger(str(gated_inj_logdir)):
                gated_inj = benchmark_suite_with_injections(
                    agent_pipeline=gated_pipeline,
                    suite=suite,
                    attack=gated_attack,
                    logdir=gated_inj_logdir,
                    force_rerun=False,
                    benchmark_version=BENCHMARK_VERSION,
                )

            baseline_attack_sr = _security_rate(baseline_inj)
            gated_attack_sr = _security_rate(gated_inj)

            baseline_result.attack_success_rate = baseline_attack_sr
            gated_result.attack_success_rate = gated_attack_sr

            print(f"  Baseline attack success: {baseline_attack_sr:.2%}")
            print(f"  Gated attack success:    {gated_attack_sr:.2%}")
            print(f"  Safety Delta:            {baseline_attack_sr - gated_attack_sr:+.2%}")

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print summary of results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    by_suite = {}
    for r in results:
        if r.suite_name not in by_suite:
            by_suite[r.suite_name] = {}
        by_suite[r.suite_name][r.agent_type] = r

    for suite_name in sorted(by_suite.keys()):
        baseline = by_suite[suite_name].get("baseline")
        gated = by_suite[suite_name].get("gated")

        print(f"{suite_name.upper()}")
        if baseline:
            print(f"  Baseline: {baseline.success_rate:.2%}" + (f" | Attack: {baseline.attack_success_rate:.2%}" if baseline.attack_success_rate else ""))
        if gated:
            print(f"  Gated:    {gated.success_rate:.2%}" + (f" | Attack: {gated.attack_success_rate:.2%}" if gated.attack_success_rate else ""))
        print()


if __name__ == "__main__":
    results = run_agent_dojo_eval(suite_names=["workspace"], include_injections=False)
    print_summary(results)
