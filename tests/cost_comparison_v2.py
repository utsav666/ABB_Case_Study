"""
Cost Comparison v2: Directly invokes the orchestrator with token tracking.

Patches ChatAnthropic.invoke to intercept all LLM calls and accumulate token usage.

Compares:
  - Graph path (with connection_id) → real DB schema
  - Preserve path (without connection_id) → schema_preserve.json
"""

import sys
import time
import json
from typing import Any
from unittest.mock import patch

sys.path.insert(0, ".")

from app.envsetup.env_loader import load_env
load_env()

from app.orchestrator.orch_graph import GraphOrchestrator
from app.utils.llm_tracker import extract_langchain_usage
from langchain_anthropic import ChatAnthropic

CONNECTION_ID = "99bfd7ae-e09e-4649-8e0e-17f6c120d50e"

# Claude Opus 4 pricing (per token)
INPUT_PRICE = 15.0 / 1_000_000   # $15 per 1M input tokens
OUTPUT_PRICE = 75.0 / 1_000_000  # $75 per 1M output tokens

QUESTIONS = [
    "How many movies are in the database?",
    "List all movies released in 2018.",
    "What are the top 10 highest-rated movies?",
    "How many distinct languages are available?",
    "List all countries in the database.",
    "Show movies released between 2010 and 2015.",
    "How many female persons are in the database?",
    "Which genres have the most movies?",
    "Show the top 5 directors by number of movies directed.",
    "Which actor-director pairs have collaborated the most?",
]


def run_with_tracking(question: str, connection_id: str | None = None) -> dict[str, Any]:
    """Run orchestrator and capture all LLM token usage."""
    acc = {"input": 0, "output": 0, "calls": 0}
    original_invoke = ChatAnthropic.invoke

    def patched_invoke(self, *args, **kwargs):
        result = original_invoke(self, *args, **kwargs)
        usage = extract_langchain_usage(result)
        acc["input"] += usage["prompt_tokens"]
        acc["output"] += usage["completion_tokens"]
        acc["calls"] += 1
        return result

    start = time.time()
    try:
        with patch.object(ChatAnthropic, "invoke", patched_invoke):
            orch = GraphOrchestrator(
                question, thread_id="cost-test", connection_id=connection_id
            )
            result = orch.executor()
        elapsed = time.time() - start
        sql = result.get("sql", "") if isinstance(result, dict) else ""
        return {"ok": True, "time": round(elapsed, 2), "tokens": acc, "sql": sql}
    except Exception as e:
        elapsed = time.time() - start
        return {"ok": False, "time": round(elapsed, 2), "tokens": acc, "error": str(e)}


def cost(inp, out):
    return inp * INPUT_PRICE + out * OUTPUT_PRICE


def main():
    print("=" * 90, flush=True)
    print("COST COMPARISON: Graph (connection_id) vs Preserve (no connection)", flush=True)
    print(f"Model: claude-opus-4-6 | Pricing: $15/M input, $75/M output", flush=True)
    print(f"Connection ID: {CONNECTION_ID}", flush=True)
    print(f"Questions: {len(QUESTIONS)}", flush=True)
    print("=" * 90, flush=True)
    print(flush=True)

    results = []

    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i:02d}/{len(QUESTIONS)}] {q}", flush=True)

        g = run_with_tracking(q, connection_id=CONNECTION_ID)
        gc = cost(g["tokens"]["input"], g["tokens"]["output"])

        p = run_with_tracking(q, connection_id=None)
        pc = cost(p["tokens"]["input"], p["tokens"]["output"])

        gs = "OK" if g["ok"] else "ERR"
        ps = "OK" if p["ok"] else "ERR"

        print(f"  Graph:    {gs} | {g['time']:.1f}s | {g['tokens']['calls']} calls | "
              f"{g['tokens']['input']} in + {g['tokens']['output']} out | ${gc:.4f}", flush=True)
        print(f"  Preserve: {ps} | {p['time']:.1f}s | {p['tokens']['calls']} calls | "
              f"{p['tokens']['input']} in + {p['tokens']['output']} out | ${pc:.4f}", flush=True)
        print(flush=True)

        results.append({
            "question": q,
            "graph": {
                "status": gs, "time_s": g["time"],
                "llm_calls": g["tokens"]["calls"],
                "input_tokens": g["tokens"]["input"],
                "output_tokens": g["tokens"]["output"],
                "cost_usd": round(gc, 6),
                "sql": g.get("sql", ""),
            },
            "preserve": {
                "status": ps, "time_s": p["time"],
                "llm_calls": p["tokens"]["calls"],
                "input_tokens": p["tokens"]["input"],
                "output_tokens": p["tokens"]["output"],
                "cost_usd": round(pc, 6),
                "sql": p.get("sql", ""),
            },
        })

    # Summary
    print("=" * 90, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 90, flush=True)

    g_in = sum(r["graph"]["input_tokens"] for r in results)
    g_out = sum(r["graph"]["output_tokens"] for r in results)
    g_cost = sum(r["graph"]["cost_usd"] for r in results)
    g_calls = sum(r["graph"]["llm_calls"] for r in results)
    g_time = sum(r["graph"]["time_s"] for r in results)

    p_in = sum(r["preserve"]["input_tokens"] for r in results)
    p_out = sum(r["preserve"]["output_tokens"] for r in results)
    p_cost = sum(r["preserve"]["cost_usd"] for r in results)
    p_calls = sum(r["preserve"]["llm_calls"] for r in results)
    p_time = sum(r["preserve"]["time_s"] for r in results)

    n = len(QUESTIONS)
    print(f"\n{'Metric':<30} {'Graph (conn_id)':<25} {'Preserve (no conn)':<25}", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Total LLM calls':<30} {g_calls:<25} {p_calls}", flush=True)
    print(f"{'Total input tokens':<30} {g_in:<25,} {p_in:,}", flush=True)
    print(f"{'Total output tokens':<30} {g_out:<25,} {p_out:,}", flush=True)
    print(f"{'Total tokens':<30} {g_in + g_out:<25,} {p_in + p_out:,}", flush=True)
    print(f"{'Total cost (USD)':<30} ${g_cost:<24.4f} ${p_cost:.4f}", flush=True)
    print(f"{'Avg cost per question':<30} ${g_cost/n:<24.4f} ${p_cost/n:.4f}", flush=True)
    print(f"{'Total time':<30} {g_time:<25.1f}s {p_time:.1f}s", flush=True)
    print(f"{'Avg time per question':<30} {g_time/n:<25.1f}s {p_time/n:.1f}s", flush=True)

    diff = g_cost - p_cost
    if p_cost > 0:
        pct = (diff / p_cost) * 100
        print(f"\n{'Cost difference':<30} ${diff:+.4f} ({pct:+.1f}% graph vs preserve)", flush=True)

    # Per-question table
    print(f"\n{'#':<4} {'Question':<40} {'G_in':<7} {'G_out':<6} {'G_$':<9} {'P_in':<7} {'P_out':<6} {'P_$':<9}", flush=True)
    print("-" * 88, flush=True)
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r['question'][:38]:<40} "
              f"{r['graph']['input_tokens']:<7} {r['graph']['output_tokens']:<6} "
              f"${r['graph']['cost_usd']:<8.4f} "
              f"{r['preserve']['input_tokens']:<7} {r['preserve']['output_tokens']:<6} "
              f"${r['preserve']['cost_usd']:<8.4f}", flush=True)

    # Save
    out_path = "tests/cost_comparison_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
