"""
Cost Comparison: FastAPI (with connection_id) vs schema_preserve (no connection_id).

Sends sample questions to both endpoints with a workspace_id so LLM token usage
gets tracked in settings.llm_calls. After all questions run, queries the DB for
actual token counts per thread.

FastAPI: http://localhost:8001/chat
  - WITH connection_id  -> uses the graph schema (real DB)
  - WITHOUT connection_id -> uses schema_preserve.json (sample data path)
"""

import json
import sys
import time
import uuid
from collections import defaultdict
from typing import Any

import requests

# Add project root so we can import app modules for DB querying
sys.path.insert(0, ".")

FASTAPI_URL = "http://localhost:8001/chat"
CONNECTION_ID = "99bfd7ae-e09e-4649-8e0e-17f6c120d50e"
WORKSPACE_ID = "982a89a5-9058-4d6b-a3ca-140ee04dd4f2"  # APEX Hyd workspace

# Sample questions from comparison_output_movie.txt (mix of Easy/Medium/Hard)
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


def call_chat(
    question: str,
    connection_id: str | None = None,
    workspace_id: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Call the /chat endpoint and return response + timing."""
    payload: dict[str, Any] = {"question": question}
    if connection_id:
        payload["connection_id"] = connection_id
    if workspace_id:
        payload["workspace_id"] = workspace_id
    if thread_id:
        payload["thread_id"] = thread_id

    start = time.time()
    try:
        resp = requests.post(FASTAPI_URL, json=payload, timeout=120)
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            return {"success": True, "elapsed": elapsed, "response": data}
        else:
            return {"success": False, "elapsed": elapsed, "error": resp.text}
    except Exception as e:
        elapsed = time.time() - start
        return {"success": False, "elapsed": elapsed, "error": str(e)}


def query_thread_token_usage(thread_id: str) -> dict[str, Any]:
    """Query the settings.llm_calls table for token usage by thread."""
    from app.envsetup.env_loader import load_env

    load_env()
    from app.settings_db.gateway import SettingsDatabaseGateway
    from sqlalchemy import text

    gw = SettingsDatabaseGateway()
    with gw.get_session() as s:
        rows = s.execute(
            text("""
                SELECT
                    llm_purpose,
                    llm_model,
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost_usd) as cost_usd,
                    COUNT(*) as call_count
                FROM settings.llm_calls
                WHERE thread_id = :tid
                GROUP BY llm_purpose, llm_model
                ORDER BY llm_purpose
            """),
            {"tid": thread_id},
        ).fetchall()

        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "calls": 0,
        }
        breakdown = []
        for r in rows:
            entry = {
                "purpose": r[0],
                "model": r[1],
                "prompt_tokens": r[2],
                "completion_tokens": r[3],
                "total_tokens": r[4],
                "cost_usd": float(r[5]),
                "call_count": r[6],
            }
            breakdown.append(entry)
            totals["prompt_tokens"] += r[2]
            totals["completion_tokens"] += r[3]
            totals["total_tokens"] += r[4]
            totals["cost_usd"] += float(r[5])
            totals["calls"] += r[6]

        return {"totals": totals, "breakdown": breakdown}


def main():
    print("=" * 80)
    print("COST COMPARISON: Graph (connection_id) vs schema_preserve (no connection)")
    print(f"FastAPI URL: {FASTAPI_URL}")
    print(f"Connection ID: {CONNECTION_ID}")
    print(f"Workspace ID: {WORKSPACE_ID}")
    print(f"Questions: {len(QUESTIONS)}")
    print("=" * 80)
    print()

    results = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i:02d}/{len(QUESTIONS)}] {question}")

        # Use unique thread IDs so we can query costs per-thread
        graph_thread = str(uuid.uuid4())
        preserve_thread = str(uuid.uuid4())

        # --- With connection_id (Graph approach) ---
        graph_result = call_chat(
            question,
            connection_id=CONNECTION_ID,
            workspace_id=WORKSPACE_ID,
            thread_id=graph_thread,
        )

        # --- Without connection_id (schema_preserve approach) ---
        preserve_result = call_chat(
            question,
            connection_id=None,
            workspace_id=WORKSPACE_ID,
            thread_id=preserve_thread,
        )

        graph_status = "OK" if graph_result["success"] else "ERROR"
        preserve_status = "OK" if preserve_result["success"] else "ERROR"

        graph_sql = (
            graph_result.get("response", {}).get("sql", "N/A")
            if graph_result["success"]
            else "N/A"
        )
        preserve_sql = (
            preserve_result.get("response", {}).get("sql", "N/A")
            if preserve_result["success"]
            else "N/A"
        )

        # Query actual token usage from DB
        graph_tokens = query_thread_token_usage(graph_thread)
        preserve_tokens = query_thread_token_usage(preserve_thread)

        g_total = graph_tokens["totals"]["total_tokens"]
        p_total = preserve_tokens["totals"]["total_tokens"]
        g_cost = graph_tokens["totals"]["cost_usd"]
        p_cost = preserve_tokens["totals"]["cost_usd"]

        print(
            f"  Graph:    {graph_status} ({graph_result['elapsed']:.1f}s) "
            f"tokens={g_total} cost=${g_cost:.4f}"
        )
        print(
            f"  Preserve: {preserve_status} ({preserve_result['elapsed']:.1f}s) "
            f"tokens={p_total} cost=${p_cost:.4f}"
        )
        print()

        results.append(
            {
                "question": question,
                "graph": {
                    "status": graph_status,
                    "time_s": round(graph_result["elapsed"], 2),
                    "sql": graph_sql,
                    "thread_id": graph_thread,
                    "tokens": graph_tokens,
                },
                "preserve": {
                    "status": preserve_status,
                    "time_s": round(preserve_result["elapsed"], 2),
                    "sql": preserve_sql,
                    "thread_id": preserve_thread,
                    "tokens": preserve_tokens,
                },
            }
        )

    # --- Summary ---
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    graph_times = [r["graph"]["time_s"] for r in results if r["graph"]["status"] == "OK"]
    preserve_times = [
        r["preserve"]["time_s"] for r in results if r["preserve"]["status"] == "OK"
    ]

    graph_successes = sum(1 for r in results if r["graph"]["status"] == "OK")
    preserve_successes = sum(1 for r in results if r["preserve"]["status"] == "OK")

    graph_total_tokens = sum(
        r["graph"]["tokens"]["totals"]["total_tokens"] for r in results
    )
    preserve_total_tokens = sum(
        r["preserve"]["tokens"]["totals"]["total_tokens"] for r in results
    )
    graph_total_cost = sum(
        r["graph"]["tokens"]["totals"]["cost_usd"] for r in results
    )
    preserve_total_cost = sum(
        r["preserve"]["tokens"]["totals"]["cost_usd"] for r in results
    )

    print(
        f"\n{'Metric':<30} {'Graph (conn_id)':<20} {'Preserve (no conn)':<20}"
    )
    print("-" * 70)
    print(
        f"{'Success rate':<30} "
        f"{graph_successes}/{len(QUESTIONS):<19} "
        f"{preserve_successes}/{len(QUESTIONS)}"
    )

    if graph_times:
        avg_g = sum(graph_times) / len(graph_times)
        avg_p = sum(preserve_times) / len(preserve_times) if preserve_times else 0
        print(f"{'Avg response time':<30} {avg_g:.1f}s{'':<16} {avg_p:.1f}s")
        print(
            f"{'Total time':<30} {sum(graph_times):.1f}s{'':<16} "
            f"{sum(preserve_times):.1f}s"
        )

    print(f"{'Total tokens':<30} {graph_total_tokens:<20} {preserve_total_tokens}")
    print(
        f"{'Total cost (USD)':<30} ${graph_total_cost:.4f}{'':<14} "
        f"${preserve_total_cost:.4f}"
    )
    print(
        f"{'Avg tokens/question':<30} "
        f"{graph_total_tokens // len(QUESTIONS):<20} "
        f"{preserve_total_tokens // len(QUESTIONS)}"
    )

    # Per-question detail table
    print(
        f"\n{'#':<3} {'Question':<45} {'G-Time':<7} {'P-Time':<7} "
        f"{'G-Tok':<7} {'P-Tok':<7} {'G-Cost':<9} {'P-Cost':<9}"
    )
    print("-" * 95)
    for i, r in enumerate(results, 1):
        q = r["question"][:43]
        gt = f"{r['graph']['time_s']:.1f}" if r["graph"]["status"] == "OK" else "ERR"
        pt = (
            f"{r['preserve']['time_s']:.1f}"
            if r["preserve"]["status"] == "OK"
            else "ERR"
        )
        g_tok = r["graph"]["tokens"]["totals"]["total_tokens"]
        p_tok = r["preserve"]["tokens"]["totals"]["total_tokens"]
        g_cost = f"${r['graph']['tokens']['totals']['cost_usd']:.4f}"
        p_cost = f"${r['preserve']['tokens']['totals']['cost_usd']:.4f}"
        print(
            f"{i:<3} {q:<45} {gt:<7} {pt:<7} {g_tok:<7} {p_tok:<7} "
            f"{g_cost:<9} {p_cost:<9}"
        )

    # Token breakdown by purpose
    print("\n\nTOKEN BREAKDOWN BY LLM PURPOSE (aggregated across all questions):")
    print("-" * 80)

    graph_by_purpose: dict[str, dict] = defaultdict(
        lambda: {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "calls": 0}
    )
    preserve_by_purpose: dict[str, dict] = defaultdict(
        lambda: {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "calls": 0}
    )

    for r in results:
        for entry in r["graph"]["tokens"]["breakdown"]:
            p = entry["purpose"]
            graph_by_purpose[p]["prompt"] += entry["prompt_tokens"]
            graph_by_purpose[p]["completion"] += entry["completion_tokens"]
            graph_by_purpose[p]["total"] += entry["total_tokens"]
            graph_by_purpose[p]["cost"] += entry["cost_usd"]
            graph_by_purpose[p]["calls"] += entry["call_count"]
        for entry in r["preserve"]["tokens"]["breakdown"]:
            p = entry["purpose"]
            preserve_by_purpose[p]["prompt"] += entry["prompt_tokens"]
            preserve_by_purpose[p]["completion"] += entry["completion_tokens"]
            preserve_by_purpose[p]["total"] += entry["total_tokens"]
            preserve_by_purpose[p]["cost"] += entry["cost_usd"]
            preserve_by_purpose[p]["calls"] += entry["call_count"]

    all_purposes = sorted(
        set(list(graph_by_purpose.keys()) + list(preserve_by_purpose.keys()))
    )

    print(
        f"\n{'Purpose':<20} {'Graph Tokens':<14} {'Graph Cost':<12} "
        f"{'Preserve Tok':<14} {'Preserve Cost':<12}"
    )
    print("-" * 72)
    for purpose in all_purposes:
        g = graph_by_purpose[purpose]
        p = preserve_by_purpose[purpose]
        print(
            f"{purpose:<20} {g['total']:<14} ${g['cost']:<11.4f} "
            f"{p['total']:<14} ${p['cost']:.4f}"
        )

    # Save detailed results
    output_path = "tests/cost_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
