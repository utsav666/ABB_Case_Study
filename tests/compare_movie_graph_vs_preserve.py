"""
Comparison Test: Graph Approach (connection_id) vs schema_preserve.json (Streamlit path)
Database: Movie DB (postgres)
Connection ID: 99bfd7ae-e09e-4649-8e0e-17f6c120d50e

Runs questions through both approaches and compares:
- Success rate (SQL generated without error)
- Execution rate (SQL runs on DB without error)
- Non-empty result rate (returns rows)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.envsetup.env_loader import load_env

load_env()

from app.agent.sql_generation_agent import generate_sql
from app.agent.table_identification_agent import identify_tables
from app.target_db.controllers import (
    render_graph_summary,
    render_graph_table_structure,
)
from app.utils.schema_summarizer import Summarizer
from app.utils.sql_executor import SqlExec
from app.utils.sql_validator import validator
from app.utils.table_struc_creation import TabStrucCreation
from app.sampledata.data_loader import DataLoader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

CONNECTION_ID = "99bfd7ae-e09e-4649-8e0e-17f6c120d50e"

# Load graph from local JSON file
GRAPH_FILE = os.path.join(os.path.dirname(__file__), "movie_schema_graph.json")


def _load_graph() -> dict[str, Any] | None:
    """Load the schema graph from a local JSON file."""
    if not os.path.exists(GRAPH_FILE):
        print(f"ERROR: Graph file not found at {GRAPH_FILE}")
        return None
    with open(GRAPH_FILE, "r") as f:
        return json.load(f)


QUESTIONS = [
    # Easy (Single Table) - 10
    "How many movies are in the database?",
    "List all movies released in 2018.",
    "What are the top 10 highest-rated movies?",
    "Show all movies with more than 100,000 votes.",
    "How many distinct languages are available?",
    "List all countries in the database.",
    "What is the average movie rating?",
    "Show movies released between 2010 and 2015.",
    "How many female persons are in the database?",
    "What is the oldest movie in the database?",
    # Medium (Joins, Aggregations) - 10
    "List all movies made in India.",
    "How many movies are in Hindi?",
    "Show the top 5 directors by number of movies directed.",
    "Which genres have the most movies?",
    "List all movies shot in Mumbai locations.",
    "How many movies does each country have?",
    "What is the average rating of movies released after 2010?",
    "Show all actors who appeared in more than 50 movies.",
    "List movies that were produced in the USA and rated above 8.",
    "Which language has the highest average movie rating?",
    # Hard (Multi-Join, Subqueries, Complex Logic) - 10
    "Who are the top 5 actors by average movie rating (minimum 10 movies)?",
    "List directors who have directed movies in more than 3 different genres.",
    "Find movies where the director also acted (present in both m_director and m_cast).",
    "What is the year-over-year trend of average movie ratings from 2000 to 2018?",
    "Which actor-director pairs have collaborated the most?",
    "Find the most prolific producer-director combination.",
    "List actors who have worked in movies across 5+ different languages.",
    "What are the top-rated movies (rating > 8) that were shot in locations in India and released after 2005?",
    "For each genre, what is the highest-rated movie and who directed it?",
    "Find persons who have been both a director and a producer for the same movie.",
    # Expert (Analytics, Window Functions, CTEs) - 5
    "Rank directors by their average movie rating, but only include those with at least 5 movies, and show their best and worst rated film.",
    "For each year, show the genre that had the highest average rating along with the number of movies in that genre.",
    "Find the breakout year for each director — the year their movie first crossed 8.0 rating.",
    "Which actors have appeared in movies spanning the widest range of years (longest career)?",
    "Calculate the percentage contribution of each country to total movie production per decade.",
    # NULL handling - 2
    "How many movies have no filming location recorded?",
    "How many persons have unknown gender?",
    # Negation / NOT EXISTS - 3
    "Which movies have no cast listed at all?",
    "Find directors who have never produced a movie.",
    "List languages that have no movies associated with them.",
    # Self-joins - 2
    "Find pairs of movies with the same title but different release years.",
    "Which actors have appeared together in more than 3 movies?",
    # UNION / Set operations - 2
    "List all person names who are either directors or producers (combined, deduplicated).",
    "Which persons are directors but NOT actors?",
    # String pattern matching - 2
    "Find all movies with love in the title.",
    "List all filming locations that mention Mumbai.",
    # Date/time edge cases - 2
    "How many movies were released per decade?",
    "Which decade had the highest average rating?",
    # COUNT DISTINCT / complex aggregation - 2
    "For each director, how many distinct genres have they worked in?",
    "Which country has the most diverse set of genres (most distinct genre combinations)?",
    # Correlated subqueries - 2
    "For each movie, show the number of cast members vs the average cast size across all movies.",
    "Find movies whose rating is above the average rating for their genre.",
    # LIMIT + ORDER + ties - 2
    "What are the 3 lowest-rated movies with more than 1000 votes?",
    "Show the 5 most recent movies per language.",
    # Multi-value / genre queries - 2
    "How many movies contain Drama as one of their genres?",
    "Find movies that are both Comedy and Romance.",
]

DIFFICULTY = (
    ["Easy"] * 10
    + ["Medium"] * 10
    + ["Hard"] * 10
    + ["Expert"] * 5
    + ["NULL"] * 2
    + ["Negation"] * 3
    + ["Self-join"] * 2
    + ["UNION"] * 2
    + ["Pattern"] * 2
    + ["Date"] * 2
    + ["Aggregation"] * 2
    + ["Correlated"] * 2
    + ["Limit"] * 2
    + ["Multi-val"] * 2
)


def run_graph_approach(question: str) -> dict[str, Any]:
    """Run question using graph approach (connection_id path)."""
    graph = _load_graph()
    if graph is None:
        return {"sql_generated": False, "error": "No schema graph found"}

    # Step 1: Identify tables
    graph_summary = render_graph_summary(graph)
    all_tables = [node["qualified_name"] for node in graph.get("nodes", [])]

    table_result = identify_tables(question, schema_summary=graph_summary)
    tables = table_result.get("tables", all_tables) if table_result.get("success") else all_tables

    # Step 2: Generate SQL
    table_structure, _ = render_graph_table_structure(graph)
    sql_result = generate_sql(
        question, tables, table_structure=table_structure
    )

    if not sql_result.get("success"):
        return {"sql_generated": False, "error": sql_result.get("error", "Unknown")}

    sql = sql_result["sql"]

    # Step 3: Validate
    val = validator(sql)
    if val is not True:
        return {"sql_generated": True, "sql": sql, "valid": False, "validation_error": str(val)}

    # Step 4: Execute
    try:
        executor = SqlExec()
        result = executor.sql_executor(sql)
        if hasattr(result, "shape"):
            return {
                "sql_generated": True,
                "sql": sql,
                "valid": True,
                "executed": True,
                "rows": len(result),
                "empty": len(result) == 0,
            }
        else:
            return {
                "sql_generated": True,
                "sql": sql,
                "valid": True,
                "executed": False,
                "exec_error": str(result),
            }
    except Exception as e:
        return {
            "sql_generated": True,
            "sql": sql,
            "valid": True,
            "executed": False,
            "exec_error": str(e),
        }


def run_preserve_approach(question: str) -> dict[str, Any]:
    """Run question using schema_preserve.json approach (Streamlit path)."""
    # Step 1: Identify tables
    schema_summary = Summarizer.summarize_schema()
    data = DataLoader.load_data()
    all_tables = list(data.get("tables", {}).keys())

    table_result = identify_tables(question, schema_summary=schema_summary)
    tables = table_result.get("tables", all_tables) if table_result.get("success") else all_tables

    # Step 2: Generate SQL (uses schema_preserve.json via TabStrucCreation)
    sql_result = generate_sql(question, tables)

    if not sql_result.get("success"):
        return {"sql_generated": False, "error": sql_result.get("error", "Unknown")}

    sql = sql_result["sql"]

    # Step 3: Validate
    val = validator(sql)
    if val is not True:
        return {"sql_generated": True, "sql": sql, "valid": False, "validation_error": str(val)}

    # Step 4: Execute
    try:
        executor = SqlExec()
        result = executor.sql_executor(sql)
        if hasattr(result, "shape"):
            return {
                "sql_generated": True,
                "sql": sql,
                "valid": True,
                "executed": True,
                "rows": len(result),
                "empty": len(result) == 0,
            }
        else:
            return {
                "sql_generated": True,
                "sql": sql,
                "valid": True,
                "executed": False,
                "exec_error": str(result),
            }
    except Exception as e:
        return {
            "sql_generated": True,
            "sql": sql,
            "valid": True,
            "executed": False,
            "exec_error": str(e),
        }


def main():
    print("=" * 80)
    print("COMPARISON: Graph Approach vs schema_preserve.json (MOVIE DB)")
    print(f"Connection ID: {CONNECTION_ID}")
    print(f"Questions: {len(QUESTIONS)}")
    print("=" * 80)

    graph_results = []
    preserve_results = []

    for i, question in enumerate(QUESTIONS):
        difficulty = DIFFICULTY[i] if i < len(DIFFICULTY) else "Unknown"
        print(f"\n[{i+1:02d}/{len(QUESTIONS)}] [{difficulty}] {question[:70]}...")

        # Graph approach
        print("  Graph:    ", end="", flush=True)
        start = time.time()
        g_result = run_graph_approach(question)
        g_time = time.time() - start
        g_result["time"] = round(g_time, 2)
        graph_results.append(g_result)

        if g_result.get("executed"):
            status = f"OK ({g_result['rows']} rows)" if not g_result.get("empty") else "OK (0 rows - EMPTY)"
        elif g_result.get("sql_generated") and not g_result.get("valid", True):
            status = f"INVALID SQL"
        elif g_result.get("sql_generated") and not g_result.get("executed", False):
            status = f"EXEC ERROR: {g_result.get('exec_error', '')[:50]}"
        else:
            status = f"GENERATION FAILED: {g_result.get('error', '')[:50]}"
        print(f"{status} ({g_time:.1f}s)")

        # Preserve approach
        print("  Preserve: ", end="", flush=True)
        start = time.time()
        p_result = run_preserve_approach(question)
        p_time = time.time() - start
        p_result["time"] = round(p_time, 2)
        preserve_results.append(p_result)

        if p_result.get("executed"):
            status = f"OK ({p_result['rows']} rows)" if not p_result.get("empty") else "OK (0 rows - EMPTY)"
        elif p_result.get("sql_generated") and not p_result.get("valid", True):
            status = f"INVALID SQL"
        elif p_result.get("sql_generated") and not p_result.get("executed", False):
            status = f"EXEC ERROR: {p_result.get('exec_error', '')[:50]}"
        else:
            status = f"GENERATION FAILED: {p_result.get('error', '')[:50]}"
        print(f"{status} ({p_time:.1f}s)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    def compute_stats(results, label):
        total = len(results)
        generated = sum(1 for r in results if r.get("sql_generated"))
        valid = sum(1 for r in results if r.get("valid"))
        executed = sum(1 for r in results if r.get("executed"))
        non_empty = sum(1 for r in results if r.get("executed") and not r.get("empty"))
        avg_time = sum(r.get("time", 0) for r in results) / total

        print(f"\n  [{label}]")
        print(f"  SQL Generated:    {generated}/{total} ({generated/total*100:.0f}%)")
        print(f"  Valid SQL:        {valid}/{total} ({valid/total*100:.0f}%)")
        print(f"  Executed on DB:   {executed}/{total} ({executed/total*100:.0f}%)")
        print(f"  Non-empty result: {non_empty}/{total} ({non_empty/total*100:.0f}%)")
        print(f"  Avg time/query:   {avg_time:.1f}s")
        return {"generated": generated, "valid": valid, "executed": executed, "non_empty": non_empty}

    g_stats = compute_stats(graph_results, "Graph (connection_id)")
    p_stats = compute_stats(preserve_results, "schema_preserve.json")

    # Breakdown by category
    print("\n" + "-" * 80)
    print("BREAKDOWN BY CATEGORY")
    print("-" * 80)

    categories = {}
    for i, diff in enumerate(DIFFICULTY):
        if diff not in categories:
            categories[diff] = []
        categories[diff].append(i)

    for cat, indices in categories.items():
        g_slice = [graph_results[i] for i in indices]
        p_slice = [preserve_results[i] for i in indices]
        n = len(indices)
        g_exec = sum(1 for r in g_slice if r.get("executed") and not r.get("empty"))
        p_exec = sum(1 for r in p_slice if r.get("executed") and not r.get("empty"))
        g_err = sum(1 for r in g_slice if not r.get("executed"))
        p_err = sum(1 for r in p_slice if not r.get("executed"))
        print(f"  {cat:12s}: Graph {g_exec}/{n} success, {g_err}/{n} errors | Preserve {p_exec}/{n} success, {p_err}/{n} errors")

    # Save detailed results
    output_path = os.path.join(os.path.dirname(__file__), "comparison_results_movie.json")
    with open(output_path, "w") as f:
        json.dump({
            "graph_results": graph_results,
            "preserve_results": preserve_results,
            "questions": QUESTIONS,
            "summary": {
                "graph": g_stats,
                "preserve": p_stats,
            }
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
