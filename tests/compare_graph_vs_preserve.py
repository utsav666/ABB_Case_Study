"""
Comparison Test: Graph Approach (connection_id) vs schema_preserve.json (Streamlit path)

Runs all 40 production schema questions through both approaches and compares:
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

CONNECTION_ID = "3403e1d0-c16c-450a-886b-4c5a929f21bb"

# Load graph from file (since settings.schema_graphs table may not exist locally)
GRAPH_FILE = os.path.join(os.path.dirname(__file__), "production_schema_graph.json")


def _load_graph() -> dict[str, Any] | None:
    """Load the schema graph from a local JSON file."""
    if not os.path.exists(GRAPH_FILE):
        print(f"ERROR: Graph file not found at {GRAPH_FILE}")
        return None
    with open(GRAPH_FILE, "r") as f:
        return json.load(f)

QUESTIONS = [
    # Easy (1-2 tables)
    "List all products in the Mountain product line.",
    "How many products are there in each color?",
    "What are the 4 product categories?",
    "Find all products with a list price greater than $1000.",
    "Which products have never been sold (sell end date is in the past)?",
    "How many products are manufactured in-house vs purchased?",
    "List all manufacturing locations and their hourly cost rates.",
    "What are the different scrap reasons for defective work orders?",
    "Find all products that weigh more than 100 pounds.",
    "Show all products available in size Large or Extra Large.",
    # Medium (2-3 tables)
    "How many products are in each subcategory?",
    "What is the average list price for Bikes vs Clothing vs Components vs Accessories?",
    "Which products are currently in inventory at the Final Assembly location?",
    "Show the total inventory quantity by product category.",
    "List all products that have had a price increase (multiple entries in list price history).",
    "Find the top 10 most expensive components (class = High) by list price.",
    "What is the total quantity currently on each shelf in the Metal Storage location?",
    "Find products that have reviews with a rating of 5.",
    "How many products in each category are currently for sale (sellenddate IS NULL)?",
    "Show the products with the highest margin (listprice - standardcost), top 10.",
    # Hard (3-4 tables)
    "For each product category, what is the average number of days to manufacture?",
    "Which products had their cost increase by more than 20% between any two cost history periods?",
    "Show the bill of materials tree for all assemblies: the assembly product name, component name, quantity needed, and BOM level.",
    "For each location, show the total inventory value (quantity times product list price).",
    "Find all products that are components in a bill of materials but are NOT finished goods.",
    "What is the scrap rate (scrapped qty / order qty) by product subcategory for work orders?",
    "Rank products within each subcategory by their list price using DENSE_RANK.",
    "Show products whose current list price differs from their standard cost by more than 200% markup.",
    "For each product model, show how many products use it and the English description.",
    "Find products that have zero inventory across all locations.",
    # Expert (4-5+ tables)
    "For each product category, show the month-over-month sales transaction volume and a 3-month moving average.",
    "Find the top 5 products by total work order quantity in each category, and show their scrap rate and average actual routing cost.",
    "Build a complete product cost analysis: for each product currently for sale, show the category, subcategory, current list price, current standard cost, total inventory quantity, total inventory value, and profit margin percentage.",
    "Compare the actual vs planned cost for work order routings, grouped by location and product category. Show the variance and rank locations by total cost overrun.",
    "Find all products that are BOTH a component in a bill of materials AND have their own work orders. Show the BOM level, assembly product name, and total work order quantity.",
    "For products with multiple price history records, calculate the cumulative price change percentage from their first list price to their current price, grouped by subcategory.",
    "Create a routing efficiency report: for each location, show the average actual hours vs scheduled hours, the most commonly routed product category, and the total number of operations.",
    "Using the transaction history and archive combined (UNION), find the year-over-year change in total transaction value by product category, identifying categories with declining trends.",
    "Identify at-risk products: those where current inventory is below the reorder point, they are still for sale, and they have had work order scraps in the last 6 months. Show their category, current stock, reorder point, and scrap reasons.",
    "For the Bikes category, show a hierarchical bill of materials explosion: list all components recursively through BOM levels, their quantities per assembly, total cost contribution, and what percentage of the final product cost each component represents.",
]

DIFFICULTY = (
    ["Easy"] * 10 + ["Medium"] * 10 + ["Hard"] * 10 + ["Expert"] * 10
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
    print("COMPARISON: Graph Approach vs schema_preserve.json")
    print(f"Connection ID: {CONNECTION_ID}")
    print(f"Questions: {len(QUESTIONS)}")
    print("=" * 80)

    graph_results = []
    preserve_results = []

    for i, question in enumerate(QUESTIONS):
        difficulty = DIFFICULTY[i]
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

    # Breakdown by difficulty
    print("\n" + "-" * 80)
    print("BREAKDOWN BY DIFFICULTY")
    print("-" * 80)

    for diff, start_idx, end_idx in [("Easy", 0, 10), ("Medium", 10, 20), ("Hard", 20, 30), ("Expert", 30, 40)]:
        g_slice = graph_results[start_idx:end_idx]
        p_slice = preserve_results[start_idx:end_idx]
        g_exec = sum(1 for r in g_slice if r.get("executed") and not r.get("empty"))
        p_exec = sum(1 for r in p_slice if r.get("executed") and not r.get("empty"))
        g_err = sum(1 for r in g_slice if not r.get("executed"))
        p_err = sum(1 for r in p_slice if not r.get("executed"))
        print(f"  {diff:8s}: Graph {g_exec}/10 success, {g_err}/10 errors | Preserve {p_exec}/10 success, {p_err}/10 errors")

    # Save detailed results
    output_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
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
