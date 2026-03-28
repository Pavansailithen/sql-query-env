"""
tasks.py — Single source of truth for all RL task metadata.

TASKS is built at import time by reading data/queries.json, so queries are
never duplicated. Each entry augments the raw JSON with success_criteria
(plain-English pass condition) and max_steps based on difficulty.
"""

from __future__ import annotations

import json
from pathlib import Path

_QUERIES_PATH = Path(__file__).parent / "data" / "queries.json"

_MAX_STEPS_BY_DIFFICULTY: dict[str, int] = {
    "easy":   3,
    "medium": 4,
    "hard":   5,
}

_SUCCESS_CRITERIA: dict[str, str] = {
    "task_1": (
        "The submitted query must be free of syntax errors and return all four "
        "columns (id, name, email, created_at) from the users table without "
        "any modification to the WHERE clause or table name."
    ),
    "task_2": (
        "The submitted query must include an explicit JOIN condition linking "
        "employees.department_id to departments.id, eliminating the cartesian "
        "product and returning exactly one row per employee with their correct "
        "department name."
    ),
    "task_3": (
        "The submitted query must replace the correlated subquery with a JOIN "
        "and GROUP BY aggregation, compute total view_count per author in a "
        "single pass, and return the top 10 authors ordered by total views "
        "descending."
    ),
}

# ---------------------------------------------------------------------------
# Build TASKS from the JSON source of truth
# ---------------------------------------------------------------------------

def _load_tasks() -> dict[str, dict]:
    with open(_QUERIES_PATH) as f:
        raw: list[dict] = json.load(f)

    tasks: dict[str, dict] = {}
    for i, entry in enumerate(raw, start=1):
        key = f"task_{i}"
        task_id = entry["task_id"]
        difficulty = entry["difficulty"]
        tasks[key] = {
            "difficulty":       difficulty,
            "schema_name":      entry["schema_name"],
            "broken_query":     entry["broken_query"],
            "correct_query":    entry["correct_query"],
            "description":      entry["description"],
            "expected_improvement": entry.get("expected_improvement", ""),
            "success_criteria": _SUCCESS_CRITERIA.get(task_id, ""),
            "max_steps":        _MAX_STEPS_BY_DIFFICULTY[difficulty],
        }
    return tasks


TASKS: dict[str, dict] = _load_tasks()
