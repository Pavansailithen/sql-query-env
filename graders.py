"""
graders.py — Per-task scoring functions for the SQL Query RL Environment.

grade(task_id, action, success, result_rows, execution_time_ms) returns a float
in [0.0, 1.0].  All graders are deterministic: the same inputs always produce
the same score.

Grading rubrics
---------------
task_1 (easy)   — syntax fix
  1.0  executed OK  AND  correct column count (4)
  0.5  executed OK  BUT  wrong column count
  0.0  execution error or no rows returned

task_2 (medium) — JOIN condition fix
  1.0  executed OK  AND  row count == expected join rows (5, one per employee)
  0.5  executed OK  BUT  row count > expected (cartesian product)
  0.3  executed OK  BUT  fewer rows than expected
  0.0  execution error

task_3 (hard)   — correlated subquery → JOIN+GROUP BY
  1.0  executed OK  AND  execution_time_ms < 50
  0.7  executed OK  AND  execution_time_ms >= 50  (correct but not optimised)
  0.3  executed OK  BUT  wrong number of rows returned
  0.0  execution error
"""

from __future__ import annotations

from models import SQLAction

# Expected constants for deterministic grading
_TASK1_EXPECTED_COLUMNS = 4          # id, name, email, created_at
_TASK2_EXPECTED_ROW_COUNT = 5        # 5 employees, each matched to one dept
_TASK3_EXPECTED_ROW_COUNT = 5        # 5 authors in sample data → all returned
_TASK3_FAST_THRESHOLD_MS = 50.0


def _grade_task_1(
    action: SQLAction,
    success: bool,
    result_rows: list,
    execution_time_ms: float,
) -> float:
    """Easy: SELECT statement syntax fix — checks execution and column count."""
    if not success:
        return 0.0

    if not result_rows:
        return 0.0

    actual_columns = len(result_rows[0])

    if actual_columns == _TASK1_EXPECTED_COLUMNS:
        return 1.0
    return 0.5


def _grade_task_2(
    action: SQLAction,
    success: bool,
    result_rows: list,
    execution_time_ms: float,
) -> float:
    """Medium: JOIN condition fix — checks that cartesian product is gone."""
    if not success:
        return 0.0

    row_count = len(result_rows)

    if row_count == _TASK2_EXPECTED_ROW_COUNT:
        return 1.0
    if row_count > _TASK2_EXPECTED_ROW_COUNT:
        return 0.5      # cartesian product still happening
    # Fewer rows than expected — unexpected, treat as wrong
    return 0.3


def _grade_task_3(
    action: SQLAction,
    success: bool,
    result_rows: list,
    execution_time_ms: float,
) -> float:
    """Hard: correlated subquery → JOIN+GROUP BY, also rewarding speed."""
    if not success:
        return 0.0

    row_count = len(result_rows)

    if row_count != _TASK3_EXPECTED_ROW_COUNT:
        return 0.3      # executed but wrong result set

    if execution_time_ms < _TASK3_FAST_THRESHOLD_MS:
        return 1.0      # correct AND fast
    return 0.7          # correct but slow (subquery may still be present)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_GRADERS = {
    "task_1": _grade_task_1,
    "task_2": _grade_task_2,
    "task_3": _grade_task_3,
}


def grade(
    task_id: str,
    action: SQLAction,
    success: bool,
    result_rows: list,
    execution_time_ms: float,
) -> float:
    """
    Return a score in [0.0, 1.0] for the given task and execution result.

    Parameters
    ----------
    task_id:
        One of "task_1", "task_2", "task_3".
    action:
        The SQLAction submitted by the agent.
    success:
        True if the query executed without errors.
    result_rows:
        List of dictionaries representing the query results.
    execution_time_ms:
        Wall-clock query execution time in milliseconds.

    Returns
    -------
    float in [0.0, 1.0].
    """
    grader = _GRADERS.get(task_id)
    if grader is None:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {list(_GRADERS)}"
        )
    return grader(action, success, result_rows, execution_time_ms)

