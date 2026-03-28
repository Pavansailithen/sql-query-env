"""
rewards.py — Shaped reward function for the SQL Query RL Environment.

compute_reward() layers multiple partial-credit signals on top of the
deterministic grader score, then clamps the final value to [-0.5, 1.0].

Reward breakdown
----------------
+0.3  query executed without a syntax error            (partial progress)
+0.3  result rows match expected output (grader >= 0.5 AND success)
+0.4  perfect solution  (grader score == 1.0)
-0.1  per step beyond step 2                           (efficiency penalty)
-0.2  submitted query identical to broken query        (no-op penalty)
────────────────────────────────────────────────────────
Clamped to [-0.5, 1.0]
"""

from __future__ import annotations

from models import SQLAction
from graders import grade

# ---------------------------------------------------------------------------
# Thresholds / weights
# ---------------------------------------------------------------------------

_EXECUTION_BONUS = 0.3      # reward for any error-free execution
_PARTIAL_BONUS   = 0.3      # reward for correct-ish result rows
_PERFECT_BONUS   = 0.4      # reward for a perfect grader score
_STEP_PENALTY    = 0.1      # subtracted per step beyond step 2
_NOOP_PENALTY    = 0.2      # subtracted when query is unchanged
_REWARD_MIN      = -0.5
_REWARD_MAX      =  1.0


def compute_reward(
    task_id: str,
    action: SQLAction,
    execution_result: str,
    success: bool,
    result_rows: list[dict],
    execution_time_ms: float,
    step_count: int,
    max_steps: int,
    original_query: str = "",
) -> tuple[float, str]:
    """
    Compute a shaped reward and a human-readable feedback string.

    Parameters
    ----------
    task_id:
        One of "task_001", "task_002", "task_003".
    action:
        The SQLAction submitted by the agent.
    execution_result:
        Dict produced by SQLEnvironment.step() with keys:
        ``success`` (bool), ``result_rows`` (list[dict]),
        ``execution_result`` (str).
    execution_time_ms:
        Wall-clock query execution time in milliseconds.
    step_count:
        Current step number within the episode (1-indexed).
    max_steps:
        Maximum steps allowed for this episode.
    original_query:
        The original broken query to detect no-op submissions.

    Returns
    -------
    (reward: float, feedback: str)
    """
    # Parameters success and result_rows are now passed directly.

    reward = 0.0
    feedback_parts: list[str] = []

    # ---- No-op penalty ------------------------------------------------
    if original_query and action.query.strip() == original_query.strip():
        reward -= _NOOP_PENALTY
        feedback_parts.append("Penalty: query is identical to the broken query.")

    # ---- Execution bonus ----------------------------------------------
    if success:
        reward += _EXECUTION_BONUS
        feedback_parts.append("Good: query executed without a syntax error.")
    else:
        feedback_parts.append(
            f"Error: query failed to execute — {execution_result}"
        )

    # ---- Grader score -------------------------------------------------
    grader_score = 0.0
    if success:
        grader_score = grade(
            task_id=task_id,
            action=action,
            success=success,
            result_rows=result_rows,
            execution_time_ms=execution_time_ms,
        )

        if grader_score >= 0.5:
            reward += _PARTIAL_BONUS
            feedback_parts.append("Good: result rows match the expected output.")
        else:
            feedback_parts.append(
                "Partial: query ran but result rows do not match expected output."
            )

        if grader_score == 1.0:
            reward += _PERFECT_BONUS
            feedback_parts.append("Excellent: perfect solution!")

    # ---- Efficiency penalty -------------------------------------------
    steps_over = max(0, step_count - 2)
    if steps_over > 0:
        penalty = steps_over * _STEP_PENALTY
        reward -= penalty
        feedback_parts.append(
            f"Efficiency penalty: -{penalty:.1f} ({steps_over} step(s) beyond step 2)."
        )

    # ---- Clamp --------------------------------------------------------
    reward = max(_REWARD_MIN, min(_REWARD_MAX, reward))

    feedback = " ".join(feedback_parts) if feedback_parts else "No feedback available."
    return round(reward, 4), feedback
