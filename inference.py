"""
inference.py — LLM inference loop for the SQL Query RL Environment.

Runs all three tasks sequentially using an OpenAI-compatible client, then
prints a summary results table.

Required environment variables
-------------------------------
  API_BASE_URL   Base URL of the OpenAI-compatible inference endpoint
  HF_TOKEN       API key / HuggingFace token
  MODEL_NAME     Model identifier to use for completions

Run
---
  python inference.py

Expected to complete within 20 minutes on 2 vCPU / 8 GB RAM.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import sys
import time
from pathlib import Path
from textwrap import dedent

from openai import OpenAI

from client import SQLEnv
from tasks import TASKS

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

_client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["HF_TOKEN"],
)
_MODEL = os.environ["MODEL_NAME"]

# HTTP environment client (expects server already running on localhost:8000)
_env = SQLEnv(base_url="http://localhost:8000")

# Load schemas for prompt context
_SCHEMAS_PATH = Path(__file__).parent / "data" / "schemas.json"
with open(_SCHEMAS_PATH) as _f:
    _SCHEMAS: dict = json.load(_f)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _schema_ddl(schema_name: str) -> str:
    """Return a human-readable table listing for the given schema."""
    schema = _SCHEMAS.get(schema_name, {})
    lines: list[str] = []
    for table, meta in schema.get("tables", {}).items():
        cols = ", ".join(meta["columns"])
        lines.append(f"  {table}({cols})")
    return "\n".join(lines)


def _build_prompt(
    broken_query: str,
    current_query: str,
    feedback: str,
    schema_name: str,
    description: str,
) -> str:
    """Construct the system+user prompt for the LLM."""
    ddl = _schema_ddl(schema_name)
    return dedent(f"""
        You are an expert SQL debugger. Your task is to fix a broken SQL query.

        ## Database schema ({schema_name})
        {ddl}

        ## Task description
        {description}

        ## Original broken query
        ```sql
        {broken_query}
        ```

        ## Your last submitted query
        ```sql
        {current_query}
        ```

        ## Feedback from the environment
        {feedback}

        ## Instructions
        - Return ONLY the corrected SQL query.
        - Do NOT include explanations, markdown fences, or any other text.
        - The query must be valid SQL that can run against the schema above.
    """).strip()


# ---------------------------------------------------------------------------
# SQL extraction
# ---------------------------------------------------------------------------

def _extract_sql(response_text: str) -> str:
    """
    Pull the first SQL statement from the model's response.

    Handles both naked SQL and markdown code fences (``` or ```sql).
    """
    # Strip markdown fences if present
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    # Return raw text, stripped
    return response_text.strip()


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(task_key: str) -> dict:
    """
    Run one task end-to-end and return a result summary dict.

    Parameters
    ----------
    task_key : one of "task_1", "task_2", "task_3"
    """
    task = TASKS[task_key]
    # Map task_key ("task_1") → task_id ("task_001") used by the server
    task_id = task_key
    max_steps = task["max_steps"]
    difficulty = task["difficulty"]

    print(f"\n{'='*60}")
    print(f"Task: {task_key} | Difficulty: {difficulty} | Max steps: {max_steps}")
    print(f"Description: {task['description']}")
    print("="*60)

    # Reset episode
    obs = _env.reset(task_name=task_id)
    current_query = task["broken_query"]
    feedback = obs.get("feedback", "") if isinstance(obs, dict) else obs.feedback
    final_score = 0.0
    steps_taken = 0

    for step_num in range(1, max_steps + 1):
        print(f"\n  [Step {step_num}/{max_steps}]")

        # Build prompt and call LLM
        prompt = _build_prompt(
            broken_query=task["broken_query"],
            current_query=current_query,
            feedback=feedback,
            schema_name=task["schema_name"],
            description=task["description"],
        )

        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw_output = response.choices[0].message.content or ""
        fixed_query = _extract_sql(raw_output)

        print(f"  LLM query: {fixed_query[:120]}{'...' if len(fixed_query) > 120 else ''}")

        # Submit action to environment
        result = _env.step(
            fixed_query=fixed_query,
            explanation="",
            optimization_notes=""
        )

        steps_taken = step_num
        result_dict = result if isinstance(result, dict) else result.__dict__
        feedback = result_dict.get("feedback", "")
        final_score = result_dict.get("reward", 0.0)
        current_query = fixed_query

        done = result_dict.get("done", False)
        print(f"  Reward: {final_score:.4f} | Done: {done}")
        if done:
            break
        print(f"  Feedback: {feedback}")



    return {
        "task_id":     task_id,
        "task_key":    task_key,
        "difficulty":  difficulty,
        "final_score": round(final_score, 4),
        "steps_taken": steps_taken,
        "max_steps":   max_steps,
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    col_w = [10, 12, 10, 13, 12, 10]
    headers = ["task_id", "difficulty", "steps", "max_steps", "final_score", "status"]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)
    for r in results:
        status = "DONE" if r["steps_taken"] < r["max_steps"] else "TIMEOUT"
        print(row_fmt.format(
            r["task_id"],
            r["difficulty"],
            str(r["steps_taken"]),
            str(r["max_steps"]),
            str(r["final_score"]),
            status,
        ))
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()
    results: list[dict] = []

    for task_key in ["task_1", "task_2", "task_3"]:
        result = run_task(task_key)
        results.append(result)

    _print_summary(results)

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
