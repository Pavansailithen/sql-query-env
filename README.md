---
title: SQL Query RL Environment
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - sql
  - debugging
  - optimization
---
 
# SQL Query RL Environment
 
An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment where an AI agent debugs and optimizes broken SQL queries against a live SQLite database.
 
## Environment Description
 
The agent receives a broken or slow SQL query plus a database schema. It must fix syntax errors, eliminate query plan problems (e.g. cartesian products), and rewrite inefficient patterns (e.g. correlated subqueries). The environment executes every submitted query against a real in-memory SQLite database and returns deterministic, objective feedback.
 
## Action Space
 
Each step the agent submits a `SQLAction`:
 
| Field | Type | Required | Description |
|---|---|---|---|
| `fixed_query` | string | ✅ | The corrected SQL query to execute |
| `explanation` | string | ❌ | Human-readable description of changes made |
| `optimization_notes` | string | ❌ | Notes about performance improvements |
 
## Observation Space
 
After each step the environment returns a `SQLObservation`:
 
| Field | Type | Description |
|---|---|---|
| `done` | bool | True when episode ends |
| `reward` | float | Scalar reward for this step (−0.5 to 1.0) |
| `original_query` | string | The original broken query |
| `last_query` | string | The agent's most recent submission |
| `execution_result` | string | Query output or error message |
| `success` | bool | True if query executed without error |
| `execution_time_ms` | float | Wall-clock execution time in milliseconds |
| `result_rows` | list[dict] | Rows returned by the query |
| `feedback` | string | Natural-language explanation of result |
 
## Tasks
 
### Task 1 — Easy: Missing Comma (Syntax Error)
- **Schema**: E-commerce (`users`, `orders`, `order_items`)
- **Problem**: `SELECT id name email` — missing comma between columns causes syntax error
- **Success**: Query executes and returns correct columns
- **Max steps**: 3
- **Grader**: 1.0 if correct columns, 0.5 if runs but wrong columns, 0.0 on syntax error
 
### Task 2 — Medium: Missing JOIN Condition (Cartesian Product)
- **Schema**: HR (`employees`, `departments`, `attendance`)
- **Problem**: `FROM employees, departments` with no WHERE/JOIN condition multiplies every row pair
- **Success**: Correct row count with proper JOIN condition
- **Max steps**: 4
- **Grader**: 1.0 if correct row count, 0.5 if too many rows, 0.0 on error
 
### Task 3 — Hard: Correlated Subquery Optimization
- **Schema**: Blog (`posts`, `authors`, `comments`)
- **Problem**: Correlated subquery in SELECT re-executes full scan per author row — O(n²)
- **Success**: Rewrite using JOIN + GROUP BY, execute under 50ms
- **Max steps**: 5
- **Grader**: 1.0 if correct and fast, 0.7 if correct but slow, 0.3 if wrong rows, 0.0 on error
 
## Reward Function
 
| Signal | Value |
|---|---|
| Query executes without syntax error | +0.3 |
| Result rows match expected | +0.3 |
| Grader score is 1.0 | +0.4 |
| Each step beyond step 2 | −0.1 |
| Submitted query identical to broken query | −0.2 |
 
Total reward is clamped between **−0.5** and **1.0**.
 
## API Endpoints
 
| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset?task_name=task_1` | Start a new episode |
| POST | `/step` | Submit an SQLAction |
| GET | `/state` | Current episode metadata |
| GET | `/health` | Liveness check |
 
## Setup Instructions
 
### Local (without Docker)
 
```bash
git clone https://huggingface.co/spaces/Lithinpavansai/sql-query-env
cd sql-query-env
python -m venv venv
venv/Scripts/activate  # Windows
pip install -r requirements.txt
pip install git+https://github.com/meta-pytorch/OpenEnv.git
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
 
### Local (with Docker)
 
```bash
docker build -t sql-query-env .
docker run -p 8000:8000 sql-query-env
```
 
### Run Baseline Inference
 
```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export HF_TOKEN=your_token_here
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
python inference.py
```
 
## Baseline Scores
 
Tested with `mistralai/Mistral-7B-Instruct-v0.3` via HuggingFace Inference API:
 
| Task | Difficulty | Steps Used | Max Steps | Score | Status |
|---|---|---|---|---|---|
| task_1 | Easy | 1 | 3 | 1.0 | ✅ DONE |
| task_2 | Medium | 1 | 4 | 1.0 | ✅ DONE |
| task_3 | Hard | 1 | 5 | 1.0 | ✅ DONE |
 
**Total runtime**: 14.8 seconds
 
## OpenEnv Validation
 
```
[OK] sql-query-env: Ready for multi-mode deployment
```
