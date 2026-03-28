import requests

class SQLEnv:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def reset(self, task_name="task_1"):
        r = requests.post(f"{self.base_url}/reset", params={"task_name": task_name})
        return r.json()

    def step(self, fixed_query, explanation="", optimization_notes=""):
        payload = {
            "metadata": {},
            "fixed_query": fixed_query,
            "query": fixed_query,
            "explanation": explanation,
            "optimization_notes": optimization_notes
        }
        r = requests.post(f"{self.base_url}/step", json=payload)
        return r.json()

    def state(self):
        r = requests.get(f"{self.base_url}/state")
        return r.json()
