
from typing import Dict

from agents.react_agent import ReActAgent


# =========================
# BASE AGENT WRAPPER
# =========================

class SpecializedAgent:
    """
    Wraps ReActAgent with a domain specialization.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.agent = ReActAgent()

    def handle(self, task: str) -> str:
        """
        Adds domain context before execution.
        """

        enriched_task = f"""
You are a specialist in {self.domain}.
Handle the following task carefully:

{task}
"""
        return self.agent.run(enriched_task)


# =========================
# ROUTER
# =========================

class TaskRouter:
    """
    Simple rule-based routing system.
    Designed to be replaceable with ML classifier later.
    """

    def __init__(self):
        self.routes = {
            "analytics": ["analyze", "trend", "statistics", "metrics"],
            "retrieval": ["search", "find", "lookup", "retrieve"],
            "operations": ["execute", "run", "perform", "task"]
        }

    def route(self, task: str) -> str:
        task_lower = task.lower()

        for domain, keywords in self.routes.items():
            if any(k in task_lower for k in keywords):
                return domain

        return "general"


# =========================
# MULTI AGENT COORDINATOR
# =========================

class MultiAgentSystem:


    def __init__(self):
        self.router = TaskRouter()

        self.agents: Dict[str, SpecializedAgent] = {
            "analytics": SpecializedAgent("Data Analytics"),
            "retrieval": SpecializedAgent("Information Retrieval"),
            "operations": SpecializedAgent("Task Execution"),
            "general": SpecializedAgent("General Assistance")
        }

    # =========================
    # MAIN ENTRYPOINT
    # =========================

    def execute(self, task: str) -> str:
        """
        Routes and executes task via appropriate agent.
        """

        domain = self.router.route(task)
        agent = self.agents.get(domain)

        return agent.handle(task)


if __name__ == "__main__":
    print("Testing Multi-Agent System...")

    system = MultiAgentSystem()

    task = "Analyze sales trends for electronics"

    result = system.execute(task)

    print("\nTask:", task)
    print("Result:", result)
