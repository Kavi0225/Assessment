from typing import List, Dict, Any

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from agents.tool_agent import ToolRegistry
from agents.memory_system import MemoryManager
import sys



# =========================
# LOCAL LLM FACTORY
# =========================

def build_local_llm():
    

    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=128,
        temperature=0
    )

    return HuggingFacePipeline(pipeline=pipe)


# =========================
# PROMPT TEMPLATE
# =========================

REACT_PROMPT = """
You are an intelligent AI assistant that can use tools.

Available Tools:
{tools}

Follow this format:

Question: {question}

Thought:
Action:
Action Input:

OR

Final Answer:
"""


# =========================
# REACT AGENT
# =========================

class ReActAgent:

    def __init__(self, llm=None):
        self.llm = llm or build_local_llm()
        self.tools = ToolRegistry()
        self.memory = MemoryManager()

    # -------------------------

    def _build_prompt(self, question: str) -> str:
        tool_desc = "\n".join(
            f"{name}: {tool.description}"
            for name, tool in self.tools.tools.items()
        )

        context = self.memory.retrieve_context(question)
        memory_context = "\n".join(context["recent"] + context["semantic"])

        return REACT_PROMPT.format(
            tools=tool_desc,
            question=f"{memory_context}\n\n{question}"
        )

    # -------------------------

    def _parse_output(self, text: str):
        action = None
        action_input = None

        for line in text.split("\n"):
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            if line.startswith("Action Input:"):
                action_input = line.replace("Action Input:", "").strip()

        return action, action_input

    # -------------------------

    def run(self, question: str, max_steps: int = 5) -> str:

        prompt = self._build_prompt(question)

        for _ in range(max_steps):

            response = self.llm(prompt)

            if "Final Answer:" in response:
                answer = response.split("Final Answer:")[-1].strip()
                self.memory.store(question)
                self.memory.store(answer)
                return answer

            action, action_input = self._parse_output(response)

            if not action:
                return response

            observation = self.tools.execute(action, action_input)
            prompt += f"\nObservation: {observation}\n"

        return "Unable to complete reasoning."


# =========================
# LOCAL TEST
# =========================

if __name__ == "__main__":
    print("Testing Local ReAct Agent...")

    agent = ReActAgent()

    response = agent.run("What is 2+2?")
    print("\nAgent Response:", response)
