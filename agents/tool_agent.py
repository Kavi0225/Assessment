from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
import json


# Tool Schema Definition

@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class Tool:
    schema: ToolSchema
    function: Callable


# Tool Registry

class ToolRegistry:
    """
    Central registry for all agent tools.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.schema.name] = tool

    def remove_tool(self, name: str) -> None:
        self.tools.pop(name, None)

    def list_tools(self) -> List[ToolSchema]:
        return [tool.schema for tool in self.tools.values()]

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)


# Input Validation

class ToolValidator:

    @staticmethod
    def validate(schema: ToolSchema, inputs: Dict[str, Any]) -> None:
        required = schema.parameters.get("required", [])

        for param in required:
            if param not in inputs:
                raise ValueError(f"Missing required parameter: {param}")


# Tool Execution Engine

class ToolExecutor:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.registry.get_tool(tool_name)

        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            ToolValidator.validate(tool.schema, inputs)
            result = tool.function(**inputs)

            return {"success": True, "result": result}

        except Exception as e:
            return {"error": str(e)}


# Multi-Step Tool Chaining Agent

class ToolAgent:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.executor = ToolExecutor(registry)

    def process_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected tool_call format:
        {
            "name": "tool_name",
            "arguments": {...}
        }
        """
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})

        return self.executor.execute(name, args)

    def run_sequence(self, tool_calls: List[Dict[str, Any]]) -> List[Dict]:
        """
        Executes multi-step tool chain.
        """
        results = []

        for call in tool_calls:
            result = self.process_tool_call(call)
            results.append(result)

            if "error" in result:
                break

        return results


# Result Formatter for LLM Consumption

class ResultFormatter:

    @staticmethod
    def format(result: Dict[str, Any]) -> str:
        return json.dumps(result, indent=2)


if __name__ == "__main__":
    print("Testing Tool Agent...")

    # 1️⃣ Create registry
    registry = ToolRegistry()

    # 2️⃣ Define sample tools
    def math_tool(expression: str):
        return eval(expression)

    def search_tool(query: str):
        return f"Mock search results for: {query}"

    # 3️⃣ Register tools
    registry.add_tool(Tool(
        schema=ToolSchema(
            name="math",
            description="Performs math calculations",
            parameters={"required": ["expression"]}
        ),
        function=math_tool
    ))

    registry.add_tool(Tool(
        schema=ToolSchema(
            name="search",
            description="Searches knowledge base",
            parameters={"required": ["query"]}
        ),
        function=search_tool
    ))

    # 4️⃣ Create agent
    agent = ToolAgent(registry)

    print("\nAvailable Tools:")
    for schema in registry.list_tools():
        print("-", schema.name)

    # 5️⃣ Execute tools
    calls = [
        {"name": "math", "arguments": {"expression": "2 + 3 * 5"}},
        {"name": "search", "arguments": {"query": "electronics"}}
    ]

    results = agent.run_sequence(calls)

    print("\nExecution Results:")
    for r in results:
        print(r)

