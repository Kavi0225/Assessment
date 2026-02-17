from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import re
import numpy as np


# =====================================================
# Prompt Template System
# =====================================================

class PromptTemplate:
    """
    Handles variable substitution and validation.
    """

    def __init__(self, template: str, required_vars: List[str]):
        self.template = template
        self.required_vars = set(required_vars)

    def format(self, variables: Dict[str, Any]) -> str:
        missing = self.required_vars - set(variables.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")

        return self.template.format(**variables)


# =====================================================
# Few-Shot Example Management
# =====================================================

@dataclass
class Example:
    input_text: str
    output_text: str


class FewShotManager:

    def __init__(self, model):
        self.examples: List[Example] = []
        self.embedder = model
        self.embeddings = None

    def add_examples(self, examples: List[Example]) -> None:
        self.examples.extend(examples)

        texts = [e.input_text for e in self.examples]
        self.embeddings = self.embedder.encode(texts, normalize_embeddings=True)

    def retrieve_similar(self, query: str, k: int = 3) -> List[Example]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec

        top_idx = np.argsort(-scores)[:k]
        return [self.examples[i] for i in top_idx]


# =====================================================
# Chain-of-Thought Wrapper
# =====================================================

class ChainOfThought:

    @staticmethod
    def wrap(prompt: str) -> str:
        return (
            "Think step-by-step before answering.\n"
            "Provide reasoning and then final answer.\n\n"
            + prompt
        )


# =====================================================
# Structured Output Parser
# =====================================================

class OutputParser:

    @staticmethod
    def parse_json(text: str) -> Dict:
        """
        Extract JSON from LLM output safely.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError("Unable to parse JSON output")


# =====================================================
# Retry + Fallback Logic
# =====================================================

class RetryHandler:

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def execute(self, func, *args, **kwargs):
        last_error = None

        for _ in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e

        raise RuntimeError(f"Max retries exceeded: {last_error}")


# =====================================================
# Token Counting & Context Truncation
# =====================================================

class TokenManager:

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    @staticmethod
    def truncate(text: str, max_tokens: int) -> str:
        tokens = text.split()
        return " ".join(tokens[:max_tokens])


# =====================================================
# Prompt Orchestrator
# =====================================================

class PromptEngine:

    def __init__(self, template: PromptTemplate, fewshot: FewShotManager):
        self.template = template
        self.fewshot = fewshot
        self.retry = RetryHandler()

    def build_prompt(self, query: str, variables: Dict[str, Any]) -> str:
        base_prompt = self.template.format(variables)

        examples = self.fewshot.retrieve_similar(query)
        fewshot_block = ""

        for ex in examples:
            fewshot_block += f"Input: {ex.input_text}\nOutput: {ex.output_text}\n\n"

        final_prompt = fewshot_block + base_prompt
        return ChainOfThought.wrap(final_prompt)

    def parse_output(self, output: str) -> Dict:
        return self.retry.execute(OutputParser.parse_json, output)
