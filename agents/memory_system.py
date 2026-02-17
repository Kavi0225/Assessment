"""
Memory System for Agent Framework

Provides:
1. Short-term conversational memory
2. Long-term semantic memory using vector storage
3. Retrieval interface for agents

Designed for production extensibility.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import deque

from sentence_transformers import SentenceTransformer
import numpy as np


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class MemoryEntry:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================
# SHORT TERM MEMORY
# =========================

class ShortTermMemory:
    """
    Sliding window conversational memory.
    """

    def __init__(self, max_size: int = 20):
        self.memory = deque(maxlen=max_size)

    def add(self, text: str, metadata: Dict = None):
        self.memory.append(
            MemoryEntry(text=text, metadata=metadata or {})
        )

    def get_recent(self, k: int = 5) -> List[str]:
        return [entry.text for entry in list(self.memory)[-k:]]

    def clear(self):
        self.memory.clear()


# =========================
# LONG TERM MEMORY
# =========================

class LongTermMemory:
    """
    Semantic vector memory for persistent knowledge.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.entries: List[MemoryEntry] = []
        self.embeddings: List[np.ndarray] = []

    def add(self, text: str, metadata: Dict = None):
        embedding = self.encoder.encode(text)
        self.entries.append(MemoryEntry(text=text, metadata=metadata or {}))
        self.embeddings.append(embedding)

    def _cosine_similarity(self, query_vec, vectors):
        vectors = np.array(vectors)
        return np.dot(vectors, query_vec) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
        )

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self.entries:
            return []

        query_vec = self.encoder.encode(query)
        scores = self._cosine_similarity(query_vec, self.embeddings)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.entries[i].text for i in top_indices]


# =========================
# HYBRID MEMORY MANAGER
# =========================

class MemoryManager:
    """
    Unified interface combining short and long-term memory.
    """

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def store(self, text: str, metadata: Dict = None):
        self.short_term.add(text, metadata)
        self.long_term.add(text, metadata)

    def retrieve_context(self, query: str) -> Dict[str, List[str]]:
        return {
            "recent": self.short_term.get_recent(),
            "semantic": self.long_term.search(query)
        }

    def reset_session(self):
        self.short_term.clear()


if __name__ == "__main__":
    print("Testing Memory System...")

    memory = MemoryManager()

    memory.store("User asked about electronics warranty")
    memory.store("System replied: warranty is 1 year")

    context = memory.retrieve_context("electronics")

    print("\nRecent Memory:", context["recent"])
    print("Semantic Memory:", context["semantic"])
