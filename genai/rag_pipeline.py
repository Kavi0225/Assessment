from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import re
import os
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from prompt import PromptTemplate, FewShotManager, Example, PromptEngine
from llm_evaluation import LLMEvaluator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
load_dotenv()

# Document Schema

@dataclass
class Document:
    text: str
    metadata: Dict

class Textcleaner:
    
    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"\s+", " ", text)    
        text = re.sub(r"Page \d+ of \d+", "", text) 
        return text.strip()
    
class TextChunker:

    @staticmethod
    def fixed_size_chunk(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        return chunks

    @staticmethod
    def recursive_chunk(text: str, max_length: int = 700) -> List[str]:
        sentences = text.split(". ")
        chunks, current = [], ""

        for sentence in sentences:
            if len(current) + len(sentence) < max_length:
                current += sentence + ". "
            else:
                chunks.append(current.strip())
                current = sentence + ". "

        if current:
            chunks.append(current.strip())

        return chunks


# Embedding Wrapper

class EmbeddingModel:

    def __init__(self, model):
        self.model = model

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


# Vector Store 

class VectorStore:

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.documents: List[Document] = []

    def add(self, docs: List[Document]) -> None:
        texts = [d.text for d in docs]
        vectors = self.embedding_model.encode(texts)

        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])

        self.documents.extend(docs)

    def delete(self, indices: List[int]) -> None:
        mask = np.ones(len(self.documents), dtype=bool)
        mask[indices] = False

        self.embeddings = self.embeddings[mask]
        self.documents = [doc for i, doc in enumerate(self.documents) if mask[i]]

    def search(self, query: str, k: int = 5, metadata_filter: Optional[Dict] = None):
        query_vec = self.embedding_model.encode([query])[0]

        similarities = self.embeddings @ query_vec
        top_indices = np.argsort(-similarities)[:k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]

            if metadata_filter:
                if not all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            results.append((doc, similarities[idx]))

        return results


# Hybrid Retriever (Dense + Sparse)

class HybridRetriever:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.tfidf = TfidfVectorizer()
        self.sparse_matrix = None

    def build_sparse_index(self):
        texts = [doc.text for doc in self.vector_store.documents]
        self.sparse_matrix = self.tfidf.fit_transform(texts)

    def search(self, query: str, k: int = 5):
        # Dense results
        dense_results = self.vector_store.search(query, k=k)

        # Sparse results
        query_vec = self.tfidf.transform([query])
        sparse_scores = (self.sparse_matrix @ query_vec.T).toarray().ravel()

        top_sparse = np.argsort(-sparse_scores)[:k]

        # Combine scores
        combined = {}

        for doc, score in dense_results:
            combined[doc.text] = combined.get(doc.text, 0) + score

        for idx in top_sparse:
            text = self.vector_store.documents[idx].text
            combined[text] = combined.get(text, 0) + sparse_scores[idx]

        ranked = sorted(combined.items(), key=lambda x: -x[1])[:k]

        return ranked


# Context Window Management

class ContextBuilder:

    @staticmethod
    def build_context(chunks: List[Tuple[str, float]], max_tokens: int = 1500) -> str:
        context = ""
        token_count = 0

        for text, _ in chunks:
            tokens = len(text.split())

            if token_count + tokens > max_tokens:
                break

            context += text + "\n\n"
            token_count += tokens

        return context.strip()


# Re-Ranking Layer

class ReRanker:

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def rerank(self, query: str, candidates: List[str]) -> List[str]:
        query_vec = self.embedding_model.encode([query])[0]
        candidate_vecs = self.embedding_model.encode(candidates)

        scores = candidate_vecs @ query_vec
        ranked_indices = np.argsort(-scores)

        return [candidates[i] for i in ranked_indices]


# RAG Pipeline Orchestrator

class RAGPipeline:

    def __init__(self, model):
        self.embedding_model = EmbeddingModel(model)
        self.vector_store = VectorStore(self.embedding_model)
        self.retriever = HybridRetriever(self.vector_store)
        self.reranker = ReRanker(self.embedding_model)

    def ingest_documents(self, raw_docs: List[str]) -> None:
        chunked_docs = []

        for doc in raw_docs:
            chunks = TextChunker.recursive_chunk(doc)
            for chunk in chunks:
                chunked_docs.append(Document(chunk, {}))

        self.vector_store.add(chunked_docs)
        self.retriever.build_sparse_index()

    def retrieve_context(self, query: str) -> str:
        hybrid_results = self.retriever.search(query)

        texts = [text for text, _ in hybrid_results]
        reranked = self.reranker.rerank(query, texts)

        return ContextBuilder.build_context([(t, 1.0) for t in reranked])



if __name__ == "__main__":

    shared_model = SentenceTransformer("all-MiniLM-L6-v2")
    # 1. Load PDF
    print("Loading PDF...")
    reader = PdfReader(".pdf") #use your own pdf for testing, make sure to place it in the same directory as this script
    raw_text = ""

    for page in reader.pages:
        raw_text += page.extract_text() + "\n"
    clean = Textcleaner
    raw_text = clean.clean(raw_text)
    load_dotenv()
    cohere_api = os.getenv("COHERE_API_KEY")
    llm = ChatCohere(
            cohere_api_key=cohere_api,
            temperature=0,
            max_token=150,
            model_name="command-r-plus"
        )

    # # use your openai key for testing
    # openai_api = os.getenv("OPENAI_API_KEY")
    # llm = ChatOpenAI(
    
    #     model="gpt-4o-mini",
    #     temperature=0
    # )

    # 2. Build RAG Pipeline
    print("Building RAG index...")
    rag = RAGPipeline(shared_model)
    rag.ingest_documents([raw_text])

    # 3. Query
    query = "What is insured?"
    print("Retrieving context...")
    context = rag.retrieve_context(query)

    print("\n--- Retrieved Context ---\n")
    print(context[:1000])

    # 4. Prompt System Setup
    template = PromptTemplate(
        template="""
    SYSTEM ROLE:
    You are an AI Reinsurance Broker Assistant specialized in underwriting,
    policy, and treaty document analysis. Your responsibility is to read
    retrieved document context and answer broker queries accurately using
    only the provided information.

    OBJECTIVE:
    Understand the broker’s question and extract the exact information
    requested from the provided context.

    STRICT RULES:
    - Use ONLY the information present in the context
    - Do NOT guess, infer, or fabricate missing values
    - If the requested data is not present, respond exactly: NOT FOUND
    - Return answers in a clean structured format suitable for underwriting workflows
    - When a field exists (e.g., UMR, Policy Number, Insured Name, Effective Date),
    extract it exactly as written in the document
    Context:
    {context}

    Question:
    {question}
    
    Return JSON:
    {{
        "answer": "...",
        "confidence": "high/medium/low"
    }}

    """,
        required_vars=["context", "question"],
    )


    fewshot = FewShotManager(shared_model)
    fewshot.add_examples([
        Example(
            input_text="What is umr?",
            output_text='{"answer": "UMR : B1724WLS20A034", "confidence": "high"}'
        )
    ])

    prompt_engine = PromptEngine(template, fewshot)

    final_prompt = prompt_engine.build_prompt(
        query,
        {
            "context": context,
            "question": query
        }
    )

    # print("\n--- Final Prompt ---\n")
    # print(final_prompt[:1000])

    # 5. Simulated LLM Response (Replace with real LLM later)
    print("\nGenerating response...")

    simulated_response = """
    {
        "answer": "The policy excludes damages caused by intentional acts, war, and natural disasters.",
        "confidence": "medium"
    }
    """

    # print("\n--- LLM Raw Output ---\n")
    # print(simulated_response)

    parsed_output = prompt_engine.parse_output(simulated_response)

    print("\n--- Parsed JSON ---\n")
    print(parsed_output)

    # 6. Evaluation
    evaluator = LLMEvaluator(shared_model)

    ground_truth = "Coverage exclusions include intentional damage, war events, and specific disaster categories."

    result = evaluator.evaluate(
        query=query,
        response=parsed_output["answer"],
        ground_truth=ground_truth,
        context=context,
    )

    print("\n--- Evaluation Metrics ---\n")
    print(result)


    # response = llm.invoke(final_prompt)
    response = llm.invoke([
            HumanMessage(content=final_prompt)
        ])

    llm_response = response.content

    print("\n--- LLM Raw Output ---\n")
    print(llm_response)