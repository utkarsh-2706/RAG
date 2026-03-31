# simple_rag.py
# Simple RAG implementation — Theory-focused, minimal dependencies
# Simulates embedding + retrieval without requiring API keys

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────
# 1. DOCUMENT STORE (simulated corpus)
# ─────────────────────────────────────────────

DOCUMENTS = [
    "RAG stands for Retrieval-Augmented Generation. It combines retrieval systems with LLMs.",
    "Vector databases store embeddings and support approximate nearest neighbor search.",
    "FAISS is a library by Meta for efficient similarity search over dense vectors.",
    "Chunking splits large documents into smaller pieces before embedding.",
    "Cosine similarity measures the angle between two vectors, used in semantic search.",
    "LLMs can hallucinate facts when they lack grounding context.",
    "Prompt engineering involves crafting inputs to guide LLM behavior effectively.",
]


# ─────────────────────────────────────────────
# 2. EMBEDDING MODEL (simulated via hashing)
# In production: use OpenAI / sentence-transformers
# ─────────────────────────────────────────────

def simulate_embedding(text: str, dim: int = 64) -> np.ndarray:
    """
    Deterministic pseudo-embedding using character-level hashing.
    Replace with real model in production.
    """
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dim).astype(np.float32)


def embed_documents(docs: List[str]) -> np.ndarray:
    return np.array([simulate_embedding(d) for d in docs])


def embed_query(query: str) -> np.ndarray:
    return simulate_embedding(query)


# ─────────────────────────────────────────────
# 3. RETRIEVER — Cosine Similarity Search
# ─────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve(query: str, docs: List[str], doc_embeddings: np.ndarray, top_k: int = 3) -> List[Tuple[float, str]]:
    """
    Returns top-K documents ranked by cosine similarity to query.
    """
    query_vec = embed_query(query)
    scores = [cosine_similarity(query_vec, doc_emb) for doc_emb in doc_embeddings]
    ranked = sorted(zip(scores, docs), reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────
# 4. PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(query: str, retrieved_docs: List[Tuple[float, str]]) -> str:
    context = "\n".join([f"- {doc}" for _, doc in retrieved_docs])
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""
    return prompt


# ─────────────────────────────────────────────
# 5. LLM CALL (simulated)
# Replace with: openai.chat.completions.create(...)
# ─────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """
    Simulated LLM response.
    In production, replace with actual API call:

    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
    """
    print("\n[LLM receives this prompt]")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    return "[Simulated LLM Answer based on retrieved context above]"


# ─────────────────────────────────────────────
# 6. SIMPLE RAG PIPELINE
# ─────────────────────────────────────────────

class SimpleRAG:
    def __init__(self, documents: List[str], top_k: int = 3):
        self.documents = documents
        self.top_k = top_k
        print("Indexing documents...")
        self.doc_embeddings = embed_documents(documents)
        print(f"Indexed {len(documents)} documents.\n")

    def query(self, user_query: str) -> str:
        # Step 1: Retrieve
        retrieved = retrieve(user_query, self.documents, self.doc_embeddings, self.top_k)

        print(f"Query: {user_query}")
        print(f"\nTop-{self.top_k} Retrieved Documents:")
        for score, doc in retrieved:
            print(f"  [{score:.3f}] {doc}")

        # Step 2: Build Prompt
        prompt = build_prompt(user_query, retrieved)

        # Step 3: Generate
        answer = call_llm(prompt)
        return answer


# ─────────────────────────────────────────────
# 7. DEMO RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = SimpleRAG(DOCUMENTS, top_k=3)
    answer = rag.query("What is RAG and why is it used?")
    print(f"\nFinal Answer: {answer}")
