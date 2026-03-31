# hyde_rag.py
# HyDE — Hypothetical Document Embeddings
# Core idea: embed a generated hypothesis, not the raw query

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────
# 1. DOCUMENT CORPUS
# ─────────────────────────────────────────────

DOCUMENTS = [
    "Transformer attention complexity scales quadratically with sequence length, causing memory issues.",
    "Sparse attention mechanisms like Longformer reduce complexity from O(n²) to O(n).",
    "Positional encodings in transformers encode token order using sinusoidal functions.",
    "BERT uses bidirectional attention while GPT uses causal (left-to-right) attention.",
    "Flash Attention optimizes memory access patterns to speed up attention computation.",
    "Retrieval-Augmented Generation grounds LLMs in retrieved external documents.",
    "Chunking strategy and embedding quality are the two biggest levers in RAG systems.",
    "Cross-encoder rerankers score query-document pairs jointly for higher precision.",
]


# ─────────────────────────────────────────────
# 2. SIMULATED EMBEDDING
# ─────────────────────────────────────────────

def simulate_embedding(text: str, dim: int = 64) -> np.ndarray:
    """
    Deterministic pseudo-embedding.
    Replace with: sentence_transformers, openai embeddings, etc.
    """
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dim).astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def retrieve(query_or_hyde: str, docs: List[str], top_k: int = 3) -> List[Tuple[float, str]]:
    q_vec = simulate_embedding(query_or_hyde)
    doc_vecs = [simulate_embedding(d) for d in docs]
    scored = [(cosine_similarity(q_vec, dv), d) for dv, d in zip(doc_vecs, docs)]
    return sorted(scored, reverse=True)[:top_k]


# ─────────────────────────────────────────────
# 3. HYPOTHESIS GENERATOR (Step 1 of HyDE)
# This is an LLM call in production.
# ─────────────────────────────────────────────

def generate_hypothesis(query: str) -> str:
    """
    Generate a hypothetical document that would answer the query.

    In production:
    ─────────────────────
    prompt = f\"\"\"Write a short technical passage (2-3 sentences) that
    directly answers the following question. Be specific and use domain language.
    Do NOT say 'I' or 'This passage'. Write as if you are the document.

    Question: {query}

    Passage:\"\"\"
    return call_llm(prompt)
    ─────────────────────

    NOTE: The hypothesis may contain hallucinated facts.
    It is used ONLY for retrieval — never injected into the final prompt.
    """
    # Simulated hypotheses for demo
    hypotheses = {
        "transformer attention": (
            "Transformer attention fails on long sequences because the "
            "self-attention mechanism has O(n²) time and memory complexity, "
            "making it computationally infeasible beyond a few thousand tokens."
        ),
        "rag retrieval": (
            "RAG systems improve answer quality by retrieving relevant document "
            "chunks at query time using dense vector similarity, grounding the "
            "LLM in factual external context rather than parametric memory."
        ),
    }
    # Match by keyword for demo
    for key, hyp in hypotheses.items():
        if any(word in query.lower() for word in key.split()):
            print(f"\n[HyDE Generator]")
            print(f"  Query     : {query}")
            print(f"  Hypothesis: {hyp}")
            return hyp

    # Default fallback hypothesis
    fallback = f"This document provides a comprehensive explanation of {query}."
    print(f"\n[HyDE Generator] Fallback hypothesis: {fallback}")
    return fallback


# ─────────────────────────────────────────────
# 4. PROMPT BUILDER
# CRITICAL: Use original query + REAL docs, NOT the hypothesis
# ─────────────────────────────────────────────

def build_prompt(original_query: str, real_docs: List[Tuple[float, str]]) -> str:
    context = "\n".join([f"- {doc}" for _, doc in real_docs])
    return f"""Answer using ONLY the retrieved context below.

Context (real retrieved documents):
{context}

Question: {original_query}

Answer:"""


# ─────────────────────────────────────────────
# 5. SIMULATED LLM
# ─────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    print("\n[Final LLM Prompt — using REAL docs, NOT hypothesis]")
    print("─" * 60)
    print(prompt)
    print("─" * 60)
    return "[Simulated HyDE-grounded answer]"


# ─────────────────────────────────────────────
# 6. HyDE RAG PIPELINE
# ─────────────────────────────────────────────

class HyDERAG:
    def __init__(self, documents: List[str], top_k: int = 3):
        self.documents = documents
        self.top_k = top_k

    def query(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")

        # ── Standard RAG (baseline comparison) ──
        print("\n--- [BASELINE] Standard RAG Retrieval (query embedded directly) ---")
        standard_results = retrieve(user_query, self.documents, self.top_k)
        for score, doc in standard_results:
            print(f"  [{score:.3f}] {doc[:70]}...")

        # ── HyDE RAG ──
        print("\n--- [HyDE] Step 1: Generate Hypothesis ---")
        hypothesis = generate_hypothesis(user_query)

        print("\n--- [HyDE] Step 2: Retrieve using Hypothesis Embedding ---")
        hyde_results = retrieve(hypothesis, self.documents, self.top_k)
        for score, doc in hyde_results:
            print(f"  [{score:.3f}] {doc[:70]}...")

        # ── Generation: Use ORIGINAL query + REAL docs (discard hypothesis) ──
        print("\n--- [HyDE] Step 3: Generate Answer (hypothesis DISCARDED) ---")
        prompt = build_prompt(user_query, hyde_results)
        answer = call_llm(prompt)

        print(f"\nFINAL ANSWER: {answer}")
        return answer


# ─────────────────────────────────────────────
# 7. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = HyDERAG(DOCUMENTS, top_k=3)
    rag.query("Why does transformer attention fail on long sequences?")
    print("\n\n")
    rag.query("How does RAG improve retrieval quality?")