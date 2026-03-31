# adaptive_rag.py
# Adaptive RAG — Query classification → conditional retrieval strategy
# Core idea: decide WHETHER and HOW to retrieve based on query type

import numpy as np
from typing import List, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# 1. QUERY STRATEGY TYPES
# ─────────────────────────────────────────────

class QueryStrategy(Enum):
    DIRECT = "direct"           # LLM answers from parametric knowledge
    STANDARD_RAG = "standard"   # Single retrieval pass
    ITERATIVE_RAG = "iterative" # Multi-step retrieval + reasoning


# ─────────────────────────────────────────────
# 2. DOCUMENT CORPUS
# ─────────────────────────────────────────────

DOCUMENTS = [
    "Our Q3 2024 revenue was $4.2 billion, up 18% year-over-year.",
    "The refund policy allows returns within 30 days with original receipt.",
    "Our API has a rate limit of 1000 requests per minute per API key.",
    "The enterprise plan includes SSO, audit logs, and dedicated support.",
    "Competitor XYZ reported Q3 revenue of $3.1 billion in their earnings call.",
    "Our Q2 2024 revenue was $3.6 billion, up 12% year-over-year.",
    "Customer satisfaction score (CSAT) for Q3 was 4.3 out of 5.",
]


# ─────────────────────────────────────────────
# 3. QUERY CLASSIFIER
# ─────────────────────────────────────────────

class QueryClassifier:
    """
    Rule-based query classifier.
    In production: replace with fine-tuned classifier or LLM prompt.

    LLM-based classification prompt:
    ─────────────────────────────────
    prompt = f\"\"\"Classify the following query into one of:
    - DIRECT: Answerable from general knowledge (no retrieval needed)
    - STANDARD_RAG: Requires retrieving from a specific knowledge base
    - ITERATIVE_RAG: Requires multi-step reasoning across multiple sources

    Query: {query}
    Classification (one word):\"\"\"
    ─────────────────────────────────
    """

    # Signals for each category
    DIRECT_SIGNALS = [
        "what is", "define", "explain", "how does", "what are",
        "tell me about", "describe", "meaning of"
    ]
    ITERATIVE_SIGNALS = [
        "compare", "versus", "vs", "difference between",
        "analyze", "trend", "over time", "both", "all quarters"
    ]
    RAG_SIGNALS = [
        "our", "we", "company", "policy", "plan", "revenue", "customer",
        "product", "api", "enterprise", "q1", "q2", "q3", "q4", "2024"
    ]

    def classify(self, query: str) -> QueryStrategy:
        q = query.lower()

        # Multi-hop signals take priority
        if any(sig in q for sig in self.ITERATIVE_SIGNALS):
            strategy = QueryStrategy.ITERATIVE_RAG
        # Domain/company-specific signals → standard RAG
        elif any(sig in q for sig in self.RAG_SIGNALS):
            strategy = QueryStrategy.STANDARD_RAG
        # General knowledge → direct LLM
        elif any(sig in q for sig in self.DIRECT_SIGNALS):
            strategy = QueryStrategy.DIRECT
        else:
            # Default: safer to retrieve than hallucinate
            strategy = QueryStrategy.STANDARD_RAG

        print(f"\n[Classifier] Query: '{query}'")
        print(f"[Classifier] Strategy: {strategy.value.upper()}")
        return strategy


# ─────────────────────────────────────────────
# 4. RETRIEVER
# ─────────────────────────────────────────────

def simulate_embedding(text: str, dim: int = 64) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dim).astype(np.float32)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def retrieve(query: str, docs: List[str], top_k: int = 3) -> List[Tuple[float, str]]:
    q_vec = simulate_embedding(query)
    scored = [(cosine_similarity(q_vec, simulate_embedding(d)), d) for d in docs]
    return sorted(scored, reverse=True)[:top_k]


# ─────────────────────────────────────────────
# 5. STRATEGY HANDLERS
# ─────────────────────────────────────────────

def handle_direct(query: str) -> str:
    """No retrieval — LLM answers from parametric knowledge."""
    print("\n[Strategy: DIRECT] No retrieval. LLM answers from memory.")
    prompt = f"Answer this question from your general knowledge:\n\n{query}\n\nAnswer:"
    print(f"Prompt: {prompt}")
    return "[Simulated direct LLM answer — no retrieval used]"


def handle_standard_rag(query: str, docs: List[str]) -> str:
    """Single retrieval pass — standard RAG."""
    print("\n[Strategy: STANDARD RAG] Single retrieval pass.")
    retrieved = retrieve(query, docs, top_k=3)
    print("Retrieved:")
    for score, doc in retrieved:
        print(f"  [{score:.3f}] {doc}")

    context = "\n".join([f"- {doc}" for _, doc in retrieved])
    prompt = f"Answer using this context:\n{context}\n\nQuestion: {query}\nAnswer:"
    print(f"\nPrompt sent to LLM:\n{prompt}")
    return "[Simulated standard RAG answer]"


def handle_iterative_rag(query: str, docs: List[str]) -> str:
    """
    Multi-step retrieval — retrieve, reason, then retrieve again.
    Each step uses the result of the previous to refine the next query.
    """
    print("\n[Strategy: ITERATIVE RAG] Multi-step retrieval.")

    # Step 1: First retrieval pass
    print("\n  [Iteration 1] Initial retrieval...")
    results_1 = retrieve(query, docs, top_k=2)
    for score, doc in results_1:
        print(f"    [{score:.3f}] {doc}")

    # Step 2: Reason over first pass → generate refined sub-query
    # In production: LLM generates follow-up query based on first results
    sub_query = f"additional context for: {query}"
    print(f"\n  [Iteration 2] Refined sub-query: '{sub_query}'")
    results_2 = retrieve(sub_query, docs, top_k=2)
    for score, doc in results_2:
        print(f"    [{score:.3f}] {doc}")

    # Step 3: Merge all retrieved docs
    all_docs = list({doc for _, doc in results_1 + results_2})
    context = "\n".join([f"- {doc}" for doc in all_docs])
    prompt = f"Using multi-step retrieved context:\n{context}\n\nQuestion: {query}\nAnswer:"
    print(f"\n  Merged context sent to LLM:\n{prompt}")
    return "[Simulated iterative RAG answer — multi-hop reasoning]"


# ─────────────────────────────────────────────
# 6. ADAPTIVE RAG PIPELINE
# ─────────────────────────────────────────────

class AdaptiveRAG:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.classifier = QueryClassifier()

    def query(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")

        # Step 1: Classify
        strategy = self.classifier.classify(user_query)

        # Step 2: Route to appropriate strategy
        if strategy == QueryStrategy.DIRECT:
            answer = handle_direct(user_query)
        elif strategy == QueryStrategy.STANDARD_RAG:
            answer = handle_standard_rag(user_query, self.documents)
        else:
            answer = handle_iterative_rag(user_query, self.documents)

        print(f"\nFINAL ANSWER: {answer}")
        return answer


# ─────────────────────────────────────────────
# 7. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = AdaptiveRAG(DOCUMENTS)

    print("\n--- Test 1: General knowledge → DIRECT ---")
    rag.query("What is machine learning?")

    print("\n\n--- Test 2: Domain-specific → STANDARD RAG ---")
    rag.query("What is our refund policy?")

    print("\n\n--- Test 3: Multi-hop → ITERATIVE RAG ---")
    rag.query("Compare our Q3 revenue vs Q2 revenue and identify the trend.")