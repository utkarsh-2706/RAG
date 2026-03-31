# branched_rag.py
# Branched RAG — Multi-retriever with query routing and result merging
# Demonstrates: router + parallel retrieval + merge + unified generation

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


# ─────────────────────────────────────────────
# 1. RETRIEVER TYPES
# ─────────────────────────────────────────────

class RetrieverType(Enum):
    VECTOR = "vector"           # Unstructured docs — semantic search
    STRUCTURED = "structured"   # SQL / tabular data — keyword/exact match
    WEB = "web"                 # Live web / APIs (simulated)


# ─────────────────────────────────────────────
# 2. DATA SOURCES
# ─────────────────────────────────────────────

# Source A: Unstructured document corpus
DOCS_CORPUS = [
    "RAG combines retrieval with generation to ground LLM responses.",
    "Vector embeddings encode semantic meaning for similarity search.",
    "FAISS enables fast approximate nearest neighbor search at scale.",
    "Chunking strategy directly impacts RAG retrieval quality.",
    "Prompt injection is a security risk in RAG systems.",
]

# Source B: Structured / tabular data (simulated as key-value store)
STRUCTURED_DB = {
    "revenue": "Q3 2024 revenue was $4.2 billion, up 18% YoY.",
    "employees": "The company has 12,400 employees as of 2024.",
    "product_count": "The catalog contains 8,200 active SKUs.",
    "latency": "Average API latency is 240ms at p99.",
    "uptime": "System uptime SLA is 99.95% over the last 12 months.",
}

# Source C: Web / live data (simulated)
WEB_RESULTS = [
    "Latest AI research shows multimodal RAG outperforms text-only by 23%.",
    "OpenAI released GPT-4o with improved instruction following in 2024.",
    "New vector DB benchmarks rank Qdrant first in recall at 1M vectors.",
]


# ─────────────────────────────────────────────
# 3. QUERY ROUTER
# ─────────────────────────────────────────────

def route_query(query: str) -> List[RetrieverType]:
    """
    Rule-based router — classifies which retrievers to invoke.

    In production, replace with:
    ─────────────────────────────
    prompt = f'''Classify which retriever(s) should handle this query.
    Options: vector, structured, web (can choose multiple).
    Return JSON: {{"retrievers": ["vector", "structured"]}}

    Query: {query}'''
    response = call_llm(prompt)
    return parse_json(response)["retrievers"]
    ─────────────────────────────
    """
    query_lower = query.lower()
    retrievers = []

    # Structured data signals
    structured_keywords = ["revenue", "employee", "count", "how many", "metrics",
                           "latency", "uptime", "number", "stats", "data"]
    if any(kw in query_lower for kw in structured_keywords):
        retrievers.append(RetrieverType.STRUCTURED)

    # Live/web data signals
    web_keywords = ["latest", "recent", "current", "2024", "news", "today", "now"]
    if any(kw in query_lower for kw in web_keywords):
        retrievers.append(RetrieverType.WEB)

    # Default: always include vector for semantic/conceptual queries
    if not retrievers or any(kw in query_lower for kw in ["what", "how", "explain", "why", "rag"]):
        retrievers.append(RetrieverType.VECTOR)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for r in retrievers:
        if r not in seen:
            seen.add(r)
            result.append(r)

    print(f"\n[Router] Query: '{query}'")
    print(f"[Router] Dispatching to: {[r.value for r in result]}")
    return result


# ─────────────────────────────────────────────
# 4. INDIVIDUAL RETRIEVERS
# ─────────────────────────────────────────────

def simulate_embedding(text: str, dim: int = 64) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dim).astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@dataclass
class RetrievedChunk:
    source: str       # Which retriever produced this
    content: str
    score: float


def vector_retriever(query: str, top_k: int = 2) -> List[RetrievedChunk]:
    """Dense semantic search over unstructured docs."""
    q_vec = simulate_embedding(query)
    scored = []
    for doc in DOCS_CORPUS:
        doc_vec = simulate_embedding(doc)
        score = cosine_similarity(q_vec, doc_vec)
        scored.append(RetrievedChunk(source="vector_store", content=doc, score=score))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def structured_retriever(query: str, top_k: int = 2) -> List[RetrievedChunk]:
    """Keyword match over structured DB (simulates SQL / metadata filter)."""
    query_lower = query.lower()
    results = []
    for key, value in STRUCTURED_DB.items():
        if key in query_lower or any(word in query_lower for word in key.split("_")):
            results.append(RetrievedChunk(
                source="structured_db",
                content=value,
                score=1.0  # Exact match = highest score
            ))
    # If no direct match, return most relevant by key overlap
    if not results:
        for key, value in list(STRUCTURED_DB.items())[:top_k]:
            results.append(RetrievedChunk(source="structured_db", content=value, score=0.5))
    return results[:top_k]


def web_retriever(query: str, top_k: int = 2) -> List[RetrievedChunk]:
    """Simulated live web/API retrieval."""
    # In production: call Tavily, SerpAPI, or a live search API
    q_vec = simulate_embedding(query)
    scored = []
    for result in WEB_RESULTS:
        r_vec = simulate_embedding(result)
        score = cosine_similarity(q_vec, r_vec)
        scored.append(RetrievedChunk(source="web_search", content=result, score=score))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────
# 5. RESULT MERGER + RE-RANKER
# ─────────────────────────────────────────────

def merge_and_rank(results: Dict[RetrieverType, List[RetrievedChunk]]) -> List[RetrievedChunk]:
    """
    Merge results from multiple retrievers.

    Strategies:
    - Simple: concatenate all, sort by score
    - Advanced: Reciprocal Rank Fusion (RRF) — normalizes ranks across retrievers
      RRF score = sum(1 / (k + rank_i)) for each retriever, k=60 is standard

    We use simple merge here. RRF is production-preferred.
    """
    all_chunks = []
    for retriever_type, chunks in results.items():
        all_chunks.extend(chunks)

    # Deduplicate by content
    seen_content = set()
    deduped = []
    for chunk in all_chunks:
        if chunk.content not in seen_content:
            seen_content.add(chunk.content)
            deduped.append(chunk)

    # Sort by score descending
    deduped.sort(key=lambda x: x.score, reverse=True)
    return deduped


# ─────────────────────────────────────────────
# 6. PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(query: str, merged_chunks: List[RetrievedChunk]) -> str:
    context_lines = []
    for chunk in merged_chunks:
        context_lines.append(f"[{chunk.source.upper()} | score={chunk.score:.3f}]\n  {chunk.content}")
    context = "\n\n".join(context_lines)

    return f"""You are a knowledgeable assistant with access to multiple data sources.
Answer the question using the retrieved context below.
Note which source each piece of information came from.

=== Multi-Source Retrieved Context ===
{context}

=== Question ===
{query}

Answer:"""


# ─────────────────────────────────────────────
# 7. SIMULATED LLM
# ─────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    print("\n[LLM Prompt]")
    print("─" * 60)
    print(prompt)
    print("─" * 60)
    return "[Simulated multi-source grounded answer]"


# ─────────────────────────────────────────────
# 8. BRANCHED RAG PIPELINE
# ─────────────────────────────────────────────

RETRIEVER_MAP = {
    RetrieverType.VECTOR: vector_retriever,
    RetrieverType.STRUCTURED: structured_retriever,
    RetrieverType.WEB: web_retriever,
}

class BranchedRAG:
    def __init__(self, top_k: int = 2):
        self.top_k = top_k

    def query(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")

        # Step 1: Route
        selected_retrievers = route_query(user_query)

        # Step 2: Retrieve in parallel (sequential here for clarity)
        all_results: Dict[RetrieverType, List[RetrievedChunk]] = {}
        for retriever_type in selected_retrievers:
            retriever_fn = RETRIEVER_MAP[retriever_type]
            chunks = retriever_fn(user_query, self.top_k)
            all_results[retriever_type] = chunks
            print(f"\n[{retriever_type.value.upper()} Retriever] — {len(chunks)} chunks")
            for c in chunks:
                print(f"  [{c.score:.3f}] {c.content[:80]}...")

        # Step 3: Merge + Re-rank
        merged = merge_and_rank(all_results)
        print(f"\n[Merger] Final ranked chunks: {len(merged)}")

        # Step 4: Prompt + Generate
        prompt = build_prompt(user_query, merged)
        answer = call_llm(prompt)

        print(f"\nFINAL ANSWER: {answer}")
        return answer


# ─────────────────────────────────────────────
# 9. DEMO — Mixed intent queries
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = BranchedRAG(top_k=2)

    print("\n--- Test 1: Conceptual query → Vector only ---")
    rag.query("What is RAG and how does it work?")

    print("\n\n--- Test 2: Metrics query → Structured only ---")
    rag.query("What is the revenue and employee count?")

    print("\n\n--- Test 3: Mixed intent → Multiple retrievers ---")
    rag.query("What are the latest RAG techniques and our current API latency metrics?")
