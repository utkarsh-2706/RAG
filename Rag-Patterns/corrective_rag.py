# corrective_rag.py
# Corrective RAG (CRAG) — Validate retrieved docs before generation
# Core idea: relevance gate → corrective action → grounded generation

import numpy as np
from typing import List, Tuple
from enum import Enum
from dataclasses import dataclass


# ─────────────────────────────────────────────
# 1. RELEVANCE CLASSIFICATION
# ─────────────────────────────────────────────

class RelevanceLabel(Enum):
    CORRECT = "correct"       # Score >= 0.75 → use directly
    AMBIGUOUS = "ambiguous"   # Score 0.45–0.74 → supplement with web
    WRONG = "wrong"           # Score < 0.45 → discard, rewrite, re-retrieve


@dataclass
class ScoredDoc:
    content: str
    score: float
    label: RelevanceLabel
    source: str = "vector_store"


# ─────────────────────────────────────────────
# 2. DOCUMENT CORPUS
# ─────────────────────────────────────────────

DOCUMENTS = [
    "Transformer attention scales quadratically with sequence length.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "FAISS supports both exact and approximate nearest neighbor search.",
    "Python was created by Guido van Rossum in 1991.",
    "Our enterprise plan includes SSO, audit logging, and SLA guarantees.",
    "RAG systems reduce hallucination by grounding LLMs in retrieved context.",
    "The mitochondria is the powerhouse of the cell.",
]

# Simulated web search results (in production: call Tavily/SerpAPI)
WEB_RESULTS = [
    "Recent research shows CRAG improves RAG answer faithfulness by 15-20%.",
    "Corrective RAG introduces a retrieval evaluator trained on relevance judgments.",
    "Web-augmented RAG combines vector retrieval with live search for freshness.",
]


# ─────────────────────────────────────────────
# 3. EMBEDDING + RETRIEVAL
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
# 4. RELEVANCE EVALUATOR
# ─────────────────────────────────────────────

def evaluate_relevance(query: str, doc: str, raw_score: float) -> ScoredDoc:
    """
    Scores how relevant a retrieved document is to the query.

    In production, use a cross-encoder or LLM evaluator:
    ─────────────────────────────────────────────────────
    # Cross-encoder (recommended):
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    score = model.predict([[query, doc]])[0]

    # LLM evaluator:
    prompt = f\"\"\"Is the following document relevant to answering the query?
    Query: {query}
    Document: {doc}
    Answer with: CORRECT, AMBIGUOUS, or WRONG\"\"\"
    label = call_llm(prompt).strip()
    ─────────────────────────────────────────────────────
    """
    # Using cosine similarity as a proxy for relevance (demo only)
    if raw_score >= 0.75:
        label = RelevanceLabel.CORRECT
    elif raw_score >= 0.45:
        label = RelevanceLabel.AMBIGUOUS
    else:
        label = RelevanceLabel.WRONG

    return ScoredDoc(content=doc, score=raw_score, label=label)


# ─────────────────────────────────────────────
# 5. KNOWLEDGE REFINER
# Strips low-quality sentences from CORRECT but noisy docs
# ─────────────────────────────────────────────

def refine_document(doc: str, query: str) -> str:
    """
    Remove sentences from a doc that are irrelevant to the query.

    In production: sentence-level scoring using a cross-encoder.
    Here: simulated by returning the full doc (production would filter).
    """
    # Simulated: in practice, split into sentences and score each
    print(f"    [Refiner] Refining doc (keeping query-relevant sentences)...")
    return doc  # Return full doc in simulation


# ─────────────────────────────────────────────
# 6. QUERY REWRITER (for WRONG case)
# ─────────────────────────────────────────────

def rewrite_query(original_query: str) -> str:
    """
    Rewrite the query to improve retrieval on retry.

    In production:
    ─────────────────────────────────────────────────
    prompt = f\"\"\"The following query failed to retrieve relevant documents.
    Rewrite it to be more specific and likely to match relevant content.

    Original query: {original_query}
    Rewritten query:\"\"\"
    return call_llm(prompt)
    ─────────────────────────────────────────────────
    """
    rewritten = f"detailed explanation of {original_query}"
    print(f"    [Query Rewriter] '{original_query}' → '{rewritten}'")
    return rewritten


# ─────────────────────────────────────────────
# 7. WEB SEARCH (simulated)
# ─────────────────────────────────────────────

def web_search(query: str, top_k: int = 2) -> List[str]:
    """
    In production: call Tavily, SerpAPI, or Bing Search API.
    """
    print(f"    [Web Search] Searching for: '{query}'")
    # Simulated: return pseudo-relevant web results
    q_vec = simulate_embedding(query)
    scored = [(cosine_similarity(q_vec, simulate_embedding(r)), r) for r in WEB_RESULTS]
    top = sorted(scored, reverse=True)[:top_k]
    results = [doc for _, doc in top]
    for r in results:
        print(f"      [WEB] {r}")
    return results


# ─────────────────────────────────────────────
# 8. CRAG DECISION LOGIC
# ─────────────────────────────────────────────

def crag_decision(query: str, scored_docs: List[ScoredDoc]) -> List[str]:
    """
    Core CRAG logic: decide what to do based on evaluation results.

    Returns a list of final context strings to pass to the LLM.
    """
    correct = [d for d in scored_docs if d.label == RelevanceLabel.CORRECT]
    ambiguous = [d for d in scored_docs if d.label == RelevanceLabel.AMBIGUOUS]
    wrong = [d for d in scored_docs if d.label == RelevanceLabel.WRONG]

    print(f"\n[CRAG Evaluator] Results: {len(correct)} CORRECT | {len(ambiguous)} AMBIGUOUS | {len(wrong)} WRONG")

    final_context = []

    if correct:
        # Case 1: At least some correct docs — refine and use them
        print("\n[CRAG Action: CORRECT] Refining and using correct documents...")
        for doc in correct:
            refined = refine_document(doc.content, query)
            final_context.append(f"[VECTOR | {doc.score:.2f}] {refined}")

        # If also ambiguous, supplement with web
        if ambiguous:
            print("[CRAG Action: SUPPLEMENT] Ambiguous docs found — adding web results...")
            web = web_search(query, top_k=1)
            final_context.extend([f"[WEB] {r}" for r in web])

    elif ambiguous:
        # Case 2: Only ambiguous — supplement heavily with web
        print("\n[CRAG Action: AMBIGUOUS] Low confidence — supplementing with web search...")
        for doc in ambiguous:
            final_context.append(f"[VECTOR | {doc.score:.2f}] {doc.content}")
        web = web_search(query, top_k=2)
        final_context.extend([f"[WEB] {r}" for r in web])

    else:
        # Case 3: All wrong — discard, rewrite, re-retrieve from web
        print("\n[CRAG Action: WRONG] All documents irrelevant — rewriting query and searching web...")
        rewritten = rewrite_query(query)
        web = web_search(rewritten, top_k=3)
        final_context.extend([f"[WEB] {r}" for r in web])

    return final_context


# ─────────────────────────────────────────────
# 9. PROMPT BUILDER + LLM
# ─────────────────────────────────────────────

def build_prompt(query: str, context: List[str]) -> str:
    ctx = "\n".join(context)
    return f"""Answer using the validated context below.
Each source is labeled [VECTOR] or [WEB].

Context:
{ctx}

Question: {query}
Answer:"""

def call_llm(prompt: str) -> str:
    print("\n[LLM Final Prompt]")
    print("─" * 60)
    print(prompt)
    print("─" * 60)
    return "[Simulated CRAG-validated answer]"


# ─────────────────────────────────────────────
# 10. CORRECTIVE RAG PIPELINE
# ─────────────────────────────────────────────

class CorrectiveRAG:
    def __init__(self, documents: List[str], top_k: int = 3):
        self.documents = documents
        self.top_k = top_k

    def query(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")

        # Step 1: Retrieve
        print("\n[Step 1] Retrieving documents...")
        retrieved = retrieve(user_query, self.documents, self.top_k)

        # Step 2: Evaluate each retrieved doc
        print("\n[Step 2] Evaluating relevance...")
        scored_docs = []
        for score, doc in retrieved:
            scored = evaluate_relevance(user_query, doc, score)
            scored_docs.append(scored)
            print(f"  [{scored.label.value.upper():9s} | {score:.3f}] {doc[:60]}...")

        # Step 3: CRAG decision → corrective actions
        print("\n[Step 3] CRAG decision logic...")
        final_context = crag_decision(user_query, scored_docs)

        # Step 4: Generate
        print("\n[Step 4] Generating answer with validated context...")
        prompt = build_prompt(user_query, final_context)
        answer = call_llm(prompt)

        print(f"\nFINAL ANSWER: {answer}")
        return answer


# ─────────────────────────────────────────────
# 11. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = CorrectiveRAG(DOCUMENTS, top_k=3)

    print("\n--- Test 1: Good retrieval → CORRECT path ---")
    rag.query("What is FAISS used for?")

    print("\n\n--- Test 2: Mixed retrieval → AMBIGUOUS path + web supplement ---")
    rag.query("Latest improvements in corrective RAG systems?")

    print("\n\n--- Test 3: Poor retrieval → WRONG path → rewrite + web ---")
    rag.query("Quantum computing error correction techniques?")

