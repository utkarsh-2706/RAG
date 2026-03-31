# self_rag.py
# Self-RAG — LLM reflects on retrieval need, relevance, and output faithfulness
# Core idea: reflection tokens drive self-evaluation and candidate selection

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


# ─────────────────────────────────────────────
# 1. REFLECTION TOKEN TYPES
# ─────────────────────────────────────────────

class RetrieveDecision(Enum):
    YES = "yes"        # Retrieval needed
    NO = "no"          # LLM can answer directly
    CONTINUE = "continue"  # Mid-generation: retrieve more

class IsRel(Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"

class IsSup(Enum):
    FULLY = "fully"        # Score: 1.0
    PARTIALLY = "partially"  # Score: 0.5
    NO = "no"              # Score: 0.0

class IsUse(Enum):
    HIGH = 5
    MEDIUM = 3
    LOW = 1


# ─────────────────────────────────────────────
# 2. REFLECTION TOKEN SCORES
# ─────────────────────────────────────────────

ISSUP_SCORES = {IsSup.FULLY: 1.0, IsSup.PARTIALLY: 0.5, IsSup.NO: 0.0}
ISUSE_SCORES = {IsUse.HIGH: 1.0, IsUse.MEDIUM: 0.6, IsUse.LOW: 0.2}
ISREL_SCORES = {IsRel.RELEVANT: 1.0, IsRel.IRRELEVANT: 0.0}

# Weights for composite score (faithfulness weighted highest)
WEIGHTS = {"isrel": 0.2, "issup": 0.5, "isuse": 0.3}


# ─────────────────────────────────────────────
# 3. CORPUS
# ─────────────────────────────────────────────

DOCUMENTS = [
    "Self-RAG trains the LLM to generate reflection tokens that evaluate retrieval and generation quality.",
    "Reflection tokens like [ISREL], [ISSUP], [ISUSE] allow the model to critique its own outputs.",
    "Hallucination in LLMs occurs when generated text is not grounded in provided context.",
    "CRAG uses an external evaluator while Self-RAG uses the LLM itself for self-critique.",
    "Fine-tuning is required to make LLMs reliably generate reflection tokens.",
    "RAG systems reduce hallucination by grounding generation in retrieved documents.",
    "The capital of Australia is Canberra, not Sydney as commonly assumed.",
]


# ─────────────────────────────────────────────
# 4. SIMULATED EMBEDDING + RETRIEVAL
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
# 5. SIMULATED REFLECTION TOKEN GENERATORS
# In production: fine-tuned LLM generates these inline
# ─────────────────────────────────────────────

def predict_retrieve_token(query: str) -> RetrieveDecision:
    """
    Should we retrieve for this query?
    In production: LLM generates [Retrieve] token as part of its output.
    """
    general_knowledge_signals = ["what is", "define", "who invented", "explain"]
    if any(sig in query.lower() for sig in general_knowledge_signals):
        # Ambiguous — might know it, might not → default retrieve
        decision = RetrieveDecision.YES
    else:
        decision = RetrieveDecision.YES  # Default: always retrieve for safety
    print(f"\n[Retrieve Token] → {decision.value.upper()}")
    return decision


def predict_isrel(query: str, doc: str, retrieval_score: float) -> IsRel:
    """
    Is this document relevant to the query?
    Simulated via score threshold. Production: LLM generates [ISREL] token.
    """
    label = IsRel.RELEVANT if retrieval_score > 0.5 else IsRel.IRRELEVANT
    return label


def predict_issup(generated_passage: str, doc: str) -> IsSup:
    """
    Is the generated passage supported by the document?
    Simulated via keyword overlap. Production: LLM generates [ISSUP] token.
    """
    gen_words = set(generated_passage.lower().split())
    doc_words = set(doc.lower().split())
    overlap = len(gen_words & doc_words) / (len(gen_words) + 1e-9)

    if overlap > 0.3:
        return IsSup.FULLY
    elif overlap > 0.1:
        return IsSup.PARTIALLY
    else:
        return IsSup.NO


def predict_isuse(passage: str) -> IsUse:
    """
    Is the generated passage useful?
    Simulated: longer, structured passages score higher.
    Production: LLM generates [ISUSE] score token.
    """
    if len(passage.split()) > 20:
        return IsUse.HIGH
    elif len(passage.split()) > 10:
        return IsUse.MEDIUM
    else:
        return IsUse.LOW


# ─────────────────────────────────────────────
# 6. CANDIDATE GENERATION (per retrieved doc)
# ─────────────────────────────────────────────

@dataclass
class Candidate:
    doc: str
    passage: str
    isrel: IsRel
    issup: IsSup
    isuse: IsUse
    composite_score: float

def generate_candidate(query: str, doc: str, retrieval_score: float) -> Candidate:
    """
    For each retrieved doc, generate a candidate response + evaluate it.
    In production: LLM generates passage + reflection tokens in one forward pass.
    """
    # Simulate passage generation
    passage = f"Based on evidence: '{doc[:60]}...' — the answer to '{query}' relates to {doc.split()[0]} and {doc.split()[-1]}."

    # Generate reflection tokens
    isrel = predict_isrel(query, doc, retrieval_score)
    issup = predict_issup(passage, doc)
    isuse = predict_isuse(passage)

    # Compute composite score
    score = (
        WEIGHTS["isrel"] * ISREL_SCORES[isrel] +
        WEIGHTS["issup"] * ISSUP_SCORES[issup] +
        WEIGHTS["isuse"] * ISUSE_SCORES[isuse]
    )

    return Candidate(
        doc=doc,
        passage=passage,
        isrel=isrel,
        issup=issup,
        isuse=isuse,
        composite_score=score
    )


# ─────────────────────────────────────────────
# 7. SELF-RAG PIPELINE
# ─────────────────────────────────────────────

class SelfRAG:
    def __init__(self, documents: List[str], top_k: int = 3, max_iterations: int = 2):
        self.documents = documents
        self.top_k = top_k
        self.max_iterations = max_iterations

    def query(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")

        # Step 1: Decide whether to retrieve
        retrieve_decision = predict_retrieve_token(user_query)

        if retrieve_decision == RetrieveDecision.NO:
            print("[Self-RAG] Answering directly without retrieval.")
            return "[Direct LLM answer — no retrieval needed]"

        best_candidate = None
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Self-RAG Iteration {iteration} ---")

            # Step 2: Retrieve
            retrieved = retrieve(user_query, self.documents, self.top_k)
            print(f"Retrieved {len(retrieved)} documents.")

            # Step 3: Generate candidate + reflect for each doc
            candidates = []
            for score, doc in retrieved:
                candidate = generate_candidate(user_query, doc, score)
                candidates.append(candidate)
                print(f"\n  Doc: {doc[:55]}...")
                print(f"  [ISREL={candidate.isrel.value}] [ISSUP={candidate.issup.value}] "
                      f"[ISUSE={candidate.isuse.name}] → Score: {candidate.composite_score:.3f}")

            # Step 4: Select best candidate
            best = max(candidates, key=lambda c: c.composite_score)

            # Step 5: Check stopping condition
            # Stop if: fully supported AND high usefulness
            if best.issup == IsSup.FULLY and best.isuse == IsUse.HIGH:
                print(f"\n[Self-RAG] Stopping: best candidate is FULLY supported and HIGH usefulness.")
                best_candidate = best
                break
            elif iteration == self.max_iterations:
                print(f"\n[Self-RAG] Max iterations reached. Using best available candidate.")
                best_candidate = best
            else:
                print(f"\n[Self-RAG] Best score {best.composite_score:.3f} insufficient — iterating...")

        print(f"\n[Self-RAG] Final Selection:")
        print(f"  Passage : {best_candidate.passage}")
        print(f"  ISSUP   : {best_candidate.issup.value}")
        print(f"  ISUSE   : {best_candidate.isuse.name}")
        print(f"  Score   : {best_candidate.composite_score:.3f}")
        return best_candidate.passage


# ─────────────────────────────────────────────
# 8. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = SelfRAG(DOCUMENTS, top_k=3, max_iterations=2)

    print("\n--- Test 1: Factual query needing retrieval ---")
    rag.query("How does Self-RAG use reflection tokens?")

    print("\n\n--- Test 2: Hallucination-prone query ---")
    rag.query("What is the capital of Australia?")