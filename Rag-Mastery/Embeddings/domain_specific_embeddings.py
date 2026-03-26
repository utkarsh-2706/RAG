"""
Domain-Specific Embeddings — Concept #4
=========================================
Why general models fail on specialized domains.
How fine-tuning fixes it. What embedding drift looks like.

What this file teaches:
    1. Gap between general model and domain model on technical terms
    2. How to fine-tune a sentence-transformer with (query, pos, neg) triplets
    3. Before/after comparison on domain pairs
    4. Embedding drift simulation — what happens when new terms appear
    5. Practical: how to build training pairs from your own data
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Domain pair evaluator — compare model before and after fine-tuning
# ---------------------------------------------------------------------------

def evaluate_domain_pairs(
    model: SentenceTransformer,
    pairs: List[Tuple[str, str, float]],   # (text_a, text_b, expected_similarity)
    label: str = "",
) -> None:
    """
    Score each pair and compare to expected similarity.
    expected_similarity: 1.0 = should be similar, 0.0 = should be dissimilar
    """
    print(f"\n  [{label}]")
    print(f"  {'Expected':>10}  {'Got':>8}  Gap  Pair")
    print(f"  {'-'*70}")
    for a, b, expected in pairs:
        embs = model.encode([a, b], show_progress_bar=False)
        got = cosine_sim(embs[0], embs[1])
        gap = got - expected
        flag = " <-- BAD" if abs(gap) > 0.3 else ""
        print(f"  {expected:>10.2f}  {got:>8.4f}  {gap:+.3f}  '{a[:30]}' vs '{b[:30]}'{flag}")


# ---------------------------------------------------------------------------
# Fine-tuning with MultipleNegativesRankingLoss
# ---------------------------------------------------------------------------

def build_training_pairs(domain: str = "rag") -> List[InputExample]:
    """
    Build (anchor, positive) pairs for MultipleNegativesRankingLoss.
    In production: extract these from your domain data automatically.
    Here: handcrafted pairs for RAG/ML domain.

    MultipleNegativesRankingLoss treats all other items in the batch as negatives.
    No need to specify negative pairs explicitly.
    """
    if domain == "rag":
        pairs = [
            ("How does RAG retrieve documents?",
             "RAG uses vector similarity to find relevant chunks from a knowledge base."),
            ("What is embedding drift?",
             "Embedding drift occurs when new domain terms appear that the model wasn't trained on."),
            ("Why does chunking affect retrieval quality?",
             "Chunk size determines the granularity of the embedded unit; small chunks retrieve precisely but lose context."),
            ("What is HNSW?",
             "HNSW is a graph-based algorithm for approximate nearest neighbor search in vector databases."),
            ("How does cosine similarity work?",
             "Cosine similarity measures the angle between two vectors, ignoring magnitude."),
            ("What causes hallucinations in LLMs?",
             "LLMs hallucinate when their parametric knowledge is incorrect or when no relevant context is provided."),
            ("What is a vector database?",
             "A vector database stores high-dimensional embeddings and supports fast similarity search."),
            ("How do you fine-tune an embedding model?",
             "Fine-tuning uses contrastive loss on (query, relevant_doc) pairs to align domain-specific semantics."),
        ]
    return [InputExample(texts=[a, b]) for a, b in pairs]


def fine_tune_model(
    base_model_name: str = "all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 4,
    output_path: str = "./fine_tuned_rag_model",
) -> SentenceTransformer:
    """
    Fine-tune a sentence-transformer on domain-specific pairs.

    Loss: MultipleNegativesRankingLoss
        - Takes (anchor, positive) pairs
        - All other positives in the batch serve as in-batch negatives
        - Fast, effective, needs no explicit negative mining

    Args:
        base_model_name : starting checkpoint
        epochs          : full passes over training data
        batch_size      : smaller = more updates, larger = more in-batch negatives
        output_path     : where to save the fine-tuned model

    Returns:
        fine-tuned SentenceTransformer
    """
    print(f"\n  Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    train_examples = build_training_pairs("rag")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"  Training on {len(train_examples)} pairs | epochs={epochs} | batch={batch_size}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=2,
        show_progress_bar=False,
        output_path=output_path,
    )
    print(f"  Fine-tuned model saved to: {output_path}")
    return model


# ---------------------------------------------------------------------------
# Embedding drift simulation
# ---------------------------------------------------------------------------

def simulate_embedding_drift(model: SentenceTransformer) -> None:
    """
    Show what happens when a new term appears that the model hasn't seen.

    'Embedding drift' has two meanings:
        1. New terms → model maps them to generic/incorrect vectors
        2. Your domain language evolves → old embeddings become stale

    We simulate both.
    """
    print("\n  Drift Type 1: New term the model doesn't know")
    print("  " + "-" * 55)

    # Made-up technical term — model can't know this
    new_term = "ContextForge pipeline"
    related_real = "RAG document retrieval pipeline"
    unrelated = "The weather is sunny today"

    embs = model.encode([new_term, related_real, unrelated], show_progress_bar=False)
    sim_related   = cosine_sim(embs[0], embs[1])
    sim_unrelated = cosine_sim(embs[0], embs[2])

    print(f"  New term:    '{new_term}'")
    print(f"  vs related:  {sim_related:.4f}  (should be HIGH if model understood the term)")
    print(f"  vs unrelated:{sim_unrelated:.4f}  (should be LOW)")
    gap = sim_related - sim_unrelated
    print(f"  Gap:         {gap:.4f}  {'(model distinguishes OK)' if gap > 0.2 else '(model confused - drift!)'}")

    print("\n  Drift Type 2: Same word, shifted meaning over time")
    print("  " + "-" * 55)

    # 'Transformer' meant electrical device before 2017, now means AI architecture
    old_meaning = "The electrical transformer stepped up the voltage to 220V."
    new_meaning = "The transformer model uses self-attention to encode sequences."
    query = "How does a transformer process input tokens?"

    embs = model.encode([query, old_meaning, new_meaning], show_progress_bar=False)
    sim_old = cosine_sim(embs[0], embs[1])
    sim_new = cosine_sim(embs[0], embs[2])

    print(f"  Query:      '{query}'")
    print(f"  Old meaning (electrical): {sim_old:.4f}")
    print(f"  New meaning (AI):         {sim_new:.4f}")
    print(f"  Model correctly prefers AI meaning: {sim_new > sim_old}")


# ---------------------------------------------------------------------------
# Main — experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("=" * 65)
    print("EXP 1: General model on domain-specific pairs")
    print("=" * 65)

    general_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Pairs that a domain-trained model should score well on
    domain_pairs = [
        # Technical synonyms — should be HIGH
        ("HNSW approximate nearest neighbor", "graph-based ANN indexing algorithm", 1.0),
        ("embedding drift", "model staleness from domain vocabulary shift", 1.0),
        ("MultipleNegativesRankingLoss", "contrastive training with in-batch negatives", 1.0),
        # Truly unrelated — should be LOW
        ("vector database indexing", "the Eiffel Tower is in Paris", 0.0),
        ("RAG retrieval pipeline", "photosynthesis in plants", 0.0),
    ]

    evaluate_domain_pairs(general_model, domain_pairs, "BEFORE fine-tuning (general model)")

    # -------------------------------------------------------------------
    # EXPERIMENT 2: Fine-tune on domain pairs, compare after
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 2: Fine-tune on RAG/ML domain pairs")
    print("=" * 65)

    ft_path = "./fine_tuned_rag_model"
    if os.path.exists(ft_path):
        print(f"  Loading cached fine-tuned model from {ft_path}")
        fine_tuned = SentenceTransformer(ft_path)
    else:
        fine_tuned = fine_tune_model(epochs=5, batch_size=4, output_path=ft_path)

    evaluate_domain_pairs(fine_tuned, domain_pairs, "AFTER fine-tuning (domain model)")

    # -------------------------------------------------------------------
    # EXPERIMENT 3: Embedding drift simulation
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 3: Embedding drift")
    print("=" * 65)
    simulate_embedding_drift(general_model)

    # -------------------------------------------------------------------
    # EXPERIMENT 4: How to build training pairs from your own data
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 4: Practical — generating training pairs from your corpus")
    print("=" * 65 + "\n")

    print("  Strategies to build (query, positive_doc) pairs:\n")
    strategies = [
        ("BM25 hard negatives",    "Use BM25 to find docs that look relevant (keyword match) but aren't → hard negatives"),
        ("User click data",         "Query + clicked doc = positive pair. Non-clicked = negative."),
        ("LLM-generated queries",   "For each doc chunk, use an LLM to generate 3 questions it answers."),
        ("Paraphrase augmentation", "Use back-translation to create paraphrases as positive pairs."),
        ("Section headers",         "Header = query proxy. Body under header = positive doc."),
    ]
    for name, desc in strategies:
        print(f"  [{name}]")
        print(f"    {desc}\n")

    print("  Most practical for RAG: LLM-generated queries per chunk.")
    print("  Feed: prompt = 'Write 3 questions this passage answers: {chunk}'")
    print("  Each (question, chunk) pair becomes a training example.")
