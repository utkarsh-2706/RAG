# Retrieval Techniques in RAG

> **Learning Path:** Each technique builds on the previous. Do NOT skip ahead.
> After each section, run the code, tweak parameters, and observe before moving on.

---

## What Is Retrieval, Really?

Before diving into techniques, internalize this:

> **Retrieval is a ranking problem, not a search problem.**

Every single chunk in your corpus gets a score. Retrieval decides *which scores are high enough* and *how many to return*. The LLM then generates an answer based on ONLY what retrieval returned — garbage in, garbage out.

The two fundamental knobs in retrieval:
- **What to score against** (embedding similarity, keyword overlap, metadata match)
- **How many to return / what threshold to use** (K value, similarity cutoff)

---

## 1. Top-K Retrieval

### Intuition

The simplest retrieval strategy: embed everything, embed the query, compute similarity, return the K closest chunks. This is the *baseline* every other technique improves upon.

Think of it like a leaderboard — every chunk gets a score, you take the top K winners regardless of how good or bad their absolute scores are.

### How It Works

```
Query (text)
    │
    ▼
Query Embedding (dense vector, e.g. 384-dim)
    │
    ▼
Cosine Similarity against ALL chunk embeddings
    │  score(chunk_i) = cos(query_vec, chunk_i_vec)
    ▼
Sort all chunks by score descending
    │
    ▼
Return top K chunks  ← these go into the LLM's context window
```

**Cosine Similarity** measures the *angle* between two vectors — not their magnitude.
- Score = 1.0 → identical direction (very similar)
- Score = 0.0 → orthogonal (unrelated)
- Score < 0.5 → likely irrelevant

### Ranking Behavior

- Chunks semantically closest to the query rise to the top
- Topic drift increases as K grows — rank #1 is nearly always relevant, rank #10 may not be
- **Top-K always returns exactly K results**, even if the corpus is entirely unrelated to the query

### Tradeoffs (Interview Critical)

| Aspect | Small K (e.g. 3) | Large K (e.g. 10) |
|---|---|---|
| Precision | High — returned chunks are very relevant | Low — irrelevant chunks creep in |
| Recall | Low — might miss the right chunk | High — better chance the answer is included |
| Context window | Efficient | Expensive / noisy |
| Failure mode | Answer chunk not retrieved | LLM confused by irrelevant context |

**When Top-K works well:**
- Corpus is clean, well-chunked, and topically diverse
- Query is specific and focused
- K is tuned to the embedding model's capability

**When Top-K fails:**
- Corpus has many near-duplicate chunks (returns redundant results)
- Query is vague or short (embedding captures little signal)
- Completely irrelevant query — Top-K still returns K chunks with no warning
- Long, noisy documents where a chunk mixes many topics

**Key failure insight:** Top-K has NO concept of "nothing is relevant." It will always return K chunks. This leads to the LLM hallucinating based on weakly-related content — solved by Similarity Threshold Filtering.

### Key Takeaways

- Always precompute and cache chunk embeddings — embedding at query time is wasteful
- Normalize embeddings so cosine similarity = dot product (faster computation)
- K is a hyperparameter: start at 3–5, tune based on your chunk size and corpus
- The ranking itself is more important than the absolute score values
- A score of 0.75 in one corpus might mean something different than 0.75 in another

### Implementation

See [top_k_retrieval.py](top_k_retrieval.py)

Run it:
```bash
python top_k_retrieval.py
```

**Experiments to run:**
1. Try K=1, 3, 5, 8 on the same query — watch precision degrade
2. Try a completely unrelated query — observe Top-K still returns results
3. Check the score values — notice the gap between rank #1 and rank #5

---

## 2. Similarity Threshold Filtering

*(Coming after you experiment with Top-K)*

---

## 3. MMR (Max Marginal Relevance)

*(Coming after Similarity Threshold)*

---

## 4. Metadata Filtering

*(Coming after MMR)*

---

## 5. Hybrid Search (BM25 + Dense)

*(Coming after Metadata Filtering)*

---

## Key Tradeoffs Summary

*(Will be filled as we go)*

- Large K → higher recall, lower precision
- Small K → higher precision, lower recall
- Relevance vs Diversity (MMR)
- Dense vs Sparse retrieval differences

---

## My Observations

*(Fill this after each experiment)*

### Top-K Observations
- [ ] What happened when K=1 vs K=8?
- [ ] What scores did irrelevant chunks get?
- [ ] Did you see any duplicate/similar chunks in the top results?
