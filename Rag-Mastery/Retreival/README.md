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

### Think About

- **What happens if K is too large?**
  More chunks enter the LLM's context. The answer might be in there, but so is noise. The LLM struggles to focus — precision drops, cost goes up, and irrelevant chunks can override correct ones.

- **Why do we get duplicate or similar chunks?**
  If the corpus has near-identical passages (same info chunked differently, or repeated sections), their embeddings will be very close together. Top-K will rank them all high — returning redundant content that wastes context window space without adding new information.

- **When would a different technique outperform Top-K?**
  When results are redundant → MMR. When scores are too low to be trusted → Threshold filtering. When you need exact keyword match → Hybrid search.

- **What does a "bad" Top-K result look like?**
  K=8 on a specific query where chunks 1–3 are perfect but chunks 4–8 are loosely related tangents. The LLM gets confused and blends correct + incorrect information.

---

## 2. Similarity Threshold Filtering

### Intuition

Top-K's fatal flaw: it **always** returns K chunks, even if nothing in the corpus is relevant to the query. The system has no concept of "I don't know."

Threshold filtering adds a **minimum score gate** — a chunk must earn a similarity score ≥ threshold to be returned at all. If nothing clears the bar, retrieval returns zero chunks. That's the correct, honest behavior.

Think of it like a job interview cutoff — candidates below the minimum score don't get called back, regardless of how many slots are open.

### How It Works

```
Query → Embedding → Cosine Similarity vs ALL chunks
    │
    ▼
Filter: keep only chunks where score >= threshold
    │
    ▼
Sort surviving chunks by score DESC
    │
    ▼
Return top min(K, survivors) chunks
    ← can return 0 chunks — that is CORRECT for irrelevant queries
```

Two parameters now work **together**:
- **threshold** — the minimum score gate (eliminates irrelevant chunks entirely)
- **K** — upper bound on how many to return (no longer a guarantee)

### Ranking Behavior

| Scenario | Top-K | Threshold Filtering |
|---|---|---|
| Relevant query | Returns K chunks | Returns K chunks (scores pass gate) |
| Vague query | Returns K with score drop-off | Returns fewer — rejects weak ones |
| Irrelevant query | Returns K garbage chunks | Returns 0 — correct refusal |
| Duplicate corpus | Redundant chunks in top K | Duplicates still pass (not solved here) |

Key shift: retrieval can now return **fewer than K results, including zero**. This enables honest RAG — the system can say "I don't have relevant information" rather than hallucinating.

### Tradeoffs (Interview Critical)

**When it works well:**
- Corpus topics are distinct — scores cluster clearly into "relevant" vs "not relevant"
- You want the system to abstain rather than hallucinate on irrelevant queries
- Embedding model is well-calibrated to your domain

**When it fails:**
- Threshold too high → rejects genuinely relevant chunks (false negatives — answer is there but blocked)
- Threshold too low → same as plain Top-K (threshold has no real effect)
- Short or vague queries produce lower scores across the board — valid chunks may fall below threshold
- **Score distributions are NOT portable across embedding models.** A threshold of 0.6 on `all-MiniLM` is not the same as 0.6 on `text-embedding-3-large`. Always calibrate per model.

**The calibration problem (critical for interviews):**
> There is no universal "right" threshold. You must inspect your score distribution empirically — look for the natural gap between relevant and irrelevant chunk scores, and set the threshold there.

### Key Takeaways

- Threshold filtering is **not a replacement** for Top-K — it's a gate that sits in front of it
- Always inspect the full score distribution before setting a threshold (see `inspect_score_distribution()`)
- The natural score "gap" between relevant and irrelevant chunks is where your threshold belongs
- A well-set threshold is one of the cheapest, highest-impact improvements to retrieval quality
- In production, monitor how often retrieval returns 0 results — too frequent means threshold is too aggressive

### Implementation

See [similarity_threshold.py](similarity_threshold.py)

Run it:
```bash
python similarity_threshold.py
```

**Experiments to run:**
1. Same relevant query — try threshold=0.2, 0.4, 0.6, 0.75 — watch chunks get eliminated
2. Irrelevant query — compare Top-K output vs Threshold output side by side
3. Run `inspect_score_distribution()` — find the natural gap and set threshold there
4. Set threshold=0.85 on a relevant query — observe false negatives (answer blocked)

### Think About

- **What is the right threshold — and how do you find it?**
  There is no universal answer. Run `inspect_score_distribution()` on representative queries from your corpus. Find where scores naturally drop off — the gap between the "good" cluster and the "noise" cluster is your threshold. Typical ranges: 0.3–0.5 (loose), 0.5–0.65 (moderate), 0.65+ (strict).

- **What happens when threshold is too high?**
  The system starts returning 0 results for legitimate queries. The LLM has no context and will either refuse to answer or hallucinate entirely. This is the "false negative" failure — the answer was in the corpus but the threshold blocked it.

- **Why can't you copy a threshold from one project to another?**
  Cosine similarity scores are relative to the embedding model's geometry. Different models place vectors differently in space — a score of 0.6 means something completely different between models. Always re-calibrate for each new model or domain.

- **Does threshold fix the duplicate chunk problem?**
  No. If two near-identical chunks both score 0.8, both clear the threshold and both get returned. Threshold only removes irrelevant chunks — it does nothing about redundancy. That's what MMR is for.

---

## 3. MMR (Max Marginal Relevance)

### Intuition

Top-K and Threshold give you the K most *relevant* chunks. But what if the top 5 chunks all say the same thing in slightly different words? Your context window fills with redundancy — and the one chunk that covers a different angle of the answer never makes it in.

> MMR asks: *"What is most relevant to the query AND most different from what I've already selected?"*

Each selection changes what the next selection should be. It's a greedy iterative process — not a one-shot ranking.

### How It Works

```
MMR_score(chunk) = λ · sim(chunk, query)
                 - (1-λ) · max_sim(chunk, already_selected)
```

- `sim(chunk, query)` — relevance to the query (maximize)
- `max_sim(chunk, already_selected)` — similarity to any already-picked chunk (minimize)
- `λ` — the diversity dial: 1.0 = pure Top-K, 0.0 = pure diversity

**Greedy step-by-step:**
```
Step 1: Pick chunk with highest sim(chunk, query)  → same as Top-K rank #1
Step 2: For each remaining chunk, compute MMR score
        Pick the best one (most relevant AND most different from selected)
Step 3: Repeat until K chunks selected
```

### Ranking Behavior

| λ value | Behavior |
|---|---|
| λ = 1.0 | Identical to Top-K |
| λ = 0.7 | Mostly relevant, gentle diversity push (production default) |
| λ = 0.5 | Equal weight on relevance and diversity |
| λ = 0.0 | Pure diversity — ignores query relevance entirely |

- Rank #1 is always the same as Top-K (no penalty yet — nothing selected)
- From rank #2 onward, near-duplicate chunks get penalized and pushed down
- You get **broader topic coverage** with the same K slots

### Tradeoffs (Interview Critical)

**When MMR works well:**
- Corpus has many near-duplicate or overlapping chunks
- Query is broad — you want coverage across multiple angles
- Context window is small — every slot must add new information
- Long documents chunked into overlapping sliding windows

**When MMR fails:**
- Query is very specific — you want the single best chunk, not diversity
- λ too low → retrieved chunks are diverse but weakly relevant (the wrong tradeoff)
- Small corpus with genuinely distinct chunks — diversity penalty fires unnecessarily
- Computationally heavier than Top-K: O(K·N) similarity lookups vs O(N)

**MMR vs Top-K — the interview answer:**
> Use Top-K when corpus chunks are naturally distinct. Use MMR when chunks cluster around similar information or when redundant retrievals are suspected. MMR wins on coverage; Top-K wins on raw relevance for specific queries.

### Key Takeaways

- MMR rank #1 is always the same as Top-K — the difference shows from rank #2 onward
- λ=0.7 is a strong default — mostly relevant, avoids the worst redundancy
- The `redundancy` score printed per chunk tells you *how much* diversity pressure was applied
- MMR is most valuable when chunking creates overlap (sliding window, recursive splitters)
- In interviews: explain MMR as "greedy iterative selection with a diversity penalty"

### Implementation

See [mmr.py](mmr.py)

Run it:
```bash
python mmr.py
```

**Experiments to run:**
1. Run λ=1.0, 0.7, 0.5, 0.2 on the same query — watch which chunks get swapped out
2. Use `compare_topk_vs_mmr()` — see exactly which chunks MMR replaces and why
3. Try a broad query (`"What are the main components of a RAG system?"`) — MMR should cover more topics
4. Run λ=0.0 — observe pure diversity (and why it's a failure case in practice)

### Think About

- **When would MMR hurt more than help?**
  When the query is very specific and there's one correct chunk. MMR might swap out the 2nd most relevant chunk (which is also correct) for a less relevant but "different" chunk. Diversity is a liability for precision-critical queries.

- **Why does MMR always pick the same rank #1 as Top-K?**
  Because at step 1, the "already selected" set is empty — the redundancy penalty is zero for every chunk. So the first pick is purely by relevance, identical to Top-K.

- **What does a high redundancy score tell you about your corpus?**
  It means many chunks are semantically similar — your chunking strategy is creating overlapping or near-duplicate chunks. This is a signal to fix your chunking, not just rely on MMR to paper over it.

- **How is λ different from the K and threshold parameters?**
  K and threshold control *how many* chunks to return. λ controls *which* chunks to return given K. They are orthogonal dimensions of retrieval tuning — a production system should tune all three.

- **Why can't you set λ=0.0 in production?**
  Pure diversity ignores relevance entirely. Chunk selection becomes almost random with respect to the query — you'd retrieve maximally spread-out chunks, most of which have nothing to do with the question.

---

## 4. Metadata Filtering

### Intuition

Every technique so far is blind to structure — the vector space knows nothing about document dates, types, languages, or access levels. Metadata filtering scopes retrieval to a specific **subset of the corpus before similarity is ever computed**.

Think of it as a SQL `WHERE` clause that runs before the vector search:
```
WHERE source = 'rag_guide' AND year >= 2024
THEN rank by cosine similarity
```

### How It Works

Each chunk stores both an embedding and a metadata dict:
```python
chunk = {
    "text": "...",
    "metadata": { "topic": "retrieval", "source": "rag_guide",
                  "year": 2024, "difficulty": "advanced" }
}
```

**Two-stage pipeline:**
```
Query + Filter conditions
    │
    ▼
Stage 1: Apply metadata filter → candidate pool (hard constraint)
    │  Eliminates non-matching chunks entirely
    ▼
Stage 2: Cosine similarity vs candidate pool ONLY
    │
    ▼
Top-K (or MMR) from filtered candidates
```

**Supported filter types:**
```python
{"topic": "retrieval"}                          # exact match
{"topic": {"$in": ["retrieval","embeddings"]}}  # multi-value
{"year": {"$gte": 2023}}                        # range
{"source": {"$ne": "draft"}}                    # exclude
{"source": "rag_guide", "year": {"$gte": 2024}} # combined AND
```

### Ranking Behavior

- Ranking behavior within the filtered pool is identical to Top-K — cosine similarity still determines order
- What changes is *what's eligible to rank* — the candidate pool shrinks from N to M (M ≤ N)
- A chunk with score 0.98 but wrong metadata is **rejected**; a chunk with score 0.41 but correct metadata **survives**
- Metadata filter is a **hard constraint**; similarity is a **soft ranking** — they compose naturally

### Tradeoffs (Interview Critical)

**When it works well:**
- Large, heterogeneous corpus (multiple doc types, dates, languages, departments)
- Queries have implicit or explicit scope (e.g., "this year's policy", "English docs only")
- Access control requirements — only show chunks from authorized sources
- Dynamic user context (subscription tier, active product version, region)

**When it fails:**
- **Over-filtering**: too strict → candidate pool near zero → retrieval degrades regardless of similarity
- **Missing metadata**: chunks tagged inconsistently → correct answers silently excluded
- **Wrong filter logic**: `topic=retrieval AND topic=embeddings` returns nothing — you need `$in` with OR
- Metadata quality is the silent killer — filters behave unpredictably on poorly-tagged corpora

**The over-filtering trap (interview critical):**
> Metadata filters are hard constraints that override similarity entirely. If your filter is logically wrong or too strict, no amount of good embeddings saves you. Always validate filter logic and handle empty pools explicitly.

**Combining with other techniques:**
> Metadata filtering is almost never used alone. Think of it as Stage 1 in a pipeline:
> `Metadata filter → Top-K → Threshold → MMR → Reranker`
> Each stage narrows or reorders the candidate set.

### Key Takeaways

- Always handle the empty-pool case — return "no results" rather than crashing or falling back silently
- `AND` logic across multiple fields can dramatically shrink the pool — use `$in` for OR within a field
- Metadata quality must be treated as a first-class concern — garbage tags = garbage retrieval
- In production, log how many chunks each filter passes — sudden drops signal tagging regressions
- Metadata filtering + MMR is a powerful combo: filter to the right domain, then diversify within it

### Implementation

See [metadata_filtering.py](metadata_filtering.py)

Run it:
```bash
python metadata_filtering.py
```

**Experiments to run:**
1. Same query — no filter vs `topic=retrieval` filter — compare what enters the result set
2. Add `year >= 2024` — how many chunks survive? What's excluded?
3. Combine `source=rag_guide` AND `difficulty=advanced` — observe the AND narrowing effect
4. Try `topic=retrieval` AND `source=history_book` — trigger the empty pool failure case

### Think About

- **What happens when your metadata filter is too strict?**
  The candidate pool shrinks to near zero. The surviving chunks may be barely relevant but they're the only ones left — retrieval returns low-quality results with no warning. Always monitor candidate pool size.

- **Why is AND logic dangerous across multiple metadata fields?**
  Each additional AND condition multiplies the exclusion. Three conditions that each pass 50% of chunks independently leave only 12.5% of the corpus. Combinatorial narrowing is non-obvious and easy to over-do.

- **How is metadata filtering different from similarity threshold?**
  Threshold filters by *score quality* (how relevant is this chunk?). Metadata filters by *structural attributes* (what kind of document is this?). They solve completely different problems and are complementary — use both together.

- **How would you handle a user query that implies a metadata filter?**
  This is query parsing / intent extraction. E.g., "What did we decide in Q1 2024?" implies `year=2024`. In production systems, an LLM or rule-based parser extracts filter conditions from the query before retrieval runs — this is called "self-querying retrieval."

- **What would happen if a chunk is missing a metadata field your filter checks?**
  It depends on implementation. Most systems treat a missing field as a non-match (the chunk is excluded). This is a silent failure — correct chunks get dropped because they were never tagged. Always audit your tagging pipeline.

---

## 5. Hybrid Search (BM25 + Dense)

### Intuition

Every technique so far has used **dense retrieval** — embed the query, embed the chunks, cosine similarity. Dense retrieval is great at capturing *meaning*. But it has a blind spot:

> **Dense retrieval fails when the answer depends on exact words, not meaning.**

Query: `"What is FAISS?"` — an embedding model may return chunks about "vector databases" and "similarity search" because they're semantically close. But the chunk that literally says *"FAISS is..."* — the exact definition — might rank lower because semantically broader chunks outscore it.

This is where **sparse retrieval (BM25)** wins. BM25 is a classical keyword-based ranking algorithm. It doesn't understand meaning — it counts and weights term frequencies. If the query word appears in the chunk, BM25 scores it highly. Period.

> **Hybrid Search fuses both signals: BM25 for exact term matching + Dense embeddings for semantic understanding.**

Neither alone is complete. Together they cover each other's blind spots.

---

### How BM25 Works (First Principles)

BM25 scores a chunk based on:
1. **Term frequency (TF)** — how often does the query term appear in the chunk?
2. **Inverse document frequency (IDF)** — how rare is the query term across the corpus? (rare terms are more informative)
3. **Document length normalization** — penalizes longer documents to avoid length bias

```
BM25(chunk, query) = Σ  IDF(term) · TF(term, chunk) · (k1 + 1)
                    terms        ─────────────────────────────────
                                 TF(term, chunk) + k1·(1 - b + b·|chunk|/avgdl)
```

- `k1` — term frequency saturation (default ~1.5): diminishing returns on repeated terms
- `b`  — length normalization (default ~0.75): how much to penalize long chunks
- `avgdl` — average document length across corpus

You don't need to memorize the formula. What matters: **BM25 rewards exact keyword overlap, penalizes term stuffing, and normalizes for length.**

---

### How Hybrid Search Works

Two retrievers run in parallel, then their scores are fused:

```
Query
 │
 ├──► BM25 retriever  → sparse scores  (keyword match)  → top N candidates
 │
 └──► Dense retriever → dense scores   (semantic match)  → top N candidates
             │
             ▼
      Score Fusion (combine both signals)
             │
             ▼
      Final ranked list → top K returned
```

**Score Fusion — Reciprocal Rank Fusion (RRF):**

The most common fusion method. Instead of combining raw scores (which live on different scales), RRF combines **ranks**:

```
RRF_score(chunk) = Σ   ────────────────
                 lists  k + rank(chunk, list)
```

- `k` = smoothing constant (default 60)
- A chunk ranked #1 in BM25 and #3 in dense gets a higher combined score than one ranked #5 in both
- Works well because ranks are comparable across retrievers; raw scores are not

**Alternative: Weighted Linear Combination**
```
final_score = α · normalize(dense_score) + (1 - α) · normalize(bm25_score)
```
- `α` controls which signal to trust more
- Requires normalizing scores to the same range first (min-max or softmax)
- More tunable but more fragile than RRF

---

### Ranking Behavior

| Query type | BM25 behavior | Dense behavior | Winner |
|---|---|---|---|
| Exact keyword: `"What is FAISS?"` | High — "FAISS" is rare, exact match | Medium — may surface related chunks | BM25 |
| Semantic: `"How do I avoid duplicate results?"` | Low — no exact overlap with "MMR" | High — captures "redundancy", "diversity" | Dense |
| Acronym / jargon: `"BM25 vs TF-IDF"` | High — exact term match | Low — embedding may not know niche terms | BM25 |
| Paraphrase: `"chunks that say the same thing"` | Low — no lexical overlap | High — semantic similarity captures intent | Dense |
| Mixed: `"FAISS approximate nearest neighbor search"` | Medium | Medium | Hybrid wins |

**The key insight for interviews:**
> Dense retrieval generalizes; BM25 specializes. Hybrid search gets the best of both — it won't miss an exact keyword match AND won't miss a semantic match.

---

### Tradeoffs (Interview Critical)

**When Hybrid Search works well:**
- Corpus contains a mix of technical jargon, acronyms, and natural language
- Queries vary widely — some keyword-specific, some conceptual
- You can't predict query type in advance (general-purpose RAG systems)
- Domain with specialized terminology not well-represented in embedding training data

**When it adds unnecessary complexity:**
- Corpus is purely conversational / natural language — dense alone is sufficient
- Latency is critical and you can't afford two retrievers
- Small corpus where exhaustive search is fast anyway
- Embedding model was specifically fine-tuned on your domain

**The fusion calibration problem:**
> In weighted fusion (`α·dense + (1-α)·bm25`), α must be tuned empirically. α=0.5 is not "balanced" — BM25 and dense scores have different distributions. Always normalize before combining, or use RRF (rank-based) to avoid scale mismatch.

**BM25 failure cases:**
- Semantic queries with no lexical overlap (BM25 returns garbage)
- Synonyms: query says "car", corpus says "automobile" — BM25 misses it
- Short chunks: fewer terms → less signal for TF-IDF weighting

**Dense failure cases:**
- Rare proper nouns, model names, version numbers (e.g., `"GPT-4o-mini"`, `"v2.3.1"`)
- Out-of-distribution domain (medical, legal, code) if model wasn't trained on it
- Very short queries (single word) — embedding captures too little signal

---

### Key Takeaways

- BM25 and dense retrieval have **complementary failure modes** — that's exactly why hybrid works
- RRF is the safest fusion method — rank-based, no score normalization needed, robust default
- Weighted fusion (`α`) is more powerful but requires careful calibration
- In production: run both retrievers, deduplicate the candidate pool, fuse scores, then apply Top-K or MMR
- Hybrid search is the **industry default** for production RAG — pure dense retrieval is a starting point, not an endpoint
- The full pipeline: `Metadata filter → BM25 + Dense → RRF fusion → Threshold → MMR → Reranker`

### Implementation

See [hybrid_search.py](hybrid_search.py)

Run it:
```bash
python hybrid_search.py
```

**Experiments to run:**
1. Query with an exact keyword (`"FAISS"`) — compare BM25 rank vs dense rank vs hybrid rank
2. Query with a semantic paraphrase (`"chunks that repeat the same information"`) — dense wins, observe BM25 rank
3. Vary the RRF `k` constant (20, 60, 100) — how stable are the final ranks?
4. Try weighted fusion at α=0.3, 0.5, 0.7 — when does BM25 start dominating?

### Think About

- **Why does dense retrieval fail on exact keyword queries like `"What is FAISS?"`**
  The embedding of `"What is FAISS?"` captures a general "define X" intent. Chunks about similar concepts (vector search, ANN) score high because they're *semantically close*, even though they don't define FAISS. The chunk that says "FAISS is..." may actually score lower than broader overview chunks.

- **When is BM25 better than a fine-tuned embedding model?**
  When the corpus contains rare proper nouns, version strings, model names, or domain-specific acronyms that the embedding model has never seen during training. BM25 doesn't care about training — if the word appears, it matches.

- **Why can't you just add the raw BM25 and dense scores together?**
  BM25 scores can be in the range [0, 15+] depending on corpus statistics. Dense cosine scores are in [-1, 1]. Adding them directly lets BM25 dominate completely. You must normalize both to the same scale (e.g., min-max to [0,1]) before combining — or avoid the problem entirely with RRF.

- **What does RRF's `k=60` constant actually do?**
  It prevents top-ranked items from having a disproportionately large score advantage. Without it, rank #1 scores infinitely better than rank #2 as N grows. `k=60` smooths the curve — a chunk at rank #1 gets score `1/61 ≈ 0.016`, rank #2 gets `1/62 ≈ 0.016`. The difference is small, so rank order matters but doesn't dominate.

- **If hybrid search is so good, why doesn't everyone use it?**
  Operational complexity. Two retrievers mean two indices to maintain (a vector index AND an inverted BM25 index), two sets of embeddings to update, two latencies to manage, and a fusion layer to tune. For many use cases, dense-only retrieval with good chunking and a reranker achieves comparable results with far less infrastructure.

---

## Key Tradeoffs Summary

| Technique | Solves | Fails When |
|---|---|---|
| Top-K | Baseline semantic ranking | Irrelevant query, redundant corpus |
| Threshold | Irrelevant query returns garbage | Threshold miscalibrated, short queries |
| MMR | Redundant/duplicate chunks | Specific query needs precision not diversity |
| Metadata Filter | Wrong doc type/date/source retrieved | Over-filtering, missing tags |
| Hybrid Search | Exact keyword + semantic gap | High latency budget, simple corpus |

**The precision-recall dial:**
- Large K → higher recall, lower precision
- Small K → higher precision, lower recall

**The relevance-diversity dial (MMR):**
- λ → 1.0: pure relevance (Top-K equivalent)
- λ → 0.0: pure diversity (relevance ignored)

**Dense vs Sparse:**
- Dense wins on semantics, paraphrase, intent
- BM25 wins on exact terms, jargon, rare proper nouns
- Hybrid wins on mixed, unpredictable query types

**The full production pipeline:**
```
Query
 │
 ├── Metadata filter     (scope to right subset)
 │
 ├── BM25 + Dense        (two retrievers in parallel)
 │
 ├── RRF fusion          (combine rankings)
 │
 ├── Threshold gate      (reject low-confidence chunks)
 │
 ├── MMR                 (diversify if redundant)
 │
 └── Reranker            (cross-encoder final pass)
         │
         ▼
      Top K → LLM context
```

---

## My Observations

*(Fill this after each experiment)*

### Top-K Observations
- [ ] What happened when K=1 vs K=8?
- [ ] What scores did irrelevant chunks get?
- [ ] Did you see any duplicate/similar chunks in the top results?

### Similarity Threshold Observations
- [ ] At what threshold did the irrelevant query return 0 results?
- [ ] Did any relevant chunks get falsely rejected at high threshold?
- [ ] Where was the natural score gap in the distribution?

### MMR Observations
- [ ] Which chunks did MMR swap out vs Top-K at λ=0.7?
- [ ] Did the swapped-in chunks cover a genuinely different angle?
- [ ] What happened to result quality at λ=0.2?

### Metadata Filtering Observations
- [ ] How many chunks survived the combined source + year filter?
- [ ] Did the over-filtering experiment return 0 results as expected?
- [ ] Which filter combination gave the best precision?

### Hybrid Search Observations
- [ ] On which query type did BM25 clearly outrank dense?
- [ ] Did RRF produce a noticeably different ranking than dense alone?
- [ ] What α value felt most balanced for your test queries?
