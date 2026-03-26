# Embeddings in RAG

> **Goal:** Understand how text is converted into vectors, how similarity is measured, and when embeddings fail — deeply enough to implement, debug, and explain in interviews.

Embeddings are the bridge between raw text and mathematical operations. Without them, "semantic search" doesn't exist — you're back to keyword matching. Every RAG system lives or dies on the quality of its embeddings.

---

## 1. Dense Embeddings

### Concept

A dense embedding converts a sentence into a fixed-size vector of floats — a single point in high-dimensional space. The model is trained so that **semantically similar text lands close together**, regardless of exact word choice.

```
"heart attack"          → [0.21, -0.43, 0.87, ...]   ←─┐
"myocardial infarction" → [0.19, -0.41, 0.85, ...]   ←─┘  close

"quantum physics"       → [-0.72, 0.33, -0.12, ...]       far
```

This is why semantic search works: retrieval becomes a geometry problem — find the nearest points to the query vector.

### How It Works (Pipeline)

```
Input text
    │
    ▼
Tokenizer → token IDs
    │
    ▼
Transformer encoder (12-24 layers of self-attention)
    │
    ▼
Token-level vectors: one vector per token
    │
    ▼
Pooling (mean pooling or CLS token)
    │
    ▼
Single vector: [v₁, v₂, ..., v₃₈₄]  ← the sentence embedding
```

**Contrastive training:** The model was trained by pushing similar sentence pairs together and dissimilar pairs apart in vector space. After training, closeness = semantic similarity.

### Mathematical Understanding

A sentence embedding is a point in ℝᴺ (N-dimensional real space).

```
v = [v₁, v₂, v₃, ..., v₃₈₄]    (all-MiniLM-L6-v2: 384 dims)
```

**What does each dimension mean?** Nothing individually. Meaning is distributed across all dimensions collectively. You cannot interpret dimension 47 as "sentiment". The geometry as a whole is what encodes semantics.

**Vector norm:**
```
||v|| = √(v₁² + v₂² + ... + v₃₈₄²)
```
Raw embeddings from sentence-transformers are NOT unit-normalized (norm ≠ 1.0). Cosine similarity normalizes internally. Dot product does not — we'll exploit this distinction in Concept #3.

**Dimensionality:**

| Model                      | Dimensions | Notes                          |
|----------------------------|------------|--------------------------------|
| all-MiniLM-L6-v2           | 384        | Fast, good general baseline    |
| all-mpnet-base-v2          | 768        | Better quality, 2x cost        |
| text-embedding-3-small     | 1536       | OpenAI, strong general purpose |
| text-embedding-3-large     | 3072       | OpenAI, best quality, 8x cost  |

Higher dimensions = more capacity to encode fine distinctions, but also higher storage and compute cost. Diminishing returns apply.

### Experiments & Observations

**Exp 1 — Embedding anatomy:**
- Raw norm is NOT 1.0 (not unit-normalized by default)
- Values span both positive and negative floats
- Shape is exactly (384,) for all-MiniLM-L6-v2

**Exp 2 — Similar pairs:**
- "heart attack" vs "myocardial infarction" → high cosine similarity (~0.7-0.9)
- Different words, same concept → model captures semantic equivalence

**Exp 3 — Dissimilar pairs:**
- Unrelated topics → low similarity (~0.0-0.2)
- The gap between similar and dissimilar is the retrieval signal

**Exp 4 — FAILURE CASE: Negation trap:**
```
"The drug is effective for treating depression."
"The drug is not effective for treating depression."
→ Similarity: HIGH (~0.85-0.95)
```
Embeddings capture topic and context, not logical operators. "Not" barely shifts the vector because the model has seen "effective" and "treating depression" in similar contexts during training. **This is a fundamental limitation** — embeddings cannot reliably distinguish negation.

**Exp 5 — Short vs long:**
- Short: "RAG reduces hallucinations." (4 words)
- Long: full paragraph about RAG (41 words)
- Similarity is still high — mean pooling compresses all token information regardless of length

**Exp 6 — Ranking:**
The query "How does RAG work?" correctly ranked:
1. "RAG retrieves relevant documents before generating answers." (most similar)
2. "Retrieval-Augmented Generation combines search with generation."
...
6. "The Eiffel Tower is located in Paris, France." (least similar)
Semantic ranking from pure vector geometry — no keywords, no rules.

### Tradeoffs

| Situation                        | Dense embedding behavior                          |
|----------------------------------|---------------------------------------------------|
| Paraphrase detection             | Excellent — model trained for this                |
| Domain-specific jargon           | Weaker — unseen terms map to generic vectors      |
| Negation / logical operators     | **Fails** — "effective" ≈ "not effective"         |
| Very short text (<5 words)       | Weaker signal — less context for pooling          |
| Cross-language retrieval         | Needs multilingual model (mE5, multilingual-e5)   |
| Real-time indexing (speed)       | Fast with small models, slow with large ones      |

### When To Use

- Default first choice for semantic search in RAG
- Any task where synonyms or paraphrases should match
- When keyword overlap is insufficient (domain variation, user query rewriting)

### Interview Insight

> "Dense embeddings are produced by mean-pooling the token-level outputs of a transformer encoder
> that was fine-tuned with contrastive loss. The contrastive objective pulls semantically similar
> pairs together and pushes dissimilar pairs apart. The key failure mode is logical operators —
> 'A' and 'not A' embed very similarly because the model is sensitive to topic and context,
> not to logical structure. For safety-critical applications (medical, legal), you must add a
> post-retrieval re-ranker or classifier to catch negation errors."

### Think About This

1. The norm of a raw embedding is not 1.0 — it varies by sentence. If you use dot product (not cosine) to compare two embeddings, what happens? Which sentence would score higher — a shorter or longer one, and why?
2. In Exp 4, "effective" and "not effective" scored ~0.85+. What does this say about using dense embeddings alone for a medical QA system?
3. In Exp 6, the ranking felt intuitive. Now think of a case where the ranking would *feel wrong* — what type of query would produce a misleading top result?

---

## 2. OpenAI vs Open-source Embeddings

Same job, different tradeoffs.

```
OpenAI API                     Open-source (local)
───────────────────────        ───────────────────────
~100-300ms latency             ~5-50ms latency
$0.02/1M tokens                $0 after download
Data leaves your machine       Fully private
Needs internet                 Works offline
1536 / 3072 dims               384 / 768 dims
```

### Models to know

| Model | Dims | Notes |
|---|---|---|
| `text-embedding-3-small` | 1536 | Best OpenAI cost/quality |
| `text-embedding-ada-002` | 1536 | Legacy, still common |
| `all-MiniLM-L6-v2` | 384 | Fastest open-source baseline |
| `all-mpnet-base-v2` | 768 | Better quality, 6x slower |
| `BAAI/bge-small-en-v1.5` | 384 | Strong for retrieval tasks |
| `intfloat/e5-base-v2` | 768 | Great for RAG, needs prefixes |

### Key results from experiments

**Exp 1 — Rankings diverge across models:**
- All three agreed on rank #1
- `mpnet-base` ranked "generator uses retrieved context" as #2 — more semantically aware
- `bge-small` scores are much higher overall (0.83 vs 0.49) — different scale, not better quality

**Exp 2 — Speed (102 texts):**
```
MiniLM-L6  (384d)  →  0.16s   ← fastest
bge-small  (384d)  →  0.30s
mpnet-base (768d)  →  0.99s   ← 6x slower for 2x dims
```
768-dim is not 2x better — but it is 6x slower. Choose carefully.

**Exp 3 — OpenAI cost:**
```
1k docs   → $0.008   (essentially free)
50k docs  → $0.40
500k docs → $4.00
Re-embed monthly (500k) → $4.00/month
```
Cost only becomes painful with millions of docs or frequent re-embedding.

**Exp 4 — E5 prefix trick:**
- `intfloat/e5-base-v2` expects `"query: "` and `"passage: "` prefixes
- On this dataset, delta was small (-0.016), but on specialized data it matters significantly
- **Always read the model card**

### When to use what

| Situation | Choice |
|---|---|
| Prototype / local dev | `all-MiniLM-L6-v2` |
| Need best open-source quality | `BAAI/bge-large-en-v1.5` or `e5-large-v2` |
| Data privacy required | Any local model |
| Production, quality matters most | `text-embedding-3-small` |
| Specialized domain (medical, legal) | Fine-tuned domain model (Concept #4) |

### Interview Insight

> "The choice between OpenAI and open-source is almost never pure quality — it's cost × privacy × latency × domain fit. For a 500k-doc knowledge base re-embedded monthly, OpenAI costs $4/month — negligible. But if data can't leave your infrastructure, local models are the only option regardless of quality. The real trap is assuming bigger dims = better retrieval. `bge-small` at 384 dims outperforms `mpnet-base` at 768 dims on retrieval benchmarks."

### Think About This

1. `bge-small` gave similarity scores of 0.83 while `MiniLM` gave 0.49 for the same pair. Does a higher score mean `bge` is better? What would you need to test to actually compare quality?
2. Re-embedding 500k docs costs $4. But what triggers a re-embed? If your embedding model becomes outdated and you switch models, you must re-embed everything. How would you design your system to make model swaps painless?
3. E5 needs `"query: "` and `"passage: "` prefixes. What happens at query time if you forget the prefix but used it during indexing? Would retrieval quality drop, or would it not matter?

---

## 3. Similarity Metrics

Three ways to measure closeness. Different answers when vectors aren't normalized.

```
cosine(a,b)  =  (a·b) / (||a|| × ||b||)    # angle only,  -1 to +1
dot(a,b)     =  Σ(aᵢ × bᵢ)                 # angle × magnitude
l2(a,b)      =  √Σ(aᵢ - bᵢ)²               # distance,  0 to +∞
```

**When normalized** → all three give identical rankings. Dot = cosine (denominator = 1). L2² = 2 − 2·cosine (proven in Exp 1: `match=True` for all).

**When NOT normalized** → dot product inflates high-magnitude vectors regardless of angle.

### Results

**Exp 1:** cos=dot=0.7419, l2²=0.5162, 2-2·cos=0.5162. Match confirmed. Model is already normalized.

**Exp 2 — Magnitude trap:**
```
Same direction, 1x magnitude:    cosine=0.49  dot=0.49   (agree)
Same direction, 5x magnitude:    cosine=0.49  dot=2.45   (dot inflated)
Different direction, 10x mag:    cosine=0.03  dot=0.35   (dot misleading!)
```
Dot product ranked the 10x unrelated vector higher than the 1x relevant one. Cosine didn't flinch.

**Exp 4 — Speed:**
```
Cosine (with norm compute) : 7.19ms  per 10k vectors
Dot product (normalized)   : 0.48ms  per 10k vectors
Speedup: 14.9x
```
This is why production systems normalize at index time and use dot product at query time.

**Exp 5 — Long doc bias:**
```
Short doc (RAG reduces hallucinations):        cosine=0.95  dot=1.89
Long doc  (full paragraph about RAG):          cosine=0.50  dot=4.00
```
Dot product makes the long doc 2x better than the short one. Cosine correctly scores the short doc higher (more focused).

### When to use which

| Metric | Use when |
|---|---|
| Cosine | Default safe choice, unnormalized vectors |
| Dot product | Vectors are pre-normalized (faster, identical to cosine) |
| L2 | Some clustering algorithms, FAISS flat index default |

### Interview Insight

> "Cosine and dot product are identical on unit-normalized vectors — and most sentence-transformer models normalize their output. So production systems normalize at ingestion and use dot product at query time for a 15x speedup with no quality loss. The failure case is skipping normalization — then dot product biases toward longer/verbose documents because they have higher-magnitude embeddings."

### Think About This

1. Exp 4 showed 14.9x speedup from dot product. But this is CPU. On a GPU with FAISS, both are extremely fast. At what scale does this 15x speedup actually matter — 1k docs or 100M docs?
2. Exp 5: cosine scored the short focused doc higher (0.95 vs 0.50). Is that always correct? Can you think of a case where you'd actually *want* the longer, more comprehensive doc ranked higher?
3. Vector databases like FAISS default to L2. Since L2 and cosine rank identically on normalized vectors, does it matter which distance function your vector DB uses — as long as you normalize before inserting?

---

## 4. Domain-Specific Embeddings & Embedding Drift

General models fail on specialized jargon — they've never seen `HNSW`, `HbA1c`, or `indemnification` in the right context. They map these to generic vectors close to nothing useful.

### The gap (Exp 1)

```
HNSW vs graph-based ANN:              expected=1.0  got=0.27  ← BAD
MultipleNegativesRankingLoss vs ...:  expected=1.0  got=0.34  ← BAD
vector DB vs Eiffel Tower:            expected=0.0  got=-0.03 ← fine
```

General model is nearly useless on technical synonyms.

### Fine-tuning (Exp 2)

With only 8 training pairs, improvement was minimal (0.27→0.25 for HNSW). You need **hundreds of pairs minimum** for meaningful gains. Rule of thumb: more pairs > more epochs.

**Loss: `MultipleNegativesRankingLoss`**
- Give it `(anchor, positive)` pairs
- All other positives in the batch act as in-batch negatives — no manual negative mining
- Most practical loss for RAG fine-tuning

### Three adaptation strategies

```
1. Use a domain pre-trained model      zero cost if one exists
   BioBERT, LegalBERT, CodeBERT

2. Fine-tune with contrastive pairs    medium cost
   Need 500+ (query, relevant_doc) pairs

3. Continued pretraining               high cost
   Run MLM on domain corpus first, then fine-tune
   Best for very niche vocabulary (clinical notes, patent filings)
```

### Embedding drift

Two flavors:

**Type 1 — Unknown term:** New product/acronym the model never saw. Gets mapped to a generic vector. Gap between "related" and "unrelated" similarity collapses.

**Type 2 — Shifted meaning:** "Transformer" meant electrical device before 2017. Model trained today correctly scores AI meaning higher (0.53) than electrical (0.36) — but a model from 2016 would flip that.

**Fix:** Periodic re-evaluation on a held-out pair set. If domain pair scores drop, re-fine-tune.

### How to build training pairs (Exp 4)

| Strategy | How |
|---|---|
| LLM-generated queries | `"Write 3 questions this passage answers: {chunk}"` → (question, chunk) pairs |
| Section headers | Header = query proxy, body = positive doc |
| User click data | Query + clicked result = positive pair |
| BM25 hard negatives | Find keyword-matching but semantically wrong docs as negatives |

Most practical: **LLM-generated queries per chunk**. Cheapest way to bootstrap domain training data.

### Interview Insight

> "Domain adaptation matters when your corpus contains terminology the base model wasn't trained on. The gap is visible: general models score 0.27 on technical synonym pairs that should score 0.9+. Fine-tuning with `MultipleNegativesRankingLoss` on (query, passage) pairs is the standard fix — but you need hundreds of pairs, not tens. The cheapest data source is using an LLM to generate questions for each chunk at indexing time."

### Think About This

1. Fine-tuning with 8 pairs barely moved the needle. At what point would you stop fine-tuning and instead switch to a domain-specific pre-trained model like BioBERT?
2. Embedding drift means your vector index becomes stale as language evolves. If you re-fine-tune and switch models, you must re-embed everything. How would you handle this in a live production system with zero downtime?
3. The "transformer" word correctly maps to AI meaning now. But what about a RAG system for electrical engineering? How would you prevent the model from returning AI papers when someone asks about electrical transformers?

---

## 5. Multi-Vector Embeddings (ColBERT-style)

Single-vector pools all tokens into one point. Token-level detail is lost. Multi-vector keeps one vector per token — no pooling.

```
Single-vector: "RAG retrieves documents" -> [one 384-dim vector]

Multi-vector:  "RAG retrieves documents" -> (10, 384) matrix
               [CLS] rag retrieve ##s documents using vector search . [SEP]
                each gets its own context-aware vector
```

### MaxSim scoring

For each query token, find its best-matching document token. Sum those scores.

```
query token "vector"     -> best match "vectors"     -> 0.73
query token "similarity" -> best match "similarity"  -> 0.86
query token "search"     -> best match "find"        -> 0.67
                            Total MaxSim score:         3.44
```

No pooling = no information loss. Each query token gets to vote independently.

### Results

Exp 3 & 4 both agreed on the top rankings with single-vector for clear cases. The real advantage emerges when one specific token (e.g. a rare domain term) in a long query gets diluted by mean-pooling — MaxSim preserves that token's signal.

Exp 5 — Storage cost:
```
Single-vector : 147 MB    (100k chunks)
Multi-vector  : 3125 MB   (100k chunks, 64 tokens, 128-dim)
Ratio         : 21x more storage
```

### Tradeoffs

| | Single-vector | Multi-vector (ColBERT) |
|---|---|---|
| Storage | Low | 21x higher |
| Query latency | Fast (one dot product) | Slower (MaxSim over all tokens) |
| Precision on rare terms | Loses signal in pooling | Preserves token-level signal |
| Indexing | Simple | Needs token-aware index |

### Interview Insight

> "ColBERT's key innovation is late interaction — the query and document token vectors are only compared at scoring time, not collapsed into a single vector at indexing time. This preserves token-level expressiveness at the cost of 21x storage. In practice, ColBERT v2 uses vector compression (128-dim + binarization) to bring this down to 3-4x over single-vector, making it viable for production."

### Think About This

1. In Exp 2, `[CLS]` matched `[CLS]` with 0.67 — a special token matching itself. Does this add noise to the score, or is it a useful signal? Should you mask special tokens in MaxSim?
2. Multi-vector stores 64 vectors per chunk vs 1. If you need to retrieve top-1000 candidates from 10M chunks before re-ranking, does MaxSim actually run on all 10M? How does ColBERT handle this at scale?
3. Single-vector and multi-vector agreed on rankings for clean queries. Under what exact conditions would they diverge — what kind of query would expose the pooling weakness?

---

## 6. Late Interaction vs Single Vector

The architecture triangle — the most interview-critical concept here.

```
Bi-encoder      encode separately -> one dot product     Fast, indexable, lower quality
Cross-encoder   encode together   -> full attention      Slow, NOT indexable, highest quality
Late interaction encode separately -> MaxSim at query time   Middle ground
```

### Why cross-encoder can't scale (Exp 4)

```
Corpus size    Cross-enc latency
1,000 docs     7,432 ms    <- 7 seconds per query
100,000 docs   743,216 ms  <- impossible
```

Can't index. Can't pre-compute. Scores one pair at a time. Never use alone at scale.

### Bi-encoder weakness (Exp 2)

Query: *"What are the limitations of RAG?"*
- Bi-encoder ranked "Chunking strategy affects retrieval quality" as #1 (keyword overlap with RAG)
- Cross-encoder correctly pushed it to #3 and surfaced "RAG struggles with multi-hop reasoning" as #1

Cross-encoder sees the query when reading the document — it understands intent, not just overlap.

### The production pattern

```
10M docs
  -> bi-encoder retrieval (top-100, fast, indexed)
  -> cross-encoder re-rank (top-5, slow but only on 100 docs)
  -> LLM
```

Cross-encoder at 100 docs = 743ms. Acceptable. This is how every serious RAG system works.

### Architecture summary

| | Bi-encoder | Cross-encoder | Late Interaction |
|---|---|---|---|
| Indexable | Yes | No | Yes |
| Quality | Lower | Highest | High |
| Speed | O(1) | O(n) | O(tokens) |
| Use case | First-stage retrieval | Re-ranking | Precision retrieval |

### Interview Insight

> "Bi-encoder and cross-encoder solve different problems. Bi-encoder enables indexing — you pre-compute doc vectors once and retrieve with a single dot product. Cross-encoder enables accuracy — it sees both query and doc simultaneously, so it can catch subtle relevance signals. The standard pattern is to use both: bi-encoder retrieves top-k candidates, cross-encoder re-ranks them. Late interaction (ColBERT) is the middle ground: indexable like bi-encoder, token-level precision like cross-encoder, at 21x storage cost."

### Think About This

1. In Exp 1, cross-encoder gave `"The Eiffel Tower"` a score of -11.36 and `"Vector databases"` a score of -11.43. They're both very negative but the *order swapped* between bi-encoder and cross-encoder. What does a negative cross-encoder score mean — and is the small difference between -11.36 and -11.43 meaningful?
2. The two-stage pipeline retrieved top-5 with bi-encoder, then re-ranked. What if the correct answer was ranked #6 by bi-encoder? It never reaches the cross-encoder. How would you set `retrieve_k` in production — and what's the tradeoff with making it larger?
3. Late interaction is indexable because doc token vectors are computed offline. But MaxSim still runs over all doc tokens at query time. For a corpus of 10M chunks with 64 tokens each, how many similarity computations does MaxSim require per query — and why does this still require a two-stage approach?

---

## Key Concepts

*(Filled in as we progress)*

- **Cosine similarity vs Dot product vs L2:** → Concept #3
- **Embedding dimensionality:** Higher = richer, but diminishing returns and higher cost
- **Vector normalization:** Dot product on normalized vectors = cosine similarity
- **Embedding drift:** → Concept #4

---

## Observations (fill this after experiments)

```
Exp 1 — Raw embedding norm:


Exp 2 — Similar pair scores:


Exp 4 — Negation scores (surprising?):


Exp 6 — Did the ranking feel correct? Any surprises?:

```
