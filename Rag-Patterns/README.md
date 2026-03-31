# RAG Patterns

## Table of Contents
1. [Simple RAG](#1-simple-rag)
2. [RAG with Memory](#2-rag-with-memory-conversational-rag)
3. [Branched RAG](#3-branched-rag-multi-retriever--adaptive-rag)
4. [Quick Cheatsheet](#quick-cheatsheet)

---

## 1. Simple RAG

### Concept & Intuition

**Problem it solves:**
LLMs have a knowledge cutoff and no access to private/domain-specific data.
Simple RAG grounds the LLM in *retrieved facts*, reducing hallucination and
enabling answers over custom corpora without fine-tuning.

**Core Idea:**
Instead of asking the LLM to recall from memory, we *retrieve* relevant
documents at query time and inject them into the prompt as context.

---

### Architecture — Step-by-Step Flow

```
User Query
    │
    ▼
[Embedding Model]         ← Convert query to vector
    │
    ▼
[Vector Store / Index]    ← ANN search over document embeddings
    │
    ▼
[Top-K Documents]         ← Retrieved chunks (k=3~5 typically)
    │
    ▼
[Prompt Builder]          ← "Answer using this context: {docs}\n\nQ: {query}"
    │
    ▼
[LLM]                     ← Generates grounded answer
    │
    ▼
Response
```

---

### Components Breakdown

| Component       | Role                                              | Examples                        |
|----------------|---------------------------------------------------|---------------------------------|
| Embeddings      | Encode query + docs into dense vectors            | OpenAI ada-002, sentence-transformers |
| Vector Store    | Index + similarity search (cosine/dot product)    | FAISS, Chroma, Pinecone         |
| Retriever       | Fetches top-K semantically similar chunks         | Dense retrieval (ANN search)    |
| Prompt Builder  | Injects retrieved context into LLM prompt         | f-string / template             |
| LLM             | Reads context + generates answer                  | GPT-4, Claude, Mistral          |

---

### When to Use

- Internal knowledge bases (HR docs, product manuals)
- FAQ bots over private documents
- Legal / medical document Q&A
- Any domain where LLM training data is insufficient

---

### Limitations / Failure Cases

| Failure                     | Reason                                                      |
|-----------------------------|-------------------------------------------------------------|
| Retrieval miss              | Query vector ≠ document vector (semantic gap)               |
| Context window overflow     | Too many chunks exceed LLM token limit                      |
| Hallucination still occurs  | LLM ignores retrieved context or context is irrelevant      |
| No conversation memory      | Each query is stateless — follow-up questions fail          |
| Chunk boundary problem      | Relevant info split across chunks, neither chunk sufficient |
| Embedding model mismatch    | Query and docs encoded with different models                |

---

## 2. RAG with Memory (Conversational RAG)

### Concept & Intuition

**Problem it solves:**
Simple RAG is stateless — each query is independent. In a real conversation,
users ask follow-up questions like "What about its limitations?" or "Can you
elaborate on that?" — with no explicit subject. Without memory, the retriever
gets a vague query and retrieves irrelevant documents.

**Core Idea:**
Maintain a chat history. Before retrieval, *rewrite the user's latest query*
by incorporating prior conversation context into a standalone, self-contained
query. Then run standard RAG on the rewritten query.

---

### Architecture — Step-by-Step Flow

```
Chat History + New User Query
            │
            ▼
  [Query Rewriter / Condenser]    ← LLM rewrites query using history
            │
            ▼
   Standalone Rewritten Query
            │
            ▼
     [Embedding Model]
            │
            ▼
     [Vector Store Search]
            │
            ▼
      [Top-K Documents]
            │
            ▼
  [Prompt Builder]                ← history + context + rewritten query
            │
            ▼
          [LLM]
            │
            ▼
         Response  ──────────────► Appended to Chat History
```

---

### Components Breakdown

| Component         | Role                                                     |
|------------------|----------------------------------------------------------|
| Chat History      | Stores (role, content) pairs — user + assistant turns    |
| Query Rewriter    | LLM call to make latest query standalone                 |
| Embeddings        | Same as Simple RAG                                       |
| Vector Store      | Same as Simple RAG                                       |
| Prompt Builder    | Now includes history + retrieved context + query         |
| LLM               | Answers in context of full conversation                  |

---

### When to Use

- Multi-turn chatbots (customer support, tutoring, documentation assistants)
- Any product where users ask follow-up or clarifying questions
- Long research sessions where context builds across turns

---

### Limitations / Failure Cases

| Failure                        | Reason                                                         |
|--------------------------------|----------------------------------------------------------------|
| History grows unbounded        | Token limits hit after many turns — need truncation/summarization |
| Rewriter introduces errors     | LLM misinterprets history and rewrites query incorrectly       |
| Slow (extra LLM call)          | Query rewriting adds latency — 2 LLM calls per turn            |
| Topic shift not detected       | Old history poisons retrieval for a completely new topic       |
| Memory is shallow (window)     | Only last N turns retained — very old context lost             |

---

## 3. Branched RAG (Multi-Retriever / Adaptive RAG)

### Concept & Intuition

**Problem it solves:**
A single retriever over a single vector store cannot handle queries that span
multiple data sources (e.g., structured DB + unstructured docs + live web).
Branched RAG routes each query — or parts of a query — to the *most appropriate
retriever*, then merges results before generation.

**Core Idea:**
A router (rule-based or LLM-based) classifies the query and dispatches it to
one or more specialist retrievers in parallel. Results are merged, ranked, and
injected into a unified prompt.

---

### Architecture — Step-by-Step Flow

```
User Query
    │
    ▼
[Query Router]               ← Classifies intent: "which retriever(s)?"
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[Retriever A]                        [Retriever B]
Vector Store                         SQL / Structured DB
(unstructured docs)                  (tables, metrics)
    │                                      │
    ▼                                      ▼
[Top-K Docs A]                      [Top-K Docs B]
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
         [Result Merger / Ranker]    ← Deduplicate + re-rank
                   │
                   ▼
         [Unified Prompt Builder]
                   │
                   ▼
                 [LLM]
                   │
                   ▼
               Response
```

---

### Components Breakdown

| Component         | Role                                                          |
|------------------|---------------------------------------------------------------|
| Query Router      | Classifies query → selects retriever(s) to invoke            |
| Retriever A       | Dense vector search over unstructured docs (semantic)         |
| Retriever B       | Structured/keyword search — SQL, BM25, metadata filter        |
| Retriever C       | (Optional) Web search / live API for real-time data           |
| Result Merger     | Deduplicates + re-ranks results across retrievers             |
| LLM               | Generates answer from merged, multi-source context            |

---

### Routing Strategies

| Strategy         | How it works                                          | Best for                     |
|-----------------|-------------------------------------------------------|------------------------------|
| Rule-based       | Keyword/regex patterns → route to retriever           | Fast, deterministic, simple  |
| Classifier model | Fine-tuned classifier predicts retriever label        | High-volume production       |
| LLM Router       | Prompt LLM to output retriever choice as JSON         | Flexible, handles nuance     |
| Parallel (all)   | Always call all retrievers, merge results             | Max recall, higher cost      |

---

### When to Use

- Enterprise assistants spanning docs + databases + APIs
- Legal/medical bots needing structured (case law DB) + unstructured (free text)
- E-commerce: product specs (structured) + reviews (unstructured) + live inventory
- Any system where query intent varies widely across different data modalities

---

### Limitations / Failure Cases

| Failure                      | Reason                                                          |
|------------------------------|-----------------------------------------------------------------|
| Misrouting                   | Router sends query to wrong retriever — bad context             |
| Result merging is hard        | How do you rank a SQL row vs a doc chunk on same scale?         |
| Latency compounds             | Parallel calls help but merging + re-ranking adds overhead      |
| Context explosion             | Multiple retrievers × top-K = huge context, hits token limits   |
| Router becomes bottleneck     | LLM router adds cost + latency on every query                   |
| Conflicting answers           | Different retrievers return contradictory information           |

---

## Quick Cheatsheet

| Pattern          | Stateful? | Multi-source? | Use When                                      |
|------------------|-----------|---------------|-----------------------------------------------|
| Simple RAG       | ❌        | ❌            | Single-turn Q&A over one corpus               |
| RAG with Memory  | ✅        | ❌            | Multi-turn chat over one corpus               |
| Branched RAG     | Optional  | ✅            | Mixed data types / multiple sources           |

---

## Repository Structure

```
rag-patterns/
│
├── README.md                        ← This file
│
├── 01_simple_rag/
│   └── simple_rag.py
│
├── 02_rag_with_memory/
│   └── rag_with_memory.py
│
└── 03_branched_rag/
    └── branched_rag.py
```

---

## Interview Q&A Summary

### Simple RAG
- **Q:** What is RAG and why is it better than pure LLM generation?
- **A:** RAG retrieves relevant documents at query time and injects them into the prompt, grounding the LLM in real, current, domain-specific facts. Pure LLM generation relies solely on parametric memory frozen at training time and cannot access private data without hallucinating.

- **Q:** What are the two failure modes you'd look for first?
- **A:** Retrieval failure (right doc not retrieved) and Generation failure (right doc retrieved but LLM ignores it). Log retrieved chunks alongside answers to isolate which layer failed.

- **Q:** How does chunking strategy affect RAG quality?
- **A:** Too large = exceeds context limits. Too small = loses context. Sweet spot: 256–512 tokens with 10–20% overlap, using sentence-aware splitting.

### RAG with Memory
- **Q:** Why can't you just pass full chat history into the retriever?
- **A:** History creates a noisy, unfocused embedding that doesn't match any specific chunk. Query rewriting distills history into one precise, targeted query.

- **Q:** How do you handle very long conversations?
- **A:** Sliding window (keep last N turns), summarization (compress old turns), or episodic memory (store history in vector store, retrieve relevant turns).

- **Q:** What's the tradeoff of using a query rewriter?
- **A:** Pro: improves retrieval for follow-ups. Con: adds a second LLM call per turn (~2x cost/latency). Mitigation: use a smaller model for rewriting.

### Branched RAG
- **Q:** How does a query router decide which retriever to use?
- **A:** Rule-based (keywords), classifier model (fast, robust), or LLM router (flexible, adds latency). Hybrid: rule-based with LLM fallback.

- **Q:** How do you merge results from different retrievers with incompatible scores?
- **A:** Reciprocal Rank Fusion (RRF): score = Σ 1/(k + rank_i) across retrievers. Normalizes across scoring systems without needing raw scores.

- **Q:** When to choose Branched RAG over a single vector store?
- **A:** When data modalities are incompatible in one embedding space — structured data loses queryability, live data can't be pre-indexed, mixed corpora hurt precision.




---

## 4. HyDE — Hypothetical Document Embeddings

### Concept & Intuition

**Problem it solves:**
In standard RAG, you embed the *user's query* and compare it against *document
embeddings*. But queries and documents live in different linguistic spaces:
- Query: `"What causes transformer attention to fail on long sequences?"`
- Document: `"Attention complexity scales quadratically with sequence length..."`

These two sentences are semantically related but *stylistically very different*.
The query is a question; the document is an answer. The embedding distance between
them can be surprisingly large — causing retrieval to miss highly relevant docs.

**Core Idea:**
Instead of embedding the raw query, use the LLM to *hallucinate a plausible
answer* (a "hypothetical document") first. Then embed *that answer* and search.
An imagined answer is stylistically closer to real documents than the original
question — dramatically improving retrieval alignment.

**Key Insight:**
> "A well-formed hypothetical answer lives in the same embedding space as real
> answers — even if its specific facts are wrong."

---

### Architecture — Step-by-Step Flow
```
User Query
    │
    ▼
[LLM — Hypothesis Generator]     ← "Write a passage that would answer: {query}"
    │
    ▼
Hypothetical Document (HyDoc)    ← May contain hallucinated facts — that's OK
    │
    ▼
[Embedding Model]                ← Embed the HyDoc, NOT the query
    │
    ▼
[Vector Store Search]            ← Search using HyDoc embedding
    │
    ▼
[Top-K Real Documents]           ← Retrieved real, grounded docs
    │
    ▼
[Prompt Builder]                 ← Original query + real retrieved docs
    │
    ▼
[LLM — Final Answer]             ← Answers from grounded real context
    │
    ▼
Response
```

> ⚠️ The HyDoc is used ONLY for retrieval. It is discarded before generation.
> The final LLM sees only the original query + real retrieved documents.

---

### Query Transformation vs Query Expansion

| Technique          | What it does                                      | HyDE uses? |
|--------------------|---------------------------------------------------|------------|
| Query Rewriting    | Rephrase query for clarity                        | ❌         |
| Query Expansion    | Add synonyms / related terms to query             | ❌         |
| Query Decomposition| Break complex query into sub-questions            | ❌         |
| **HyDE**           | **Generate a full hypothetical answer to embed**  | ✅         |

HyDE is a *query transformation* strategy — it transforms the query into a
document-shaped artifact before retrieval, bridging the query-document gap.

---

### Why Direct Query Embedding Sometimes Fails

1. **Asymmetric encoding**: Short queries vs long documents have different
   statistical properties in embedding space
2. **Intent gap**: "What is X?" and "X is defined as..." have low cosine
   similarity despite being semantically aligned
3. **Domain terminology**: Domain docs use jargon; user queries use plain language
4. **Cross-lingual mismatch**: Multilingual settings amplify the gap

---

### Hallucination vs Useful Hypothesis Tradeoff

| HyDoc Quality   | Impact on Retrieval                                         |
|----------------|--------------------------------------------------------------|
| Good hypothesis | Embeds close to correct document → excellent retrieval       |
| Wrong facts     | Still OK — style/topic alignment drives retrieval, not facts |
| Completely off  | Rare, but can retrieve irrelevant docs                       |
| Confident wrong | Dangerous if HyDoc somehow bleeds into generation prompt     |

**Mitigation**: Always discard the HyDoc before the generation step. Never
inject a hypothetical document into the final prompt.

---

### When to Use

- Long-form document retrieval (research papers, legal texts, technical manuals)
- Domain-specific corpora where query and doc language diverge significantly
- Cross-lingual retrieval (generate hypothesis in document's language)
- When you have a powerful LLM available for the hypothesis step

---

### Limitations / Failure Cases

| Failure                        | Reason                                                           |
|--------------------------------|------------------------------------------------------------------|
| Adds LLM call cost + latency   | Every query requires an extra generation step before retrieval   |
| Hypothesis is totally wrong    | Weak LLM generates off-topic hypothesis → retrieval fails        |
| Overkill for simple queries    | "What is Python?" doesn't need hypothesis generation             |
| Non-determinism                | Same query → different hypotheses → inconsistent retrieval       |
| HyDoc injected into prompt     | Common implementation bug — hallucinations corrupt generation    |

### INTERVIEW Q&A — HyDE

Q1. What problem does HyDE solve that standard RAG cannot?

Standard RAG compares a short natural-language question against long answer-shaped document chunks. These live in different statistical regions of embedding space even when semantically related — a phenomenon called the query-document asymmetry gap. HyDE bridges this by first generating a full-length, answer-shaped hypothesis using an LLM, then embedding that for retrieval. The hypothesis mirrors the style and vocabulary of real documents, pulling retrieval vectors into the right neighborhood even when the LLM's specific facts are wrong.


Q2. If the hypothesis contains hallucinated facts, doesn't that corrupt the final answer?

No — the hypothesis is used only to find the right documents. It is explicitly discarded before the generation step. The final LLM receives the original user query plus real retrieved documents. Think of it as a search probe: its job is to navigate embedding space, not to provide facts. A well-shaped but factually wrong hypothesis can still find the right neighborhood in the vector store.


Q3. When would you NOT use HyDE?

Three cases: (1) Simple factual queries where the query already uses document-like language. (2) Latency-sensitive systems — HyDE adds a full LLM call before retrieval. (3) Weak LLMs that generate off-topic hypotheses — the hypothesis then points retrieval away from relevant documents. The rule of thumb: use HyDE when retrieval precision is poor and you have a capable LLM; skip it when the retrieval gap is small or speed is critical.


---

## 5. Adaptive RAG

### Concept & Intuition

**Problem it solves:**
Not all questions benefit from retrieval. Asking "What is 2 + 2?" through a
full RAG pipeline wastes time and money. Conversely, asking "What were our Q3
sales?" without retrieval produces hallucinations. Adaptive RAG adds a
*decision layer* that routes each query to the right strategy:
- No retrieval (LLM answers from parametric knowledge)
- Standard retrieval (single-shot RAG)
- Iterative retrieval (multi-step reasoning over multiple retrieved chunks)

**Core Idea:**
Classify the query *before* deciding whether and how to retrieve. Retrieval
is expensive — make it conditional on actual need.

**Key Insight:**
> "Retrieval is a tool, not a requirement. A smart system knows when not to use it."

---

### Architecture — Step-by-Step Flow
```
User Query
    │
    ▼
[Query Classifier]              ← What type of query is this?
    │
    ├─── General Knowledge ──────────────────────► [LLM Direct]──► Response
    │         (no retrieval needed)
    │
    ├─── Factual / Domain ───────────────────────► [Standard RAG]──► Response
    │         (single retrieval pass)
    │
    └─── Complex / Multi-hop ────────────────────► [Iterative RAG]
              (requires reasoning                        │
               across multiple docs)              [Retrieve → Reason → Retrieve]
                                                        │
                                                      Response
```

---

### Query Classification

| Query Type       | Examples                              | Strategy          | Why                             |
|-----------------|---------------------------------------|-------------------|---------------------------------|
| General knowledge| "What is photosynthesis?"             | LLM Direct        | LLM already knows this          |
| Current events   | "What happened in the 2024 election?" | Web RAG           | LLM knowledge is stale          |
| Domain-specific  | "Summarize our return policy"         | Standard RAG      | Private/domain data needed      |
| Multi-hop        | "Compare our Q3 sales vs competitor"  | Iterative RAG     | Requires multiple retrieval steps|
| Conversational   | "Thanks!" / "Tell me more"            | Memory / History  | Not a retrieval task             |

---

### Classification Strategies

| Method            | How                                        | Tradeoff                           |
|-------------------|--------------------------------------------|------------------------------------|
| Keyword rules     | Regex patterns for date/entity/etc.        | Fast, brittle                      |
| Small classifier  | Fine-tuned BERT/logistic regression        | Fast, robust, needs labeled data   |
| LLM prompt        | Ask LLM to classify before answering       | Flexible, adds latency             |
| Self-RAG signal   | LLM predicts its own confidence            | Elegant, requires fine-tuned model |

---

### Cost vs Latency Tradeoff
```
Strategy          Cost      Latency    Accuracy
─────────────────────────────────────────────────
LLM Direct         Low       Fast       High (general knowledge)
Standard RAG       Medium    Medium     High (factual/domain)
Iterative RAG      High      Slow       Highest (complex queries)
```

The classifier itself must be *cheap* — otherwise you spend more classifying
than you save by skipping retrieval.

---

### When to Use

- High-traffic production systems where retrieval cost is significant
- Mixed query workloads (conversational + factual + domain-specific)
- Systems with strict latency SLAs that can't afford retrieval on every call
- Any product where >30% of queries are answerable without retrieval

---

### Limitations / Failure Cases

| Failure                        | Reason                                                          |
|--------------------------------|-----------------------------------------------------------------|
| Classifier misroutes           | Sends domain query to "LLM Direct" → hallucination             |
| False confidence               | LLM answers confidently from memory when it should retrieve     |
| Complexity underestimation     | Routes multi-hop query to single RAG → incomplete answer        |
| Classifier adds latency        | If using LLM classifier, partially negates the savings          |
| Training data dependency       | Fine-tuned classifiers need labeled query type data             |


### INTERVIEW Q&A — Adaptive RAG
Q1. Why does every RAG system eventually need adaptive routing?

At scale, query workloads are never homogeneous. Some questions are general knowledge, some need live retrieval, some require multi-hop reasoning. Applying the same retrieval strategy to all query types is wasteful (over-retrieves for simple queries) and inaccurate (under-retrieves for complex ones). Adaptive RAG is essentially the query planner — it minimizes cost while maximizing accuracy by matching strategy to query complexity.


Q2. What is the risk of defaulting to "LLM Direct" for ambiguous queries?

The LLM may answer confidently from stale or incorrect parametric memory — a subtle hallucination that's hard to detect because it sounds correct. The safer default is always "Standard RAG": if a document exists and is relevant, grounding is almost always better than trusting LLM memory. Reserve "Direct" only for unambiguously general-knowledge queries with high classifier confidence.


Q3. How would you implement query classification in production without an LLM router?

Train a lightweight text classifier (distilBERT or even logistic regression over TF-IDF) on a labeled set of query types. Collect 500–1000 examples of DIRECT / RAG / ITERATIVE queries from your production logs, label them, fine-tune. This classifier runs in <5ms vs ~200ms for an LLM router. Combine with a rule-based first pass (keyword matching) to handle the clearest cases instantly, with the ML classifier handling ambiguous ones.


---

## 6. Corrective RAG (CRAG)

### Concept & Intuition

**Problem it solves:**
Standard RAG blindly trusts the retriever. It always passes the top-K results
to the LLM — even when those results are irrelevant, outdated, or wrong. The
LLM then either hallucinates an answer or fabricates connections between the
query and irrelevant context. CRAG adds a **validation gate** between retrieval
and generation: check whether what was retrieved is actually useful before
using it.

**Core Idea:**
After retrieval, run a *relevance evaluator* on each retrieved document. Based
on confidence scores, decide whether to: (1) use the documents as-is, (2)
selectively filter and supplement, or (3) discard entirely and re-retrieve via
web search or query rewriting.

**Key Insight:**
> "Bad context is worse than no context. A validation gate prevents the LLM
> from confidently answering with irrelevant evidence."

---

### Architecture — Step-by-Step Flow
```
User Query
    │
    ▼
[Retriever]                      ← Standard dense retrieval (top-K docs)
    │
    ▼
[Relevance Evaluator]            ← Score each doc: Correct / Ambiguous / Wrong
    │
    ├── ALL CORRECT ─────────────────────────────► [Refine Docs]──► [LLM]──► Response
    │   (high confidence)               ↑
    │                           Knowledge Refinement
    │                           (strip irrelevant sentences)
    │
    ├── AMBIGUOUS ──────────────► [Web Search]──► [Merge Results]──► [LLM]──► Response
    │   (mixed confidence)       (supplement with
    │                             live web results)
    │
    └── ALL WRONG ──────────────► [Query Rewriter]──► [Web Search]──► [LLM]──► Response
        (low confidence)         (reformulate query,
                                  re-retrieve externally)
```

---

### Retrieval Validation Techniques

| Technique               | How It Works                                              | Speed   |
|------------------------|-----------------------------------------------------------|---------|
| Cosine score threshold  | If similarity < threshold → mark as wrong                 | Fast    |
| Cross-encoder reranker  | Score (query, doc) pair jointly — much more accurate      | Medium  |
| LLM evaluator           | "Is this document relevant to this query? Yes/No/Partial" | Slow    |
| NLI classifier          | Check if doc entails / contradicts / is neutral to query  | Medium  |

**Production recommendation**: Cross-encoder for precision, cosine threshold
as a fast pre-filter to reduce cross-encoder calls.

---

### Confidence Scoring
```
Score Range    Classification    Action
──────────────────────────────────────────────────────────
0.8 – 1.0      CORRECT          Use doc, optionally refine
0.4 – 0.8      AMBIGUOUS        Supplement with web search
0.0 – 0.4      WRONG            Discard, rewrite query, re-retrieve
```

Thresholds are tunable per domain. Medical/legal contexts demand higher
thresholds than general knowledge assistants.

---

### Retry Strategies

| Strategy           | When to Use                              | Risk                            |
|-------------------|------------------------------------------|---------------------------------|
| Use as-is          | High confidence docs                     | Low                             |
| Knowledge refine   | High confidence but noisy docs           | Removes useful adjacent context |
| Web supplement     | Ambiguous — partial coverage             | Adds latency, possible noise    |
| Query rewrite      | Complete retrieval miss                  | May still miss; adds LLM call   |
| Fallback to direct | All strategies fail                      | Hallucination risk              |

---

### When to Use

- High-stakes Q&A where wrong context is dangerous (medical, legal, finance)
- Domains with rapidly changing facts where the vector index may be stale
- Any system where you can log retrieval failures and attribution is important
- When you want measurable retrieval quality metrics in production

---

### Limitations / Failure Cases

| Failure                          | Reason                                                      |
|----------------------------------|-------------------------------------------------------------|
| Evaluator itself is wrong        | Classifier may incorrectly score relevant docs as wrong     |
| Web search adds noise            | External results may be low quality or off-topic            |
| Cascade latency                  | Validate → retry → web search → merge = many round trips    |
| Over-correction                  | Discarding ambiguous docs loses partially useful context    |
| Threshold sensitivity            | Wrong threshold kills precision or recall of validation     |


### INTERVIEW Q&A — Corrective RAG

**Q1. Why is CRAG considered a major leap over standard RAG architectures?**

> Standard RAG has no feedback loop — it retrieves and blindly generates regardless of retrieval quality. CRAG introduces a *validation gate* modeled after how a careful researcher works: read what you found, assess if it's relevant, and if not, go look somewhere else. This breaks the silent failure mode of standard RAG where irrelevant context produces fluent but wrong answers — the hardest failure to detect in production.

---

**Q2. How do you implement the relevance evaluator in production?**

> Three tiers by accuracy vs speed: **(1) Cosine threshold** — fast pre-filter, low accuracy, use as a gating check. **(2) Cross-encoder** (e.g., `ms-marco-MiniLM`) — scores (query, doc) pairs jointly with much higher accuracy than bi-encoders, ~20ms per pair, production-viable. **(3) LLM evaluator** — highest accuracy, can provide rationale, but ~200ms and costly at scale. Recommended stack: cosine pre-filter → cross-encoder for survivors → LLM evaluator only for high-stakes ambiguous cases.

---

**Q3. What is the "knowledge refinement" step in CRAG and why does it matter?**

> Even when a document is marked CORRECT, it may contain irrelevant surrounding sentences that dilute the signal or confuse the LLM. Knowledge refinement splits retrieved documents into sentence-level chunks, scores each sentence independently against the query, and strips low-scoring sentences before injecting into the prompt. This reduces context noise, compresses token usage, and improves generation precision — especially important when retrieved documents are long or loosely structured.

---

## Updated Repository Structure
```
rag-patterns/
│
├── README.md                           ← Full guide (all 6 patterns)
│
├── 01_simple_rag/
│   └── simple_rag.py
│
├── 02_rag_with_memory/
│   └── rag_with_memory.py
│
├── 03_branched_rag/
│   └── branched_rag.py
│
├── 04_hyde_rag/
│   └── hyde_rag.py
│
├── 05_adaptive_rag/
│   └── adaptive_rag.py
│
└── 06_corrective_rag/
    └── corrective_rag.py