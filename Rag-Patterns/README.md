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

---

## 7. Self-RAG

### Concept & Intuition

**Problem it solves:**
Every RAG pattern so far makes a critical assumption: *the LLM accepts
retrieved context passively and generates an answer*. But what if the LLM
could evaluate its own retrieval need, critique the quality of what was
retrieved, and assess whether its own generated answer is faithful to the
evidence? Self-RAG makes the LLM an *active participant* in the RAG loop —
not just a consumer of retrieved context.

**Why previous approaches aren't enough:**
- Simple RAG: blind trust in retrieval + generation
- CRAG: external evaluator validates retrieval, but generation is still passive
- Adaptive RAG: classifies query type, but doesn't reflect on output quality
- None of these ask: *"Is my answer actually supported by the evidence?"*

**Core Idea:**
Train (or prompt) the LLM to generate special **reflection tokens** inline
with its output. These tokens express the model's own judgment about:
1. *Should I retrieve at all?*
2. *Is this retrieved document relevant?*
3. *Is my generated passage supported by this document?*
4. *Is my final answer useful?*

**Key Insight:**
> "Self-RAG turns the LLM into both the writer and the editor — it generates
> and simultaneously critiques its own output using reflection tokens."

---

### Architecture — Step-by-Step Flow
```
User Query
    │
    ▼
[LLM — Retrieve Decision]        ← Generates [Retrieve] or [No Retrieve] token
    │
    ├── [No Retrieve] ───────────────────────────► [LLM generates directly]
    │                                                       │
    │                                               [IsUSEFUL? token]
    │                                                       │
    │                                               Final Response
    │
    └── [Retrieve] ──────────────────────────────► [Retriever → Top-K Docs]
                                                            │
                                              ┌─────────────▼──────────────┐
                                              │  For each retrieved doc:    │
                                              │  [ISREL] — is doc relevant? │
                                              │  [ISSUP] — does generated   │
                                              │            passage use doc?  │
                                              │  [ISUSE] — is output useful? │
                                              └─────────────┬──────────────┘
                                                            │
                                              [Score + Rank all candidates]
                                                            │
                                                    Best Response
```

---

### Reflection Tokens (The Core Mechanism)

| Token       | Question Asked                                        | Values                          |
|-------------|-------------------------------------------------------|---------------------------------|
| `[Retrieve]`| Should retrieval happen for this query?               | `yes` / `no` / `continue`       |
| `[ISREL]`   | Is this retrieved document relevant to the query?     | `relevant` / `irrelevant`       |
| `[ISSUP]`   | Is the generated passage supported by the document?   | `fully` / `partially` / `no`    |
| `[ISUSE]`   | Is the final response useful to the user?             | Score 1–5                       |

---

### Self-Evaluation / Critique Loop
```
For each retrieved document d_i:
    1. Generate candidate passage p_i using (query + d_i)
    2. Generate [ISREL] token: is d_i relevant?
    3. Generate [ISSUP] token: is p_i supported by d_i?
    4. Generate [ISUSE] token: is p_i useful?
    5. Compute composite score: score_i = w1*ISREL + w2*ISSUP + w3*ISUSE

Select passage with highest composite score → Final Response
```

---

### Iterative Refinement

Self-RAG can run in multiple passes:
1. **Pass 1**: Generate initial answer with reflection tokens
2. **Critique**: If [ISSUP] = "partially" or [ISUSE] < 3 → trigger refinement
3. **Pass 2**: Retrieve additional documents, regenerate with critique context
4. **Stop condition**: [ISSUP] = "fully" AND [ISUSE] >= 4, OR max iterations hit

---

### Confidence Scoring
```
Composite Score Formula:
score = α × P(ISREL=relevant) + β × P(ISSUP=fully) + γ × P(ISUSE=high)

where α + β + γ = 1
Typical weights: α=0.2, β=0.5, γ=0.3
(faithfulness/ISSUP weighted highest — hallucination prevention)
```

---

### When to Use

- High-stakes generation where hallucination is unacceptable (medical, legal)
- Long-form answer generation that requires multi-source synthesis
- Systems that need explainable, auditable generation with faithfulness scores
- When you want the model itself to be the quality gate

---

### Limitations / Failure Cases

| Failure                         | Reason                                                          |
|---------------------------------|-----------------------------------------------------------------|
| Requires fine-tuned model       | Reflection tokens need special training — can't use any LLM    |
| Slow (multiple generation passes)| Each doc requires a separate generation + scoring pass        |
| Self-critique is unreliable     | LLM may score its own hallucinations as "fully supported"      |
| Prompt-based simulation is weak | Without fine-tuning, reflection tokens are easy to fake        |
| High token cost                 | Multiple candidate generations × top-K docs = large token bill |


✅ INTERVIEW Q&A — Self-RAG
Q1. How does Self-RAG differ from CRAG in its approach to quality control?

CRAG uses an external evaluator model to validate retrieved documents before generation. Self-RAG internalizes evaluation — the same LLM that generates the answer also generates reflection tokens that assess retrieval relevance, generation faithfulness, and output usefulness inline during generation. CRAG is modular (swap the evaluator independently); Self-RAG is tightly coupled (requires a fine-tuned model) but more elegant since evaluation and generation happen in a single forward pass.


Q2. What makes the [ISSUP] token the most important reflection token?

[ISSUP] directly measures hallucination risk. If a generated passage is "not supported" by the retrieved document, the model fabricated the content — it's a hallucination by definition. [ISREL] only tells us about the document's value, and [ISUSE] measures user experience. But [ISSUP] is the faithfulness gate: a passage that is relevant and useful but not supported is actively dangerous. This is why ISSUP carries the highest weight (0.5) in the composite scoring formula.


Q3. Can you implement Self-RAG without fine-tuning the LLM?

You can approximate it with prompting — instruct the LLM to output structured JSON with reflection scores alongside its answer. But this is brittle: the model wasn't trained to generate these tokens reliably, so the scores are often miscalibrated or inconsistent. True Self-RAG requires supervised fine-tuning on a dataset where reflection tokens are annotated by a stronger model or human judges. The prompt-based version is useful for prototyping but should not be trusted for production quality control.


---

## 8. Agentic RAG

### Concept & Intuition

**Problem it solves:**
All previous RAG patterns follow a fixed pipeline: query → retrieve → generate.
Even adaptive and corrective variants are *reactive* — they respond to a single
query with at most a few retrieval passes. But real-world questions often require
*plans*: "Research competitor pricing, compare it to ours, identify gaps, and
draft a strategy memo." No single retrieval pass can answer this. The system
needs to *plan*, *decompose*, *act*, *observe*, and *iterate* — like an agent.

**Why previous approaches aren't enough:**
- Single-pass RAG: one query, one retrieval, one answer
- Iterative RAG: multiple retrievals but on the same query
- None plan across multiple tasks, use multiple tools, or decide when to stop

**Core Idea:**
The LLM becomes an **agent** with retrieval as one of many tools. The agent
follows a Reasoning + Acting (ReAct) loop: reason about what to do next,
act using a tool, observe the result, reason again — until the task is complete.

**Key Insight:**
> "In Agentic RAG, retrieval is not the pipeline — retrieval is a *tool the
> agent calls when it decides it needs information*."

---

### Architecture — Step-by-Step Flow
```
User Task (complex, multi-step)
    │
    ▼
[Agent — Task Planner]           ← Decomposes task into sub-goals
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  AGENT LOOP (ReAct)                  │
│                                                      │
│  [Thought] ←── What do I need to do next?            │
│      │                                               │
│      ▼                                               │
│  [Action] ←── Which tool should I call?              │
│      │                                               │
│      ├── retrieve(query)     ← Vector search         │
│      ├── web_search(query)   ← Live web              │
│      ├── sql_query(query)    ← Structured DB         │
│      ├── code_executor(code) ← Run calculations      │
│      └── summarize(docs)     ← Compress context      │
│                                                      │
│  [Observation] ←── Tool returns result               │
│      │                                               │
│      ▼                                               │
│  [Thought] ←── Do I have enough? What's next?        │
│      │                                               │
│      └── [Repeat or FINISH]                          │
└─────────────────────────────────────────────────────┘
    │
    ▼
Final Synthesized Response
```

---

### Planning vs Execution

| Layer      | Role                                                        | Example                              |
|-----------|-------------------------------------------------------------|--------------------------------------|
| Planner    | Decomposes the goal into ordered sub-tasks                  | "Step 1: find our pricing. Step 2…"  |
| Executor   | Calls tools and collects observations                       | `retrieve("enterprise pricing")`     |
| Memory     | Maintains scratchpad of all thoughts + observations         | Full ReAct trace                     |
| Stopper    | Decides when the task is sufficiently complete              | "I now have enough to answer."       |

---

### Decision-Making Loop: When to Retrieve vs When to Stop
```
RETRIEVE when:
  - Required information is not in current context
  - A previous retrieval returned low-relevance results
  - A sub-task requires new factual grounding

STOP when:
  - All sub-tasks in the plan are complete
  - Enough context exists to answer fully
  - Max iteration limit reached (safety guardrail)
  - LLM generates "Final Answer:" token
```

---

### Multi-Step Reasoning Example
```
Task: "Compare our Q3 sales to competitor XYZ and suggest 3 strategies."

Thought 1: I need our Q3 sales data.
Action 1: retrieve("Q3 sales internal report")
Observation 1: "Our Q3 revenue was $4.2B, up 18% YoY."

Thought 2: I need competitor XYZ's Q3 data.
Action 2: web_search("XYZ competitor Q3 2024 earnings")
Observation 2: "XYZ reported $3.1B in Q3, up 8% YoY."

Thought 3: I have both. Now I need strategy frameworks.
Action 3: retrieve("competitive strategy frameworks")
Observation 3: "Porter's 5 Forces... Blue Ocean... etc."

Thought 4: I have enough. Synthesize the answer.
Action 4: FINISH → Generate comparative analysis + 3 strategies
```

---

### When to Use

- Multi-step research tasks spanning multiple data sources
- Autonomous report generation (gather + synthesize + write)
- Customer support bots that must query CRM, docs, and policy simultaneously
- Any task that a junior analyst would need multiple steps to complete

---

### Limitations / Failure Cases

| Failure                        | Reason                                                          |
|--------------------------------|-----------------------------------------------------------------|
| Infinite loops                 | Agent keeps retrieving without converging — needs max_iter cap |
| Planning hallucination         | Agent plans steps that don't lead to the goal                   |
| Tool call errors               | Tool returns error; agent must handle gracefully or abort       |
| Context window exhaustion      | Long ReAct traces + multiple retrievals exceed token limit      |
| Latency                        | 5-10 tool calls × retrieval latency = slow response            |
| Unpredictable behavior         | Hard to test — agent path depends on LLM stochasticity          |

### INTERVIEW Q&A — Agentic RAG
Q1. What is the fundamental difference between Agentic RAG and Iterative RAG?

Iterative RAG performs multiple retrieval passes on a single query, each pass refining the same question. Agentic RAG has a planner that decomposes a task into different sub-goals, calls different tools for each, and synthesizes across all observations. Iterative RAG is a loop over one question; Agentic RAG is a graph of decisions over a complex task. The agent also decides which tool to use — retrieval is just one option alongside web search, calculators, and APIs.


Q2. How do you prevent an agent from looping infinitely?

Three hard stops: (1) Max iteration cap — hard limit (e.g., 10 steps) regardless of task completion. (2) Repetition detection — if the agent calls the same tool with the same input twice, abort that branch. (3) Confidence gate — if the LLM generates a FINISH token or a "I have enough information" signal, stop immediately. In production, combine all three. The max iteration cap is the most critical safety net — it prevents infinite loops from burning API budget.


Q3. How do you design the tool interface for an Agentic RAG system?

Each tool should be a function with: a clear name, a one-sentence description the LLM uses for selection, typed inputs/outputs, and error handling that returns a structured error message rather than raising an exception. The LLM selects tools based on the description — so description quality directly determines routing accuracy. Tools should be atomic and composable: a retrieve tool and a summarize tool are better than a single retrieve_and_summarize tool because the agent can choose to summarize or not based on context.


---

## 9. Multimodal RAG

### Concept & Intuition

**Problem it solves:**
Real-world knowledge isn't just text. Product catalogs have images. Research
papers have charts. Medical records have scans. Legal documents have tables
and diagrams. Standard RAG operates entirely in text space — it cannot retrieve
based on visual content or answer queries that require understanding images
alongside text.

**Why previous approaches aren't enough:**
- All previous RAG patterns embed only text
- A query like "Find products that look like this image" is completely impossible
- Even "What does Figure 3 in this paper show?" requires visual understanding

**Core Idea:**
Map multiple modalities (text, images, audio, video) into a *shared embedding
space* where cross-modal similarity search becomes possible. A text query can
retrieve images, and an image query can retrieve text — because they live in
the same vector space.

**Key Insight:**
> "Multimodal RAG treats modality as a translation problem: everything gets
> mapped to a common language of vectors where semantic similarity is
> modality-agnostic."

---

### Architecture — Step-by-Step Flow
```
User Input (text query OR image OR both)
    │
    ▼
[Multimodal Encoder]             ← CLIP / LLaVA / ImageBind
    │                            Maps input to shared embedding space
    ▼
Unified Query Vector
    │
    ▼
[Multimodal Vector Store]        ← Contains text embeddings + image embeddings
    │                            All in the same space
    ▼
[Top-K Mixed Results]            ← May return: text chunks, images, tables
    │
    ▼
[Context Assembler]              ← Convert non-text modalities to LLM-readable form
    │                               Images → captions / base64
    │                               Tables → markdown
    │                               Charts → description
    ▼
[Multimodal LLM]                 ← GPT-4V, LLaVA, Claude 3 (vision-capable)
    │
    ▼
Response (text, or text + image references)
```

---

### Cross-Modal Embeddings — CLIP Intuition
```
CLIP Training Objective:
──────────────────────────────────────────────────────────
Image: [photo of a dog running]   ──► Image Encoder ──► Vector A
Text:  "A dog running in a park"  ──► Text Encoder  ──► Vector B

Training: Minimize distance(Vector A, Vector B)
          Maximize distance(Vector A, Vector of unrelated text)
──────────────────────────────────────────────────────────

Result: Images and their descriptions end up NEAR each other in vector space.
        "A dog" image and "dog" text share a neighborhood — cross-modal search works.
```

---

### The Alignment Problem (Text ↔ Image Space)

| Challenge                  | Description                                              | Mitigation                         |
|---------------------------|----------------------------------------------------------|------------------------------------|
| Modality gap               | Text and image vectors cluster separately in joint space | Better contrastive training        |
| Fine-grained visual detail | Embeddings miss small but critical visual differences    | Higher-resolution patch encoders   |
| Abstract concepts          | "Justice" or "irony" hard to ground in images            | Use text-primary retrieval          |
| Domain shift               | CLIP trained on web data → poor on medical/legal images  | Domain fine-tuning                  |
| Asymmetric quality         | Text descriptions richer than visual encoding            | Hybrid text+image indexing          |

---

### Indexing Strategies

| Strategy              | How                                              | Best for                         |
|----------------------|--------------------------------------------------|----------------------------------|
| Image embeddings only | Embed raw images using CLIP vision encoder       | Visual search, product images    |
| Caption-based         | Generate captions, embed captions as text        | When text retrieval is stronger  |
| Dual index            | Separate text + image indices, merged at query   | High precision, complex docs     |
| Hybrid (text+image)   | Concatenate or average text+image embeddings     | Documents with both modalities   |

---

### When to Use

- E-commerce: "Find products similar to this image"
- Medical imaging: retrieve relevant case studies given a scan
- Technical documentation: "Show me diagrams related to this architecture"
- Legal/financial: retrieve charts and tables alongside text evidence
- Manufacturing: defect detection with visual + specification text lookup

---

### Limitations / Failure Cases

| Failure                        | Reason                                                          |
|--------------------------------|-----------------------------------------------------------------|
| Modality gap persists          | Text and image vectors not perfectly aligned even with CLIP     |
| High indexing cost             | Encoding images is expensive vs text                            |
| Storage overhead               | Image embeddings + thumbnails = large index                     |
| Weak fine-grained retrieval    | CLIP misses subtle visual differences (two similar products)    |
| LLM context limits             | Multiple high-res images exhaust token/context budget rapidly   |
| Domain mismatch                | General CLIP model fails on specialized domains (X-rays, MRI)  |


### INTERVIEW Q&A — Multimodal RAG
Q1. How does CLIP enable cross-modal retrieval without explicit image-text matching?

CLIP is trained with contrastive learning on 400M image-text pairs from the web. For each pair, it minimizes the distance between the image vector and text vector while maximizing distance to all other pairs in the batch. After training, the image encoder and text encoder both produce vectors in the same 512-dimensional space where semantic similarity is preserved across modalities. This means "a dog running" as text and an actual photo of a dog running land in the same neighborhood — enabling a text query to retrieve relevant images without any explicit linking.


Q2. What is the "modality gap" and how does it affect retrieval quality?

Even after CLIP training, text and image vectors don't fully intermingle — they form two separate clusters in the shared space with a measurable gap between them. This means cross-modal similarity scores are systematically lower than within-modal scores, making it harder to rank text and image results on the same scale. Mitigations: (1) apply modality-specific score normalization, (2) use separate indices and merge results via RRF, (3) fine-tune CLIP on domain-specific data to better align modalities.


Q3. How would you design a Multimodal RAG system for a medical imaging use case?

You cannot use off-the-shelf CLIP — it's trained on natural images and fails on X-rays and MRI scans. Design: (1) Fine-tune a medical CLIP model on radiology report + scan pairs using contrastive learning. (2) Build a dual index: one for scan embeddings, one for report text. (3) At query time, encode the patient scan through the medical vision encoder, retrieve similar historical cases by both visual similarity AND text (symptoms, diagnoses). (4) Pass top-K images + reports to a vision LLM (GPT-4V or Med-PaLM) for generation. (5) Add HIPAA-compliant metadata filtering to restrict access by department.


---

## 10. Graph RAG

### Concept & Intuition

**Problem it solves:**
Vector databases excel at finding semantically similar isolated chunks. But
they fundamentally cannot answer questions about *relationships*: "Who reports
to the CEO?", "Which drugs interact with this compound?", "How is Company A
connected to the tax scandal through Company B?" These are **graph questions** —
they require traversing connections between entities, not measuring similarity
between vectors.

**Why previous approaches aren't enough:**
- Dense retrieval: finds similar text, not connected entities
- Sparse retrieval: keyword match, no structural traversal
- Neither can answer: "Find all diseases caused by genes that interact with BRCA1"

**Core Idea:**
Build a **knowledge graph** from documents: extract entities (nodes) and
relationships (edges), store them in a graph database, and at query time
*traverse the graph* to gather multi-hop context before passing to the LLM.
Vector similarity finds the *entry point* into the graph; graph traversal
provides the *connected context* that embeddings miss.

**Key Insight:**
> "Vector search finds WHAT is relevant. Graph traversal finds HOW things are
> CONNECTED. Graph RAG combines both — enter the graph by similarity,
> explore it by relationship."

---

### Architecture — Step-by-Step Flow
```
Source Documents
    │
    ▼
[Entity + Relationship Extractor]   ← NER + relation extraction (LLM or spaCy)
    │
    ├── Entities: [Apple Inc, Tim Cook, iPhone 15, AAPL]
    └── Relations: [Tim Cook]─CEO_OF→[Apple Inc], [iPhone 15]─MADE_BY→[Apple Inc]
    │
    ▼
[Knowledge Graph DB]               ← Neo4j, NetworkX, Amazon Neptune
    │
    (also embed entities/nodes for vector entry)
    ▼
[Vector Index over entity embeddings]

─────────────────────────────────────────────────────

User Query
    │
    ▼
[Query Entity Extractor]           ← "What companies did Tim Cook work at before Apple?"
    │                               Extracts: [Tim Cook]
    ▼
[Vector Search over Entity Index]  ← Find entry node: Tim_Cook_node
    │
    ▼
[Graph Traversal]                  ← Hop outward: Tim Cook → WORKED_AT → [companies]
    │                               Control: depth, direction, relationship type filter
    ▼
[Subgraph / Path Collection]       ← Relevant nodes + edges gathered
    │
    ▼
[Context Serializer]               ← Convert graph paths to text: "Tim Cook → CEO_OF → Apple Inc"
    │
    ▼
[LLM]                              ← Reads graph-structured context
    │
    ▼
Response
```

---

### Knowledge Graphs vs Vector Databases

| Dimension              | Vector DB                          | Knowledge Graph                        |
|-----------------------|------------------------------------|----------------------------------------|
| Data structure         | Dense vectors (chunks)             | Nodes + typed edges (entities)         |
| Query type             | "Similar to X"                     | "Connected to X via relationship Y"    |
| Multi-hop reasoning    | ❌ Not native                       | ✅ Graph traversal (BFS/DFS)           |
| Relationship modeling  | Implicit (via co-occurrence)        | Explicit (typed edges)                 |
| Structured queries     | Approximation only                 | Precise (Cypher, SPARQL, Gremlin)      |
| Construction cost      | Low (embed chunks)                 | High (extract + validate entities)     |
| Maintenance            | Add new embeddings                 | Add/update nodes + edges carefully     |
| Best for               | Semantic similarity                | Relational + hierarchical reasoning    |

---

### Entity + Relationship Modeling
```
Entities (Nodes):
  Person: Tim Cook, Satya Nadella
  Company: Apple Inc, Microsoft
  Product: iPhone 15, Azure
  Event: WWDC 2024, Q3 Earnings

Relationships (Edges):
  Tim Cook ──[CEO_OF]──────────► Apple Inc
  iPhone 15 ──[MADE_BY]──────► Apple Inc
  Apple Inc ──[COMPETITOR_OF]──► Microsoft
  Tim Cook ──[SPOKE_AT]────────► WWDC 2024
  Azure ──[OWNED_BY]───────────► Microsoft

Graph query (Cypher):
  MATCH (p:Person)-[:CEO_OF]->(c:Company)-[:COMPETITOR_OF]->(rival:Company)
  WHERE p.name = "Tim Cook"
  RETURN rival.name
  → "Microsoft"
```

---

### Traversal vs Similarity Search

| Approach           | Mechanism                        | Answers                                      |
|-------------------|----------------------------------|----------------------------------------------|
| Similarity search  | Cosine distance in vector space  | "What is semantically similar to X?"         |
| Graph traversal    | BFS / DFS / Dijkstra on edges    | "What is structurally connected to X?"       |
| Combined (GraphRAG)| Entry via vector, explore via graph | "Find X, then explore all connections"    |

---

### When Graphs Outperform Embeddings

- Multi-hop reasoning: "Company A → invested by → VC B → also invested → Company C"
- Hierarchical queries: org charts, taxonomies, class hierarchies
- Compliance/audit trails: "Who approved X, who reports to them, what policies apply?"
- Drug interactions: compound → interacts_with → enzyme → regulates → disease
- Any domain where *relationship type* matters as much as *semantic content*

---

### When to Use

- Knowledge management over highly structured domains (legal, medical, finance)
- Enterprise systems needing org chart + policy traversal
- Fraud detection: entity relationship traversal across transactions
- Research assistants needing citation graph + concept hierarchy navigation
- Any system where "how are these things related?" is a primary query type

---

### Limitations / Failure Cases

| Failure                        | Reason                                                          |
|--------------------------------|-----------------------------------------------------------------|
| Graph construction is expensive| Entity extraction + validation + deduplication is costly        |
| Entity resolution errors       | "Apple Inc" and "Apple" treated as different nodes              |
| Relationship extraction noise  | LLM-extracted relations contain errors at scale                 |
| Graph staleness                | New documents require re-extraction, not just re-embedding      |
| Sparse graphs miss connections | Relationships only exist if explicitly extracted from text      |
| Traversal explosion            | Deep multi-hop traversal returns too many nodes                 |


### ✅ INTERVIEW Q&A — Graph RAG

**Q1. When would you recommend Graph RAG over standard vector RAG?**

> The decisive signal is whether the query is *relational* or *semantic*. If users ask "What is similar to X?" — use vectors. If users ask "How is X connected to Y?", "Who owns X?", "What caused X?" — use a graph. The clearest real-world indicator: when correct answers require combining information from multiple entities that aren't in the same document chunk. A vector search will find each chunk individually but cannot traverse the relationship path between them. Graph RAG shines in enterprise knowledge management, fraud detection, drug discovery, and compliance systems.

---

**Q2. How do you handle entity resolution — the problem where "Apple" and "Apple Inc" and "AAPL" all refer to the same node?**

> Entity resolution is the hardest engineering problem in Graph RAG. Three approaches: **(1) Canonical name normalization** — map all surface forms to a canonical ID during extraction (regex + lookup tables). **(2) Embedding deduplication** — embed all extracted entity names, cluster near-duplicates, merge clusters into single nodes. **(3) LLM-assisted resolution** — after extraction, prompt an LLM: "Are these the same entity? Apple, Apple Inc, AAPL." In production, combine all three with a human-in-the-loop review for high-confidence merges. The cost of getting this wrong is high: duplicate nodes create disconnected subgraphs and break traversal.

---

**Q3. How does Microsoft's GraphRAG paper approach graph construction differently from naive extraction?**

> Microsoft's GraphRAG builds a hierarchical community structure over the knowledge graph. First, it extracts entities and relationships at chunk level. Then it applies a graph community detection algorithm (Leiden) to find clusters of densely connected entities. Each community is summarized by an LLM into a "community report." At query time, for global queries ("What are the main themes?"), it retrieves community reports — capturing macro-level structure that individual chunks miss. For local queries, it falls back to entity-level traversal. This addresses the key weakness of naive Graph RAG: the inability to answer questions that require understanding the entire document corpus, not just individual entity paths.

