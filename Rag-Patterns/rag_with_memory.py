# rag_with_memory.py
# RAG with Conversational Memory — stateful multi-turn Q&A
# Demonstrates: query rewriting + history injection

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Message:
    role: str   # "user" or "assistant"
    content: str

@dataclass
class ChatHistory:
    messages: List[Message] = field(default_factory=list)
    max_turns: int = 6  # Keep last N messages to avoid token overflow

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        # Sliding window — drop oldest turns beyond limit
        if len(self.messages) > self.max_turns:
            self.messages = self.messages[-self.max_turns:]

    def format(self) -> str:
        return "\n".join([f"{m.role.upper()}: {m.content}" for m in self.messages])

    def is_empty(self) -> bool:
        return len(self.messages) == 0


# ─────────────────────────────────────────────
# 2. SIMULATED EMBEDDING (same as Simple RAG)
# ─────────────────────────────────────────────

DOCUMENTS = [
    "RAG stands for Retrieval-Augmented Generation. It grounds LLMs in retrieved facts.",
    "Vector databases store embeddings and enable fast similarity search.",
    "FAISS is Meta's library for approximate nearest neighbor search.",
    "Chunking splits documents into smaller pieces before embedding for better retrieval.",
    "Cosine similarity is used to measure semantic closeness between query and document vectors.",
    "LLMs hallucinate when they lack grounding context from a retrieval system.",
    "Query rewriting reformulates vague follow-up questions into standalone queries.",
    "Chat history stores previous turns so the assistant can maintain conversation context.",
    "Memory in RAG systems enables multi-turn coherent conversations over documents.",
    "Token limits constrain how much history can be injected into an LLM prompt.",
]

def simulate_embedding(text: str, dim: int = 64) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dim).astype(np.float32)

def embed_documents(docs: List[str]) -> np.ndarray:
    return np.array([simulate_embedding(d) for d in docs])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def retrieve(query: str, docs: List[str], doc_embeddings: np.ndarray, top_k: int = 3) -> List[Tuple[float, str]]:
    q_vec = simulate_embedding(query)
    scores = [cosine_similarity(q_vec, emb) for emb in doc_embeddings]
    return sorted(zip(scores, docs), reverse=True)[:top_k]


# ─────────────────────────────────────────────
# 3. QUERY REWRITER
# This is the KEY addition over Simple RAG.
# In production: make an LLM call with history.
# ─────────────────────────────────────────────

def rewrite_query(history: ChatHistory, latest_query: str) -> str:
    """
    Rewrites the latest user query into a standalone question
    using the conversation history.

    In production, replace with:
    ─────────────────────────────
    prompt = f'''Given this conversation history:
    {history.format()}

    Rewrite the latest question as a fully standalone question
    that makes sense without any prior context.

    Latest question: {latest_query}
    Standalone question:'''
    return call_llm(prompt)
    ─────────────────────────────
    """
    if history.is_empty():
        return latest_query  # No history — query is already standalone

    # Simulated rewrite: prepend last assistant topic as context signal
    last_assistant = next(
        (m.content for m in reversed(history.messages) if m.role == "assistant"),
        ""
    )
    # Simple heuristic simulation for demo purposes
    rewritten = f"{latest_query} (in the context of: {last_assistant[:80]}...)"
    print(f"\n[Query Rewriter]")
    print(f"  Original : {latest_query}")
    print(f"  Rewritten: {rewritten}")
    return rewritten


# ─────────────────────────────────────────────
# 4. PROMPT BUILDER (now history-aware)
# ─────────────────────────────────────────────

def build_prompt(history: ChatHistory, context_docs: List[Tuple[float, str]], query: str) -> str:
    context = "\n".join([f"- {doc}" for _, doc in context_docs])
    history_text = history.format() if not history.is_empty() else "None"

    return f"""You are a helpful assistant engaged in a multi-turn conversation.
Use the retrieved context AND conversation history to answer accurately.

=== Conversation History ===
{history_text}

=== Retrieved Context ===
{context}

=== Current Question ===
{query}

Answer:"""


# ─────────────────────────────────────────────
# 5. SIMULATED LLM
# ─────────────────────────────────────────────

def call_llm(prompt: str, label: str = "LLM") -> str:
    print(f"\n[{label} Prompt Sent]")
    print("─" * 60)
    print(prompt)
    print("─" * 60)
    return f"[Simulated answer from {label}]"


# ─────────────────────────────────────────────
# 6. CONVERSATIONAL RAG PIPELINE
# ─────────────────────────────────────────────

class RAGWithMemory:
    def __init__(self, documents: List[str], top_k: int = 3, max_history_turns: int = 6):
        self.documents = documents
        self.top_k = top_k
        self.history = ChatHistory(max_turns=max_history_turns)
        print("Indexing documents...")
        self.doc_embeddings = embed_documents(documents)
        print(f"Indexed {len(documents)} documents.\n")

    def chat(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"USER: {user_query}")

        # Step 1: Rewrite query using history
        standalone_query = rewrite_query(self.history, user_query)

        # Step 2: Retrieve using the rewritten query
        retrieved = retrieve(standalone_query, self.documents, self.doc_embeddings, self.top_k)
        print(f"\nTop-{self.top_k} Retrieved Docs:")
        for score, doc in retrieved:
            print(f"  [{score:.3f}] {doc}")

        # Step 3: Build history-aware prompt
        prompt = build_prompt(self.history, retrieved, user_query)

        # Step 4: Generate answer
        answer = call_llm(prompt, label="Conversational LLM")

        # Step 5: Update history
        self.history.add("user", user_query)
        self.history.add("assistant", answer)

        print(f"\nASSISTANT: {answer}")
        return answer


# ─────────────────────────────────────────────
# 7. DEMO — Multi-turn conversation
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = RAGWithMemory(DOCUMENTS, top_k=3, max_history_turns=6)

    # Turn 1 — standalone question
    rag.chat("What is RAG?")

    # Turn 2 — follow-up (requires memory to resolve "it")
    rag.chat("What are its limitations?")

    # Turn 3 — another follow-up
    rag.chat("How does query rewriting help with that?")
