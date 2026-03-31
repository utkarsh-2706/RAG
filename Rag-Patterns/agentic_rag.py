# agentic_rag.py
# Agentic RAG — ReAct loop with retrieval as a tool
# Core idea: LLM plans, calls tools, observes, and decides when to stop

import numpy as np
from typing import List, Dict, Callable, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# 1. AGENT MEMORY / SCRATCHPAD
# ─────────────────────────────────────────────

@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: str
    observation: str

@dataclass
class AgentScratchpad:
    steps: List[AgentStep] = field(default_factory=list)

    def add(self, thought: str, action: str, action_input: str, observation: str):
        self.steps.append(AgentStep(thought, action, action_input, observation))

    def format(self) -> str:
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}:")
            lines.append(f"  Thought    : {step.thought}")
            lines.append(f"  Action     : {step.action}({step.action_input})")
            lines.append(f"  Observation: {step.observation}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 2. TOOL DEFINITIONS
# ─────────────────────────────────────────────

# Internal document corpus
INTERNAL_DOCS = [
    "Q3 2024: Our revenue was $4.2 billion, up 18% YoY. Operating margin 22%.",
    "Q2 2024: Our revenue was $3.6 billion, up 12% YoY. Operating margin 19%.",
    "Enterprise plan: includes SSO, audit logs, dedicated support, 99.9% SLA.",
    "Our pricing: Starter $29/mo, Pro $99/mo, Enterprise custom.",
    "Competitive strategy framework: differentiation, cost leadership, focus.",
    "Customer churn Q3: 4.2%, down from 5.1% in Q2. Main driver: onboarding improvements.",
]

# Simulated web results
WEB_DB = {
    "competitor pricing": "XYZ Corp: Starter $39/mo, Pro $129/mo, Enterprise $500+/mo.",
    "competitor revenue": "XYZ Corp Q3 2024: $3.1 billion, up 8% YoY.",
    "market trends": "SaaS market growing 18% annually. AI features driving premium pricing.",
}

def simulate_embedding(text: str, dim: int = 64) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dim).astype(np.float32)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ── Tool: Vector Retriever ──
def tool_retrieve(query: str) -> str:
    """Search internal knowledge base."""
    q_vec = simulate_embedding(query)
    scored = [(cosine_similarity(q_vec, simulate_embedding(d)), d) for d in INTERNAL_DOCS]
    top = sorted(scored, reverse=True)[:2]
    results = "\n".join([f"  - {doc}" for _, doc in top])
    return f"Internal knowledge base results:\n{results}"

# ── Tool: Web Search ──
def tool_web_search(query: str) -> str:
    """Search the live web for current information."""
    for key, value in WEB_DB.items():
        if any(word in query.lower() for word in key.split()):
            return f"Web search result: {value}"
    return "Web search: No highly relevant results found."

# ── Tool: Calculator ──
def tool_calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

# ── Tool: Summarizer ──
def tool_summarize(text: str) -> str:
    """Summarize long text to fit in context."""
    words = text.split()
    if len(words) > 30:
        return "Summary: " + " ".join(words[:30]) + "... [truncated]"
    return f"Summary: {text}"

# Tool registry
TOOLS: Dict[str, Callable] = {
    "retrieve": tool_retrieve,
    "web_search": tool_web_search,
    "calculate": tool_calculate,
    "summarize": tool_summarize,
}


# ─────────────────────────────────────────────
# 3. SIMULATED AGENT PLANNER / REASONER
# In production: LLM generates Thought + Action in ReAct format
# ─────────────────────────────────────────────

@dataclass
class AgentAction:
    thought: str
    tool: str          # "retrieve" | "web_search" | "calculate" | "FINISH"
    tool_input: str
    is_final: bool = False

def simulate_agent_plan(task: str, scratchpad: AgentScratchpad) -> AgentAction:
    """
    Simulates the LLM reasoning step in the ReAct loop.

    In production, this is an LLM call with a ReAct prompt:
    ─────────────────────────────────────────────────────────
    system_prompt = '''You are an agent. Use tools to answer the task.
    Format: Thought: <reasoning>
            Action: <tool_name>
            Action Input: <input>
    When done: Action: FINISH, Action Input: <final answer>'''

    user_prompt = f'''Task: {task}
    Previous steps: {scratchpad.format()}
    Next step:'''
    ─────────────────────────────────────────────────────────
    """
    step_num = len(scratchpad.steps)
    task_lower = task.lower()

    # Simulated decision tree (mimics LLM reasoning)
    if step_num == 0:
        if "competitor" in task_lower or "compare" in task_lower:
            return AgentAction(
                thought="I need our internal financial data first.",
                tool="retrieve",
                tool_input="Q3 revenue operating margin"
            )
        elif "market" in task_lower or "trend" in task_lower:
            return AgentAction(
                thought="I need current market information.",
                tool="web_search",
                tool_input="market trends SaaS"
            )
        else:
            return AgentAction(
                thought="Let me search our internal knowledge base.",
                tool="retrieve",
                tool_input=task
            )

    elif step_num == 1:
        if "competitor" in task_lower:
            return AgentAction(
                thought="Now I need competitor data to compare.",
                tool="web_search",
                tool_input="competitor pricing revenue"
            )
        elif "calculate" in task_lower or "%" in task:
            return AgentAction(
                thought="I need to run a calculation.",
                tool="calculate",
                tool_input="4.2 / 3.6 - 1"
            )
        else:
            return AgentAction(
                thought="I have enough context to answer.",
                tool="FINISH",
                tool_input="",
                is_final=True
            )

    elif step_num == 2:
        if "strategy" in task_lower:
            return AgentAction(
                thought="I have the data. Now I need strategy frameworks.",
                tool="retrieve",
                tool_input="competitive strategy differentiation"
            )
        else:
            return AgentAction(
                thought="I have gathered sufficient information to synthesize an answer.",
                tool="FINISH",
                tool_input="",
                is_final=True
            )

    else:
        return AgentAction(
            thought="All sub-tasks complete. Generating final answer.",
            tool="FINISH",
            tool_input="",
            is_final=True
        )


# ─────────────────────────────────────────────
# 4. FINAL SYNTHESIS
# ─────────────────────────────────────────────

def synthesize_answer(task: str, scratchpad: AgentScratchpad) -> str:
    """
    Synthesize a final answer from the full ReAct trace.
    In production: LLM reads full scratchpad and generates answer.
    """
    print("\n[Agent Synthesizer] Generating final answer from scratchpad...")
    print(f"\nFull ReAct Trace:\n{scratchpad.format()}")
    return (
        f"[Simulated Final Answer]\n"
        f"Task: {task}\n"
        f"Based on {len(scratchpad.steps)} tool calls, the agent has gathered all "
        f"required information and synthesized a comprehensive response."
    )


# ─────────────────────────────────────────────
# 5. AGENTIC RAG PIPELINE (ReAct Loop)
# ─────────────────────────────────────────────

class AgenticRAG:
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

    def run(self, task: str) -> str:
        print(f"\n{'='*60}")
        print(f"AGENT TASK: {task}")
        print(f"Max iterations: {self.max_iterations}")

        scratchpad = AgentScratchpad()

        for iteration in range(self.max_iterations):
            print(f"\n--- Agent Iteration {iteration + 1} ---")

            # Reason: what to do next
            action = simulate_agent_plan(task, scratchpad)
            print(f"Thought : {action.thought}")
            print(f"Action  : {action.tool}({action.tool_input[:50]})")

            # Check for stop condition
            if action.is_final or action.tool == "FINISH":
                print("[Agent] FINISH condition reached.")
                break

            # Execute tool
            if action.tool in TOOLS:
                observation = TOOLS[action.tool](action.tool_input)
            else:
                observation = f"Unknown tool: {action.tool}"

            print(f"Observation: {observation[:100]}...")

            # Store in scratchpad
            scratchpad.add(
                thought=action.thought,
                action=action.tool,
                action_input=action.tool_input,
                observation=observation
            )

        else:
            print(f"\n[Agent] Max iterations ({self.max_iterations}) reached. Stopping.")

        # Synthesize final answer from full trace
        return synthesize_answer(task, scratchpad)


# ─────────────────────────────────────────────
# 6. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent = AgenticRAG(max_iterations=5)

    print("\n--- Task 1: Multi-step competitive analysis ---")
    agent.run("Compare our Q3 revenue vs competitor XYZ and suggest 3 pricing strategies.")

    print("\n\n--- Task 2: Market research task ---")
    agent.run("Summarize current SaaS market trends and how our pricing compares.")