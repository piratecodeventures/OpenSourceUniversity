# GenAI Track: Agents & Multimodal Systems (Deep Dive)

## 📜 Story Mode: The Captain

> **Mission Date**: 2043.11.15
> **Location**: Bridge of the USS Prometheus
> **Officer**: AI Captain "Aura"
>
> **The Problem**: A Chatbot can *tell* me how to fly the ship.
> But I need it to *fly* the ship.
> It needs to see the console (Vision) and press the buttons (Action).
>
> **The Solution**: **Agentic AI**.
> A ReAct Loop: Perceive $\to$ Reason $\to$ Act.
>
> *"Computer. Observe the Navigation Chart. Plot a course to Sector 7. Engage warp drive."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**: Systems that use LLMs as a "Reasoning Engine" to control external software tools.
2.  **WHY**: LLMs are frozen in time. Agents can search the web, run python code, and query databases.
3.  **WHEN**: Tasks require multi-step logic or external data access.
4.  **WHERE**: `LangChain`, `LangGraph`, `AutoGPT`.
5.  **WHO**: Harrison Chase (LangChain), Andrew Ng (Agentic Workflow).
6.  **HOW**: `While (Goal != Achieved): Think() -> Tool() -> Observe()`.

---

## 2. Mathematical Deep Dive: Reasoning Traces

### 2.1 ReAct (Reason + Act)
Standard Prompting: Input $\to$ Output.
**ReAct**: Enforce a structure of thought.
1.  **Thought**: "I need to find the weather using the weather_api."
2.  **Action**: `weather_api("London")`
3.  **Observation**: "15C, Raining"
4.  **Thought**: "It is raining, so I should recommend an umbrella."
5.  **Final Answer**: "Take an umbrella."
*   *Why it works*: Decomposes complex problems into atomic steps.

### 2.2 Tree of Thoughts (ToT)
For hard problems (Math, Coding):
Search over a tree of possible reasoning paths.
*   **Generator**: Propose 3 next steps.
*   **Evaluator**: Rate each step (Sure/Maybe/Impossible).
*   **Search**: BFS or DFS to find the solution.

---

## 3. The Ship's Code (Polyglot: LangGraph Agent)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# LEVEL 2: Building a State Machine Agent
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def reason_node(state):
    # LLM decides what to do
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

def action_node(state):
    # Execute tool call
    last_msg = state['messages'][-1]
    tool_call = last_msg.tool_calls[0]
    result = execute_tool(tool_call)
    return {"messages": [ToolMessage(content=result, tool_call_id=tool_call.id)]}

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", reason_node)
workflow.add_node("tools", action_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue, # If tool_call exists -> "tools", else -> END
    {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

---

## 4. System Architecture: The Agentic Loop

```mermaid
graph TD
    Goal[User: "Update the database"] --> Agent
    
    subgraph "Reasoning Engine"
        Agent --> |Memory| ShortTerm[Conversation History]
        Agent --> |Knowledge| LongTerm[Vector DB]
        Agent --> |Plan| Planner[Decompose Task]
    end
    
    Planner --> |Thought| LLM
    LLM --> |"Call SQL Tool"| ToolExecutor
    
    ToolExecutor --> |"SELECT * FROM Users"| Database
    Database --> |"Rows..."| ToolExecutor
    ToolExecutor --> |Observation| LLM
    
    LLM --> |"Done"| FinalAnswer
```

---

## 13. Industry Interview Corner

### ❓ Real World Questions

**Q1: "How do you handle infinite loops in Agents?"**
*   **Answer**: "Agents often get stuck repeating the same tool call. Mitigations: 1. **Max Iterations** (Stop after 10 steps). 2. **Reflexion** (Append 'You tried this before and it failed' to the context). 3. **Human-in-the-loop** (Ask user for help)."

**Q2: "What is Function Calling (JSON Mode)?"**
*   **Answer**: "Instead of parsing free text, models like GPT-4 and Llama-3 are fine-tuned to output structured JSON when tools are defined. This guarantees parameters match the function signature (e.g., `{'location': 'London'}`)."

---

## 14. Debug Your Thinking (Misconceptions)

> [!WARNING]
> **"Agents are Autonomous."**
> *   **Correction**: They are **Semi-Autonomous**. In production, 100% autonomy usually leads to 100% chaos. Always implement "Rails" and "Approval Gates" for sensitive actions (Writing to DB, Sending Emails).
