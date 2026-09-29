"""
LangGraph Agent with Production Error Handling
Retry logic, model fallback, and structured state management.
"""

from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import Client, traceable

from app.config import get_settings

_settings = get_settings()
_langsmith_client = Client(
    api_url=_settings.langchain_endpoint,
    api_key=_settings.langchain_api_key or None,
)


# === Agent State ===

class AgentState(TypedDict):
    """
    State for the production agent.
    Uses Annotated with add_messages reducer for message accumulation.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    
# === Agent Builder ===

class ProductionAgent:
    """
    Production LangGraph agent with:
    - Retry on failure (model fallback)
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatGoogleGenerativeAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,  # We handle retries ourselves
            api_key=settings.gemini_api_key,
        )
        self.fallback_llm = ChatGoogleGenerativeAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.gemini_api_key,
        )
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine."""

        def process_message(state: AgentState) -> dict:
            """Try to process the message with the primary model."""
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary model."""
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """Return a graceful error message."""
            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, I'm having trouble processing your request "
                        "right now. Please try again in a moment."
                    ))
                ],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt."""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback attempt."""
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # Build the graph
        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )
        graph.add_edge("error", END)

        return graph.compile()

    @traceable(
        name="production_agent_invoke",
        client=_langsmith_client,
        project_name=_settings.langchain_project,
        enabled=_settings.langchain_tracing,
    )
    def invoke(self, message: str) -> dict:
        """
        Invoke the agent with a user message.
        Returns: {"response": str, "model_used": str, "error": str | None}
        """
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": "",
        })

        return {
            "response": result["messages"][-1].text,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }


"""
code explaination:

## The big picture first

Imagine a call centre. When a customer calls:

1. The call goes to your **best agent** (the primary AI model).
2. If that agent is unavailable, the call goes to a **backup agent** (the fallback model).
3. If the backup also fails, a **polite recorded message** plays: "Sorry, please try again later."

The customer always hears *something*. They never get a crash or silence.

That's this whole file: an AI agent that **doesn't break when the AI service has problems**. It's built as a LangGraph state machine, so each of those steps is a node.

The flow:

```
START → process (primary model)
           ├── worked ────────────────────────► END
           ├── failed, retries left ─► fallback (backup model)
           │                              ├── worked ─► END
           │                              └── failed ─► error ─► END
           └── failed, no retries left ────────────────► error ─► END
```

---

## Part 0: The imports

```python
from typing import Optional
from typing_extensions import TypedDict, Annotated
```
Tools for describing data shapes. `Optional[str]` means "a string, or `None`".

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
```
LangGraph's building blocks: the graph, the start/end points, and a helper for combining messages (explained below).

```python
from langchain_openai import ChatOpenAI
```
The connector that talks to OpenAI's models (GPT).

```python
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
```
Message types. A chat is a list of these: `HumanMessage` = what the user said, `AIMessage` = what the AI said. `BaseMessage` is the parent type covering both.

```python
from langsmith import traceable
```
LangSmith is a monitoring tool for AI apps. `traceable` records each run so you can inspect it later (inputs, outputs, timing, errors).

```python
from app.config import get_settings
```
Loads settings (model names, API key, retry limit) from your config, usually environment variables. This keeps secrets and settings out of the code.

---

## Part 1: The State (the shared notebook)

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
```

This is the notebook passed between every step. Four fields:

| Field | Meaning |
|---|---|
| `messages` | The conversation so far |
| `error` | The latest error message, or `None` if all is well |
| `retry_count` | How many times the primary model has failed |
| `model_used` | Who answered: `"primary"`, `"fallback"`, or `"error_handler"` |

### The tricky part: `Annotated[..., add_messages]`

When a node returns an update like `{"error": None}`, LangGraph normally **replaces** the old value with the new one.

But for messages, replacing would be bad. You'd lose the conversation history. So `add_messages` tells LangGraph: "for this field, **add** new messages to the list instead of replacing it."

Analogy: most fields are a whiteboard (erase and rewrite). `messages` is a diary (you only add new pages).

This helper is called a **reducer**: a rule for how updates combine with the existing value.

So when a node returns `{"messages": [response]}`, the AI's reply is **appended** after the user's question.

---

## Part 2: Setting up the agent (`__init__`)

```python
settings = get_settings()

self.primary_llm = ChatOpenAI(
    model=settings.primary_model,
    temperature=0,
    timeout=30,
    max_retries=0,
    api_key=settings.openai_api_key,
)
```

This creates the connection to the main AI model. The settings:

- **`model`**: which GPT model, e.g. a big, smart one.
- **`temperature=0`**: how creative/random the answers are. 0 means as consistent and predictable as possible. Good for business apps.
- **`timeout=30`**: if the model hasn't replied in 30 seconds, give up (raise an error).
- **`max_retries=0`**: the OpenAI library normally retries failed calls automatically. This switches that off, because the graph handles failures itself. Otherwise you'd have retries hidden inside retries, and a single request could take minutes.
- **`api_key`**: the password for OpenAI.

`self.fallback_llm` is the same, but with `settings.fallback_model`, usually a smaller, cheaper, or more available model.

```python
self.max_retries = settings.max_retries
self.graph = self._build_graph()
```

Stores the retry limit and builds the graph once, when the agent is created. Building it once and reusing it is efficient.

---

## Part 3: The workers (nodes)

All of these are defined inside `_build_graph`, so they can use `self.primary_llm` etc.

### `process_message`: try the main model

```python
try:
    response = self.primary_llm.invoke(state["messages"])
    return {"messages": [response], "error": None, "model_used": "primary"}
except Exception as e:
    return {"error": str(e), "retry_count": state["retry_count"] + 1, "model_used": ""}
```

- **`try`**: send the whole conversation to the primary model.
- **If it works**: add the reply to messages, clear the error, note "primary".
- **If anything goes wrong** (timeout, OpenAI down, rate limit...): don't crash. Instead, write the error into the notebook and add 1 to `retry_count`.

The key idea: **errors become data in the state instead of crashes.** The next step can then look at the notebook and decide what to do.

### `try_fallback`: try the backup model

Same pattern with the fallback model. On success, note "fallback". On failure, record the error. (It doesn't bump `retry_count`, since that only matters for the primary.)

### `handle_error`: the polite recorded message

```python
return {
    "messages": [AIMessage(content="I'm sorry, I'm having trouble ...")],
    "model_used": "error_handler",
}
```

No AI call here at all. It just adds a hard-coded apology as if the AI had said it. The user gets a friendly reply instead of a technical error.

---

## Part 4: The decision makers (routers)

These don't change the state. They only **read** it and return a label saying where to go next.

### `route_after_process`

```python
if state.get("error") is None:
    return "done"                  # primary worked
elif state["retry_count"] < self.max_retries:
    return "fallback"              # failed, but we're allowed another try
else:
    return "error"                 # failed, out of chances
```

### `route_after_fallback`

```python
if state.get("error") is None:
    return "done"
else:
    return "error"
```

---

## Part 5: Wiring the graph

```python
graph = StateGraph(AgentState)
```
Create an empty flowchart that uses our notebook shape.

```python
graph.add_node("process", process_message)
graph.add_node("fallback", try_fallback)
graph.add_node("error", handle_error)
```
Add the three workers and give each a name.

```python
graph.add_edge(START, "process")
```
Always begin at `process`.

```python
graph.add_conditional_edges(
    "process",
    route_after_process,
    {"done": END, "fallback": "fallback", "error": "error"},
)
```
After `process`, run the router. The dictionary translates its label into a destination: `"done"` goes to END, `"fallback"` goes to the fallback node, and so on.

```python
graph.add_conditional_edges("fallback", route_after_fallback, {"done": END, "error": "error"})
graph.add_edge("error", END)
```
Same idea after fallback. After the error message, always finish.

```python
return graph.compile()
```
**Compile** turns the drawing into something runnable. It also checks the wiring (for example, that every node can be reached).

---

## Part 6: Using the agent (`invoke`)

```python
@traceable(name="production_agent_invoke")
def invoke(self, message: str) -> dict:
```
This is the only method outside code needs to call. `@traceable` sends a record of each call to LangSmith.

```python
result = self.graph.invoke({
    "messages": [HumanMessage(content=message)],
    "error": None,
    "retry_count": 0,
    "model_used": "",
})
```
Fill in a fresh notebook (the user's message, no error, zero retries) and run the flowchart. `result` is the final state of the notebook.

```python
return {
    "response": result["messages"][-1].content,
    "model_used": result.get("model_used", "unknown"),
    "error": result.get("error"),
}
```
Pull out the useful parts:
- **`response`**: `[-1]` means "last item in the list", i.e. the newest message. That's the AI's answer (or the apology).
- **`model_used`**: who answered.
- **`error`**: what went wrong, if anything.

---

## Walk-through: three scenarios

Say `max_retries = 1`.

**1. Everything fine**
process → primary answers → error is None → **END**.
Returns: real answer, `"primary"`, no error.

**2. Primary down, backup fine**
process fails → `retry_count` becomes 1, error recorded → route checks: error exists, is 1 < 1? No → **error node**.

Wait, that skips the fallback! With `max_retries = 1`, the fallback never runs. You'd need `max_retries = 2` or more for the fallback to be used. With `max_retries = 2`:
process fails → retry_count 1 → 1 < 2 → **fallback** → backup answers → **END**.
Returns: real answer, `"fallback"`, no error.

**3. Both down** (`max_retries = 2`)
process fails → fallback fails → **error** → **END**.
Returns: the apology, `"error_handler"`, and the fallback's error text.

---

## Things worth noticing

**1. The word "retry" is misleading.** The primary model is only tried **once**. There's no loop back to `process`. `retry_count` and `max_retries` really just act as a switch: "is the fallback allowed?" With `max_retries` of 0 or 1, the fallback is effectively disabled (scenario 2 above). Worth checking what your config sets.

**2. So this graph is actually a DAG.** Tying back to your last question: there's no arrow going backwards, so this particular graph has no loops. A true retry would add an edge from `process` back to itself.

**3. The same API key and provider are used for both models.** If OpenAI itself is down, the fallback will likely fail too. Many teams fall back to a **different provider** (e.g. Anthropic or Azure OpenAI) for real resilience.

**4. It catches every error the same way.** A timeout is worth retrying with another model. But a bad request (e.g. the message is too long) will fail on both models too. Smarter code checks the error type first.

**5. No memory between calls.** Every `invoke` starts with a brand-new notebook containing only the one message. There's no checkpointer or `thread_id`, so the agent doesn't remember earlier messages in a conversation.

**6. Callers must check `error`.** In scenario 3, `response` contains a normal-looking sentence. If your code only reads `response`, it won't notice something failed. The `error` and `model_used` fields are there for exactly this, and they'd pair nicely with the `MetricsCollector` from the previous file (`error=True` when `model_used == "error_handler"`).

---

## One-line summary

This file builds a small LangGraph flowchart that asks the main AI model first, switches to a backup model if allowed, and falls back to a polite apology if both fail, so the user never sees a crash.
"""