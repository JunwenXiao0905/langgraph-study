# LangGraph Study

This project uses `uv` and starts with small LangGraph examples.

## Run

```bash
uv run lg-tutorial
```

Router example:

```bash
uv run python -m lg_tutorial.router_graph
```

Tool-calling example:

```bash
uv run python -m lg_tutorial.tool_calling_graph
```

Memory + tool-calling example:

```bash
uv run python -m lg_tutorial.memory_tool_calling_graph
```

Interrupt example:

```bash
uv run python -m lg_tutorial.interrupt_graph
```

## First goal

The first example teaches:

- shared state with `TypedDict`
- nodes as plain Python functions
- edges from `START` to `END`
- `compile()` and `invoke()`

## Next goal

The tool-calling example teaches:

- `MessagesState` as chat state
- `bind_tools(...)` on the chat model
- `ToolNode(...)` for executing tool calls
- `tools_condition` for routing between model and tools
- the standard loop: `chatbot -> tools -> chatbot`

## Next step after tools

The memory example teaches:

- `InMemorySaver()` as a checkpointer
- `compile(checkpointer=...)` to persist graph state
- `thread_id` in `configurable` to keep one conversation thread
- reusing the same graph across multiple `invoke(...)` calls
- `graph.get_state(...)` to inspect the saved thread state

## Next step after memory

The interrupt example teaches:

- `interrupt(...)` to pause graph execution from inside a node
- `Command(resume=...)` to continue from the paused thread
- why interrupts require a checkpointer
- that the interrupted node re-runs from the top after resume
- `graph.get_state(...)` to inspect pending interrupts

## Files

```text
src/lg_tutorial/main.py
src/lg_tutorial/first_graph.py
src/lg_tutorial/chat_graph.py
src/lg_tutorial/router_graph.py
src/lg_tutorial/tool_calling_graph.py
src/lg_tutorial/memory_tool_calling_graph.py
src/lg_tutorial/interrupt_graph.py
```
