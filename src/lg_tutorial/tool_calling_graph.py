import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(".env.local")


def build_model() -> ChatOpenAI:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to .env.local or export it in the current shell."
            )

        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v3"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}}
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env.local or export it in the current shell."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=api_key,
        temperature=0,
    )


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def lookup_concept(name: str) -> str:
    """Look up a short tutorial note for a LangGraph concept name."""
    notes = {
        "stategraph": "StateGraph is the graph builder. You add nodes and edges to it, then call compile().",
        "messagesstate": "MessagesState is LangGraph's built-in chat state. Its messages field appends messages instead of overwriting them.",
        "toolnode": "ToolNode executes tool calls produced by the model and writes ToolMessage results back into state.",
        "tools_condition": "tools_condition checks the last AIMessage. If it contains tool calls, the graph routes to the tools node.",
    }

    key = name.strip().lower().replace(" ", "")
    return notes.get(key, f"No local note found for {name}.")


TOOLS = [multiply, lookup_concept]
SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are a LangGraph study assistant. "
        "Use tools when the user asks for multiplication or asks about a specific LangGraph concept. "
        "When a tool result is enough, answer directly and concisely."
    )
)


def chatbot(state: MessagesState) -> dict:
    model = build_model().bind_tools(TOOLS)
    response = model.invoke([SYSTEM_MESSAGE, *state["messages"]])
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")

    return builder.compile()


def run_demo(user_input: str) -> dict:
    graph = build_graph()
    return graph.invoke({"messages": [HumanMessage(content=user_input)]})


if __name__ == "__main__":
    samples = [
        "What is MessagesState?",
        "Please multiply 7 and 9.",
    ]

    for text in samples:
        result = run_demo(text)
        final_message = result["messages"][-1]
        print(f"You: {text}")
        print(f"AI: {final_message.content}")
        print()
