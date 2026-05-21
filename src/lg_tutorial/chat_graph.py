import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv(".env.local")


def build_model() -> ChatOpenAI:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to .env or export it in the current shell."
            )

        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or export it in the current shell."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=api_key,
        temperature=0,
    )


def chatbot(state: MessagesState) -> dict:
    model = build_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


def chat_once(graph, messages: list[BaseMessage], user_input: str) -> dict:
    next_messages = [*messages, HumanMessage(content=user_input)]
    return graph.invoke({"messages": next_messages})


def run_demo() -> dict:
    graph = build_graph()
    return chat_once(graph, [], "请用两句话介绍 LangGraph。")


if __name__ == "__main__":
    result = run_demo()
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")
