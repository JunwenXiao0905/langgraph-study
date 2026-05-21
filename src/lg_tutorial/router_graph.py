from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class RouterState(TypedDict):
    user_input: str
    route: str
    result: str


def decide_route(state: RouterState) -> dict:
    text = state["user_input"].strip().lower()

    if "write" in text or "article" in text:
        route = "writer"
    elif "chat" in text or "hello" in text or "hi" in text:
        route = "chatbot"
    else:
        route = "fallback"

    return {"route": route}


def pick_next_node(state: RouterState) -> Literal["writer", "chatbot", "fallback"]:
    return state["route"]  # type: ignore[return-value]


def writer_node(state: RouterState) -> dict:
    text = state["user_input"]
    return {"result": f"writer node received: {text}"}


def chatbot_node(state: RouterState) -> dict:
    text = state["user_input"]
    return {"result": f"chatbot node received: {text}"}


def fallback_node(state: RouterState) -> dict:
    text = state["user_input"]
    return {"result": f"fallback node received: {text}"}


def build_graph():
    builder = StateGraph(RouterState)
    builder.add_node("decide_route", decide_route)
    builder.add_node("writer", writer_node)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "decide_route")
    builder.add_conditional_edges(
        "decide_route",
        pick_next_node,
        {
            "writer": "writer",
            "chatbot": "chatbot",
            "fallback": "fallback",
        },
    )
    builder.add_edge("writer", END)
    builder.add_edge("chatbot", END)
    builder.add_edge("fallback", END)

    return builder.compile()


def run_demo(user_input: str) -> dict:
    graph = build_graph()
    return graph.invoke({"user_input": user_input, "route": "", "result": ""})


if __name__ == "__main__":
    for text in [
        "write an article about langgraph",
        "hello there",
        "what is this",
    ]:
        result = run_demo(text)
        print(result["route"], "->", result["result"])
