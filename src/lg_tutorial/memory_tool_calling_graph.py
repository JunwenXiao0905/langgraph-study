from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lg_tutorial.tool_calling_graph import SYSTEM_MESSAGE, TOOLS, build_model


def chatbot(state: MessagesState) -> dict:
    model = build_model().bind_tools(TOOLS)
    response = model.invoke([SYSTEM_MESSAGE, *state["messages"]])
    return {"messages": [response]}


def build_graph():
    memory = InMemorySaver()

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")

    return builder.compile(checkpointer=memory)


def chat_once(graph, thread_id: str, user_input: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )


def run_demo() -> None:
    graph = build_graph()
    thread_id = "study-thread"

    turns = [
        "My name is Xiaojun. Please remember it.",
        "What is my name?",
        "Please multiply 8 and 6.",
    ]

    for text in turns:
        result = chat_once(graph, thread_id, text)
        final_message = result["messages"][-1]
        print(f"You: {text}")
        print(f"AI: {final_message.content}")
        print()

    checkpoint = graph.get_state({"configurable": {"thread_id": thread_id}})
    print(f"Saved message count for {thread_id}: {len(checkpoint.values['messages'])}")


if __name__ == "__main__":
    run_demo()
