"""LangSmith tracing: 给 LangGraph 挂载观测，查看每次 invoke 的节点调用树。

需要 LangSmith API key。如无 key，代码仍可运行（不报错），但 trace 不会上传。

run:
    uv run python -m lg_tutorial.traced_graph
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lg_tutorial.tool_calling_graph import SYSTEM_MESSAGE, TOOLS, build_model

load_dotenv(".env.local")

# ── LangSmith tracing 配置 ──────────────────────────────────────────
# 只有提供了 API key 才开启 tracing，避免无 key 时 401 报错刷屏。
langsmith_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ.setdefault("LANGCHAIN_PROJECT", "langgraph-study")
# ─────────────────────────────────────────────────────────────────────


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


def run_demo():
    graph = build_graph()

    # 通过 config 给每次调用附加 metadata，可在 LangSmith UI 中按 tags 筛选
    config = {
        "configurable": {"thread_id": "trace-demo"},
        "metadata": {"demo": "langsmith-intro", "step": 1},
        "tags": ["learning", "tracing"],
    }

    result = graph.invoke(
        {"messages": [HumanMessage(content="multiply 6 by 7")]},
        config,
    )
    print(result["messages"][-1].content)

    print()
    if os.getenv("LANGCHAIN_API_KEY"):
        print("追踪已上传。在 https://smith.langchain.com 查看 project=langgraph-study 的 trace。")
    else:
        print("未设置 LANGCHAIN_API_KEY，trace 未上报。在 .env.local 中添加该 key 即可启用。")


if __name__ == "__main__":
    run_demo()
