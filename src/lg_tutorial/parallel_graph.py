"""LangGraph Send API: 并行分发——一个节点扇出多个并行分支，汇合后继续。

run:
    uv run python -m lg_tutorial.parallel_graph
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class ArticleState(TypedDict):
    topic: str
    subtopics: list[str]
    subtopic: str
    sections: Annotated[list[str], operator.add]
    final_article: str


def planner(state: ArticleState) -> dict:
    """为 topic 生成子主题列表，供后续并行 writer 分发。"""
    topic = state["topic"]
    return {
        "subtopics": [
            f"{topic} 的定义与背景",
            f"{topic} 的核心原理",
            f"{topic} 的应用场景",
        ]
    }


def continue_to_writers(state: ArticleState) -> list[Send]:
    """为每个子主题创建一个 Send，分发到 writer 节点并行执行。"""
    return [
        Send("writer", {"subtopic": subtopic})
        for subtopic in state["subtopics"]
    ]


def writer(state: ArticleState) -> dict:
    """每个并行 writer 实例独立执行，只处理自己被分配到的 subtopic。"""
    subtopic = state["subtopic"]
    return {"sections": [f"## {subtopic}\n\n这是关于「{subtopic}」的详细阐述。"]}


def reviewer(state: ArticleState) -> dict:
    """所有 writer 完成后，汇总 sections 生成最终文章。"""
    sections_text = "\n\n".join(state["sections"])
    return {"final_article": f"# {state['topic']}\n\n{sections_text}"}


def build_graph():
    builder = StateGraph(ArticleState)
    builder.add_node("planner", planner)
    builder.add_node("writer", writer)
    builder.add_node("reviewer", reviewer)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", continue_to_writers)
    builder.add_edge("writer", "reviewer")
    builder.add_edge("reviewer", END)

    return builder.compile()


def run_demo(topic: str = "LangGraph Send API") -> dict:
    graph = build_graph()
    return graph.invoke(
        {
            "topic": topic,
            "subtopics": [],
            "subtopic": "",
            "sections": [],
            "final_article": "",
        }
    )


if __name__ == "__main__":
    result = run_demo()
    print(result["final_article"])
