"""LangGraph subgraph: 把一个编译好的 graph 当成另一个 graph 的节点。

父图把子图视为一个节点（黑盒），但子图内部有自己的节点和流转。
父图和子图通过共享 state 字段来传递数据。

run:
    uv run python -m lg_tutorial.subgraph
"""

from pprint import pformat
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# ═══════════════════════════════════════════════════════════════════
# 子图：研究模块
# ═══════════════════════════════════════════════════════════════════

class ResearchState(TypedDict):
    """子图的 state。和父图共享 topic、research_done 字段。"""
    topic: str
    research_done: bool


def search_topic(state: ResearchState) -> dict:
    """模拟搜索：根据 topic 收集资料。"""
    print(f"    [子图] search_topic: 正在研究「{state['topic']}」")
    return {"research_done": True}


def summarize_findings(state: ResearchState) -> dict:
    """模拟总结。"""
    print(f"    [子图] summarize_findings: 研究完成")
    return {}


def build_research_subgraph():
    """构建并编译研究子图。返回的编译结果可以直接当节点用。"""
    builder = StateGraph(ResearchState)
    builder.add_node("search_topic", search_topic)
    builder.add_node("summarize_findings", summarize_findings)
    builder.add_edge(START, "search_topic")
    builder.add_edge("search_topic", "summarize_findings")
    builder.add_edge("summarize_findings", END)
    return builder.compile()


# ═══════════════════════════════════════════════════════════════════
# 父图：文章生成
# ═══════════════════════════════════════════════════════════════════

class ArticleState(TypedDict):
    """父图的 state。topic 和 research_done 与子图共享。"""
    topic: str
    research_done: bool
    outline: list[str]
    article: str


def make_outline(state: ArticleState) -> dict:
    topic = state["topic"]
    return {"outline": [f"{topic} 的定义", f"{topic} 的核心原理", f"{topic} 的应用场景"]}


def write_article(state: ArticleState) -> dict:
    lines = "\n".join(f"- {item}" for item in state["outline"])
    return {"article": f"# {state['topic']}\n\n{lines}"}


def build_parent_graph():
    # 拿到编译好的子图
    research_graph = build_research_subgraph()

    builder = StateGraph(ArticleState)
    # 子图作为节点：直接传编译后的 graph 对象
    builder.add_node("research", research_graph)
    builder.add_node("make_outline", make_outline)
    builder.add_node("write_article", write_article)

    builder.add_edge(START, "research")
    builder.add_edge("research", "make_outline")
    builder.add_edge("make_outline", "write_article")
    builder.add_edge("write_article", END)

    return builder.compile()


# ═══════════════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════════════

def run_demo(topic: str = "LangGraph Subgraph") -> dict:
    graph = build_parent_graph()

    print("=" * 60)
    print("stream() — 父图视角（子图内部不可见）：")
    for chunk in graph.stream(
        {
            "topic": topic,
            "research_done": False,
            "outline": [],
            "article": "",
        }
    ):
        print(f"  父图事件: {pformat(chunk, width=80)}")

    print()

    print("stream(subgraphs=True) — 子图内部可见：")
    for chunk in graph.stream(
        {
            "topic": topic,
            "research_done": False,
            "outline": [],
            "article": "",
        },
        subgraphs=True,
    ):
        print(f"  事件: {pformat(chunk, width=80)}")

    final = graph.invoke(
        {
            "topic": topic,
            "research_done": False,
            "outline": [],
            "article": "",
        }
    )
    print()
    print("最终文章：")
    print(final["article"])
    return final


if __name__ == "__main__":
    run_demo()
