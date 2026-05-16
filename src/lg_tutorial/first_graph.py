from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ArticleState(TypedDict):
    topic: str
    outline: list[str]
    article: str


def make_outline(state: ArticleState) -> dict:
    topic = state["topic"]
    return {
        "outline": [
            f"{topic} 是什么",
            f"{topic} 解决什么问题",
            f"{topic} 最核心的组成部分",
        ]
    }


def write_article(state: ArticleState) -> dict:
    lines = "\n".join(f"- {item}" for item in state["outline"])
    return {"article": f"# {state['topic']}\n\n{lines}"}


def build_graph():
    builder = StateGraph(ArticleState)
    builder.add_node("make_outline", make_outline)
    builder.add_node("write_article", write_article)
    builder.add_edge(START, "make_outline")
    builder.add_edge("make_outline", "write_article")
    builder.add_edge("write_article", END)
    return builder.compile()


def run_demo() -> dict:
    graph = build_graph()
    return graph.invoke({"topic": "LangGraph", "outline": [], "article": ""})


if __name__ == "__main__":
    result = run_demo()
    print(result["article"])
