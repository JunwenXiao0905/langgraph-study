from typing import Any
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    task: str
    human_decision: dict[str, Any] | None
    outcome: str


def review_task(state: ApprovalState) -> dict:
    print("review_task: entered node")

    # interrupt() resumes by re-running this node from the top.
    decision = interrupt(
        {
            "question": "Do you approve this task?",
            "task": state["task"],
            "expected_resume_format": {
                "approved": True,
                "comment": "optional note",
            },
        }
    )

    print(f"review_task: resumed with {decision}")
    return {"human_decision": decision}


def apply_decision(state: ApprovalState) -> dict:
    decision = state["human_decision"] or {}
    approved = bool(decision.get("approved"))
    comment = str(decision.get("comment", "")).strip()

    if approved:
        outcome = f"Approved task: {state['task']}"
    else:
        outcome = f"Rejected task: {state['task']}"

    if comment:
        outcome += f" | Comment: {comment}"

    return {"outcome": outcome}


def build_graph():
    memory = InMemorySaver()

    builder = StateGraph(ApprovalState)
    builder.add_node("review_task", review_task)
    builder.add_node("apply_decision", apply_decision)

    builder.add_edge(START, "review_task")
    builder.add_edge("review_task", "apply_decision")
    builder.add_edge("apply_decision", END)

    return builder.compile(checkpointer=memory)


def print_stream(
    graph,
    payload: ApprovalState | Command,
    config: dict[str, Any],
) -> None:
    for chunk in graph.stream(payload, config):
        print(chunk)


def run_demo() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "approval-thread"}}

    print("First run: expect an interrupt")
    print_stream(graph, {"task": "Send the weekly report to the team."}, config)

    snapshot = graph.get_state(config)
    print(f"Pending interrupts: {[item.value for item in snapshot.interrupts]}")
    print(f"Next nodes: {snapshot.next}")
    print()

    print("Resume run: provide a human decision")
    resume_value = {"approved": True, "comment": "Looks good. Send it."}
    print_stream(graph, Command(resume=resume_value), config)

    final_state = graph.get_state(config)
    print()
    print(f"Final outcome: {final_state.values['outcome']}")


if __name__ == "__main__":
    run_demo()
