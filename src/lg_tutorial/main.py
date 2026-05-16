from lg_tutorial.chat_graph import build_graph, chat_once


def main() -> None:
    graph = build_graph()
    messages = []

    print("输入 exit 退出。")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = chat_once(graph, messages, user_input)
        assistant_message = result["messages"][-1]
        print(f"AI: {assistant_message.content}")
        messages = result["messages"]


if __name__ == "__main__":
    main()
