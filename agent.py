
import os
import time
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import RetryPolicy


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")


# --------------------------------------------------
# 2. Initialize the model
# --------------------------------------------------

model = ChatOpenRouter(
    model="anthropic/claude-sonnet-4.5",
    openrouter_api_key=api_key,
    max_tokens=1000,
)


# --------------------------------------------------
# 3. Configure SQLite persistence
# --------------------------------------------------

conn = sqlite3.connect(
    "memory_graph.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn)
checkpointer.setup()


# --------------------------------------------------
# 4. Define the graph state
# --------------------------------------------------

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    status: str


# --------------------------------------------------
# 5. Chat node
# --------------------------------------------------

def chat(state: State):
    print("\n[CHAT] Node started")

    ai_response = model.invoke(state["messages"])

    print("[CHAT] Node completed")

    return {
        "messages": [ai_response],
        "status": "chat_completed",
    }


# --------------------------------------------------
# 6. Waiting node with a simulated failure
# --------------------------------------------------

# TEST ONLY: This variable resets whenever Python restarts.
waiting_attempts = 0


def waiting_node(state: State):
    global waiting_attempts
    waiting_attempts += 1

    print(f"\n[WAIT] Attempt {waiting_attempts}")

    # Simulate a temporary failure on the first attempt.
    if waiting_attempts == 1:
        print("[WAIT] Simulating a temporary failure...")
        raise RuntimeError("Simulated temporary failure")

    print("[WAIT] Waiting for 10 seconds...")
    time.sleep(10)

    print("[WAIT] Node completed successfully")

    return {
        "status": "completed",
    }


# --------------------------------------------------
# 7. Build the graph and configure retries
# --------------------------------------------------

builder = StateGraph(State)

builder.add_node("chat", chat)

builder.add_node(
    "waiting_node",
    waiting_node,
    retry_policy=RetryPolicy(max_attempts=3),
)

builder.add_edge(START, "chat")
builder.add_edge("chat", "waiting_node")
builder.add_edge("waiting_node", END)

workflow = builder.compile(checkpointer=checkpointer)


# --------------------------------------------------
# 8. Interactive terminal chat
# --------------------------------------------------

def main():
    thread_id = input("Thread ID (e.g. retry_test_1): ").strip()

    if not thread_id:
        print("Thread ID cannot be empty.")
        return

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    print("\nChat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            continue

        try:
            result = workflow.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ],
                    "status": "running",
                },
                config=config,
            )

            print(f"\nAI: {result['messages'][-1].content}\n")
            print(f"[GRAPH] Final status: {result['status']}\n")

        except Exception as error:
            print(f"\n[GRAPH ERROR] {type(error).__name__}: {error}")
            print("[GRAPH ERROR] This invocation failed.")
            print("The terminal is still running. Try another message.\n")


# --------------------------------------------------
# 9. Start the application and close SQLite safely
# --------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close()
        print("\n[DATABASE] SQLite connection closed.")
