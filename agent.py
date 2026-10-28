import os
from langgraph.graph import StateGraph,END,START
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from typing import TypedDict, Annotated
from pydantic import Field

from langchain_core.messages import BaseMessage
import operator

# loading env variables
load_dotenv()

# model initialization
model = ChatOpenRouter(
    model="anthropic/claude-sonnet-4.5",
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=1000,
)

# state definition
class State(TypedDict):
    input: Annotated[str, Field(description="User input in chat!!!")]
    generated_text: Annotated[str, Field(description="Ai generated Message!!!")]
    messages: Annotated[list[BaseMessage], operator.add]

# stategraph initialization
graph = StateGraph(State)

initial_state = {"input":"What is the capital of pakistan?", "generated_text":"", "messages":[]}

# node 1: chat
def chat(state: State):
    ai_response = model.invoke(state["input"])
    return {"generated_text": ai_response.content, "messages": [ai_response]}


graph.add_node("chat",chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

workflow = graph.compile()

response = workflow.invoke(initial_state)
ai_message = response["generated_text"]

print(ai_message)




