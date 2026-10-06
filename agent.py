import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 1. Load environment variables (.env file)
load_dotenv()

# Initialize the model (using GPT-4o-mini for fast routing)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Define the State
class SupportState(TypedDict):
    customer_query: str
    category: str
    response: str

# 3. Define Nodes
def classify_intent_node(state: SupportState):
    query = state["customer_query"]
    
    prompt = f"""
    You are an issue classifier for a customer support team.
    Classify the following customer query into exactly ONE of two categories: 'technical' or 'escalate'.
    
    - Use 'escalate' for billing, refunds, payments, accounts, or complaints.
    - Use 'technical' for bugs, code issues, errors, setup, or app crashes.
    
    Query: {query}
    Category (reply with ONLY the word 'technical' or 'escalate'):
    """
    
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    
    return {"category": category}

def technical_agent_node(state: SupportState):
    query = state["customer_query"]
    
    prompt = f"You are an expert technical support assistant. Provide a helpful, step-by-step resolution to this issue: {query}"
    response = llm.invoke(prompt)
    
    return {"response": response.content}

def escalation_agent_node(state: SupportState):
    query = state["customer_query"]
    
    prompt = f"Draft a polite message to the customer explaining that their request regarding '{query}' has been handed off to our human billing/management team."
    response = llm.invoke(prompt)
    
    return {"response": response.content}

# 4. Routing logic function
def route_by_category(state: SupportState):
    return state["category"]

# 5. Assemble the Graph
workflow = StateGraph(SupportState)

# Add Nodes
workflow.add_node("classify", classify_intent_node)
workflow.add_node("technical", technical_agent_node)
workflow.add_node("escalation", escalation_agent_node)

# Define Flow / Edges
workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "technical": "technical",
        "escalate": "escalation"
    }
)

workflow.add_edge("technical", END)
workflow.add_edge("escalation", END)

# Compile Graph
app = workflow.compile()

# 6. Execute Test Queries
if __name__ == "__main__":
    # Test 1: Technical Issue
    tech_input = {"customer_query": "I am getting a 500 internal server error when calling the API."}
    print("--- Testing Technical Route ---")
    result_tech = app.invoke(tech_input)
    print(f"Routed To: {result_tech['category']}")
    print(f"Response: {result_tech['response']}\n")

    # Test 2: Billing / Escalation Issue
    escalate_input = {"customer_query": "I was charged twice on my credit card this month."}
    print("--- Testing Escalation Route ---")
    result_esc = app.invoke(escalate_input)
    print(f"Routed To: {result_esc['category']}")
    print(f"Response: {result_esc['response']}")