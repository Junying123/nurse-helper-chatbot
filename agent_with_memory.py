from llm import llm
from graph import graph
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from tools.hospital import hospital_cypher_qa
from tools.patient import patient_cypher_qa
from tools.physician import physician_cypher_qa
from tools.checkup import checkup_cypher_qa
from tools.review import get_review
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st 
from langchain_core.runnables import RunnableConfig  
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def explore_hospital(question: str) -> str:
    """Provide information about hospital-related questions using Cypher."""
    return hospital_cypher_qa.invoke(question)

@tool
def explore_patient(question: str) -> str:
    """Provide information about patient-related questions using Cypher."""
    return patient_cypher_qa.invoke(question)

@tool
def explore_checkup(question: str) -> str:
    """Provide information about checkup-related questions using Cypher."""
    return checkup_cypher_qa.invoke(question)

@tool
def explore_physician(question: str) -> str:
    """Provide information about physician-related questions using Cypher."""
    return physician_cypher_qa.invoke(question)
    
@tool
def explore_review(question: str) -> str:
    """Provide information about reviews."""
    return get_review(question)




tools = [
    explore_hospital,
    explore_patient,
    explore_checkup,
    explore_review,
    explore_physician,
]


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tool_node = ToolNode(tools)

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}

# Build graph
graph_builder = StateGraph(MessagesState)

graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("generate", generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)


memory = MemorySaver()
graph_memory = graph_builder.compile(checkpointer=memory)

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

system_message = SystemMessage(content="""

Thought: What action and insight you need from the context?
Tool_used: What tool you use to retrieve the information?
Relationship: Draw the relationship between the entities in the context in the markdown format (e.g. Patient Node -> Checkup Node).
Response: list the details in bulletpoint to make it more readable

Use the following format to display a response:
Thought: \n
Tool_used: \n
Relationship: \n
Response: \n
""")

langgraph_agent_executor = create_react_agent(llm, tools, state_modifier=system_message, checkpointer=memory)

import streamlit as st

def print_stream(stream):
    """Display messages from the stream in Streamlit."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            st.write(message)  # Display tuple messages directly
        else:
            st.markdown(message.pretty_print())  # Use markdown for formatted output

def generate_response(query):
    thread_id = get_session_id()
    session_id = thread_id  # Use the same ID for session and thread
    message_history = get_memory(session_id)

    checkpoint_ns = "your_namespace"  # Replace with your logic for namespace
    checkpoint_id = f"{session_id}_{thread_id}"  # Example of generating a unique checkpoint ID

    config = RunnableConfig({"configurable": {"thread_id": "1"}})

    # Call the stream and display the output in Streamlit
    stream = langgraph_agent_executor.stream({"messages": [(query)]}, config=config, stream_mode="values")
    
    response_content = ""
    tool_messages = []  # List to hold tool messages
    
    for message in stream:
        if isinstance(message["messages"][-1], AIMessage):
            response_content += message["messages"][-1].content.strip()  # Extract string content from AIMessage.
        else:
            tool_messages.append(message)  # Collect tool messages
    
    return response_content.strip(), tool_messages  # Return both content and tool messages
