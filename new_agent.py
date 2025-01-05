from llm import llm
from graph import graph
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
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
from pyvis.network import Network
import json

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
    """Provide information about reviews using Cypher."""
    return get_review(question)




tools = [
    explore_hospital,
    explore_patient,
    explore_checkup,
    explore_review,
    explore_physician,
]

system_message = SystemMessage(content="""

Thought: What action and insight you need from the context?
Cypher_query: What cypher query you use to retrieve the information?
Tool_used: What tool you use to retrieve the information?
Relationship: Draw the relationship between the entities in the context in the markdown format.
Response: list the details in bulletpoint to make it more readable

Use the following format to provide a response:
Thought: \n
Cypher_query: \n
Tool_used: \n
Relationship: \n
Response: \n
""")



langgraph_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

def generate_response(query: str):
    """
    Invoke the LangGraph agent with a user query and return only the output content.
    
    Args:
        query (str): The user query.
    
    Returns:
        str: The output content from the agent's response.
    """ 

    thread_id = get_session_id()    
    session_id = get_session_id()  
    message_history = get_memory(session_id)

    config = RunnableConfig(
        configurable={
            "session_id": session_id,
            "thread_id": thread_id,
        }
    )

    
    messages = langgraph_agent_executor.invoke({
        "messages": [HumanMessage(content=query)],
        "chat_history": message_history, 
        "config" : config

          
    })

    
    if "messages" in messages:
        for message in messages["messages"]:
            if isinstance(message, ToolMessage):
                with st.expander("Tool Message", expanded=False):  
                    st.write("Content:", message.content)  

    
    output_content = messages["messages"][-1].content if "messages" in messages else "No response generated."

    
    return output_content

