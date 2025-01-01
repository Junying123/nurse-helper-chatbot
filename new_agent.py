from llm import llm
from graph import graph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from tools.hospital import hospital_cypher_qa
from tools.patient import patient_cypher_qa
from tools.physician import physician_cypher_qa
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st 
from langchain_core.runnables import RunnableConfig  


@tool
def explore_hospital(question: str) -> str:
    """Provide information about patient questions using Cypher and context from GraphCypherQAChain"""
    return hospital_cypher_qa.invoke(question)

@tool
def explore_patient(question: str) -> str:
    """Provide information about patient questions using Cypher and context from GraphCypherQAChain."""
    return patient_cypher_qa.invoke(question)


tools = [explore_hospital, explore_patient]



system_message = SystemMessage(content="""
You are a nursing expert providing information about nursing care.
Be as helpful as possible and return as much information as possible.
Only answer questions related to checkups, hospitals, or patients.

You must rely solely on the information provided in the context and the tools available to you. 
Do not use any pre-trained knowledge.

You **must** use the tools provided for every relevant question. 
Do not decline to use a tool, as it is essential for generating accurate responses.

When generating responses:
Always follow the required format strictly.
Use the actual Cypher examples from the `GraphCypherQAChain` to construct accurate queries.
Validate the information retrieved from the tools before providing a final answer.
If the information is insufficient, indicate that further clarification is needed.

Use the following format to provide a response:
```
Thought
Tool_used (e.g. explore_patient)
Response
```

Begin!

Previous conversation history:
{{chat_history}}

New input: {{input}}
""")

memory = MemorySaver()

langgraph_agent_executor = create_react_agent(llm, tools, state_modifier=system_message, checkpointer=memory)

# Define the memory function
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent executor
agent_executor = RunnableWithMessageHistory(
    langgraph_agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and streams responses to be printed in the console.
    Returns the final output content from the last AIMessage after invocation.
    """
    # Define the messages structure using HumanMessage
    messages = [HumanMessage(content=user_input)]
    
    # Get the session ID and thread ID
    session_id = get_session_id()  # Assuming this generates a unique session ID
    thread_id = get_session_id()    # You can use the same function or create a new one for thread ID

    # Create a RunnableConfig instance
    config = RunnableConfig(
        configurable={
            "session_id": session_id,
            "thread_id": thread_id
        }
    )

    # Initialize the StreamlitCallbackHandler
    st_callback = StreamlitCallbackHandler(st.container())

    # Initialize a variable to hold the final output content
    final_output_content = ""

    # Prepare the input for the invoke method
    input_data = {
        "messages": messages,
        "input": user_input  # Ensure the input key is included
    }

    # Invoke the agent with the messages and configuration, including the callback
    response = agent_executor.invoke(input_data, config, callbacks=[st_callback])
    
    # Print the entire response to understand its structure
    st.write("Response from agent:", response)

    # Check if the response contains messages
    if 'messages' in response:
        for message in response['messages']:
            if isinstance(message, AIMessage):
               
                final_output_content = message.content  

    # Print the final output content in Streamlit
    if final_output_content:
        st.write(final_output_content)  # Display the final output content in the Streamlit app

    # Return the final output content
    return final_output_content

