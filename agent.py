from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id
from tools.review import get_review
from tools.patient import patient_cypher_qa
from tools.hospital import hospital_cypher_qa
from tools.physician import physician_cypher_qa
from tools.checkup import checkup_cypher_qa
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st 

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a nurse expert assistant providing information about nursing care."),
        ("human", "{input}"),
    ]
)

nurse_chat = chat_prompt | llm | StrOutputParser()
"""
    Tool.from_function(
        name="General Knowledge Chat for Nurse",
        description="For general nurse knowledge chat not covered by other tools",
        func=nurse_chat.invoke,
    ),
"""
tools = [

    Tool.from_function(
        name="Patients Information",  
        description="""Provide information about patient questions using Cypher and context from GraphCypherQAChain.
                       List the response concisely in bullet point for readability.
                    """,
        func=patient_cypher_qa, 
    ),

    Tool.from_function(
        name="Hospital Information",  
        description="""Provide information about hospital questions using Cypher and context from GraphCypherQAChain.
                       List the response concisely in bullet point for readability.
                    """,
        func=hospital_cypher_qa, 
    ),

    Tool.from_function(
        name="Physician Information",  
        description="""Provide information about physician questions using Cypher and context from GraphCypherQAChain.
                       List the response concisely in bullet point for readability.
                    """,
        func=physician_cypher_qa, 
    ),

     Tool.from_function(
        name="Checkup Information",  
        description="""U must use the full context from checkup_cypher_qa to answer.
                       List the response concisely in bullet point for readability.
                    """,
        func=checkup_cypher_qa, 
    ),

    Tool.from_function(
        name="Review Search",  
        description="""Provide information about reviews about hospital, environment, staff service, quality care.
                       List the response concisely in bullet point for readability.
                    """,
        func=get_review, 
    ),

]


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a nursing expert providing information about nursing care.
Be as helpful as possible and return as much information as possible.
Only answer questions related to checkups, hospitals, or patients.

You must rely solely on the information provided in the context and the tools available to you. 
Do not use any pre-trained knowledge.



TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please follow this format:
```
Thought: Do I need to use a tool? Yes 
Action: [the action to take, should be one of {tool_names}] 
Action Input: [the input to the action] 
Observation: [the result of the action]
```

You **must** use the tools provided for every relevant question. 
Do not decline to use a tool, as it is essential for generating accurate responses.

When generating responses:
Always follow the required format strictly.
Use the actual Cypher examples from the `GraphCypherQAChain` to construct accurate queries.
Validate the information retrieved from the tools before providing a final answer.
If the information is insufficient, indicate that further clarification is needed.

Use the following format to provide a response:
```
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=6,
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    
    st_callback = StreamlitCallbackHandler(st.container())

    
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}, "callbacks": [st_callback]}
    )

    
    return response['output']


