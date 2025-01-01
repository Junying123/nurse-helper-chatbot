import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import llm

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a nurse expert assistant providing information about nursing care."),
        ("human", "{input}"),
    ]
)

# Create the nurse chat chain
nurse_chat = chat_prompt | llm | StrOutputParser()

# Function to get the response from the nurse chat
def prompt_response(user_input):
    """Invoke the nurse chat model and return the response."""
    return nurse_chat.invoke({"input": user_input})