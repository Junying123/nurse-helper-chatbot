# chat_history.py
from graph import graph 
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
import streamlit as st
from utils import get_session_id  

# Initialize the chat message history using the existing graph connection
session_id = get_session_id()  
chat_history = Neo4jChatMessageHistory(
    session_id=session_id,
    graph=graph  
)

# Function to save a message to Neo4j
def save_message(role, content):
    if role == "user":
        chat_history.add_user_message(content)
    else:
        chat_history.add_ai_message(content)

# Function to load chat history from Neo4j
def load_chat_history():
    return chat_history.messages  # Returns messages in a structured format
