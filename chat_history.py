
from graph import graph 
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
import streamlit as st
from utils import get_session_id  


session_id = get_session_id()  
chat_history = Neo4jChatMessageHistory(
    session_id=session_id,
    graph=graph  
)


def save_message(role, content):
    if role == "user":
        chat_history.add_user_message(content)
    else:
        chat_history.add_ai_message(content)


def load_chat_history():
    return chat_history.messages  
