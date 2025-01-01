import streamlit as st
import uuid

def write_message(role, content, save=True):
    """
    Writes a message to the Streamlit chat interface.

    Args:
        role (str): The role of the message sender ('user' or 'assistant').
        content (str): The content of the message.
        save (bool): Whether to save the message in the session state.
    """
    if save:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": role, "content": content})

    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").write(content)


def get_session_id():
    """
    Generates or retrieves a unique session ID.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id
