import streamlit as st
from streamlit_option_menu import option_menu
from utils import write_message
from agent import generate_response
from general import prompt_response
from standard import load_vector_store, create_bm25_retriever, splitter, get_response, hybrid_query
from dotenv import load_dotenv
from chat_history import save_message, load_chat_history


load_dotenv()


st.set_page_config(page_title="Nurse Helper", page_icon="🧑‍⚕️", layout="wide")


try:
    file_path = "nurse-helper-chatbot/source/document_nurse1.txt"
    split_docs = splitter(file_path)
    vector_store = load_vector_store()
    bm25_retriever = create_bm25_retriever(split_docs)
except Exception as e:
    st.error(f"Initialization Error: {str(e)}")

primary_color = st.get_option("theme.primaryColor")
text_color = st.get_option("theme.textColor")
background_color = st.get_option("theme.backgroundColor")

# Sidebar 
with st.sidebar:
    st.title("👩‍⚕️ Nurse Helper Chatbot")
    st.subheader("Navigation")
    selected_tab = option_menu(
        menu_title=None,
        options=["Description","General Knowledge", "Work Hub Daily Operation", "Professional Standards Guideline", "Chat History"],
        icons=["info-circle", "lightbulb", "briefcase", "book", "clock-history"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "2px", "background-color": background_color},
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "4px",
                "color": text_color,
                "font-weight": "bold",
                "--hover-color": "#ddd",
            },
            "nav-link-selected": {
                "background-color": primary_color,
                "color": "white",
                "font-weight": "bold",
                "font-size": "18px",
            },
            "icon": {
                "font-size": "22px",
                "font-weight": "bold",
            },
        },
    )
    st.image("nurse-helper-chatbot/assets/logo.png", use_container_width=True)


if 'general_messages' not in st.session_state:
    st.session_state.general_messages = [{"role": "assistant", "content": "Hi, I'm the Nurse Helper Chatbot! How can I help you?"}]

if 'work_hub_messages' not in st.session_state:
    st.session_state.work_hub_messages = [{"role": "assistant", "content": "Hi, I'm the Nurse Helper Chatbot! How can I help you?"}]

if 'standards_chat_history' not in st.session_state:
    st.session_state.standards_chat_history = [{"role": "assistant", "content": "Hi, I'm the Nurse Helper Chatbot! How can I help you?"}]


if selected_tab == "Description":
    st.title("📝 Description")
    st.write("Welcome to the Nurse Helper Chatbot!")
    st.write("""
    ### Features:
    1. **💼 Work Hub (Daily Operation):** Manage hospital operations and patient info.
    2. **📘 Professional Standards Guideline:** Access nursing best practices and internal guidelines.
    """)

elif selected_tab == "General Knowledge":
    st.title("🗨️ General Knowledge")
    st.write("This page is for general nurse knowledge chat not covered by other tools.")
    
    for message in st.session_state.general_messages:
        write_message(message['role'], message['content'])

    if prompt := st.chat_input("What do you need to know about hospital details? 🏥 🩺"):
        write_message("user", prompt)
        st.session_state.general_messages.append({"role": "user", "content": prompt})
        save_message("user", prompt)  

        with st.spinner("Thinking..."):
            try:
                response = prompt_response(prompt)
                st.session_state.general_messages.append({"role": "assistant", "content": response})
                save_message("assistant", response)  
                write_message("assistant", response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif selected_tab == "Work Hub Daily Operation":
    st.title("💼 Work Hub (Daily Operation)")
    st.write("This page is for hospital operations and patient info queries.")
    
    
    for message in st.session_state.work_hub_messages:
        write_message(message['role'], message['content'])

    
    if prompt := st.chat_input("What do you need to know about hospital details? 🏥 🩺"):
        write_message("user", prompt)
        st.session_state.work_hub_messages.append({"role": "user", "content": prompt})
        save_message("user", prompt)  
        
        with st.spinner("Thinking..."):
            try:
                response = generate_response(prompt)
                st.session_state.work_hub_messages.append({"role": "assistant", "content": response})
                save_message("assistant", response)  
                write_message("assistant", response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif selected_tab == "Professional Standards Guideline":
    st.title("📘 Professional Standards Guideline")
    st.write("This page provides nursing best practices and internal guidelines.")

    
    for message in st.session_state.standards_chat_history:
        write_message(message['role'], message['content'])

       
    if user_input := st.chat_input("Ask me about nursing practice standards 📃"):
        write_message("user", user_input)
        st.session_state.standards_chat_history.append({"role": "user", "content": user_input})
        save_message("user", user_input)  

        with st.spinner("Retrieving response..."):
            try:
                results = hybrid_query(vector_store, bm25_retriever, user_input)
                retrieved_content = "\n\n".join(
                    [f"**Document {i+1}:**\n{doc.page_content}" for i, doc in enumerate(results)]
                )
                
                with st.expander("Retrieved Documents", expanded=True):
                    st.write("Here is the retrieved context:")
                    st.write(retrieved_content)

                response = get_response(
                    bm25_retriever,
                    st.session_state.standards_chat_history,
                    user_input,
                    vector_store,
                )

                st.session_state.standards_chat_history.append({"role": "assistant", "content": response})
                save_message("assistant", response)  # Save assistant response to Neo4j
                write_message("assistant", response)

            except Exception as e:
                st.error(f"Error: {str(e)}")

elif selected_tab == "Chat History":
    st.title("🗂️ Chat History")

    
    try:
        chat_history = load_chat_history()  
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        chat_history = []

    
    if chat_history:
        st.subheader("All Chat History from Neo4j")

        grouped_messages = []
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):  # Ensure there is a response for the input
                grouped_messages.append((chat_history[i], chat_history[i + 1]))

        for i, (user_message, assistant_message) in enumerate(grouped_messages):
           
            if isinstance(user_message, dict):
                user_content = user_message.get("content", "No content available")
                st.markdown(f"**Input {i + 1}:**\n{user_content}", unsafe_allow_html=True)
            elif hasattr(user_message, "content"):
                user_content = getattr(user_message, "content", "No content available")
                st.markdown(f"**Input {i + 1}:**\n{user_content}", unsafe_allow_html=True)

            
            if isinstance(assistant_message, dict):
                assistant_content = assistant_message.get("content", "No content available")
                st.markdown(f"**Response {i + 1}:**\n{assistant_content}", unsafe_allow_html=True)
            elif hasattr(assistant_message, "content"):
                assistant_content = getattr(assistant_message, "content", "No content available")
                st.markdown(f"**Response {i + 1}:**\n{assistant_content}", unsafe_allow_html=True)

            
            if i < len(grouped_messages) - 1:
                st.divider()
    else:
        st.write("No chat history found.")










