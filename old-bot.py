import streamlit as st
from streamlit_option_menu import option_menu
from utils import write_message
from new_agent import generate_response
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

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

if selected_tab == "Description":
    st.title("📝 Description")
    st.write("Welcome to the Nurse Helper Chatbot!")
    st.write("""
    ### Features:
    1. **💼 Work Hub (Daily Operation):** 
       - Manage hospital operations and patient information efficiently.
       - Retrieve patient details, handle daily tasks, and ensure smooth operations.
    <hr>

    2. **📘 Professional Standards Guideline:** 
       - Access nursing best practices and internal guidelines.
       - Stay updated with the latest standards in patient care and nursing practices.
    <hr>

    3. **🗨️ General Knowledge:** 
       - Engage in general knowledge chats about nursing topics not covered by other tools.
       - Ask questions about hospital policies, procedures, and general healthcare information.
    <hr>

    4. **🗂️ Chat History:** 
       - Review your past interactions with the chatbot.
       - Access chat history stored in Neo4j for reference and learning.
    """, unsafe_allow_html=True)

    st.markdown("""
        <hr>
        <footer style='text-align: center;'>
            <p>Created by: Ng Jun Ying | S2132214</p>
            <p>Final Year Project | Bachelor Degree of Computer Science in Artificial Intelligence at University of Malaya</p>
            <div style='display: flex; justify-content: center; gap: 20px;'>
                <p><a href="mailto:junying9999@gmail.com">Email</a></p>
                <p><a href="mailto:s2132214@siswa.edu.my.com">Institution Email</a></p>
                <p><a href="https://wa.me/60164348929" target="_blank">WhatsApp</a></p>
                <p><a href="https://www.linkedin.com/in/gareth-ng-jy/" target="_blank">LinkedIn</a></p>
            </div>
        </footer>
    """, unsafe_allow_html=True)


elif selected_tab == "General Knowledge":
    st.title("🗨️ General Knowledge")
    st.write("This page is for general nurse knowledge chat not covered by other tools.")
    
    for message in st.session_state.general_messages:
        write_message(message['role'], message['content'])

    if prompt := st.chat_input("What do you need to know about the nurse general knowledge details? 🏥 🩺"):
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

    with st.expander("🧐 No Idea to ask?", expanded=False):
        st.subheader("Medication Management")
        medication_questions = [
            "What are the common side effects of opioids, and how can they be managed in patients?",
            "How should medications like insulin be prescribed and adjusted for diabetic patients?",
            "What precautions should I take when administering high-alert medications such as anticoagulants?",
            "How can I prevent medication errors during the administration process?"
        ]
        
        for question in medication_questions:
            st.write(question)

        st.markdown("---")

        st.subheader("Nutrition")
        nutrition_questions = [
            "What nutritional advice should I give to patients recovering from surgery to promote healing?",
            "How can I help patients with chronic illnesses, like heart disease, make healthier dietary choices?"
        ]
        
        for question in nutrition_questions:
            st.write(question)

        st.markdown("---")

        st.subheader("Technology and Devices")
        tech_questions = [
            "How do I troubleshoot common issues with IV pumps or infusion devices during patient care?",
            "What are the benefits of using electronic health records (EHR) for patient documentation and communication?"
        ]
        
        for question in tech_questions:
            st.write(question)

        st.markdown("---")

        st.subheader("Communication Skills")
        communication_questions = [
            "How can I educate a patient about their new medication regimen in a way they can easily understand?",
            "What strategies can I use to improve collaboration with physicians and other healthcare team members?"
        ]
        
        for question in communication_questions:
            st.write(question)


elif selected_tab == "Work Hub Daily Operation":
    st.title("💼 Work Hub (Daily Operation)")
    st.write("This page is for hospital operations and patient info queries.")
    
    for message in st.session_state.work_hub_messages:
        write_message(message['role'], message['content'])

    
    
    # Chat input field
    prompt = st.chat_input("What do you need to know about hospital,patient,checkup,psyhician's details? 🏥 🩺", key='chat-input')

    if prompt:
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


    with st.expander("💡 No Idea to ask?", expanded=False):
            st.subheader("Hospitals")
            hospital_questions = [
                "Can you give me the Penang General Hospital info?",
                "Which hospital offers specialization in Neurology?",
                "Which hospitals are located at Penang?",
                "Does Penang General Hospital recruit physicians with specialization in Pulmonology?"
            ]
            
            for question in hospital_questions:
                st.write(question)

            st.markdown("---")

            st.subheader("Checkups")
            checkup_questions = [
                "What are the details of checkups conducted in room A314?",
                "What checkups attended by Dr. Aisyah binti Kamaruddin include patient details?",
                "What checkups has the patient with ID 321 undergone?",
                "Retrieve checkups for Dr. Ng Yee Siang that involve emergencies or abnormal test results."
            ]
            
            for question in checkup_questions:
                st.write(question)

            st.markdown("---")

            st.subheader("Patients")
            patient_questions = [
                "Can you give me the patient info with Identity Number 880519-05-9016?",
                "Can you find a patient's information and its checkup with which physician attends the checkup for patient ID 123?",
                "Can you give me patient info for ID 523?"
            ]
            
            for question in patient_questions:
                st.write(question)

            st.markdown("---")

            st.subheader("Physicians")
            physician_questions = [
                "Find doctors with 'Tan' in their name and where they work.",
                "Get information about doctor Dr. Saraswathy Pillai and the hospital where they work.",
                "What checkups attended by Dr. Aisyah binti Kamaruddin include patient details?",
                "Show doctors whose name includes 'Lim' or exactly matches Dr. Ng Yee Siang, along with their hospital details.",
                "Find doctors with license number 48720 and the hospital they work at.",
                "List doctors specializing in Cardiology or Orthopedics at Penang General Hospital and Penang Adventist Hospital."
            ]
            
            for question in physician_questions:
                st.write(question)

            # Section for Reviews
            st.markdown("---")
            st.subheader("Reviews")

            # Emotional Support
            st.write("### Emotional Support")
            emotional_support_questions = [
                "How are the reviews regarding the emotional support provided by nurses during patient care?",
                "What feedback has been shared about nurses addressing patients' fears or anxieties?"
            ]
            
            for question in emotional_support_questions:
                st.write(question)

            # Technical Skills
            st.write("### Technical Skills")
            technical_skills_questions = [
                "How do the reviews describe the technical skills of nurses in performing medical procedures?",
                "Are there any concerns mentioned about nurses handling medical equipment or following protocols?"
            ]
            
            for question in technical_skills_questions:
                st.write(question)

            # Communication
            st.write("### Communication")
            communication_questions = [
                "What do the reviews say about the clarity and effectiveness of communication from nurses?",
                "How is the feedback regarding nurses ensuring patients understand their care plans and instructions?"
            ]
            
            for question in communication_questions:
                st.write(question)

            # Service Quality
            st.write("### Service Quality")
            service_quality_questions = [
                "What is the general perception of the quality of care provided by nurses?",
                "Are there any specific comments about nurses' attentiveness and professionalism?"
            ]
            
            for question in service_quality_questions:
                st.write(question)

            # Timeliness and Efficiency
            st.write("### Timeliness and Efficiency")
            timeliness_questions = [
                "How do the reviews rate nurses' responsiveness to patient requests and call bells?",
                "What is the feedback on nurses ensuring timely administration of treatments and medications?"
            ]
            
            for question in timeliness_questions:
                st.write(question)

            # Hygiene and Cleanliness
            st.write("### Hygiene and Cleanliness")
            hygiene_questions = [
                "Are there any comments about nurses adhering to hygiene protocols, such as handwashing and glove use?",
                "How do patients perceive the nurses’ role in maintaining a clean and safe care environment?"
            ]
            
            for question in hygiene_questions:
                st.write(question)

            # Food/Cafeteria
            st.write("### Food/Cafeteria")
            food_questions = [
                "What do patients say about nurses’ involvement in ensuring meals meet dietary or nutritional needs?",
                "Are there any reviews about nurses helping patients navigate food-related challenges during their stay?"
            ]
            
            for question in food_questions:
                st.write(question)
    

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

    with st.expander("🤔 No Idea to ask?", expanded=False):
        st.subheader("Standards")
        standards_questions = [
            "1. What are the objectives of the Nursing Practice Standards, and how do they contribute to the overall quality of nursing care?",
            "2. Standard 1 for nursing services mentions providing continuous care. Can you elaborate on the shift system for nurses and the criteria for ensuring 24/7 patient care?",
            "3. Can you explain the steps involved in the autonomous nursing process, as described in Standard 2, and how each step contributes to creating a patient-centered care plan?",
            "4. Standard 4 focuses on infection prevention. What specific measures are nurses required to take to prevent hospital-acquired infections and maintain a hygienic environment?",
            "5. Standard 7 outlines the procedures for patient discharge. Can you explain the criteria for assessing a patient's readiness for discharge and the information provided to patients and their families to ensure a smooth transition home?",
            "6. What are the specific duties and responsibilities of a nurse in charge during a day shift, evening shift, and night shift, and how do these roles contribute to the efficient functioning of the ward?",
            "7. Can you walk me through the step-by-step process of admitting a patient, from preparing the necessary documents and equipment to providing initial care and recording patient information?",
            "8. What are the key differences between the eight-hour and twelve-hour working systems for nurses, and what factors might influence the choice of one system over the other?",
            "9. The document mentions 'Ward Round in accordance with the nursing process.' What are the specific recommendations for conducting effective ward rounds, and how do they relate to the nursing process and patient assessment?",
            "10. How do nurses assess the quality of their practice?"
        ]
        
        for question in standards_questions:
            st.write(question)

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