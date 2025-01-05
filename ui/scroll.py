import streamlit as st
# Function to inject JavaScript for scrolling
def inject_scroll_to_bottom_js():
    js = """
    <script>
    function scrollToBottom() {
        window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
    }
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)

# Function to inject CSS for absolute positioning
def inject_css():
    css = """
    <style>
    .scroll-to-bottom {
        position: fixed;
        bottom: 130px;  /* Adjust this value based on your layout */
        right: 20px;     /* Adjust this value based on your layout */
        z-index: 1000;   /* Ensure it appears above other elements */
        background-color: #f0f0f0;
        border: none;
        border-radius: 5px;
        padding: 10px;
        cursor: pointer;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
