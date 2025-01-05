import streamlit as st

# Create the LLM
from langchain_openai import ChatOpenAI

import streamlit as st

llm = ChatOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
)


# Create the Embedding model
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key=st.secrets["OPENAI_API_KEY"],
    model="text-embedding-3-small"
)

