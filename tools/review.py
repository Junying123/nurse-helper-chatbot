import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              
    graph=graph,                              
    index_name="review_index",                
    node_label="review",                      
    text_node_property="Review",              
    embedding_node_property="embedding"       
)



retriever = neo4jvector.as_retriever()



instructions = (
    "To answer the question when human ask about reviews. Please answer the question in details"
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)



question_answer_chain = create_stuff_documents_chain(llm, prompt)
review_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)



def get_review(input):
    return review_retriever.invoke({"input": input})

