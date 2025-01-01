import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                               # (1) Embedding function or model
    graph=graph,                              # (2) Neo4j graph connection
    index_name="review_index",                # (3) Name of the index for reviews
    node_label="review",                      # (4) Node label for reviews
    text_node_property="Review",              # (5) Property to retrieve for text (singular)
    embedding_node_property="embedding"       # (6) Property for storing embeddings
)



retriever = neo4jvector.as_retriever()


# tag::prompt[]
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
# end::prompt[]

# tag::chain[]
question_answer_chain = create_stuff_documents_chain(llm, prompt)
review_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)
# end::chain[]


def get_review(input):
    return review_retriever.invoke({"input": input})

