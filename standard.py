import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.retrievers import BM25Retriever
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(file_path, chunk_size=500, chunk_overlap=10):
    """
    Load a document from the specified file path and split it into smaller chunks.

    Parameters:
    - file_path (str): The path to the document file.
    - chunk_size (int): The maximum size of each chunk (default is 1000).
    - chunk_overlap (int): The number of overlapping characters between chunks (default is 10).

    Returns:
    - list: A list of split document chunks.
    """
    try:
        # Load the document
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(documents)

        return split_docs
    except Exception as e:
        raise RuntimeError(f"Error splitting document: {str(e)}")
    
def load_vector_store():
    """
    Initialize and load the vector store from Pinecone.

    Returns:
        vector_store: The vector store object.
    """
    try:
        # Get Pinecone API key from environment variables
        api_key = os.environ.get("PINECONE_API_KEY") or "PINECONE_API_KEY"
        if api_key == "PINECONE_API_KEY":
            raise ValueError("PINECONE_API_KEY is missing in environment variables")

        # Initialize Pinecone client
        pinecone = Pinecone(api_key=api_key)

        # Specify the index name and retrieve the index
        index_name = "nurse-practice"
        index = pinecone.Index(index_name)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create and return the vector store
        vector_store = PineconeVectorStore(index, embeddings)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Error initializing vector store: {str(e)}")


def create_bm25_retriever(documents):
    """
    Create a BM25 retriever using the given documents.

    Parameters:
        documents (list): A list of documents to initialize the retriever.

    Returns:
        bm25_retriever: The BM25 retriever object.
    """
    try:
        bm25_retriever = BM25Retriever.from_documents(documents)
        return bm25_retriever
    except Exception as e:
        raise RuntimeError(f"Error creating BM25 retriever: {str(e)}")


def hybrid_query(vector_store, bm25_retriever, query, vector_k=5, bm25_k=5):
    """
    Perform a hybrid query using vector-based and BM25 retrieval methods.

    Parameters:
        vector_store: The vector store instance.
        bm25_retriever: The BM25 retriever instance.
        query (str): The search query.
        vector_k (int): Number of results to retrieve from vector search.
        bm25_k (int): Number of results to retrieve from BM25 search.

    Returns:
        combined_results (list): Combined results from both methods.
    """
    try:
        # Perform vector-based retrieval
        vector_results = vector_store.similarity_search(query, k=vector_k)

        # Perform BM25-based retrieval
        bm25_results = bm25_retriever.invoke(query)

        # Combine the results
        combined_results = vector_results + bm25_results
        return combined_results
    except Exception as e:
        raise RuntimeError(f"Error performing hybrid query: {str(e)}")


def get_context_retrieval(retriever):
    """
    Create a context-aware retriever chain.

    Parameters:
        retriever: The retriever instance.

    Returns:
        retriever_chain: A history-aware retriever chain.
    """
    try:
        llm = ChatOpenAI()
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        return retriever_chain
    except Exception as e:
        raise RuntimeError(f"Error creating context-aware retriever: {str(e)}")


def get_conversational_rag_chain(retriever_chain):
    """
    Create a retrieval-augmented generation (RAG) chain for conversational queries.

    Parameters:
        retriever_chain: The retriever chain object.

    Returns:
        retrieval_chain: The RAG chain for conversational queries.
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are assigned as an expert professional in nurse pratice standards. Your role is to assist interim nurse in solving problems and scenarios presented to you. When they seek pratice standards, you should list the most relevant standards in points clearly and accordingly based on retriever. Your expertise will guide them in making informed decisions."),
            ("user", "Query: {input}\nContext: {context}"),
            ("system", "Example 1:\nQuery: What is the purpose of nursing practice standards?\nResponse: The nursing service standards include: continuous care for patients 24/7, using the autonomous nursing process, regular patient visits each shift, infection prevention measures, hygiene education for patients and relatives, maintaining equipment and medicines, preparing patients for discharge, helping patients adapt to hospital environments, promoting self-care for patients, and ensuring patient satisfaction."),
            ("system", "Example 2:\nQuery: How do nurses prevent infections in hospitals?\nResponse: Nurses prevent infections by washing hands before, during, and after patient contact, using protective equipment like gloves and masks, disinfecting and sterilizing equipment, keeping patient rooms clean twice daily, separating infectious patients, and properly disposing of needles and sharp objects."),
            ("system", "Example 3:\nQuery: What steps do nurses follow to discharge a patient?\nResponse: Steps for discharge include assessing patient readiness for self-care, educating on monitoring symptoms, hygiene, diet, and rest, providing medicine instructions and follow-up appointments, and ensuring referral documents are prepared if needed."),
        ])
        combine_docs_chain = create_stuff_documents_chain(
            llm=ChatOpenAI(temperature=0.9),
            prompt=prompt,
        )
        retrieval_chain = create_retrieval_chain(
            retriever_chain,
            combine_docs_chain,
        )
        return retrieval_chain
    except Exception as e:
        raise RuntimeError(f"Error creating RAG chain: {str(e)}")


def get_response(retriever, chat_history, user_input, vector_store):
    """
    Generate a response using a RAG-based conversational chain.

    Parameters:
        retriever: The retriever instance.
        chat_history (list): List of previous chat messages.
        user_input (str): User's input query.
        vector_store: The vector store instance.

    Returns:
        response (str): The generated response.
    """
    try:
        # Create retriever and RAG chain
        retriever_chain = get_context_retrieval(retriever)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

        # Perform hybrid retrieval
        results = hybrid_query(vector_store, retriever, user_input)

        # Print retrieved documents (debugging purpose)
        print("Retrieved Documents:")
        for doc in results:
            print(f"Content: {doc.page_content}")

        # Generate and return the response
        result = conversation_rag_chain.invoke({
            "chat_history": chat_history,
            "input": user_input,
        })
        return result["answer"]
    except Exception as e:
        raise RuntimeError(f"Error generating response: {str(e)}")
