import os
import time
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
    return text


def get_text_chunks(text):
    """Split the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_text(text)


def embed_with_retry(texts, retries=3, delay=2):
    """Embed text chunks with retry logic for handling rate limits."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    for attempt in range(retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error("Retry attempts exhausted.")
                raise e


def get_vector_store(text_chunks):
    """Create a FAISS vector store from the embedded text chunks."""
    embedded_texts = embed_with_retry(text_chunks)
    return FAISS.from_texts(text_chunks, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))


def get_conversational_chain(vector_store):
    """Set up a conversational retrieval chain using ConversationBufferMemory."""
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Key to store chat history
        return_messages=True,       # Return messages as part of the output
    )

    # Configure the conversational LLM
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=512,
    )

    # Create the conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )

    return conversation_chain
