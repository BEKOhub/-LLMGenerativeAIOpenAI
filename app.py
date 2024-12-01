import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain


def user_input(user_question):
    """Handle user input and update the conversation."""
    response = st.session_state.conversation({"question": user_question})
    for i, message in enumerate(response["chat_history"]):
        if i % 2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Assistant: ", message.content)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Information Retrieval System", layout="wide")
    st.header("Information Retrieval System 💁")

    # Text input for user questions
    user_question = st.text_input("Ask a question based on the uploaded PDF files")

    # Initialize Streamlit state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Handle user input
    if user_question and st.session_state.conversation:
        user_input(user_question)

    # Sidebar for file uploads
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing your files..."):
                # Extract text, chunk it, and create the conversational chain
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Processing complete! Ask your question in the input box.")


if __name__ == "__main__":
    main()
