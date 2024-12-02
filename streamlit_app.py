
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import google.generativeai as genai

from fl import get_text_chunks, get_vector_store, process_files
from map import get_mapreduce_chain

# Charger les variables d'environnement
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Fonction pour gérer les questions utilisateur
def user_input(user_question, raw_text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Recherche des documents similaires
    docs = new_db.similarity_search(user_question)

    # Diviser le texte brut en chunks
    docs = get_text_chunks(raw_text)

    # Chaîne MapReduce
    chain = get_mapreduce_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    print(response)

    st.write("Reply: ", response)

# Fonction principale Streamlit
def main():
    st.set_page_config(page_title="Chat Documents")
    st.header("Chat with Documents using Gemini💁")

    # Menu dans la barre latérale pour uploader les fichiers
    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload your PDF, Image, or Text Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if uploaded_files:
                    raw_text = process_files(uploaded_files)
                    if not raw_text.strip():
                        st.error("No valid text found in the uploaded files.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)

                    st.success("Files processed successfully!")
                else:
                    st.error("Please upload files before processing.")

    # Interface pour poser une question
    user_question = st.text_input("Ask a Question from Files")
    if user_question and uploaded_files:
        raw_text = process_files(uploaded_files)
        user_input(user_question, raw_text)
    elif user_question:
        st.error("Please upload files before asking a question.")

# Chaînes auxiliaires

if __name__ == "__main__":
    main()
