import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # Importer le type Document
# Fonction pour extraire le texte des fichiers PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Fonction pour extraire le texte des images via OCR
def get_image_text(image_files):
    text = ""
    for image_file in image_files:
        image = Image.open(image_file)
        text += pytesseract.image_to_string(image, lang="fra")
    return text

# Fonction pour traiter les fichiers (PDF et Images)
def process_files(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            all_text += get_pdf_text([file])
        elif file.name.endswith((".png", ".jpg", ".jpeg")):
            all_text += get_image_text([file])
    return all_text

# Fonction pour diviser le texte en chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

# Fonction pour créer un index vectoriel à partir des chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Extraire le contenu textuel des Documents
    text_data = [doc.page_content for doc in text_chunks]
    vector_store = FAISS.from_texts(text_data, embedding=embeddings)
    vector_store.save_local("faiss_index")