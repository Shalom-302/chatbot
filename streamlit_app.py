# Importations des bibliothèques
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document  # Importer le type Document
from dotenv import load_dotenv
import google.generativeai as genai

# Charger les variables d'environnement depuis un fichier .env pour sécuriser les clés API
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Configurer l'API Google Generative AI avec la clé API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Fonction pour extraire le texte des fichiers PDF
def get_pdf_text(pdf_docs):
    text = ""  # Initialiser la variable pour stocker le texte
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Créer un lecteur PDF pour chaque fichier
        for page in pdf_reader.pages:
            text += page.extract_text() 
            # Extraire le texte de chaque page et l'ajouter à la variable `text`
    return text  # Retourner le texte extrait

# Fonction pour extraire le texte des images via OCR
def get_image_text(image_files):
    text = ""  # Initialiser la variable pour stocker le texte extrait
    for image_file in image_files:
        image = Image.open(image_file)  # Ouvrir l'image
        text += pytesseract.image_to_string(image)# Utiliser Tesseract pour extraire le texte
        print(text)
    return text  # Retourner le texte extrait

# Fonction pour extraire le texte des fichiers texte (.txt)
def get_text_file_content(text_files):
    text = ""  # Initialiser la variable pour stocker le texte
    for text_file in text_files:
        text += text_file.read().decode("utf-8")  # Lire et décoder le contenu du fichier texte
    return text  # Retourner le texte extrait

# Fonction pour traiter tous les fichiers téléchargés et extraire le texte
def process_files(uploaded_files):
    all_text = ""  # Initialiser une variable pour stocker le texte de tous les fichiers
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            all_text += get_pdf_text([file])  # Extraire le texte des fichiers PDF
        elif file.name.endswith((".png", ".jpg", ".jpeg")):
            all_text += get_image_text([file])  # Extraire le texte des images
        elif file.name.endswith(".txt"):
            all_text += get_text_file_content([file])  # Extraire le texte des fichiers texte
    return all_text  # Retourner le texte extrait de tous les fichiers

# Fonction pour diviser le texte en segments (chunks)
def get_text_chunks(text):
    # Utiliser RecursiveCharacterTextSplitter pour découper le texte en segments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Diviser le texte en plusieurs morceaux
    return chunks  # Retourner les morceaux

# Fonction pour créer et sauvegarder un index vectoriel basé sur les chunks de texte
def get_vector_store(documents):
    # Utiliser GoogleGenerativeAIEmbeddings pour obtenir les embeddings (vecteurs) des documents
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Créer un index FAISS à partir des embeddings
    vector_store = FAISS.from_texts([doc.page_content for doc in documents], embedding=embeddings)
    
    # Sauvegarder l'index FAISS localement sous le nom "faiss_index"
    vector_store.save_local("faiss_index")

# Fonction pour configurer une chaîne de traitement conversationnelle (QA)
def get_conversational_chain():
    # Template du prompt pour poser des questions en fonction du contenu PDF
    prompt_template = """
    À partir des documents fournis, extrayez les informations suivantes :
    - Total HT (Hors Taxes)
    - Total TVA (Taxe sur la Valeur Ajoutée)
    - Total TTC (Toutes Taxes Comprises)
    
    Si ces informations ne sont pas disponibles, répondez simplement "Les informations ne sont pas disponibles dans le document".

    Document fourni : {context}
    \n
    Réponse :
    """
    
    # Utiliser le modèle Google Generative AI pour le traitement des questions et réponses
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    
    # Créer un template de prompt avec des variables d'entrée "context"
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    
    # Charger une chaîne de questions-réponses (QA chain)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain  # Retourner la chaîne

# Fonction principale de l'application
def main():
    # Configurer le titre de la page Streamlit
    st.set_page_config(page_title="Chat PDF and Documents")  
    
    # Afficher un en-tête pour l'application
    st.header("Document Chatbot (PDF, Images, TXT)")

    # Créer une barre latérale pour uploader les fichiers
    uploaded_files = st.file_uploader("Upload your PDF, Image, or Text Files", accept_multiple_files=True)
    
    # Créer un bouton pour soumettre et traiter les fichiers
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            # Traiter les fichiers et extraire le texte
            all_text = process_files(uploaded_files)
            
            # Diviser le texte en chunks
            text_chunks = get_text_chunks(all_text)
            
            # Créer des objets Document pour chaque chunk
            documents = []
            for chunk in text_chunks:
                documents.append(Document(page_content=chunk))
            
            # Créer et sauvegarder un index vectoriel basé sur les documents
            get_vector_store(documents)
            
            # Afficher un message de succès une fois le traitement terminé
            st.success("Files processed successfully!")

            # Génération automatique de la réponse
            conversational_chain = get_conversational_chain()
            response = conversational_chain.invoke(
                {"input_documents": documents, "question": "Générer un résumé des informations financières extraites."}
            )
            
            # Afficher la réponse dans l'interface Streamlit
            st.write("Generated Response: ", response["output_text"])

if __name__ == "__main__":
    main()

