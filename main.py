# from typing import Annotated, List
# from PyPDF2 import PdfReader
# from fastapi import APIRouter, Request, UploadFile, File
# import os
# import google.generativeai as genai
# from PIL import Image
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import pytesseract
# from langchain_community.vectorstores import FAISS
# from io import BytesIO
# from langchain.prompts import PromptTemplate
# from glob import glob
# import asyncio
# import concurrent.futures
# from functools import partial

# from langchain.chains.question_answering import load_qa_chain
# from dotenv import load_dotenv

# router = APIRouter(prefix="/streamlit", tags=["Streamlit"])

# load_dotenv()
# # os.getenv("AIzaSyDRwZkwArrKtGbg8GJlQE_4hnRTvzA4gI8")
# genai.configure(api_key="AIzaSyD50jeE2wcguwlL9gSk5Mc8A6G8T-CnW3o")

# def get_pdf_text(pdf_file: UploadFile):
#     text = ""
#     pdf_reader = PdfReader(pdf_file.file)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def get_image_text(image_file: UploadFile):
#     file_content =  image_file.file.read()
#     image = Image.open(BytesIO(file_content))
#     return pytesseract.image_to_string(image, lang='fra') 

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Fonction pour obtenir une chaîne conversationnelle
# def get_conversational_chain():
#     prompt_template = """
#     Répondez à la question aussi précisément que possible en utilisant uniquement le contexte fourni, qui concerne les trajets, les lignes de bus, les stations et le trafic urbain dans le Grand Abidjan. 
#     Assurez-vous de fournir tous les détails disponibles liés à ces sujets. 
#     Si la réponse ne se trouve pas dans le contexte fourni, ou si la question ne concerne pas les trajets et le trafic urbain, répondez simplement : 
#     "Je suis désolé, je ne peux répondre qu'aux questions liées aux trajets et au trafic urbain dans le Grand Abidjan."
#     Ne fournissez pas de réponse incorrecte ou hors contexte.

#     Contexte :\n {context}?\n
#     Question : \n{question}\n

#     Réponse :
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# @router.post("/", summary="Upload your PDF, Image, or Text Files")
# async def updaload_streamlit_pdf_file(
#     request: Request,
#     uploaded_files: Annotated[list[UploadFile], File(...)]
# ) -> ResponseModel:
#     all_text = "" 
#     for file in uploaded_files:
#         if file.filename.endswith(".pdf"):
#             all_text += get_pdf_text([file])
#         elif file.filename.endswith((".png", ".jpg", ".jpeg")):
#             all_text += get_image_text(file)
#         elif file.filename.endswith(".json"):
#             all_text += get_json_text(file)
#         else:
#             response_base.fail(request=request) 
#         file.file.seek(0)          

#     text_chunks = get_text_chunks(all_text)
#     get_vector_store(text_chunks)
#     return response_base.success(request=request)

# async def read_file_async(file_path: str):
#     loop = asyncio.get_event_loop()
#     if file_path.endswith(".pdf"):
#         with open(file_path, "rb") as pdf_file:
#             return await loop.run_in_executor(None, partial(get_pdf_text, pdf_file))
#     elif file_path.endswith((".png", ".jpg", ".jpeg")):
#         with open(file_path, "rb") as image_file:
#             return await loop.run_in_executor(None, partial(get_image_text, image_file))
#     elif file_path.endswith(".json"):
#         with open(file_path, "r") as json_file:
#             return await loop.run_in_executor(None, json_file.read)
#     return ""

# async def process_all_files_async(file_paths: str):
#     tasks = [read_file_async(file_path) for file_path in file_paths]
#     print("TASKS => ", tasks)
#     all_texts = await asyncio.gather(*tasks)
#     print("A")
#     return "".join(all_texts)


# @router.get("/process-directory")
# async def process_directory(request: Request):
#     root_directory = STREAMLIT_DATA_ROOT_DIR
#     file_patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.json"]

#     all_files = get_all_files_from_directory(root_directory, file_patterns)
#     all_text = await process_all_files_async(all_files)         
    
#     text_chunks = get_text_chunks(all_text)
#     print("B")
#     get_vector_store(text_chunks) 
#     print("C")

#     return response_base.success(request=request)


# @router.post("/question", summary="Ask question")
# async def ask_question(
#     request: Request,
#     obj: Question
# ) -> ResponseModel:
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = vector_store.similarity_search(obj.question)
#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents": docs, "question": obj.question},
#         return_only_outputs=True
#     )

#     return response_base.success(request=request, data=response["output_text"])

# import speech_recognition as sr

# filename = "Lesson-001-Anglais.wav"     

# # initialize the recognizer
# r = sr.Recognizer()


# # open the file
# with sr.AudioFile(filename) as source:
#     # listen for the data (load audio to memory)
#     audio_data = r.record(source)
#     # recognize (convert from speech to text)
#     text = r.recognize_google(audio_data)
#     print(text)
    
#!/usr/bin/env python3



from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import tempfile  # Pour les fichiers temporaires

# Charger le fichier audio principal
audio_file = "02-Track-2.wav"
sound = AudioSegment.from_wav(audio_file)

# Découper l'audio en segments basés sur les silences
audio_chunks = split_on_silence(
    sound, 
    min_silence_len=500,  # Silence minimal en millisecondes
    silence_thresh=sound.dBFS-14,  # Seuil pour détecter le silence
    keep_silence=500  # Garder un peu de silence autour des segments
)

# Initialiser le reconnaisseur
recognizer = sr.Recognizer()

# Transcrire chaque chunk
full_text = ""  # Contiendra le texte complet

for i, chunk in enumerate(audio_chunks):
    try:
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            # Exporter le segment dans le fichier temporaire
            chunk.export(temp_file.name, format="wav")
            temp_file.seek(0)  # Remettre le curseur au début

            # Charger le fichier temporaire dans SpeechRecognition
            with sr.AudioFile(temp_file.name) as source:
                audio = recognizer.record(source)
                # Transcrire avec Google API
                text = recognizer.recognize_google(audio, language="fr-FR")
                full_text += text + " "  # Ajouter le texte transcrit
    except sr.UnknownValueError:
        print(f"Segment {i}: Impossible de reconnaître le texte")
    except sr.RequestError as e:
        print(f"Erreur API Google : {e}")

# Afficher la transcription complète
print("Transcription complète :")
print(full_text)
