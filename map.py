from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain

# Fonction pour la phase map
def get_map_chain():
    map_template = """
   Réponds à la question de l'utilisateur uniquement en te basant sur la transcription disponible dans FAISS. 
   Utilise le texte transcrit pour comprendre le contexte et fournir une réponse claire et concise. 
   N'invente rien et ne propose pas de détails qui ne figurent pas dans la transcription. 
   Si la réponse n'est pas clairement disponible dans la transcription, indique-le.
    {docs}
    Question : {question}
    Réponse :
    """
    map_prompt = PromptTemplate(template=map_template, input_variables=["docs", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    return LLMChain(llm=model, prompt=map_prompt)

# Fonction pour la phase reduce
def get_reduce_chain():
    reduce_template = """
    Voici les réponses partielles générées :
    {docs}
    Fusionnez-les en une réponse concise et complète :
    Réponse :
    """
    reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["docs"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)

    # Utilisation de StuffDocumentsChain pour combiner les réponses
    return StuffDocumentsChain(
        llm_chain=reduce_llm_chain,
        document_variable_name="docs"
    )

# Chaîne MapReduceDocuments sans ReduceDocumentsChain
def get_mapreduce_chain():
    map_chain = get_map_chain()
    reduce_chain = get_reduce_chain()

    # Chaîne combinée MapReduce
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_chain,  # Remplace ReduceDocumentsChain par StuffDocumentsChain
        document_variable_name="docs",
        return_intermediate_steps=False
    )
