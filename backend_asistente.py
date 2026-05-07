import os
import shutil
import gc
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# IMPORTANTE: Importaciones de LangChain (Aquí es donde estaba el error)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate # Ruta corregida
from langchain_community.chains.retrieval_qa.base import RetrievalQA

# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app)

# --- Variables de Estado para el Frontend ---
ia_status = {
    "is_loading": True,
    "ia_ready": False,
    "error": None,
    "status": "online"
}

# --- 1. Configuración de API ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 2. Procesamiento de Documentos ---
PDF_FOLDER_PATH = "Archivos PDF"
PERSIST_DIRECTORY = "./chroma_db_diabetes"
qa_chain = None

def inicializar_ia():
    global qa_chain, ia_status
    try:
        # Usamos el modelo que nos funcionó: embedding-001
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=API_KEY
        )
        
        # Cargar o crear base de datos
        if os.path.exists(PDF_FOLDER_PATH):
            documents = []
            for filename in os.listdir(PDF_FOLDER_PATH):
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
                persist_directory=PERSIST_DIRECTORY
            )
            
            # Configurar LLM y Prompt
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
            
            custom_prompt_template = """
            Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela (UCV).
            Responde basándote EXCLUSIVAMENTE en el contexto para educar de forma pedagógica.
            
            Contexto: {context}
            Pregunta: {question}
            
            Respuesta:"""
            
            # Aquí se usa el PromptTemplate que daba error
            CUSTOM_PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever(),
                chain_type_kwargs={"prompt": CUSTOM_PROMPT}
            )
            
            ia_status["ia_ready"] = True
            ia_status["is_loading"] = False
            print("SISTEMA: IA Lista.")
        else:
            ia_status["error"] = "No se encontró la carpeta de PDFs"
            ia_status["is_loading"] = False
    except Exception as e:
        ia_status["error"] = str(e)
        ia_status["is_loading"] = False
        print(f"ERROR: {e}")

# Ejecutar la carga en segundo plano
import threading
threading.Thread(target=inicializar_ia).start()

# --- 3. Endpoints ---
@app.route('/')
def status():
    return jsonify(ia_status)

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"response": "La IA aún se está cargando..."}), 503
    data = request.get_json()
    respuesta = qa_chain.run(data.get('question'))
    return jsonify({"response": respuesta})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
