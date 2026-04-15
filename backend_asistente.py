import google.generativeai as genai
# Importaciones corregidas para versiones modernas
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain_community.document_loaders import PyPDFLoader
from flask import Flask, request, jsonify
from flask_cors import CORS
import shutil
import traceback
import threading
import time

# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app)

# --- Configuración de API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variables de estado global
qa_chain = None
is_initializing = False
init_error = None
thread_started = False

def background_setup():
    """
    Carga los documentos y configura la IA en un hilo separado
    para que el servidor Flask pueda responder 'Live' de inmediato.
    """
    global qa_chain, is_initializing, init_error
    
    # Pausa de seguridad para que Render detecte el puerto abierto
    time.sleep(5)
    
    print("SISTEMA: Iniciando procesamiento de documentos en segundo plano...")
    is_initializing = True
    
    try:
        if not API_KEY:
            init_error = "GOOGLE_API_KEY no configurada."
            return

        genai.configure(api_key=API_KEY)

        PDF_FOLDER_PATH = "Archivos PDF"
        PERSIST_DIRECTORY = "./chroma_db_diabetes"

        # Nombre del modelo de embeddings (sin prefijo models/ para evitar error 400)
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa para ahorrar espacio en disco
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                time.sleep(1)
            except: pass

        # Carga de PDFs
        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error cargando {filename}: {e}")
        
        if not documents:
            # Fallback si no hay archivos
            init_error = "No se encontraron PDFs en la carpeta."
            return

        # Procesamiento de texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Base vectorial
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Modelo Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        # Prompt UCV
        template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela (UCV).
        Responde basándote en el contexto para educar a otros profesionales.
        
        Contexto: {context}
        Pregunta: {question}
        
        Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Cadena RAG
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("SISTEMA: ¡IA lista para responder!")

    except Exception as e:
        init_error = str(e)
        traceback.print_exc()
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def home():
    """Ruta raíz: sirve para despertar la carga inicial"""
    global thread_started
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()
        
    return jsonify({
        "status": "online", 
        "ia_ready": qa_chain is not None,
        "loading": is_initializing,
        "error": init_error
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Recibe preguntas del frontend"""
    global qa_chain
    
    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "Estoy analizando los manuales de la UCV. Dame un minuto..."}), 503
        return jsonify({"response": f"Error de inicio: {init_error or 'Desconocido'}"}), 500

    data = request.get_json()
    question = data.get('question')

    try:
        result = qa_chain.invoke({"query": question})
        return jsonify({"response": result["result"]})
    except Exception as e:
        return jsonify({"response": f"Error procesando: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
