import google.generativeai as genai
import os
import shutil
import traceback
import threading
import time
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS

# Importaciones de LangChain modernas
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

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

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER_PATH = os.path.join(BASE_DIR, "Archivos PDF")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db_diabetes")

def background_setup():
    """Inicializa la base de datos vectorial y el modelo en segundo plano"""
    global qa_chain, is_initializing, init_error
    
    print("SISTEMA: Esperando estabilidad del servidor...")
    time.sleep(5)
    
    is_initializing = True
    init_error = None
    
    try:
        if not API_KEY:
            raise ValueError("La variable GOOGLE_API_KEY no está configurada en Render.")

        genai.configure(api_key=API_KEY)

        # Configuración de Embeddings (text-embedding-004 es el más reciente)
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa si existe para evitar archivos corruptos
        if os.path.exists(PERSIST_DIRECTORY):
            print("SISTEMA: Limpiando base de datos persistente anterior...")
            shutil.rmtree(PERSIST_DIRECTORY)
            gc.collect()

        # Cargar PDFs
        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            print(f"SISTEMA: Encontrados {len(pdf_files)} archivos PDF.")
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                    print(f"  - Cargado: {filename}")
                except Exception as e:
                    print(f"  - Error cargando {filename}: {e}")
        
        if not documents:
            raise ValueError(f"No se encontraron archivos PDF en la ruta: {PDF_FOLDER_PATH}")

        # Dividir texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        del documents
        gc.collect()

        print(f"SISTEMA: Creando base vectorial con {len(chunks)} fragmentos...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Configurar LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        # Prompt personalizado (UCV)
        template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela (UCV).
        Responde basándote en el contexto para educar a otros profesionales de forma pedagógica.
        
        Contexto: {context}
        Pregunta: {question}
        
        Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Crear cadena RAG
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("SISTEMA: ¡IA lista para responder!")

    except Exception as e:
        init_error = str(e)
        print(f"ERROR CRÍTICO: {init_error}")
        traceback.print_exc()
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def health_check():
    """Ruta para verificar estado y activar el hilo de carga"""
    global thread_started
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()
        
    return jsonify({
        "status": "online", 
        "ia_ready": qa_chain is not None,
        "is_loading": is_initializing,
        "error": init_error
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Endpoint principal para el chat"""
    global qa_chain
    
    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "Estoy analizando los manuales de diabetes. Por favor, espera 60 segundos y reintenta."}), 503
        return jsonify({"response": f"Error de inicio: {init_error or 'Desconocido'}"}), 500

    try:
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({"response": "Por favor, escribe una pregunta."}), 400
            
        result = qa_chain.invoke({"query": question})
        return jsonify({"response": result["result"]})
    except Exception as e:
        print(f"Error en /ask: {e}")
        return jsonify({"response": f"Error interno: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
