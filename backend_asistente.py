import os
import shutil
import threading
import time
import gc
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# Importaciones ajustadas para LangChain 1.x / Community 0.4.x
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
except ImportError as e:
    print(f"ERROR DE IMPORTACIÓN: {e}")

# --- Configuración de Flask ---
app = Flask(__name__)
# CORS configurado para evitar errores de seguridad en el navegador
CORS(app, resources={r"/*": {"origins": "*",
                              "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
                            }})

# --- Configuración de API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variables de estado global
qa_chain = None
is_initializing = False
init_error = None
thread_started = False

# Rutas absolutas para el entorno de Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER_PATH = os.path.join(BASE_DIR, "Archivos PDF")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db_diabetes")

def background_setup():
    """
    Configura la IA en un hilo separado. 
    Permite que Flask responda 'Live' de inmediato mientras se procesan los PDFs.
    """
    global qa_chain, is_initializing, init_error
    
    print("SISTEMA: Iniciando procesamiento de documentos en segundo plano...")
    is_initializing = True
    init_error = None
    
    try:
        if not API_KEY:
            raise ValueError("La variable GOOGLE_API_KEY no está configurada en Render.")

        # Inicialización de Embeddings (Modelo estable v4)
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa para evitar conflictos de persistencia
        if os.path.exists(PERSIST_DIRECTORY):
            print("SISTEMA: Eliminando base de datos previa...")
            shutil.rmtree(PERSIST_DIRECTORY)
            gc.collect()
            time.sleep(2)

        # Carga de documentos PDF
        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            print(f"SISTEMA: Encontrados {len(pdf_files)} archivos.")
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"  - Error en {filename}: {e}")
        
        if not documents:
            raise ValueError(f"No se encontraron PDFs en: {PDF_FOLDER_PATH}")

        # División de texto (Optimizado para memoria)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        del documents
        gc.collect()

        print(f"SISTEMA: Indexando {len(chunks)} fragmentos...")
        
        # Creación de la base vectorial Chroma
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Configuración del Modelo (Gemini 1.5 Flash)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        # Prompt académico personalizado para la UCV
        template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela (UCV).
        Responde de forma pedagógica basándote EXCLUSIVAMENTE en el contexto proporcionado.
        
        Contexto: {context}
        Pregunta: {question}
        
        Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Construcción de la cadena RAG
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("SISTEMA: ¡IA lista para recibir consultas!")

    except Exception as e:
        init_error = str(e)
        print(f"ERROR CRÍTICO: {traceback.format_exc()}")
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def home():
    """Endpoint de salud: Activa el hilo de carga al ser visitado"""
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
    """Endpoint para procesar preguntas del chat"""
    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "Estoy analizando los manuales de la UCV. Por favor, reintenta en un minuto."}), 503
        return jsonify({"response": f"El motor de IA no pudo iniciar: {init_error}"}), 500

    try:
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({"response": "La pregunta no puede estar vacía."}), 400
            
        # Invocación de la cadena RAG
        result = qa_chain.invoke({"query": question})
        return jsonify({"response": result["result"]})
    except Exception as e:
        print(f"Error en consulta: {e}")
        return jsonify({"response": f"Error interno: {str(e)}"}), 500

if __name__ == '__main__':
    # Render asigna el puerto mediante la variable de entorno PORT
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
