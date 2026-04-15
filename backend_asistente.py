import google.generativeai as genai
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

# Inicialización de Flask
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
    Configura el motor RAG en un hilo secundario para evitar timeouts en Render.
    """
    global qa_chain, is_initializing, init_error
    
    # Pausa para asegurar estabilidad inicial del servidor
    time.sleep(3)
    
    print("SISTEMA: Iniciando procesamiento de documentos...")
    is_initializing = True
    init_error = None
    
    try:
        if not API_KEY:
            init_error = "Error: GOOGLE_API_KEY no encontrada."
            print(f"CRÍTICO: {init_error}")
            return

        genai.configure(api_key=API_KEY)

        PDF_FOLDER_PATH = "Archivos PDF"
        PERSIST_DIRECTORY = "./chroma_db_diabetes"

        # CORRECCIÓN ERROR 400: Se usa el nombre del modelo sin "models/"
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa
        if os.path.exists(PERSIST_DIRECTORY):
            print("SISTEMA: Limpiando base de datos anterior...")
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                time.sleep(1)
            except Exception as e:
                print(f"Aviso: No se pudo eliminar DB: {e}")

        # Carga de documentos
        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            print(f"SISTEMA: Encontrados {len(pdf_files)} PDFs.")
            
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                    print(f"SISTEMA: Cargado {filename}")
                except Exception as e:
                    print(f"Error cargando {filename}: {e}")
        
        if not documents:
            init_error = "No hay PDFs en 'Archivos PDF'."
            print(f"SISTEMA: {init_error}")
            return

        # Fragmentación
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        print(f"SISTEMA: Indexando {len(chunks)} fragmentos...")
        
        # Base vectorial
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Configuración LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        # Prompt Académico UCV
        template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela (UCV).
        Responde de forma pedagógica y profesional.
        
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
        
        print("SISTEMA: IA lista.")

    except Exception as e:
        init_error = str(e)
        print(f"ERROR: {e}")
        traceback.print_exc()
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def home():
    global thread_started
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()
        
    return jsonify({
        "status": "online", 
        "rag_ready": qa_chain is not None,
        "is_initializing": is_initializing,
        "error": init_error
    })

@app.route('/ask', methods=['POST'])
def ask():
    global qa_chain, thread_started
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()

    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "Analizando manuales. Espera un momento..."}), 503
        return jsonify({"response": f"Error: {init_error}"}), 500

    data = request.get_json()
    user_question = data.get('question')

    try:
        result = qa_chain.invoke({"query": user_question})
        return jsonify({"response": result["result"]})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
