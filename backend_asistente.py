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
import gc

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
    
    # Pausa de seguridad inicial
    time.sleep(5)
    
    print("SISTEMA: Iniciando procesamiento de documentos en segundo plano...")
    is_initializing = True
    init_error = None
    
    try:
        if not API_KEY:
            init_error = "GOOGLE_API_KEY no configurada en el entorno."
            print(f"ERROR: {init_error}")
            return

        genai.configure(api_key=API_KEY)

        PDF_FOLDER_PATH = "Archivos PDF"
        PERSIST_DIRECTORY = "./chroma_db_diabetes"

        # Nombre del modelo de embeddings (sin prefijo models/ para evitar error 400)
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa para evitar bloqueos de archivos
        if os.path.exists(PERSIST_DIRECTORY):
            print("SISTEMA: Limpiando base de datos previa...")
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                gc.collect()
                time.sleep(2)
            except Exception as e:
                print(f"Aviso: No se pudo limpiar la carpeta: {e}")

        # Carga de PDFs
        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            print(f"SISTEMA: Encontrados {len(pdf_files)} archivos.")
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                    print(f"  - Cargado: {filename}")
                except Exception as e:
                    print(f"  - Error cargando {filename}: {e}")
        
        if not documents:
            init_error = "No se encontraron documentos válidos en 'Archivos PDF'."
            print(f"ERROR: {init_error}")
            return

        # Procesamiento de texto con limpieza de memoria
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        del documents
        gc.collect()

        print(f"SISTEMA: Indexando {len(chunks)} fragmentos...")
        
        # Base vectorial
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Modelo Gemini 1.5 Flash (más ligero para Render)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        # Prompt UCV Académico
        template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela (UCV).
        Responde basándote en el contexto para educar a otros profesionales de forma pedagógica.
        
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
        
        print("SISTEMA: ¡IA lista para responder consultas!")

    except Exception as e:
        init_error = f"Excepción en background_setup: {str(e)}"
        print(f"ERROR CRÍTICO: {init_error}")
        traceback.print_exc()
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def home():
    """Ruta raíz para salud del servidor y despertar el hilo"""
    global thread_started
    if not thread_started:
        thread_started = True
        # Usamos un hilo daemon para que no bloquee el cierre del proceso
        threading.Thread(target=background_setup, daemon=True).start()
        
    return jsonify({
        "status": "online", 
        "ia_ready": qa_chain is not None,
        "is_loading": is_initializing,
        "error": init_error,
        "info": "Si ia_ready es false y is_loading es false, revisa el campo error."
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Recibe preguntas del frontend"""
    global qa_chain, thread_started
    
    # Asegurar que el hilo se inicie si la primera petición es /ask
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()

    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "Estoy analizando los manuales. Por favor, espera 60 segundos y reintenta."}), 503
        return jsonify({"response": f"El motor de IA falló al iniciar: {init_error}"}), 500

    try:
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({"response": "Pregunta vacía."}), 400
            
        result = qa_chain.invoke({"query": question})
        return jsonify({"response": result["result"]})
    except Exception as e:
        print(f"Error en /ask: {e}")
        return jsonify({"response": f"Error interno: {str(e)}"}), 500

if __name__ == '__main__':
    # Render asigna el puerto mediante la variable de entorno PORT
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
