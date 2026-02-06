import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
from langchain_community.document_loaders import PyPDFLoader
from flask import Flask, request, jsonify
from flask_cors import CORS
import shutil
import traceback
import threading
import time

app = Flask(__name__)
CORS(app)

# --- 1. Configuración de la API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variables globales de estado
retrieval_chain = None
is_initializing = False
init_error = None
thread_started = False

def background_setup():
    """Proceso de carga en segundo plano para evitar el timeout de Render/Gunicorn"""
    global retrieval_chain, is_initializing, init_error
    
    # Pequeña pausa para permitir que Flask se vincule al puerto primero
    time.sleep(2)
    
    print("SISTEMA: Iniciando procesamiento de documentos en segundo plano...")
    is_initializing = True
    init_error = None
    
    try:
        if not API_KEY:
            init_error = "No hay GOOGLE_API_KEY configurada."
            print(f"CRÍTICO: {init_error}")
            return

        genai.configure(api_key=API_KEY)
        os.environ["GOOGLE_API_KEY"] = API_KEY

        PDF_FOLDER_PATH = "Archivos PDF"
        PERSIST_DIRECTORY = "./chroma_db_diabetes"

        # Modelo de Embeddings
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa para evitar conflictos de esquema
        if os.path.exists(PERSIST_DIRECTORY):
            print("SISTEMA: Limpiando base de datos persistente anterior...")
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except:
                pass

        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            print(f"SISTEMA: Encontrados {len(pdf_files)} archivos PDF.")
            
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                    print(f"SISTEMA: Cargado {filename}")
                except Exception as e:
                    print(f"Error cargando {filename}: {e}")
        
        if not documents:
            init_error = "No se encontraron PDFs en la carpeta 'Archivos PDF'."
            print(f"SISTEMA: {init_error}")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        print(f"SISTEMA: Creando base vectorial con {len(chunks)} fragmentos...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        # Prompt académico personalizado (UCV)
        template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran conocimiento y autoeficacia en el manejo de su condición. 

        Utiliza el siguiente contexto para responder de forma pedagógica. Si la respuesta no está en el contexto, indícalo claramente.
        
        Contexto: {context}
        Pregunta: {question}
        
        Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # ✅ NUEVO: Crear cadena con enfoque moderno de LangChain v0.2+
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(
            retriever=vector_db.as_retriever(),
            combine_docs_chain=document_chain
        )
        
        print("SISTEMA: ¡Asistente RAG listo!")

    except Exception as e:
        init_error = str(e)
        print(f"ERROR EN RAG: {e}")
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
        "rag_ready": retrieval_chain is not None,
        "is_initializing": is_initializing,
        "error": init_error
    })

@app.route('/ask', methods=['POST'])
def ask():
    global retrieval_chain, thread_started
    
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()

    if retrieval_chain is None:
        if is_initializing:
            return jsonify({"response": "El asistente se está iniciando y procesando los documentos PDF. Por favor, espera unos segundos e intenta de nuevo."}), 503
        if init_error:
            return jsonify({"response": f"Error de configuración: {init_error}"}), 500
        return jsonify({"response": "El motor RAG no está listo."}), 503

    data = request.get_json()
    user_question = data.get('question')

    try:
        # ✅ NUEVO: Usar invoke() con la nueva cadena
        response = retrieval_chain.invoke({"input": user_question})
        return jsonify({"response": response["answer"]})
    except Exception as e:
        return jsonify({"response": f"Error al procesar la consulta: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
