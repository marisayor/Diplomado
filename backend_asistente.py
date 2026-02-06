import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

app = Flask(__name__)
CORS(app)

# --- 1. Configuración de la API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variables globales de estado
qa_chain = None
is_initializing = False
init_error = None

def background_setup():
    """Proceso de carga en segundo plano para evitar el timeout de Render/Gunicorn"""
    global qa_chain, is_initializing, init_error
    
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
            model="models/text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa para forzar uso de nueva clave y evitar corrupción
        if os.path.exists(PERSIST_DIRECTORY):
            print("SISTEMA: Limpiando base de datos persistente anterior...")
            shutil.rmtree(PERSIST_DIRECTORY)

        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            print(f"SISTEMA: Buscando archivos en {PDF_FOLDER_PATH}...")
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                    print(f"SISTEMA: Cargado con éxito {filename}")
                except Exception as e:
                    print(f"Error cargando {filename}: {e}")
        
        if not documents:
            init_error = "No se encontraron PDFs en la carpeta especificada."
            print(f"SISTEMA: {init_error}")
            return

        # Procesamiento de texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Creación de la base vectorial (Este es el paso que suele tardar)
        print(f"SISTEMA: Creando base vectorial con {len(chunks)} fragmentos...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Configurar el modelo Gemini más reciente y estable
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        template = """Eres un experto en educación en diabetes de la UCV. 
        Utiliza exclusivamente el siguiente contexto para responder de forma pedagógica.
        Si la información no está en el contexto, indícalo claramente.
        
        Contexto: {context}
        Pregunta: {question}
        Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("SISTEMA: ¡Asistente RAG listo para recibir consultas!")

    except Exception as e:
        init_error = str(e)
        print(f"ERROR EN RAG: {e}")
        traceback.print_exc()
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def home():
    status = "online"
    if is_initializing:
        status = "inicializando documentos"
    elif qa_chain:
        status = "listo"
    elif init_error:
        status = f"error: {init_error}"
        
    return jsonify({
        "status": status, 
        "rag_ready": qa_chain is not None,
        "is_initializing": is_initializing
    })

@app.route('/ask', methods=['POST'])
def ask():
    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "El asistente aún está procesando los documentos (esto toma 1-2 minutos al iniciar). Por favor, intenta de nuevo en un momento."}), 503
        if init_error:
            return jsonify({"response": f"El asistente no pudo iniciarse correctamente: {init_error}"}), 500
        
        # Si por alguna razón no se ha lanzado el hilo (failsafe)
        thread = threading.Thread(target=background_setup)
        thread.start()
        return jsonify({"response": "Iniciando el motor de búsqueda. Por favor, espera un minuto."}), 503

    data = request.get_json()
    user_question = data.get('question')

    if not user_question:
        return jsonify({"response": "No enviaste ninguna pregunta."}), 400

    try:
        answer = qa_chain.run(user_question)
        return jsonify({"response": answer})
    except Exception as e:
        print(f"Error en consulta: {e}")
        return jsonify({"response": f"Ocurrió un error al procesar tu pregunta: {str(e)}"}), 500

# Iniciar la carga en segundo plano apenas arranca el script
# Esto ocurre ANTES de que Flask tome el control del puerto
setup_thread = threading.Thread(target=background_setup)
setup_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
