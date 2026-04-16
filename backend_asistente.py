import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
import shutil
import traceback
import threading
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GOOGLE_API_KEY")
retrieval_chain = None
is_initializing = False
init_error = None
thread_started = False

def background_setup():
    global retrieval_chain, is_initializing, init_error
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

        # ✅ RUTA ABSOLUTA CORRECTA PARA RENDER
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PDF_FOLDER_PATH = os.path.join(BASE_DIR, "Archivos PDF")
        PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db_diabetes")

        # ✅ DIAGNÓSTICO: Ver qué ve el sistema
        print(f"SISTEMA: Directorio base: {BASE_DIR}")
        print(f"SISTEMA: Ruta de PDFs: {PDF_FOLDER_PATH}")
        print(f"SISTEMA: ¿Existe la carpeta?: {os.path.exists(PDF_FOLDER_PATH)}")

        if not os.path.exists(PDF_FOLDER_PATH):
            init_error = "Carpeta 'Archivos PDF' no encontrada en el servidor."
            print(f"SISTEMA: ¡ERROR! {init_error}")
            return

        pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
        print(f"SISTEMA: Archivos PDF detectados: {pdf_files}")

        if not pdf_files:
            init_error = "No hay archivos PDF en la carpeta 'Archivos PDF'."
            print(f"SISTEMA: {init_error}")
            return

        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",  # ✅ Formato correcto
            google_api_key=API_KEY
        )

        # Limpiar base vectorial anterior
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)

        # Cargar documentos
        documents = []
        for filename in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                documents.extend(loader.load())
                print(f"SISTEMA: Cargado: {filename}")
            except Exception as e:
                print(f"Error cargando {filename}: {e}")

        if not documents:
            init_error = "No se pudo extraer texto de los PDFs."
            print(f"SISTEMA: {init_error}")
            return

        # Dividir y crear base vectorial
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"SISTEMA: Creando base vectorial con {len(chunks)} fragmentos...")

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Configurar LLM y cadena
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)  # ✅ Modelo real y gratuito

        template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran conocimiento y autoeficacia en el manejo de su condición. 

Utiliza el siguiente contexto para responder de forma pedagógica. Si la respuesta no está en el contexto, indícalo claramente.
        
Contexto: {context}
Pregunta: {question}
        
Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
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
            return jsonify({"response": "El asistente se está iniciando. Por favor, espera unos segundos e intenta de nuevo."}), 503
        if init_error:
            return jsonify({"response": f"Error de configuración: {init_error}"}), 500
        return jsonify({"response": "El motor RAG no está listo."}), 503

    data = request.get_json()
    user_question = data.get('question')

    try:
        response = retrieval_chain.invoke({"input": user_question})
        return jsonify({"response": response["answer"]})
    except Exception as e:
        return jsonify({"response": f"Error al procesar la consulta: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)  # ✅ Correcto para Render
