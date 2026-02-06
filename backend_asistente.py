import google.generativeai as genai
# Importaciones actualizadas para compatibilidad con las últimas versiones de LangChain
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

app = Flask(__name__)
CORS(app)

# --- 1. Configuración de la API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variables globales de estado
qa_chain = None
is_initializing = False
init_error = None
thread_started = False

def background_setup():
    """Proceso de carga en segundo plano para evitar el timeout de Render/Gunicorn"""
    global qa_chain, is_initializing, init_error
    
    # Pequeña pausa para dejar que el servidor Flask se asiente en el puerto
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
            model="models/text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa
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

        template = """Eres un experto en educación en diabetes de la UCV. 
        Utiliza el contexto para responder de forma pedagógica.
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
    # Iniciar el hilo en la primera visita si no ha empezado
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
    
    # Failsafe para iniciar el hilo si la primera petición es /ask
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()

    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "El asistente se está iniciando. Por favor, espera 60 segundos."}), 503
        if init_error:
            return jsonify({"response": f"Error de inicio: {init_error}"}), 500
        return jsonify({"response": "El motor RAG se está activando, intenta en un momento."}), 503

    data = request.get_json()
    user_question = data.get('question')

    try:
        answer = qa_chain.run(user_question)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"response": f"Error en consulta: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
```
eof

### Acción requerida en tu `requirements.txt`:
Para que el código anterior funcione, asegúrate de que tu archivo `requirements.txt` en GitHub incluya estas líneas (o actualízalas):

```text
flask
flask-cors
gunicorn
google-generativeai
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
pypdf
chromadb
