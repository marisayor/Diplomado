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
import gc

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GOOGLE_API_KEY")

qa_chain = None
is_initializing = False
init_error = None
thread_started = False

# Rutas absolutas para evitar errores en Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER_PATH = os.path.join(BASE_DIR, "Archivos PDF")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db_diabetes")

def background_setup():
    global qa_chain, is_initializing, init_error
    time.sleep(5)
    is_initializing = True
    
    try:
        if not API_KEY:
            init_error = "API_KEY no encontrada"
            return

        genai.configure(api_key=API_KEY)

        # Nombre del modelo sin "models/"
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=API_KEY
        )

        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            gc.collect()

        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error en {filename}: {e}")
        
        if not documents:
            init_error = f"No hay PDFs en {PDF_FOLDER_PATH}"
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        del documents
        gc.collect()

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        template = """Eres un profesor de la Universidad Central de Venezuela.
        Usa el contexto para responder: {context}
        Pregunta: {question}
        Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        print("SISTEMA: IA lista")

    except Exception as e:
        init_error = str(e)
        traceback.print_exc()
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def home():
    global thread_started
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()
    return jsonify({"status": "online", "ia_ready": qa_chain is not None, "error": init_error})

@app.route('/ask', methods=['POST'])
def ask():
    if qa_chain is None:
        return jsonify({"response": "Iniciando sistema..."}), 503
    data = request.get_json()
    try:
        result = qa_chain.invoke({"query": data.get('question')})
        return jsonify({"response": result["result"]})
    except Exception as e:
        return jsonify({"response": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
