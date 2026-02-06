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
import gc
import traceback

app = Flask(__name__)
CORS(app)

# --- 1. Configuración de la API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variable global para la cadena RAG
qa_chain = None

def inicializar_asistente():
    global qa_chain
    try:
        if not API_KEY:
            print("CRÍTICO: No hay GOOGLE_API_KEY en las variables de entorno.")
            return

        genai.configure(api_key=API_KEY)
        os.environ["GOOGLE_API_KEY"] = API_KEY

        PDF_FOLDER_PATH = "Archivos PDF"
        PERSIST_DIRECTORY = "./chroma_db_diabetes"

        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)

        # Forzar limpieza si hay error previo
        if os.path.exists(PERSIST_DIRECTORY):
            print("Limpiando base de datos previa para evitar conflictos...")
            shutil.rmtree(PERSIST_DIRECTORY)

        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            print(f"Cargando PDFs desde {PDF_FOLDER_PATH}...")
            for filename in os.listdir(PDF_FOLDER_PATH):
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
        
        if not documents:
            print("ADVERTENCIA: No se encontraron PDFs. Usando modo básico.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=API_KEY)

        template = """Responde como experto en diabetes usando el contexto: {context}. Pregunta: {question}"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        print("SISTEMA: Asistente RAG listo.")

    except Exception as e:
        print(f"ERROR DURANTE INICIALIZACIÓN: {e}")
        traceback.print_exc()

# Intentar inicializar al cargar el módulo
inicializar_asistente()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "rag_active": qa_chain is not None})

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"response": "El asistente se está iniciando o tuvo un error. Revisa los logs."}), 503
    
    data = request.get_json()
    try:
        answer = qa_chain.run(data.get('question'))
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
