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

app = Flask(__name__)
CORS(app)

# --- 1. Configuración de la API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variables globales
qa_chain = None
is_initializing = False

def setup_rag():
    """Función para inicializar el motor RAG solo cuando sea necesario"""
    global qa_chain, is_initializing
    
    if qa_chain is not None:
        return True
    
    is_initializing = True
    try:
        if not API_KEY:
            print("CRÍTICO: No hay GOOGLE_API_KEY configurada.")
            return False

        genai.configure(api_key=API_KEY)
        os.environ["GOOGLE_API_KEY"] = API_KEY

        PDF_FOLDER_PATH = "Archivos PDF"
        PERSIST_DIRECTORY = "./chroma_db_diabetes"

        # Modelo de Embeddings
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa para asegurar que la nueva clave funcione
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)

        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            print(f"SISTEMA: Cargando documentos desde {PDF_FOLDER_PATH}...")
            for filename in os.listdir(PDF_FOLDER_PATH):
                if filename.lower().endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Error cargando {filename}: {e}")
        
        if not documents:
            print("SISTEMA: No se encontraron PDFs. Usando base de conocimiento vacía.")
            return False

        # Dividir textos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Crear base vectorial
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Configurar LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

        template = """Eres un experto en educación en diabetes de la UCV. 
        Usa el siguiente contexto para responder la pregunta de forma educativa.
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
        
        print("SISTEMA: RAG inicializado con éxito.")
        return True

    except Exception as e:
        print(f"ERROR EN RAG: {e}")
        traceback.print_exc()
        return False
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online", 
        "rag_ready": qa_chain is not None,
        "initializing": is_initializing
    })

@app.route('/ask', methods=['POST'])
def ask():
    global qa_chain
    
    # Si no está listo, intentar inicializar ahora
    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "El asistente se está despertando y procesando los documentos. Por favor, espera 30 segundos e intenta de nuevo."}), 503
        
        success = setup_rag()
        if not success:
            return jsonify({"response": "Hubo un problema configurando la base de datos. Verifica la clave de API y los PDFs en los logs."}), 500

    data = request.get_json()
    user_question = data.get('question')

    try:
        answer = qa_chain.run(user_question)
        return jsonify({"response": answer})
    except Exception as e:
        print(f"Error en consulta: {e}")
        return jsonify({"response": f"Lo siento, ocurrió un error: {str(e)}"}), 500

if __name__ == '__main__':
    # Importante: No llamamos a setup_rag aquí para que el boot sea instantáneo
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
