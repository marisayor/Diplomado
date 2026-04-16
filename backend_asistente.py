import os
import gc
import shutil
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app)

# --- 1. Configuración de la API de Google Gemini ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: La variable de entorno GOOGLE_API_KEY no está configurada.")

genai.configure(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 2. Rutas robustas (funcionan en Render y Netlify) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER_PATH = os.path.join(BASE_DIR, "Archivos PDF")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db_diabetes")

# --- 3. Inicialización de componentes ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # ✅ Como en 2025
    google_api_key=API_KEY
)
vector_db = None
qa_chain = None

# --- 4. Carga o creación de la base vectorial ---
def initialize_vector_db():
    global vector_db, qa_chain
    db_needs_recreation = False

    # Intentar cargar base existente
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            print(f"Intentando cargar base vectorial desde '{PERSIST_DIRECTORY}'...")
            vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)
            if not vector_db.get()['ids']:
                print("Base cargada pero vacía. Se recreará.")
                db_needs_recreation = True
                del vector_db
                vector_db = None
                gc.collect()
        except Exception as e:
            print(f"Error al cargar base persistente: {e}. Se recreará.")
            db_needs_recreation = True
            if 'vector_db' in locals():
                del vector_db
            vector_db = None
            gc.collect()
    else:
        print(f"Directorio '{PERSIST_DIRECTORY}' no existe. Se creará nueva base.")
        db_needs_recreation = True

    # Recrear si es necesario
    if db_needs_recreation:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                print("Base antigua eliminada.")
            except Exception as e:
                print(f"Error al eliminar base antigua: {e}")
                return

        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            print(f"Cargando documentos desde: {PDF_FOLDER_PATH}")
            for filename in os.listdir(PDF_FOLDER_PATH):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(PDF_FOLDER_PATH, filename)
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        if docs:
                            documents.extend(docs)
                            print(f"  - Cargado: {filename} ({len(docs)} páginas)")
                        else:
                            print(f"  - Advertencia: {filename} no tiene texto extraíble")
                    except Exception as e:
                        print(f"  - Error al cargar {filename}: {e}")
        else:
            print(f"ADVERTENCIA: Carpeta '{PDF_FOLDER_PATH}' no encontrada.")
            return

        if not documents:
            print("ERROR: No se encontraron documentos válidos.")
            return

        chunks = text_splitter.split_documents(documents)
        print(f"Divididos en {len(chunks)} fragmentos. Creando base vectorial...")
        try:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
                collection_name="diabetes_diploma_docs",
                persist_directory=PERSIST_DIRECTORY
            )
            print("Base vectorial creada y persistida.")
        except Exception as e:
            print(f"ERROR al crear base vectorial: {e}")
            return

    # --- 5. Configuración del modelo LLM y cadena RAG ---
    if vector_db:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=API_KEY)
        custom_prompt_template = """
Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran conocimiento y autoeficacia en el manejo de su condición.

Utiliza el siguiente contexto para responder de forma pedagógica. Si la respuesta no está en el contexto, indícalo claramente.

Contexto: {context}
Pregunta: {question}

Respuesta:"""
        CUSTOM_PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )
        print("Cadena RAG inicializada correctamente.")
    else:
        print("ADVERTENCIA: No se pudo inicializar la cadena RAG.")

# --- 6. Endpoints de la API ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "rag_ready": qa_chain is not None
    })

@app.route('/ask', methods=['POST'])
def ask_assistant_api():
    if not qa_chain:
        return jsonify({"response": "El asistente aún no está listo. Por favor, espera unos segundos."}), 503

    data = request.get_json()
    user_question = data.get('question')
    if not user_question:
        return jsonify({"response": "Por favor, proporcione una pregunta."}), 400

    try:
        answer = qa_chain.run(user_question)
        return jsonify({"response": answer})
    except Exception as e:
        print(f"ERROR al procesar pregunta: {e}")
        traceback.print_exc()
        return jsonify({"response": f"Error interno: {str(e)}"}), 500

# --- 7. Inicialización y ejecución ---
if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    initialize_vector_db()  # Inicializa la base al arrancar
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
