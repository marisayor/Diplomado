import google.generativeai as genai
# Importaciones de LangChain actualizadas para evitar ModuleNotFoundError
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

# Inicialización de Flask
app = Flask(__name__)
CORS(app)

# --- Configuración de API ---
API_KEY = os.getenv("GOOGLE_API_KEY")

# Variables de estado global para monitoreo
qa_chain = None
is_initializing = False
init_error = None
thread_started = False

def background_setup():
    """
    Configura el motor RAG en un hilo secundario para evitar el 'Port scan timeout' en Render.
    """
    global qa_chain, is_initializing, init_error
    
    # Pausa de cortesía para asegurar que el proceso principal de Flask esté listo
    time.sleep(5)
    
    print("SISTEMA: Iniciando procesamiento de documentos en segundo plano...")
    is_initializing = True
    init_error = None
    
    try:
        if not API_KEY:
            init_error = "Error: GOOGLE_API_KEY no configurada en las variables de entorno."
            print(f"CRÍTICO: {init_error}")
            return

        # Configuración inicial de Google
        genai.configure(api_key=API_KEY)

        PDF_FOLDER_PATH = "Archivos PDF"
        PERSIST_DIRECTORY = "./chroma_db_diabetes"

        # FIX ERROR 400: Se debe usar solo el nombre del modelo. 
        # La librería añade 'models/' internamente.
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            google_api_key=API_KEY
        )

        # Limpiar base de datos previa para evitar errores de persistencia o esquema
        if os.path.exists(PERSIST_DIRECTORY):
            print("SISTEMA: Eliminando base de datos previa para regenerar...")
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                gc.collect()
                time.sleep(2)
            except Exception as e:
                print(f"Aviso: No se pudo borrar la DB: {e}")

        # Carga de documentos PDFs
        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith(".pdf")]
            print(f"SISTEMA: Procesando {len(pdf_files)} archivos PDF.")
            
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(PDF_FOLDER_PATH, filename))
                    documents.extend(loader.load())
                    print(f"  - Cargado: {filename}")
                except Exception as e:
                    print(f"  - Error en {filename}: {e}")
        
        if not documents:
            init_error = "No se encontraron documentos en la carpeta 'Archivos PDF'."
            print(f"SISTEMA: {init_error}")
            return

        # Dividir texto en fragmentos (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        print(f"SISTEMA: Indexando {len(chunks)} fragmentos en Chroma...")
        
        # Crear base vectorial
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Configurar el modelo de lenguaje (Gemini 1.5 Flash)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        # Prompt Académico UCV completo
        custom_prompt_template = """
        Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran **conocimiento** y **autoeficacia** en el manejo de su condición. 
        Para ello, integrarás y aplicarás los principios de la neurociencia del aprendizaje, la teoría de la carga cognitiva, la teoría de la autoeficacia de Bandura, la escucha activa y las herramientas de las precauciones universales de alfabetización en salud.

        Cuando un educador te pregunte cómo enseñar un aspecto específico de la diabetes debe:
        1. Sugerir métodos didácticos adecuados y concretos.
        2. Justificar tus sugerencias basándote en las teorías mencionadas.
        3. Ofrecer ejemplos prácticos y aplicables.
        4. Enfatizar la diferencia entre 'dar información' y 'educar'.
        
        Debes basar todas tus respuestas EXCLUSIVAMENTE en el contexto proporcionado. Si la información no está en el contexto, indícalo. No inventes.

        Contexto: {context}
        Pregunta: {question}

        Respuesta:
        """
        
        CUSTOM_PROMPT = PromptTemplate(
            template=custom_prompt_template, 
            input_variables=["context", "question"]
        )

        # Crear la cadena RAG final
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )
        
        print("SISTEMA: ¡Asistente RAG inicializado y listo para consultas!")

    except Exception as e:
        init_error = str(e)
        print(f"ERROR CRÍTICO: {e}")
        traceback.print_exc()
    finally:
        is_initializing = False

@app.route('/', methods=['GET'])
def home():
    """Endpoint para verificar el estado y despertar el proceso de carga"""
    global thread_started
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()
        
    return jsonify({
        "status": "online", 
        "rag_ready": qa_chain is not None,
        "is_initializing": is_initializing,
        "error": init_error,
        "message": "Bienvenido al Backend del Asistente UCV"
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Endpoint para recibir preguntas"""
    global qa_chain, thread_started
    
    # Por si acaso se llama directamente a /ask
    if not thread_started:
        thread_started = True
        threading.Thread(target=background_setup, daemon=True).start()

    if qa_chain is None:
        if is_initializing:
            return jsonify({"response": "Estoy analizando los manuales de diabetes de la UCV. Dame un minuto para terminar de cargar..."}), 503
        return jsonify({"response": f"El motor de IA no pudo iniciar: {init_error}"}), 500

    data = request.get_json()
    user_question = data.get('question')

    if not user_question:
        return jsonify({"response": "Por favor escribe una pregunta."}), 400

    try:
        # Usamos invoke() que es el método estándar actual de LangChain
        result = qa_chain.invoke({"query": user_question})
        return jsonify({"response": result["result"]})
    except Exception as e:
        print(f"Error en consulta: {e}")
        return jsonify({"response": f"Lo siento, ocurrió un error procesando tu pregunta: {str(e)}"}), 500

if __name__ == '__main__':
    # Render usa la variable de entorno PORT
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
