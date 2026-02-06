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

# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app)

# --- 1. Configuración de la API de Google Gemini ---
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERROR: La variable de entorno GOOGLE_API_KEY no está configurada. El asistente no funcionará.")

genai.configure(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 2. Carga y Procesamiento de Documentos ---
PDF_FOLDER_PATH = "Archivos PDF"
PERSIST_DIRECTORY = "./chroma_db_diabetes"

# Inicializar text_splitter una sola vez
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# CAMBIO 1: Actualizar el modelo de embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)

vector_db = None
db_needs_recreation = False

# --- Lógica para cargar o crear la base de datos vectorial (sin cambios) ---
if os.path.exists(PERSIST_DIRECTORY) and os.path.isdir(PERSIST_DIRECTORY):
    try:
        print(f"Intentando cargar la base de datos vectorial existente desde '{PERSIST_DIRECTORY}'...")
        vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)
        if not vector_db.get()['ids']:
            print(f"Base de datos cargada desde '{PERSIST_DIRECTORY}' pero parece vacía. Se marcará para recreación.")
            db_needs_recreation = True
            del vector_db
            vector_db = None
            gc.collect()
    except Exception as e:
        print(f"Error al cargar la base de datos persistente: {e}. Se marcará para recreación.")
        db_needs_recreation = True
        del vector_db
        vector_db = None
        gc.collect()
else:
    print(f"Directorio de persistencia '{PERSIST_DIRECTORY}' no encontrado. Se procederá a crear una nueva base de datos.")
    db_needs_recreation = True

if db_needs_recreation:
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            print(f"Intentando eliminar el directorio persistente '{PERSIST_DIRECTORY}' para recrearlo...")
            shutil.rmtree(PERSIST_DIRECTORY)
            print("Directorio eliminado exitosamente.")
        except Exception as e:
            print(f"ERROR al intentar eliminar el directorio persistente '{PERSIST_DIRECTORY}': {e}")
            vector_db = None

    if vector_db is None:
        documents = []
        if os.path.exists(PDF_FOLDER_PATH):
            print(f"Cargando documentos desde: {PDF_FOLDER_PATH}")
            for filename in os.listdir(PDF_FOLDER_PATH):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(PDF_FOLDER_PATH, filename)
                    try:
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                        print(f"  - Cargado: {filename}")
                    except Exception as e:
                        print(f"  - Error al cargar {filename}: {e}")
        else:
            print(f"ADVERTENCIA: La carpeta '{PDF_FOLDER_PATH}' no existe. No se cargarán PDFs.")
            mock_document_text = """
            ### Módulo 1: Fundamentos de la Diabetes Mellitus
            **Definición de Diabetes:** La diabetes mellitus es un grupo de enfermedades metabólicas caracterizadas por hiperglucemia (niveles elevados de glucosa en sangre) resultante de defectos en la secreción de insulina, en la acción de la insulina, o en ambas.
            """
            documents = text_splitter.create_documents([mock_document_text])

        if documents:
            chunks = text_splitter.split_documents(documents)
            print(f"Documentos divididos en {len(chunks)} fragmentos para la creación de la base de datos.")
            try:
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings_model,
                    collection_name="diabetes_diploma_docs",
                    persist_directory=PERSIST_DIRECTORY
                )
                print(f"Base de datos vectorial creada y persistida en '{PERSIST_DIRECTORY}'.")
            except Exception as e:
                print(f"ERROR: Falló la creación de la base de datos vectorial desde los documentos: {e}")
                vector_db = None
        else:
            print("ERROR: No se encontraron documentos válidos para procesar. El asistente no tendrá una base de conocimiento.")
            vector_db = None

# --- 3. Configuración del Modelo de Lenguaje (Gemini) ---
# CAMBIO 2: Actualizar el modelo principal a gemini-2.5-flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=API_KEY)

# --- 4. Creación de la Cadena RAG (sin cambios en la lógica) ---
qa_chain = None
if vector_db:
    custom_prompt_template = """
    Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran **conocimiento** y **autoeficacia** en el manejo de su condición. Para ello, integrarás y aplicarás los principios de la neurociencia del aprendizaje, la teoría de la carga cognitiva, la teoría de la autoeficacia de Bandura, la escucha activa y las herramientas de las precauciones universales de alfabetización en salud, tal como se definen en tus documentos de referencia. Cuando un educador te pregunte cómo enseñar un aspecto específico de la diabetes (ya sea cognitivo, afectivo o psicomotor) o cómo planificar una actividad instruccional, debes:
1. **Sugerir métodos didácticos** adecuados y concretos.
2. **Justificar tus sugerencias** explicando cómo estos métodos se alinean con las bases teóricas mencionadas (ej., cómo reducen la carga cognitiva, cómo fomentan la autoeficacia, cómo se adaptan a la alfabetización en salud, o cómo aplican principios de neurociencia).
3. **Ofrecer ejemplos prácticos y aplicables** en el contexto de la educación en diabetes.
4. **Enfatizar la diferencia entre 'dar información' y 'educar' terapéuticamente**, promoviendo un enfoque centrado en la capacitación y el empoderamiento del paciente. 5. Puedes utilizar el Ejemplo de actividad para guiarte. Debes basar todas tus respuestas EXCLUSIVAMENTE en el contexto proporcionado por los documentos. Si la información necesaria para responder no se encuentra en el contexto, indica claramente que no puedes responder a esa pregunta. No inventes.

    Contexto: {context}

    Pregunta: {question}

    Respuesta:
    """
    CUSTOM_PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    print("Cadena RAG inicializada correctamente.")
else:
    print("ADVERTENCIA: La cadena QA no se pudo inicializar porque la base de datos vectorial no está disponible.")

# --- 5. Endpoint de la API Flask (sin cambios) ---
@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({"message": "Asistente de Educación en Diabetes en línea. ¡Envía tus preguntas a /ask!"}), 200

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "¡Hola desde el backend! El servidor Flask está funcionando."}), 200


@app.route('/ask', methods=['POST'])
def ask_assistant_api():
    if not qa_chain:
        return jsonify({"response": "Lo siento, el asistente no está completamente configurado. No se pudo cargar la base de datos de conocimiento."}), 500

    data = request.get_json()
    user_question = data.get('question')

    if not user_question:
        return jsonify({"response": "Por favor, proporcione una pregunta."}), 400

    try:
        answer = qa_chain.run(user_question)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"response": f"Lo siento, ocurrió un error al procesar su pregunta. Detalle técnico: {e}"}), 500

# --- Ejecutar el servidor Flask ---
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))
