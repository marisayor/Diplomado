import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain_community.document_loaders import PyPDFLoader
from flask import Flask, request, jsonify # Importar Flask y sus utilidades
from flask_cors import CORS # Para manejar CORS, necesario para el frontend
import shutil # Importar shutil para eliminar directorios
import gc # Importar garbage collector para forzar liberación de memoria

# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app) # Habilitar CORS para permitir peticiones desde el frontend web

# --- 1. Configuración de la API de Google Gemini ---
# IMPORTANTE: ¡NO USES LA CLAVE API DIRECTAMENTE EN EL CÓDIGO!
# Se obtiene de las variables de entorno de Render.
API_KEY = os.getenv("GOOGLE_API_KEY")

# Verificar si la clave API está disponible
if not API_KEY:
    print("ERROR: La variable de entorno GOOGLE_API_KEY no está configurada. El asistente no funcionará.")
    # En un entorno de producción, podrías considerar salir o lanzar una excepción aquí.
    # Para este caso, permitiremos que el script continúe para que el error sea capturado
    # por la inicialización de genai o langchain.

genai.configure(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY # Asegura que Langchain también la use

# --- 2. Carga y Procesamiento de Documentos (Se ejecuta una vez al iniciar el servidor) ---
PDF_FOLDER_PATH = "Archivos PDF" # <--- ¡VERIFICA Y AJUSTA TU RUTA!
PERSIST_DIRECTORY = "./chroma_db_diabetes"

# Inicializar text_splitter una sola vez
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Inicializar embeddings_model una sola vez (necesita la API_KEY)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)

vector_db = None # Inicializar vector_db a None
db_needs_recreation = False # Bandera para indicar si la DB necesita ser recreada

# --- Lógica para cargar o crear la base de datos vectorial ---
# Primero, intentar cargar la base de datos existente
if os.path.exists(PERSIST_DIRECTORY) and os.path.isdir(PERSIST_DIRECTORY):
    try:
        print(f"Intentando cargar la base de datos vectorial existente desde '{PERSIST_DIRECTORY}'...")
        vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)
        # Una pequeña verificación para asegurar que la colección no esté vacía si se cargó
        if not vector_db.get()['ids']: # Si está vacía
            print(f"Base de datos cargada desde '{PERSIST_DIRECTORY}' pero parece vacía. Se marcará para recreación.")
            db_needs_recreation = True
            # Intentar liberar el objeto y forzar la recolección de basura para liberar el archivo
            del vector_db
            vector_db = None
            gc.collect()
    except Exception as e:
        print(f"Error al cargar la base de datos persistente: {e}. Se marcará para recreación.")
        db_needs_recreation = True
        # Intentar liberar el objeto y forzar la recolección de basura
        del vector_db
        vector_db = None
        gc.collect()
else:
    print(f"Directorio de persistencia '{PERSIST_DIRECTORY}' no encontrado. Se procederá a crear una nueva base de datos.")
    db_needs_recreation = True

# Si la base de datos necesita ser recreada, intentar eliminar el directorio antiguo y crear uno nuevo
if db_needs_recreation:
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            print(f"Intentando eliminar el directorio persistente '{PERSIST_DIRECTORY}' para recrearlo...")
            shutil.rmtree(PERSIST_DIRECTORY)
            print("Directorio eliminado exitosamente.")
        except PermissionError as e:
            print(f"ERROR CRÍTICO: Permiso denegado al intentar eliminar '{PERSIST_DIRECTORY}': {e}")
            print("Esto ocurre si el archivo está siendo utilizado por otro proceso (ej. una terminal anterior, explorador de archivos, antivirus).")
            print("Por favor, CIERRE TODAS LAS VENTANAS DE EXPLORADOR DE ARCHIVOS, TERMINALES O PROGRAMAS que puedan estar accediendo a la carpeta './chroma_db_diabetes'.")
            print("Luego, ELIMINE MANUALMENTE la carpeta './chroma_db_diabetes' y REINICIE el script.")
            vector_db = None # Asegurar que vector_db sea None si la eliminación falla
            # En un entorno de producción, aquí se podría considerar salir del programa o lanzar una excepción.
        except Exception as e:
            print(f"ERROR al intentar eliminar el directorio persistente '{PERSIST_DIRECTORY}': {e}")
            print("Se recomienda eliminar la carpeta manualmente y reiniciar el script.")
            vector_db = None

    # Solo proceder a crear si la eliminación fue exitosa o el directorio no existía
    if vector_db is None: # Si loading failed, or was empty, or deletion failed
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
            print("Usando texto simulado como fallback para demostración.")
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
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=API_KEY)

# --- 4. Creación de la Cadena RAG ---
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

# --- 5. Endpoint de la API Flask ---
# Ruta raíz para manejar peticiones OPTIONS y GET básicas
@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({"message": "Asistente de Educación en Diabetes en línea. ¡Envía tus preguntas a /ask!"}), 200

@app.route('/ask', methods=['POST'])
def ask_assistant_api():
    if not qa_chain:
        print("DEBUG: qa_chain no está disponible al recibir la pregunta.") # DEBUG
        return jsonify({"response": "Lo siento, el asistente no está completamente configurado. No se pudo cargar la base de datos de conocimiento."}), 500

    data = request.get_json()
    user_question = data.get('question')

    if not user_question:
        print("DEBUG: Pregunta de usuario vacía.") # DEBUG
        return jsonify({"response": "Por favor, proporcione una pregunta."}), 400

    try:
        print(f"DEBUG: Recibida pregunta: {user_question}") # DEBUG
        print("DEBUG: Intentando ejecutar qa_chain.run()...") # DEBUG
        answer = qa_chain.run(user_question)
        print(f"DEBUG: Respuesta generada: {answer}") # DEBUG
        return jsonify({"response": answer})
    except Exception as e:
        print(f"ERROR: Excepción al procesar la pregunta: {e}") # DEBUG
        # Aquí puedes añadir más detalles del error si 'e' es un objeto complejo
        import traceback
        print(f"ERROR: Traceback completo: {traceback.format_exc()}") # DEBUG: Imprime el stack trace completo
        return jsonify({"response": f"Lo siento, ocurrió un error al procesar su pregunta. Detalle técnico: {e}"}), 500

# --- Ejecutar el servidor Flask ---
if __name__ == '__main__':
    print("Iniciando servidor Flask para el asistente de Educación en Diabetes...")
    print("El servidor estará escuchando en http://127.0.0.1:5000/")
    print("Asegúrese de que su clave API y la ruta de los PDFs sean correctas.")
    # CAMBIO CLAVE: Desactivar el modo debug para evitar problemas de bloqueo de archivos
    # Render usa Gunicorn, por lo que este bloque no se ejecuta en producción.
    # El puerto 10000 es el que usa Render, no 5000.
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))
