import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader # Asegúrate de que esta línea esté presente

# --- Configuración de la API de Google Gemini ---
# La clave API debe ser obtenida de las variables de entorno de Render por seguridad.
API_KEY = os.getenv("GOOGLE_API_KEY")

# Verificar si la clave API está disponible
if not API_KEY:
    print("ERROR: La variable de entorno GOOGLE_API_KEY no está configurada en asistente_rag.py.")
    # Si la clave no está, las siguientes inicializaciones fallarán.
    # En un entorno de producción, aquí se podría considerar lanzar una excepción o salir.

# Configurar genai para el uso directo y también para LangChain
genai.configure(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY # Asegura que LangChain también la use desde el entorno

# --- Inicialización de Modelos y Funciones ---
# Inicializar embeddings_model (necesita la API_KEY)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)

# Inicializar el modelo de lenguaje (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=API_KEY)

# Inicializar text_splitter una sola vez
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# --- Funciones para la Base de Datos Vectorial ---
def load_and_process_documents(pdf_folder_path, persist_directory):
    """
    Carga y procesa documentos PDF, y crea/carga una base de datos vectorial Chroma.
    """
    vector_db = None
    db_needs_recreation = False

    # Intentar cargar la base de datos existente
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        try:
            print(f"Intentando cargar la base de datos vectorial existente desde '{persist_directory}'...")
            vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
            if not vector_db.get()['ids']: # Si está vacía
                print(f"Base de datos cargada desde '{persist_directory}' pero parece vacía. Se marcará para recreación.")
                db_needs_recreation = True
                del vector_db
                vector_db = None
                #gc.collect() # No es necesario aquí, se maneja en backend_asistente si aplica
        except Exception as e:
            print(f"Error al cargar la base de datos persistente: {e}. Se marcará para recreación.")
            db_needs_recreation = True
            del vector_db
            vector_db = None
            #gc.collect()
    else:
        print(f"Directorio de persistencia '{persist_directory}' no encontrado. Se procederá a crear una nueva base de datos.")
        db_needs_recreation = True

    # Si la base de datos necesita ser recreada
    if db_needs_recreation:
        if os.path.exists(persist_directory):
            try:
                print(f"Intentando eliminar el directorio persistente '{persist_directory}' para recrearlo...")
                shutil.rmtree(persist_directory)
                print("Directorio eliminado exitosamente.")
            except Exception as e:
                print(f"ERROR: Falló la eliminación del directorio '{persist_directory}': {e}")
                print("Por favor, asegúrese de que no haya procesos utilizando la carpeta.")
                return None # Retorna None si no se puede eliminar/crear la DB

        documents = []
        if os.path.exists(pdf_folder_path):
            print(f"Cargando documentos desde: {pdf_folder_path}")
            for filename in os.listdir(pdf_folder_path):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(pdf_folder_path, filename)
                    try:
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                        print(f"  - Cargado: {filename}")
                    except Exception as e:
                        print(f"  - Error al cargar {filename}: {e}")
        else:
            print(f"ADVERTENCIA: La carpeta '{pdf_folder_path}' no existe. No se cargarán PDFs.")
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
                    persist_directory=persist_directory
                )
                print(f"Base de datos vectorial creada y persistida en '{persist_directory}'.")
            except Exception as e:
                print(f"ERROR: Falló la creación de la base de datos vectorial desde los documentos: {e}")
                vector_db = None
        else:
            print("ERROR: No se encontraron documentos válidos para procesar. El asistente no tendrá una base de conocimiento.")
            vector_db = None
    
    return vector_db

# --- Configuración del Prompt y Cadena RAG ---
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

def initialize_qa_chain(vector_db_instance):
    """
    Inicializa la cadena RAG con la base de datos vectorial y el LLM.
    """
    if vector_db_instance:
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=vector_db_instance.as_retriever(),
            chain_type_kwargs={"prompt": CUSTOM_PROMPT}
        )
        print("Cadena RAG inicializada correctamente.")
        return qa_chain
    else:
        print("ADVERTENCIA: La cadena QA no se pudo inicializar porque la base de datos vectorial no está disponible.")
        return None

