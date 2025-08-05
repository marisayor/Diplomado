import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Importaciones actualizadas para evitar las advertencias de deprecación
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma # Importación actualizada
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain_community.document_loaders import PyPDFLoader # Importación actualizada

# --- 1. Configuración de la API de Google Gemini ---
# IMPORTANTE: Reemplace "YOUR_API_KEY" con su clave API de Google Gemini.
# Puede obtener una en Google AI Studio: https://aistudio.google.com/app/apikey
# Para un entorno de producción, es mejor usar variables de entorno.
API_KEY = "AIzaSyCA5-ZpMMKV4NeHDGgiHMpp5GgHUXUQ7Vo" # <--- ¡REEMPLAZAR CON SU CLAVE API REAL!

# --- VERIFICACIÓN CRÍTICA DE LA CLAVE API ---


# --- DEBUGGING: Imprimir la clave API (solo para verificación, luego eliminar) ---
print(f"DEBUG: La clave API cargada (primeros 5 caracteres): {API_KEY[:5]}*****")

# Configurar genai (esto es para el cliente genai directo, no estrictamente necesario si se pasa a LangChain)
genai.configure(api_key=API_KEY)
# Establecer la clave API como variable de entorno (buena práctica, aunque la pasaremos directamente)
os.environ["GOOGLE_API_KEY"] = API_KEY


# --- 2. Carga y Procesamiento de Documentos ---
import os # Asegúrate de que esta línea no esté comentada
from langchain_community.document_loaders import DirectoryLoader # Asegúrate de que esta línea no esté comentada
from langchain.text_splitter import RecursiveCharacterTextSplitter # Asegúrate de que esta línea no esté comentada

# Obtiene la ruta del directorio del script actual de forma segura
directorio_base = os.path.dirname(os.path.abspath(__file__))

# Combina la ruta base con el nombre de la carpeta de los PDFs
ruta_a_los_pdfs = os.path.join(directorio_base, "Archivos PDF")

documents = [] # Inicializa la lista de documentos

# Carga los documentos PDF de la carpeta especificada
# Asegúrate de que estas líneas NO estén comentadas
loader = DirectoryLoader(ruta_a_los_pdfs, glob="**/*.pdf")
documents.extend(loader.load())

# Inicializar text_splitter fuera del bloque if/else para que siempre esté disponible
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Tamaño de cada fragmento
    chunk_overlap=200 # Superposición entre fragmentos para mantener el contexto
)
if os.path.exists(PDF_FOLDER_PATH):
    print(f"Cargando documentos desde: {PDF_FOLDER_PATH}")
    for filename in os.listdir(PDF_FOLDER_PATH):
        if filename.lower().endswith(".pdf"): # Asegurarse de que sea un PDF
            file_path = os.path.join(PDF_FOLDER_PATH, filename)
            try:
                loader = PyPDFLoader(file_path)
                # loader.load() devuelve una lista de objetos Document,
                # extendemos nuestra lista principal con ellos.
                documents.extend(loader.load())
                print(f"  - Cargado: {filename}")
            except Exception as e:
                print(f"  - Error al cargar {filename}: {e}")
else:
    print(f"ADVERTENCIA: La carpeta '{PDF_FOLDER_PATH}' no existe. No se cargarán PDFs.")
    print("Por favor, actualice 'PDF_FOLDER_PATH' en el código con la ruta correcta a sus documentos.")
    # Si la carpeta no existe, para que el script no falle, usamos un mock_document_text
    print("\nUsando texto simulado como fallback para demostración.")
    mock_document_text = """
    ### Módulo 1: Fundamentos de la Diabetes Mellitus

    **Definición de Diabetes:** La diabetes mellitus es un grupo de enfermedades metabólicas caracterizadas por hiperglucemia (niveles elevados de glucosa en sangre) resultante de defectos en la secreción de insulina, en la acción de la insulina, o en ambas. La hiperglucemia crónica de la diabetes se asocia con daño a largo plazo, disfunción y falla de varios órganos, especialmente los ojos, riñones, nervios, corazón y vasos sanguíneos.

    ### Módulo 5: Educación del Paciente y Bases Pedagógicas

    **Principios de Educación Terapéutica:** La educación terapéutica no es solo dar información, sino capacitar al paciente para gestionar su enfermedad y mantener o mejorar su calidad de vida. Implica un proceso continuo de aprendizaje, apoyo y adaptación.

    **Teoría de la Autoeficacia (Albert Bandura):** La creencia de un individuo en su capacidad para ejecutar con éxito un determinado comportamiento. En educación en diabetes, fomentar la autoeficacia es clave para que los pacientes adopten y mantengan comportamientos saludables.
    * **Fuentes de Autoeficacia:**
        * **Experiencias de dominio:** Lograr éxitos personales en la tarea (ej., inyectarse insulina correctamente).
        * **Modelado vicario:** Observar a otros (similares) realizar la tarea con éxito.
        * **Persuasión verbal:** Recibir aliento y feedback positivo.

    **Métodos Didácticos en Educación en Diabetes:**
    * **Exposiciones:** Útiles para transmitir información fundamental, pero deben ser concisas, visuales y seguidas de interacción.
    * **Demostraciones:** Esenciales para habilidades prácticas (ej., inyección de insulina, uso de glucómetro). Deben ir seguidas de práctica supervisada.
    * **Juego de Roles (Role-playing):** Permite practicar habilidades de comunicación y resolución de problemas en un entorno seguro.
    """
    documents = text_splitter.create_documents([mock_document_text]) # Usar el mock text si no hay PDFs


# Asegurarse de que 'documents' no esté vacío antes de crear chunks
if documents:
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(chunks)} fragmentos.")
else:
    print("No se encontraron documentos válidos para procesar después de la carga. Asegúrese de que la ruta sea correcta y contenga PDFs accesibles.")
    chunks = [] # Asegurar que chunks sea una lista vacía si no hay documentos

# --- 3. Creación de Embeddings y Almacenamiento en Base de Datos Vectorial ---
# Usar el modelo de embeddings de Google (text-embedding-004 o similar)
# ¡IMPORTANTE CAMBIO AQUÍ! Pasar la API_KEY directamente.
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)

# Crear una base de datos vectorial en memoria con ChromaDB
# Para un uso persistente, es buena práctica especificar un directorio.
PERSIST_DIRECTORY = "./chroma_db_diabetes"

# Si los chunks están vacíos, no podemos crear la base de datos,
# pero podemos intentar cargar una existente si el directorio persiste.
if chunks:
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        collection_name="diabetes_diploma_docs",
        persist_directory=PERSIST_DIRECTORY
    )
    # Guardar la base de datos en disco
    vector_db.persist()
    print(f"Base de datos vectorial creada y persistida en '{PERSIST_DIRECTORY}'")
else:
    print(f"Intentando cargar la base de datos vectorial existente desde '{PERSIST_DIRECTORY}'...")
    try:
        # Intentar cargar la base de datos si ya existe y no hay nuevos chunks
        vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)
        print(f"Base de datos vectorial cargada desde '{PERSIST_DIRECTORY}'")
    except Exception as e:
        print(f"Error al cargar la base de datos persistente: {e}. No se pudo inicializar la base de datos.")
        vector_db = None # Si no se puede cargar ni crear, vector_db será None


# --- 4. Configuración del Modelo de Lenguaje (Gemini) ---
# Usar el modelo de chat de Gemini (gemini-1.5-flash o gemini-1.0-pro)
# ¡IMPORTANTE CAMBIO AQUÍ! Pasar la API_KEY directamente.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=API_KEY) # temperature baja para respuestas más directas

# --- 5. Creación de la Cadena RAG (Retrieval-Augmented Generation) ---
# Definir el prompt para guiar a Gemini a usar solo el contexto proporcionado.
# Es crucial instruir al modelo a NO inventar información y a adoptar el rol deseado.
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

# Configurar la cadena de RetrievalQA con el retriever de ChromaDB y el LLM
# Solo si vector_db se inicializó correctamente
qa_chain = None
if vector_db:
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
else:
    print("ADVERTENCIA: La cadena QA no se pudo inicializar porque la base de datos vectorial no está disponible.")


# --- 6. Función para Interactuar con el Asistente ---
def ask_assistant(question):
    """
    Función para enviar una pregunta al asistente y obtener una respuesta basada en los documentos.
    """
    if not qa_chain: # Si la cadena QA no se inicializó, no se puede responder
        return "Lo siento, el asistente no está completamente configurado. No se pudo cargar la base de datos de conocimiento."
    try:
        response = qa_chain.run(question)
        return response
    except Exception as e:
        return f"Lo siento, ocurrió un error al procesar su pregunta: {e}"

# --- Ejemplos de Uso ---
if __name__ == "__main__":
    print("Asistente de Educación en Diabetes (Basado en sus documentos) listo.")
    print("¡Haga sus preguntas o escriba 'salir' para terminar!")
    print("Ejemplos: '¿Cómo puedo enseñar sobre la inyección de insulina para fomentar la autoeficacia?', '¿Qué métodos didácticos reducen la carga cognitiva al explicar el conteo de carbohidratos?', '¿Cómo usar el juego de roles en educación en diabetes?'")

    # Si la base de datos no se pudo cargar/crear, el asistente no funcionará.
    if not vector_db:
        print("\n¡ATENCIÓN! El asistente no puede responder preguntas porque la base de datos de conocimiento no se cargó correctamente.")
        print("Por favor, revise los mensajes anteriores para solucionar el problema de carga de documentos o de la base de datos persistente.")
    else:
        while True:
            user_question = input("\nEducador: ")
            if user_question.lower() == 'salir':
                print("Asistente: ¡Hasta luego! Espero haber sido de ayuda en tu planificación educativa.")
                break

            print("Asistente: Analizando tu solicitud...")
            answer = ask_assistant(user_question)
            print(f"Asistente: {answer}")


