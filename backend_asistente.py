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
