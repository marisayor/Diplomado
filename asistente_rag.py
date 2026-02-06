import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
from langchain_community.document_loaders import PyPDFLoader
import shutil
import traceback

def crear_asistente_rag(api_key, pdf_folder_path="Archivos PDF", persist_directory="./chroma_db_diabetes"):
    """
    Crea y devuelve una cadena RAG lista para responder preguntas.
    """
    try:
        if not api_key:
            return None, "No hay GOOGLE_API_KEY configurada."

        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)

        # Limpiar base de datos previa
        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
            except:
                pass

        # Cargar documentos PDF
        documents = []
        if os.path.exists(pdf_folder_path):
            pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith(".pdf")]
            print(f"SISTEMA: Encontrados {len(pdf_files)} archivos PDF.")
            
            for filename in pdf_files:
                try:
                    loader = PyPDFLoader(os.path.join(pdf_folder_path, filename))
                    documents.extend(loader.load())
                    print(f"SISTEMA: Cargado {filename}")
                except Exception as e:
                    print(f"Error cargando {filename}: {e}")
        
        if not documents:
            return None, "No se encontraron PDFs en la carpeta 'Archivos PDF'."

        # Dividir documentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Crear embeddings y base vectorial
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        print(f"SISTEMA: Creando base vectorial con {len(chunks)} fragmentos...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_directory
        )

        # Configurar LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        # Prompt personalizado (UCV)
        template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran conocimiento y autoeficacia en el manejo de su condición. 

        Utiliza el siguiente contexto para responder de forma pedagógica. Si la respuesta no está en el contexto, indícalo claramente.
        
        Contexto: {context}
        Pregunta: {question}
        
        Respuesta:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # ✅ NUEVO: Crear cadena con arquitectura moderna
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(
            retriever=vector_db.as_retriever(),
            combine_docs_chain=document_chain
        )
        
        print("SISTEMA: ¡Asistente RAG listo!")
        return retrieval_chain, None

    except Exception as e:
        error_msg = str(e)
        print(f"ERROR EN RAG: {e}")
        traceback.print_exc()
        return None, error_msg
