from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# --- Configuración de Flask ---
app = Flask(__name__)
CORS(app) # Habilitar CORS para permitir peticiones desde el frontend web

# --- Endpoint de prueba simple ---
@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({"message": "Asistente de Educación en Diabetes en línea. ¡Envía tus preguntas a /ask!"}), 200

@app.route('/test', methods=['GET'])
def test_endpoint():
    print("DEBUG: Recibida petición en /test") # DEBUG
    return jsonify({"message": "¡Hola desde el backend! El servidor Flask está funcionando."}), 200

@app.route('/ask', methods=['POST'])
def ask_assistant_api():
    # Esta es una respuesta de prueba, la lógica RAG está deshabilitada temporalmente
    print("DEBUG: Recibida petición en /ask - (RAG deshabilitado para prueba)") # DEBUG
    data = request.get_json()
    user_question = data.get('question', 'No question provided') # Obtener la pregunta del usuario

    # Simular una respuesta sin usar la cadena RAG
    response_text = f"Has preguntado: '{user_question}'. (El asistente RAG está deshabilitado temporalmente para diagnóstico. Si esto funciona, el problema está en la inicialización de RAG/API de Google)."
    
    print(f"DEBUG: Respuesta simulada: {response_text}") # DEBUG
    return jsonify({"response": response_text})

# --- Ejecutar el servidor Flask ---
if __name__ == '__main__':
    print("Iniciando servidor Flask para el asistente de Educación en Diabetes...")
    print("El servidor estará escuchando en http://127.0.0.1:5000/")
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))
