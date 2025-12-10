import google.generativeai as genai
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import json
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Usamos el modelo Flash que es rápido y gratuito
model = genai.GenerativeModel('gemini-flash-latest')

router = APIRouter()

@router.post("/scan-problem")
async def scan_math_problem(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # --- EL NUEVO PROMPT INTELIGENTE ---
        prompt = """
        Eres un profesor experto de ingeniería. Analiza la imagen y clasifica el problema matemático.
        
        1. CLASIFICACIÓN (determina la ruta):
           - Si ves una integral (∫): ruta = "/integracion"
           - Si ves una tabla de datos (x, y) y piden "ajuste", "regresión" o "mínimos cuadrados": ruta = "/ajuste"
           - Si ves una tabla de datos (x, y) y piden "interpolar", "lagrange" o "newton": ruta = "/interpolacion/lagrange"
           - Si ves una ecuación diferencial (y', dy/dt) y condiciones iniciales: ruta = "/edo/euler" (o /edo/rk4 si lo especifican)

        2. EXTRACCIÓN (saca los datos):
           - Ecuación: conviértela a sintaxis Python (ej: 'np.exp(x)' en vez de 'e^x').
           - Variables: busca a, b, t0, y0, h, n, etc.

        RESPONDE SOLO ESTE JSON:
        {
            "ruta_sugerida": "/ruta/detectada",
            "parametros": {
                "ecuacion": "string o null",
                "t0": numero (float/int) o null,  <-- CAMBIO AQUÍ
                "y0": numero (float/int) o null,
                "h": numero (float/int) o null,
                "a": numero (float/int) o null,
                "b": numero (float/int) o null,
                "n": numero (float/int) o null,
                "x_str": "1,2,3,4" (si es tabla, como string separado por comas),
                "y_str": "2,4,6,8" (si es tabla)
            }
        }
        """

        response = model.generate_content([prompt, image])
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        datos_json = json.loads(texto_limpio)

        return datos_json

    except Exception as e:
        # ... manejo de errores ...
        return {"error": str(e)}
    
@router.get("/test-models")
def list_models():
    """Ruta temporal para ver qué modelos funcionan"""
    try:
        modelos = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                modelos.append(m.name)
        return {"modelos_disponibles": modelos}
    except Exception as e:
        return {"error": str(e)}