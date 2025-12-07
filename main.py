from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <--- IMPORTANTE: Importar esto

# Importamos los routers (Nota: Asegúrate de que el archivo routers/EMR.py exista)
from routers import interpolacion, EMR, integracion, edo

# Configuración del Servidor
app = FastAPI(
    title="Motor Matemático Harvard API",
    description="Backend profesional modularizado para métodos numéricos.",
    version="3.0.0"
)

# --- CONFIGURACIÓN DE SEGURIDAD (CORS) ---
# ESTO ES LO QUE TE FALTABA.
# Sin esto, Next.js recibirá un error "Network Error" o "CORS Error".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" permite que CUALQUIER web se conecte (ideal para desarrollo)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (POST, GET, etc.)
    allow_headers=["*"],
)

# --- CONEXIÓN DE ROUTERS (Las Zonas del Restaurante) ---

# Zona 1: Interpolación (Lagrange / Newton)
app.include_router(interpolacion.router, prefix="/interpolacion", tags=["Semana 9-10: Interpolación"])

# Zona 2: Ajuste de Curvas (Mínimos Cuadrados)
# Nota: Veo que llamaste al archivo "EMR". Asegúrate de que en la carpeta routers se llame "EMR.py"
app.include_router(EMR.router, prefix="/ajuste", tags=["Semana 11: Ajuste de Curvas"])

# Zona 3: Integración Numérica
app.include_router(integracion.router, prefix="/integracion", tags=["Semana 12: Integración"])

# Zona 4: Ecuaciones Diferenciales
app.include_router(edo.router, prefix="/edo", tags=["Semana 13-15: Ecuaciones Diferenciales"])


# --- RUTA RAÍZ ---
@app.get("/")
def home():
    return {
        "status": "online",
        "mensaje": "Bienvenido a la API. Ve a /docs para ver la documentación interactiva."
    }