import numpy as np
import math
import pandas as pd
from fastapi import FastAPI
# Importamos los routers que acabamos de crear
from routers import interpolacion, EMR, integracion, edo

# Configuración del Servidor
app = FastAPI(
    title="Motor Matemático Harvard API",
    description="Backend profesional modularizado para métodos numéricos.",
    version="3.0.0"
)

# --- CONEXIÓN DE ROUTERS (Las Zonas del Restaurante) ---

# Zona 1: Interpolación (Lagrange / Newton)
app.include_router(interpolacion.router, prefix="/interpolacion", tags=["Semana 9-10: Interpolación"])

# Zona 2: Ajuste de Curvas (Mínimos Cuadrados)
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