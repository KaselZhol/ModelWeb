from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# 1. INTERPOLACIÓN (Semana 9 y 10)
# Sirve para: Lagrange y Newton
# ==========================================
class PuntosInput(BaseModel):
    x_puntos: List[float] = Field(..., description="Lista de coordenadas X")
    y_puntos: List[float] = Field(..., description="Lista de coordenadas Y")
    x_eval: Optional[float] = Field(None, description="Punto opcional para evaluar el polinomio final")

    class Config:
        json_schema_extra = {
            "example": {
                "x_puntos": [0.0, 1.0, 3.0],
                "y_puntos": [1.0, 2.0, 0.0],
                "x_eval": 2.0
            }
        }

# ==========================================
# 2. AJUSTE DE CURVAS (Semana 11)
# Sirve para: Mínimos Cuadrados
# ==========================================
class AjusteInput(BaseModel):
    x_puntos: List[float]
    y_puntos: List[float]
    grado: int = Field(1, description="Grado del polinomio (1=Lineal, 2=Cuadrático...)")
    x_eval: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "x_puntos": [1, 2, 3, 4],
                "y_puntos": [6, 11, 18, 27],
                "grado": 2,
                "x_eval": 2.5
            }
        }

# ==========================================
# 3. INTEGRACIÓN NUMÉRICA (Semana 12)
# Sirve para: Trapecio y Simpson
# ==========================================
class IntegralInput(BaseModel):
    funcion: str = Field(..., description="La función matemática a integrar, ej: 'x**2'")
    a: float = Field(..., description="Límite inferior")
    b: float = Field(..., description="Límite superior")
    n: int = Field(..., description="Número de sub-intervalos (segmentos)")
    metodo: str = Field("trapecio", description="Método a usar: 'trapecio' o 'simpson'")

    class Config:
        json_schema_extra = {
            "example": {
                "funcion": "x**3 / (1 + sqrt(x))",
                "a": 2.0,
                "b": 5.0,
                "n": 6,
                "metodo": "simpson"
            }
        }

# ==========================================
# 4. ECUACIONES DIFERENCIALES (Semanas 13 y 15)
# Sirve para: Euler, Taylor, RK2, RK4
# ==========================================
class EDOInput(BaseModel):
    ecuacion: str = Field(..., description="Ecuación de y' (ej: '-2*y + x')")
    t0: float = Field(..., description="Tiempo inicial")
    y0: float = Field(..., description="Valor inicial de y")
    h: float = Field(..., description="Tamaño del paso")
    pasos: int = Field(..., description="Cantidad de iteraciones a realizar")
    
    # Campos Opcionales (Solo para Taylor o cálculo de error)
    ecuacion_segunda_derivada: Optional[str] = Field(None, description="Solo para Taylor: Ecuación de y'' (puedes usar 'yp')")
    solucion_exacta: Optional[str] = Field(None, description="Opcional: Para calcular el error real")

    class Config:
        json_schema_extra = {
            "example": {
                "ecuacion": "y - t**2 + 1",
                "t0": 0.0,
                "y0": 0.5,
                "h": 0.1,
                "pasos": 5,
                "ecuacion_segunda_derivada": None, 
                "solucion_exacta": None
            }
        }