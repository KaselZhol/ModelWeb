from fastapi import APIRouter, HTTPException
from models.schemas import PuntosInput
# Asegúrate de que tus funciones estén en /logica/interpolacion.py
from logica.interpolacion import resolver_lagrange_con_pasos_reales, resolver_newton_web

router = APIRouter()

@router.post("/lagrange")
def calcular_lagrange(datos: PuntosInput):
    """
    Calcula el Polinomio de Lagrange paso a paso.
    """
    try:
        return resolver_lagrange_con_pasos_reales(datos.x_puntos, datos.y_puntos, datos.x_eval)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/newton")
def calcular_newton(datos: PuntosInput):
    """
    Calcula Diferencias Divididas de Newton y genera la tabla.
    """
    try:
        return resolver_newton_web(datos.x_puntos, datos.y_puntos, datos.x_eval)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))