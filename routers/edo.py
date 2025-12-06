from fastapi import APIRouter, HTTPException
from models.schemas import EDOInput
# Importamos las 3 funciones que definimos
from logica.edo import resolver_edo_universal_web, resolver_solo_rk2, resolver_solo_rk4

router = APIRouter()

@router.post("/universal")
def calcular_euler_taylor(datos: EDOInput):
    """
    Resuelve EDOs básicas.
    Si envías 'ecuacion_segunda_derivada', usa Taylor Orden 2.
    Si no, usa Euler.
    """
    try:
        # Lógica de decisión simple
        metodo_a_usar = "euler"
        if datos.ecuacion_segunda_derivada:
            metodo_a_usar = "taylor2"

        return resolver_edo_universal_web(
            f_prime_str=datos.ecuacion,
            x0=datos.t0,
            y0=datos.y0,
            h=datos.h,
            pasos_num=datos.pasos,
            metodo=metodo_a_usar,
            f_double_prime_str=datos.ecuacion_segunda_derivada,
            sol_exacta_str=datos.solucion_exacta
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/rk2")
def calcular_rk2_heun(datos: EDOInput):
    """
    Resuelve usando Runge-Kutta Orden 2 (Método de Heun).
    """
    try:
        return resolver_solo_rk2(
            datos.ecuacion, datos.t0, datos.y0, datos.h, datos.pasos
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/rk4")
def calcular_rk4_clasico(datos: EDOInput):
    """
    Resuelve usando Runge-Kutta Orden 4 (El clásico).
    """
    try:
        return resolver_solo_rk4(
            datos.ecuacion, datos.t0, datos.y0, datos.h, datos.pasos
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))