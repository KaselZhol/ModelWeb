from fastapi import APIRouter, HTTPException
from models.schemas import IntegralInput
from logica.Integracion import resolver_integracion_web

router = APIRouter()

@router.post("/calcular")
def calcular_integral(datos: IntegralInput):
    """
    Calcula la integral definida usando Trapecio o Simpson 1/3.
    """
    try:
        return resolver_integracion_web(
            datos.funcion, 
            datos.a, 
            datos.b, 
            datos.n, 
            datos.metodo
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))