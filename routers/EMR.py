from fastapi import APIRouter, HTTPException
from models.schemas import AjusteInput
from logica.EMR import resolver_minimos_cuadrados_web

router = APIRouter()

@router.post("/minimos-cuadrados")
def calcular_minimos_cuadrados(datos: AjusteInput):
    """
    Realiza ajuste polinomial (Regresión) de cualquier grado.
    """
    try:
        return resolver_minimos_cuadrados_web(
            datos.x_puntos, 
            datos.y_puntos, 
            datos.grado, 
            datos.x_eval
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))