import numpy as np
import math
import pandas as pd
# SEMANA 11
def resolver_minimos_cuadrados_web(x_puntos, y_puntos, grado=1, x_eval=None):
    """
    Realiza un ajuste polinomial de cualquier grado (1=Lineal, 2=Cuadrático, etc.)
    y devuelve los datos listos para la web.
    """
    # 1. Preparación de datos
    xs = np.array(x_puntos, dtype=float)
    ys = np.array(y_puntos, dtype=float)
    n = len(xs)
    
    pasos = []
    pasos.append(f"--- INICIO: Ajuste por Mínimos Cuadrados (Grado {grado}) ---")
    pasos.append(f"Datos recibidos: {n} puntos.")
    
    # 2. Cálculo de Coeficientes (La "Caja Negra" de Numpy)
    # np.polyfit encuentra los coeficientes que minimizan el error cuadrático
    coeficientes = np.polyfit(xs, ys, grado)
    
    # Creamos el objeto polinomio para facilitar cálculos
    p = np.poly1d(coeficientes)
    
    # Formateamos la ecuación bonita para mostrarla
    # poly1d a veces devuelve cadenas con muchas lineas, las limpiamos
    ecuacion_str = str(p).strip().replace("\n", " ")
    
    pasos.append(f"Modelo matemático generado (Polinomio):")
    pasos.append(f"   y = {ecuacion_str}")
    
    # 3. Cálculo de Errores (Métrica Educativa)
    y_pred = p(xs) # Predicciones sobre los puntos originales
    
    # ECM (Error Cuadrático Medio)
    residuo_cuadrado = (ys - y_pred)**2
    ecm = np.mean(residuo_cuadrado)
    
    # R^2 (Coeficiente de determinación) - ¡Muy importante en ciencia!
    # R2 = 1 - (SumaResiduos / SumaTotal)
    y_promedio = np.mean(ys)
    ss_tot = np.sum((ys - y_promedio)**2)
    ss_res = np.sum(residuo_cuadrado)
    r2 = 1 - (ss_res / ss_tot)
    
    pasos.append("\n--- Evaluación del Modelo ---")
    pasos.append(f"1. Error Cuadrático Medio (ECM): {ecm:.4f}")
    pasos.append(f"   (Promedio de cuánto se aleja el modelo de los puntos al cuadrado)")
    pasos.append(f"2. Coeficiente R^2: {r2:.4f}")
    pasos.append(f"   (Un 1.0 es un ajuste perfecto. Un 0.0 es pésimo)")

    # 4. Generación de Datos para la Gráfica (Curva Suave)
    # Usamos linspace para que si es cuadrática se vea curva y no quebrada
    x_min, x_max = min(xs), max(xs)
    padding = (x_max - x_min) * 0.1
    x_grafica = np.linspace(x_min - padding, x_max + padding, 100)
    y_grafica = p(x_grafica)

    # 5. Evaluación Puntual (Predicción)
    resultado_puntual = None
    if x_eval is not None:
        resultado_puntual = p(x_eval)
        pasos.append(f"\nPREDICCIÓN: Para x={x_eval}, el modelo predice y={resultado_puntual:.4f}")

    # --- RETORNO JSON ---
    return {
        "tipo_ajuste": f"Polinomio Grado {grado}",
        "ecuacion_texto": ecuacion_str,
        "metricas": {
            "ecm": ecm,
            "r2": r2,
            "coeficientes": coeficientes.tolist()
        },
        "grafica": {
            "x_modelo": x_grafica.tolist(), # La línea azul/verde
            "y_modelo": y_grafica.tolist(),
            "x_datos": xs.tolist(),         # Los puntos rojos
            "y_datos": ys.tolist()
        },
        "evaluacion": {
            "x": x_eval,
            "y": resultado_puntual
        },
        "pasos": pasos
    }