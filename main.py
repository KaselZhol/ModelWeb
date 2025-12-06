import numpy as np

def lagrange_eval_api(x_eval, x_puntos, y_puntos):
    n = len(x_puntos)
    total = 0.0
    
    # Lista de pasos (List of strings)
    lista_pasos = []
    lista_pasos.append(f"--- INICIO: Evaluando Polinomio de Lagrange en x = {x_eval} ---")
    
    for i in range(n):
        li = 1.0  
        formula_str = "" # Para mostrar la fracción visualmente
        
        # 1. Cálculo de Li(x)
        for j in range(n):
            if j != i:
                numerador = x_eval - x_puntos[j]
                denominador = x_puntos[i] - x_puntos[j]
                
                # Validación de seguridad (división por cero)
                if denominador == 0:
                    return {"error": "Error matemático: Puntos x duplicados detectados."}
                
                term = numerador / denominador
                li *= term
        
        # 2. Contribución y_i * L_i(x)
        contribucion = y_puntos[i] * li
        total += contribucion
        
        # 3. Registro del paso (Formato Educativo)
        # Usamos HTML básico o Markdown para resaltar cosas si queremos
        paso_detalle = {
            "iteracion": i,
            "termino_Li": float(f"{li:.6f}"), # Convertimos a float puro para JSON
            "formula": f"L_{i}({x_eval})",
            "explicacion": f"Multiplicamos y_{i} ({y_puntos[i]}) por el factor L_{i} ({li:.4f}). Sumamos {contribucion:.4f} al total."
        }
        lista_pasos.append(paso_detalle)
        
    # Retorno estructurado (JSON friendly)
    return {
        "resultado_final": total,
        "x_evaluado": x_eval,
        "pasos_detallados": lista_pasos
    }