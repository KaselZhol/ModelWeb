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

def resolver_lagrange_web(x_puntos, y_puntos, x_eval=None):
    # Convertimos entradas a arrays de numpy por seguridad
    xs = np.array(x_puntos, dtype=float)
    ys = np.array(y_puntos, dtype=float)
    n = len(xs)
    
    # Lista para guardar la explicación "Paso a Paso"
    pasos_explicativos = []
    pasos_explicativos.append("--- INICIO: Método de Interpolación de Lagrange ---")
    pasos_explicativos.append(f"Fórmula general: P(x) = Σ y_i * L_i(x)")
    
    # Inicializamos coeficientes en 0
    result_coeffs = np.zeros(n)
    
    # --- BUCLE PRINCIPAL (Lógica Matemática) ---
    for i in range(n):
        # 1. Construir el polinomio base Li(x)
        # Excluimos el punto actual xs[i]
        xs_excl = np.delete(xs, i)
        
        # Numerador: (x - x0)(x - x1)...
        numer_coeffs = np.poly(xs_excl)
        
        # Denominador: (xi - x0)(xi - x1)...
        denom = np.prod(xs[i] - xs_excl)
        
        # Ajuste de tamaño (Padding) para poder sumar arrays
        if len(numer_coeffs) < len(result_coeffs):
            diferencia = len(result_coeffs) - len(numer_coeffs)
            numer_coeffs = np.pad(numer_coeffs, (diferencia, 0), 'constant')
        
        # Calculamos el término final: y_i * (Numerador / Denominador)
        peso = ys[i] / denom
        termino_actual = peso * numer_coeffs
        result_coeffs = result_coeffs + termino_actual
        
        # --- NARRATIVA EDUCATIVA (MEJORADA) ---
        # Convertimos el array del término actual a texto legible (ej: "2x^2 + 3")
        poly_str = str(np.poly1d(termino_actual)).strip().replace("\n", " ")
        
        pasos_explicativos.append(
            f"Paso {i+1}: Usando punto P{i}({xs[i]}, {ys[i]}).\n"
            f"   -> Denominador calculado Π(xi - xj): {denom:.4f}\n"
            f"   -> Peso del término (yi / denom): {ys[i]} / {denom:.4f} = {peso:.4f}\n"
            f"   -> Término agregado al polinomio final: {poly_str}"
        )

    # Creamos el objeto Polinomio final
    P = np.poly1d(result_coeffs)

    # --- PREPARACIÓN DE DATOS PARA LA GRÁFICA ---
    x_min, x_max = min(xs), max(xs)
    padding = (x_max - x_min) * 0.2 
    
    x_grafica = np.linspace(x_min - padding, x_max + padding, 100)
    y_grafica = P(x_grafica)

    # --- EVALUACIÓN PUNTUAL ---
    resultado_puntual = None
    if x_eval is not None:
        resultado_puntual = float(P(x_eval))
        pasos_explicativos.append(f"RESULTADO FINAL: Evaluando P({x_eval}) obtenemos {resultado_puntual:.6f}")

    # --- RETORNO DE DATOS (JSON) ---
    return {
        "ecuacion_texto": str(P).strip(), 
        "coeficientes": result_coeffs.tolist(),
        "evaluacion": {
            "x_solicitada": x_eval,
            "y_resultado": resultado_puntual
        },
        "datos_grafica": {
            "x": x_grafica.tolist(), 
            "y": y_grafica.tolist(), 
            "puntos_originales_x": xs.tolist(), 
            "puntos_originales_y": ys.tolist()
        },
        "pasos": pasos_explicativos # <--- Aquí viaja la explicación completa
    }