import numpy as np
import math
import pandas as pd
# SEMANA 9
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
    
def resolver_lagrange_con_pasos_reales(x_puntos, y_puntos, x_eval=None):
    xs = np.array(x_puntos, dtype=float)
    ys = np.array(y_puntos, dtype=float)
    n = len(xs)
    
    pasos_explicativos = []
    pasos_explicativos.append("--- INICIO: Construcción del Polinomio de Lagrange ---")
    pasos_explicativos.append(f"Fórmula general: P(x) = Σ y_i * L_i(x)")
    
    result_coeffs = np.zeros(n)
    
    # BUCLE PRINCIPAL
    for i in range(n):
        # 1. Preparar datos
        xi, yi = xs[i], ys[i]
        xs_excl = np.delete(xs, i) # Puntos restantes (xj)
        
        # 2. Calcular Numerador y Denominador
        numer_coeffs = np.poly(xs_excl) # Coeficientes de (x-x0)(x-x1)...
        denom = np.prod(xi - xs_excl)   # Valor escalar (xi-x0)(xi-x1)...
        
        # 3. Calcular el Polinomio de este término (Li)
        # Li_coeffs = (yi / denom) * numer_coeffs
        peso = yi / denom
        termino_coeffs = peso * numer_coeffs
        
        # Ajuste de tamaño (Padding) para sumar
        if len(termino_coeffs) < len(result_coeffs):
            diferencia = len(result_coeffs) - len(termino_coeffs)
            termino_coeffs = np.pad(termino_coeffs, (diferencia, 0), 'constant')
        
        result_coeffs = result_coeffs + termino_coeffs
        
        # --- AQUÍ ESTÁ LA MEJORA VISUAL DE LOS PASOS ---
        # Convertimos el array de coeficientes actual a texto bonito: "2x^2 + 3"
        # Usamos np.poly1d solo para formatear el texto automáticamente
        poly_str = str(np.poly1d(termino_coeffs)).strip().replace("\n", " ")
        
        paso_texto = (
            f"Paso {i}: Usando punto P{i}({xi}, {yi})\n"
            f"   -> Calculamos L_{i}(x). El denominador es: Π(xi - xj) = {denom:.2f}\n"
            f"   -> El peso (yi / denom) es: {yi} / {denom:.2f} = {peso:.4f}\n"
            f"   -> Término agregado al polinomio final: {poly_str}"
        )
        pasos_explicativos.append(paso_texto)

    # Polinomio Final
    P = np.poly1d(result_coeffs)
    
    # Datos Gráfica
    x_min, x_max = min(xs), max(xs)
    padding = (x_max - x_min) * 0.2
    x_grafica = np.linspace(x_min - padding, x_max + padding, 100)
    y_grafica = P(x_grafica)

    resultado_eval = None
    if x_eval is not None:
        resultado_eval = float(P(x_eval))
        pasos_explicativos.append(f"EVALUACIÓN: P({x_eval}) = {resultado_eval:.6f}")

    return {
        "ecuacion_final": str(P).strip(),
        "pasos": pasos_explicativos, # <--- AQUÍ ESTÁN TUS PASOS
        "grafica": {
            "x": x_grafica.tolist(),
            "y": y_grafica.tolist(),
            "puntos_x": xs.tolist(),
            "puntos_y": ys.tolist()
        }
    }
# SEMANA 10
def resolver_newton_web(x_puntos, y_puntos, x_eval=None):
    # 1. Preparación de datos
    xs = np.array(x_puntos, dtype=float)
    ys = np.array(y_puntos, dtype=float)
    n = len(xs)
    
    pasos_explicativos = []
    pasos_explicativos.append("--- INICIO: Método de Diferencias Divididas de Newton ---")
    
    # 2. Construcción de la Tabla (Lógica Matemática)
    # Creamos matriz n x n llena de ceros
    tabla = np.zeros((n, n))
    # La primera columna son las Y originales (f[x_i])
    tabla[:, 0] = ys
    
    # Bucle para llenar columnas (orden 1, orden 2, etc.)
    for j in range(1, n):
        pasos_explicativos.append(f"\n--- Calculando Diferencias de Orden {j} ---")
        for i in range(n - j):
            # Fórmula: (Valor_Abajo - Valor_Arriba) / (X_Lejos - X_Cerca)
            numerador = tabla[i+1, j-1] - tabla[i, j-1]
            denominador = xs[i+j] - xs[i]
            
            valor_calculado = numerador / denominador
            tabla[i, j] = valor_calculado
            
            # --- Narrativa Educativa Detallada ---
            pasos_explicativos.append(
                f"Celda [{i},{j}]: ({tabla[i+1, j-1]:.4f} - {tabla[i, j-1]:.4f}) / ({xs[i+j]} - {xs[i]}) = {valor_calculado:.6f}"
            )
            
    # Los coeficientes son la primera fila de la tabla (la diagonal superior conceptualmente)
    coeficientes = tabla[0, :]
    
    # 3. Construcción del Polinomio como TEXTO (Para mostrar la fórmula)
    # Formato: b0 + b1(x-x0) + b2(x-x0)(x-x1)...
    poly_str = f"{coeficientes[0]:.4f}"
    for k in range(1, n):
        signo = "+" if coeficientes[k] >= 0 else "-"
        val_abs = abs(coeficientes[k])
        
        # Construimos los paréntesis (x-x0)(x-x1)...
        terminos = ""
        for m in range(k):
            terminos += f"(x - {xs[m]})"
            
        poly_str += f" {signo} {val_abs:.4f}*{terminos}"

    # 4. Función interna para evaluar (Horner) - Se usa para gráfica y evaluación puntual
    def evaluar_newton_interno(x_val):
        result = coeficientes[n-1]
        for k in range(n-2, -1, -1):
            result = result * (x_val - xs[k]) + coeficientes[k]
        return result

    # 5. Generación de Datos para la Gráfica
    x_min, x_max = min(xs), max(xs)
    padding = (x_max - x_min) * 0.2
    x_grafica = np.linspace(x_min - padding, x_max + padding, 100)
    # Aplicamos la función de evaluación a todo el array (vectorizado)
    # Nota: Como 'evaluar_newton_interno' usa bucle for, usamos un list comprehension o vectorize
    y_grafica = [evaluar_newton_interno(val) for val in x_grafica]

    # 6. Evaluación Puntual (si se pide)
    resultado_puntual = None
    if x_eval is not None:
        resultado_puntual = evaluar_newton_interno(x_eval)
        pasos_explicativos.append(f"\nEVALUACIÓN FINAL en x={x_eval}: {resultado_puntual:.6f}")

    # --- RETORNO JSON ---
    return {
        "metodo": "Newton Diferencias Divididas",
        "coeficientes": coeficientes.tolist(),
        "tabla_completa": tabla.tolist(), # IMPORTANTE: Enviamos la tabla para que el Frontend la dibuje
        "polinomio_texto": poly_str,
        "pasos": pasos_explicativos,
        "evaluacion": {
            "x": x_eval,
            "y": resultado_puntual
        },
        "grafica": {
            "x": x_grafica.tolist(),
            "y": y_grafica,
            "puntos_x": xs.tolist(),
            "puntos_y": ys.tolist()
        }
    }

def resolver_newton_web(x_puntos, y_puntos, x_eval=None):
    # 1. Preparación de datos
    xs = np.array(x_puntos, dtype=float)
    ys = np.array(y_puntos, dtype=float)
    n = len(xs)
    
    pasos_explicativos = []
    pasos_explicativos.append("--- INICIO: Método de Diferencias Divididas de Newton ---")
    
    # 2. Construcción de la Tabla (Lógica Matemática)
    # Creamos matriz n x n para guardar TODA la historia
    tabla = np.zeros((n, n))
    tabla[:, 0] = ys # La primera columna son las Y
    
    # Bucle para llenar columnas
    for j in range(1, n):
        pasos_explicativos.append(f"\n--- Calculando Diferencias de Orden {j} ---")
        for i in range(n - j):
            # Fórmula: (Abajo - Arriba) / (X_lejos - X_cerca)
            numerador = tabla[i+1, j-1] - tabla[i, j-1]
            denominador = xs[i+j] - xs[i]
            
            valor = numerador / denominador
            tabla[i, j] = valor
            
            pasos_explicativos.append(
                f"Celda [{i},{j}]: ({tabla[i+1, j-1]:.4f} - {tabla[i, j-1]:.4f}) / ({xs[i+j]} - {xs[i]}) = {valor:.4f}"
            )
            
    # Los coeficientes son la primera fila (diagonal superior)
    coeficientes = tabla[0, :]
    
    # 3. Construcción del Polinomio (Texto Bonito)
    # Formato: b0 + b1(x-x0) + b2(x-x0)(x-x1)...
    poly_str = f"{coeficientes[0]:.4f}"
    for k in range(1, n):
        signo = "+" if coeficientes[k] >= 0 else "-"
        val_abs = abs(coeficientes[k])
        
        # Construimos (x - x0)(x - x1)...
        terminos_x = ""
        for m in range(k):
            terminos_x += f"(x - {xs[m]})"
            
        poly_str += f" {signo} {val_abs:.4f}*{terminos_x}"

    # 4. Función interna para evaluar (Horner)
    def evaluar_newton(val_x):
        res = coeficientes[n-1]
        for k in range(n-2, -1, -1):
            res = res * (val_x - xs[k]) + coeficientes[k]
        return res

    # 5. Generar Datos Gráfica
    x_min, x_max = min(xs), max(xs)
    padding = (x_max - x_min) * 0.2
    x_grafica = np.linspace(x_min - padding, x_max + padding, 100)
    y_grafica = [evaluar_newton(val) for val in x_grafica]

    # 6. Evaluación Puntual
    resultado_puntual = None
    if x_eval is not None:
        resultado_puntual = evaluar_newton(x_eval)
        pasos_explicativos.append(f"\nRESULTADO FINAL: Evaluando P({x_eval}) = {resultado_puntual:.6f}")

    # --- RETORNO JSON ---
    return {
        "metodo": "Newton",
        "coeficientes": coeficientes.tolist(),
        "tabla_completa": tabla.tolist(), # <--- IMPORTANTE: La tabla completa para el Frontend
        "polinomio_texto": poly_str,
        "evaluacion": {
            "x": x_eval,
            "y": resultado_puntual
        },
        "grafica": {
            "x": x_grafica.tolist(),
            "y": y_grafica,
            "puntos_originales_x": xs.tolist(),
            "puntos_originales_y": ys.tolist()
        },
        "pasos": pasos_explicativos
    }
    
def resolver_newton_universal_web(x_puntos, y_puntos, x_eval=None, valor_real=None):
    """
    Resuelve Newton para CUALQUIER set de datos (Seno, Exponencial, etc.)
    y devuelve el paso a paso detallado para la web.
    """
    xs = np.array(x_puntos, dtype=float)
    ys = np.array(y_puntos, dtype=float)
    n = len(xs)
    
    pasos = []
    pasos.append("--- INICIO: Interpolación de Newton ---")
    
    # 1. Construir la Tabla de Diferencias Divididas (Matriz n x n)
    # Usamos una matriz completa para poder mostrarla en el Frontend
    tabla = np.zeros((n, n))
    tabla[:, 0] = ys # Primera columna = f(x)
    
    pasos.append(f"Datos iniciales cargados: {n} puntos.")

    # --- BUCLE DE CÁLCULO DE LA TABLA ---
    for j in range(1, n):
        pasos.append(f"\n--- Generando Diferencias de Orden {j} ---")
        for i in range(n - j):
            # Fórmula: (Abajo - Arriba) / (X_lejos - X_cerca)
            numerador = tabla[i+1, j-1] - tabla[i, j-1]
            denominador = xs[i+j] - xs[i]
            
            valor = numerador / denominador
            tabla[i, j] = valor
            
            # Explicación detallada para el estudiante
            pasos.append(
                f"f[x{i}...x{i+j}] = ({tabla[i+1, j-1]:.5f} - {tabla[i, j-1]:.5f}) / ({xs[i+j]:.4f} - {xs[i]:.4f}) = {valor:.6f}"
            )

    # Los coeficientes finales (b0, b1, b2...) son la primera fila
    coeficientes = tabla[0, :]

    # 2. Construir el Texto de la Ecuación
    # P(x) = b0 + b1(x-x0) + ...
    poly_str = f"{coeficientes[0]:.4f}"
    for k in range(1, n):
        signo = "+" if coeficientes[k] >= 0 else "-"
        # Construimos los términos (x-x0)(x-x1)...
        terminos_x = ""
        for m in range(k):
            terminos_x += f"(x - {xs[m]:.2f})"
        
        poly_str += f" {signo} {abs(coeficientes[k]):.4f}*{terminos_x}"

    # 3. Función Interna de Evaluación (Horner)
    def evaluar_newton(val_x):
        res = coeficientes[n-1]
        for k in range(n-2, -1, -1):
            res = res * (val_x - xs[k]) + coeficientes[k]
        return res

    # 4. Datos para Gráfica (100 puntos)
    x_min, x_max = min(xs), max(xs)
    padding = (x_max - x_min) * 0.2
    x_grafica = np.linspace(x_min - padding, x_max + padding, 100)
    y_grafica = [evaluar_newton(val) for val in x_grafica]

    # 5. Evaluación Puntual y Cálculo de Error
    resultado_puntual = None
    error_abs = None
    
    if x_eval is not None:
        resultado_puntual = evaluar_newton(x_eval)
        pasos.append(f"\nEVALUACIÓN: Sustituyendo x={x_eval:.4f} en el polinomio...")
        pasos.append(f"Resultado aproximado P({x_eval:.4f}) = {resultado_puntual:.6f}")
        
        # Si nos dieron el valor real (ej: math.sin(x)), calculamos el error
        if valor_real is not None:
            error_abs = abs(valor_real - resultado_puntual)
            pasos.append(f"COMPARACIÓN: Valor Real = {valor_real:.6f} | Error Absoluto = {error_abs:.6f}")

    # --- RETORNO JSON ---
    return {
        "ecuacion_texto": poly_str,
        "tabla_diferencias": tabla.tolist(), # Para pintar la tabla en HTML
        "coeficientes": coeficientes.tolist(),
        "resultado": {
            "x": x_eval,
            "y_aprox": resultado_puntual,
            "y_real": valor_real,
            "error": error_abs
        },
        "grafica": {
            "x": x_grafica.tolist(),
            "y": y_grafica,
            "puntos_x": xs.tolist(),
            "puntos_y": ys.tolist()
        },
        "pasos": pasos # El historial completo
    }
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
# SEMANA 12
def resolver_integracion_web(funcion_str, a, b, n, metodo="trapecio"):
    # --- 1. Interpretación de la función ---
    def f(x):
        contexto = {"x": x, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "pi": np.pi}
        return eval(funcion_str, {"__builtins__": None}, contexto)

    # Validaciones iniciales
    if metodo == "simpson" and n % 2 != 0:
        return {"error": "Error: Para Simpson 1/3, 'n' debe ser par."}

    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = np.array([f(xi) for xi in xs])
    
    pasos = []
    pasos.append(f"--- INICIO: Integración por Método {metodo.capitalize()} ---")
    pasos.append(f"Datos: Intervalo [{a}, {b}], n={n}, paso h={h:.5f}")

    area = 0.0
    error_reporte = "No calculado"

    # ==========================================
    # CAMINO A: LÓGICA EXCLUSIVA DE TRAPECIO
    # ==========================================
    if metodo == "trapecio":
        # 1. Tu fórmula de Trapecio
        suma_interna = np.sum(ys[1:-1])
        area = (h / 2) * (ys[0] + 2 * suma_interna + ys[-1])
        
        pasos.append("Fórmula: (h/2) * [f(a) + 2*SumaInterna + f(b)]")
        pasos.append(f"   -> Términos extremos: {ys[0]:.4f} + {ys[-1]:.4f}")
        pasos.append(f"   -> Suma interna: {suma_interna:.4f}")

        # 2. TU CÁLCULO DE ERROR (Cota con segunda derivada)
        # Esto es código recuperado de TU script original
        try:
            xx_fino = np.linspace(a, b, 1001)
            yy_fino = np.array([f(xi) for xi in xx_fino])
            dx = xx_fino[1] - xx_fino[0]
            
            # Diferencia finita centrada para f''(x)
            # Usamos np.gradient dos veces para simular tu cálculo de f2
            dy = np.gradient(yy_fino, dx)
            d2y = np.gradient(dy, dx)
            M2 = np.max(np.abs(d2y))
            
            cota_error = ((b - a) / 12.0) * (h**2) * M2
            error_reporte = cota_error
            
            pasos.append("\n--- Análisis de Error (Exclusivo Trapecio) ---")
            pasos.append(f"Estimación de max |f''(x)| ≈ {M2:.6f}")
            pasos.append(f"Cota de error teórica <= {cota_error:.8f}")
        except:
            pasos.append("No se pudo calcular la derivada automáticamente.")

    # ==========================================
    # CAMINO B: LÓGICA EXCLUSIVA DE SIMPSON
    # ==========================================
    elif metodo == "simpson":
        # 1. Tu fórmula de Simpson (Impares x4, Pares x2)
        suma_impares = np.sum(ys[1:n:2])
        suma_pares = np.sum(ys[2:n:2])
        area = (h / 3) * (ys[0] + 4 * suma_impares + 2 * suma_pares + ys[-1])
        
        pasos.append("Fórmula: (h/3) * [f(a) + 4*Impares + 2*Pares + f(b)]")
        pasos.append(f"   -> Suma Impares (x4): {suma_impares:.4f}")
        pasos.append(f"   -> Suma Pares (x2): {suma_pares:.4f}")
        
        # 2. Tu comparación con REFERENCIA (Alta precisión)
        # Simpson no suele usar la cota de derivada f''''(x) en código porque es muy ruidosa,
        # así que usamos tu método de "Referencia Fina" que tenías en el script 2.
        n_ref = 10000
        h_ref = (b - a) / n_ref
        xs_ref = np.linspace(a, b, n_ref + 1)
        ys_ref = np.array([f(xi) for xi in xs_ref])
        ref_val = (h_ref/3) * (ys_ref[0] + 4*np.sum(ys_ref[1:n:2]) + 2*np.sum(ys_ref[2:n:2]) + ys_ref[-1])
        
        diff = abs(ref_val - area)
        error_reporte = diff
        
        pasos.append("\n--- Comparación con Referencia (Exclusivo Simpson) ---")
        pasos.append(f"Valor referencia (N=10000): {ref_val:.8f}")
        pasos.append(f"Diferencia: {diff:.8f}")

    pasos.append(f"\n>>> RESULTADO FINAL DEL ÁREA: {area:.8f}")

    return {
        "metodo": metodo,
        "resultado": area,
        "error_analisis": error_reporte,
        "pasos": pasos,
        "grafica": { "x": xs.tolist(), "y": ys.tolist() }
    }
# SEMANA 13
def resolver_edo_universal_web(f_prime_str, x0, y0, h, pasos_num, metodo="euler", f_double_prime_str=None, sol_exacta_str=None):
    """
    Resuelve EDOs usando Euler o Taylor Orden 2.
    Admite funciones en formato texto (para la web) y calcula errores si hay solución exacta.
    
    Args:
        f_prime_str: Ecuación de y' (ej: "x + y")
        f_double_prime_str: Ecuación de y'' (Solo requerida para Taylor, ej: "1 + yp")
        sol_exacta_str: Ecuación de la solución real para calcular error (ej: "exp(x)")
    """
    # 1. Preparar Listas de Resultados
    xs = [x0]
    ys = [y0]
    errores = [0.0] if sol_exacta_str else None
    
    # Historial detallado para el estudiante
    historial_pasos = []
    historial_pasos.append(f"--- INICIO: Método {metodo.capitalize()} ---")
    historial_pasos.append(f"Condiciones: y({x0})={y0}, h={h}, Pasos={pasos_num}")
    
    # Variables actuales
    x_actual = x0
    y_actual = y0
    
    # 2. Funciones Auxiliares para interpretar texto matemático
    def evaluar_expr(expr_str, ctx):
        # Contexto seguro + funciones matemáticas
        seguridad = {"__builtins__": None}
        librerias = {"sin": math.sin, "cos": math.cos, "exp": math.exp, "sqrt": math.sqrt, "log": math.log, "pi": math.pi}
        full_ctx = {**seguridad, **librerias, **ctx}
        return eval(expr_str, full_ctx)

    # --- BUCLE PRINCIPAL ---
    for i in range(pasos_num):
        # A. Calcular Primera Derivada (y')
        # Disponible para Euler y Taylor
        yp = evaluar_expr(f_prime_str, {"x": x_actual, "y": y_actual})
        
        y_siguiente = 0.0
        explicacion = ""
        
        # B. Aplicar Lógica del Método
        if metodo == "euler":
            # Fórmula: y_next = y + h * y'
            y_siguiente = y_actual + h * yp
            
            explicacion = (
                f"Paso {i+1} (Euler):\n"
                f"   -> Pendiente (y') = f({x_actual:.2f}, {y_actual:.4f}) = {yp:.4f}\n"
                f"   -> Nuevo y = {y_actual:.4f} + {h} * {yp:.4f} = {y_siguiente:.4f}"
            )
            
        elif metodo == "taylor2":
            if not f_double_prime_str:
                return {"error": "Para Taylor 2 se requiere la ecuación de la segunda derivada (y'')."}
            
            # Calcular Segunda Derivada (y'')
            # Nota: Permitimos usar 'yp' en la fórmula de la segunda derivada como en tus ejemplos
            ypp = evaluar_expr(f_double_prime_str, {"x": x_actual, "y": y_actual, "yp": yp})
            
            # Fórmula: y_next = y + h*y' + (h^2/2)*y''
            termino_1 = h * yp
            termino_2 = (h**2 / 2.0) * ypp
            y_siguiente = y_actual + termino_1 + termino_2
            
            explicacion = (
                f"Paso {i+1} (Taylor 2):\n"
                f"   -> y' = {yp:.4f}, y'' = {ypp:.4f}\n"
                f"   -> Términos: h*y'={termino_1:.4f}, (h^2/2)*y''={termino_2:.4f}\n"
                f"   -> Nuevo y = {y_actual:.4f} + {termino_1:.4f} + {termino_2:.4f} = {y_siguiente:.4f}"
            )

        # C. Cálculo de Error (Si hay solución exacta)
        x_siguiente = round(x_actual + h, 12) # Round para evitar 0.300000004
        error_val = None
        
        if sol_exacta_str:
            y_real = evaluar_expr(sol_exacta_str, {"x": x_siguiente})
            error_val = y_real - y_siguiente
            explicacion += f"\n   -> Valor Real: {y_real:.4f} | Error: {error_val:.6f}"
            errores.append(error_val)
            
        # D. Guardar y Avanzar
        historial_pasos.append(explicacion)
        xs.append(x_siguiente)
        ys.append(y_siguiente)
        
        x_actual = x_siguiente
        y_actual = y_siguiente

    # --- RETORNO JSON ---
    return {
        "metodo": metodo,
        "datos_tabla": {
            "x": xs,
            "y_calculada": ys,
            "error": errores if errores else []
        },
        "grafica": {
            "x": xs,
            "y": ys
        },
        "pasos": historial_pasos
    }
# SEMANA 15
