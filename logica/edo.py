import math
import numpy as np

# --- FUNCIÓN DE EVALUACIÓN SEGURA (EL ESCUDO) ---
def evaluar_seguro(expr_str, ctx):
    """
    Evalúa una expresión matemática de forma segura.
    Si la expresión es None, vacía o inválida, devuelve 0.0.
    """
    # 1. Validación de vacíos
    if not expr_str:
        return 0.0
    
    if isinstance(expr_str, str):
        if expr_str.strip() == "" or expr_str.lower() == "none" or expr_str.lower() == "null":
            return 0.0
    
    try:
        # 2. Contexto matemático permitido
        seguridad = {"__builtins__": {}} # Sin acceso a sistema
        librerias = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan, 
            "exp": math.exp, "sqrt": math.sqrt, "log": math.log, 
            "pi": math.pi, "e": math.e, "abs": abs
        }
        # Combinamos seguridad + librerías + variables (x, y, t)
        full_ctx = {**seguridad, **librerias, **ctx}
        
        # 3. Evaluación
        resultado = eval(str(expr_str), full_ctx)
        return float(resultado)
    except Exception:
        # Si la fórmula está mal escrita (ej: "2**"), devolvemos 0.0 en lugar de crashear
        return 0.0

# --- SEMANA 13: EULER Y TAYLOR ---
def resolver_edo_universal_web(f_prime_str, x0, y0, h, pasos_num, metodo="euler", f_double_prime_str=None, sol_exacta_str=None):
    # Preparar Listas
    xs = [x0]
    ys = [y0]
    
    # Solo calculamos error si hay una solución exacta válida (no "0" ni vacía)
    calcular_error = False
    if sol_exacta_str and str(sol_exacta_str).strip() != "0" and str(sol_exacta_str).strip() != "":
        calcular_error = True
        
    errores = [0.0] if calcular_error else []
    
    historial_pasos = []
    historial_pasos.append(f"--- INICIO: Método {str(metodo).capitalize()} ---")
    historial_pasos.append(f"Condiciones: y({x0})={y0}, h={h}")
    
    x_actual = x0
    y_actual = y0
    
    for i in range(pasos_num):
        # A. Derivada (y')
        # Pasamos 't' también por si el usuario usa t en vez de x
        yp = evaluar_seguro(f_prime_str, {"x": x_actual, "y": y_actual, "t": x_actual})
        
        y_siguiente = 0.0
        explicacion = ""
        
        # B. Lógica del Método
        if metodo == "taylor2":
            # Si no hay segunda derivada, evaluar_seguro devolverá 0.0 y funcionará como Euler
            ypp = evaluar_seguro(f_double_prime_str, {"x": x_actual, "y": y_actual, "yp": yp, "t": x_actual})
            
            termino_1 = h * yp
            termino_2 = (h**2 / 2.0) * ypp
            y_siguiente = y_actual + termino_1 + termino_2
            
            explicacion = (
                f"Paso {i+1} (Taylor):\n"
                f" -> y'={yp:.4f}, y''={ypp:.4f}\n"
                f" -> y_new = {y_actual:.4f} + {termino_1:.4f} + {termino_2:.4f} = {y_siguiente:.4f}"
            )
            
        else:
            # Euler (Por defecto)
            y_siguiente = y_actual + h * yp
            explicacion = (
                f"Paso {i+1} (Euler):\n"
                f" -> y'={yp:.4f}\n"
                f" -> y_new = {y_actual:.4f} + {h}*{yp:.4f} = {y_siguiente:.4f}"
            )
            
        # C. Cálculo de Error
        x_siguiente = x_actual + h
        
        if calcular_error:
            y_real = evaluar_seguro(sol_exacta_str, {"x": x_siguiente, "t": x_siguiente})
            error_val = abs(y_real - y_siguiente)
            errores.append(error_val)
            explicacion += f"\n -> Real={y_real:.4f} | Error={error_val:.6f}"
            
        # D. Guardar
        historial_pasos.append(explicacion)
        xs.append(x_siguiente)
        ys.append(y_siguiente)
        
        x_actual = x_siguiente
        y_actual = y_siguiente
        
    return {
        "metodo": metodo,
        "datos_tabla": {
            "x": xs,
            "y_calculada": ys,
            "error": errores
        },
        "grafica": {
            "x": xs,
            "y": ys
        },
        "pasos": historial_pasos
    }

# --- SEMANA 15: RK2 y RK4 (Legacy) ---
def resolver_solo_rk2(ecuacion_str, t0, y0, h, pasos_num):
    ts = [t0]
    ys = [y0]
    pasos_explicativos = [f"--- MÉTODO RK2 (HEUN) ---"]
    
    t_actual = t0
    y_actual = y0
    
    for i in range(pasos_num):
        k1 = evaluar_seguro(ecuacion_str, {"t": t_actual, "y": y_actual, "x": t_actual})
        k2 = evaluar_seguro(ecuacion_str, {"t": t_actual + h, "y": y_actual + h * k1, "x": t_actual + h})
        
        pendiente_promedio = (k1 + k2) / 2.0
        y_siguiente = y_actual + h * pendiente_promedio
        
        pasos_explicativos.append(f"Paso {i+1}: k1={k1:.4f}, k2={k2:.4f} -> y_new={y_siguiente:.4f}")
        
        t_actual += h
        y_actual = y_siguiente
        ts.append(t_actual)
        ys.append(y_actual)

    return {
        "metodo": "RK2 (Heun)",
        "grafica": {"t": ts, "y": ys},
        "resultado_final": y_actual,
        "pasos": pasos_explicativos
    }

def resolver_solo_rk4(ecuacion_str, t0, y0, h, pasos_num):
    ts = [t0]
    ys = [y0]
    pasos_explicativos = [f"--- MÉTODO RK4 ---"]
    
    t_actual = t0
    y_actual = y0
    
    for i in range(pasos_num):
        k1 = evaluar_seguro(ecuacion_str, {"t": t_actual, "y": y_actual, "x": t_actual})
        
        t_half = t_actual + h/2
        k2 = evaluar_seguro(ecuacion_str, {"t": t_half, "y": y_actual + (h/2)*k1, "x": t_half})
        k3 = evaluar_seguro(ecuacion_str, {"t": t_half, "y": y_actual + (h/2)*k2, "x": t_half})
        
        t_next = t_actual + h
        k4 = evaluar_seguro(ecuacion_str, {"t": t_next, "y": y_actual + h*k3, "x": t_next})
        
        pendiente = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        y_siguiente = y_actual + h * pendiente
        
        pasos_explicativos.append(f"Paso {i+1}: k1={k1:.2f}... k4={k4:.2f} -> y_new={y_siguiente:.4f}")
        
        t_actual += h
        y_actual = y_siguiente
        ts.append(t_actual)
        ys.append(y_actual)

    return {
        "metodo": "RK4",
        "grafica": {"t": ts, "y": ys},
        "resultado_final": y_actual,
        "pasos": pasos_explicativos
    }