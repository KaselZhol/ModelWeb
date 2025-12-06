import numpy as np
import math
import pandas as pd
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
def resolver_solo_rk2(ecuacion_str, t0, y0, h, pasos_num):
    ts = [t0]
    ys = [y0]
    pasos_explicativos = []
    
    pasos_explicativos.append(f"--- MÉTODO RK2 (HEUN) ---")
    
    t_actual = t0
    y_actual = y0
    
    def f(t, y):
        ctx = {"t": t, "y": y, "sin": math.sin, "cos": math.cos, "exp": math.exp}
        return eval(ecuacion_str, {"__builtins__": None}, ctx)

    for i in range(pasos_num):
        # Lógica RK2
        k1 = f(t_actual, y_actual)
        k2 = f(t_actual + h, y_actual + h * k1)
        
        pendiente_promedio = (k1 + k2) / 2.0
        y_siguiente = y_actual + h * pendiente_promedio
        
        texto = (
            f"Paso {i+1}:\n"
            f"   -> k1 (inicio) = {k1:.6f}\n"
            f"   -> k2 (fin estimado) = {k2:.6f}\n"
            f"   -> Promedio (k1+k2)/2 = {pendiente_promedio:.6f}\n"
            f"   -> Nuevo y = {y_actual:.6f} + {h} * {pendiente_promedio:.6f} = {y_siguiente:.6f}"
        )
        pasos_explicativos.append(texto)
        
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
    pasos_explicativos = []
    
    pasos_explicativos.append(f"--- MÉTODO RK4 ---")
    
    t_actual = t0
    y_actual = y0
    
    def f(t, y):
        ctx = {"t": t, "y": y, "sin": math.sin, "cos": math.cos, "exp": math.exp}
        return eval(ecuacion_str, {"__builtins__": None}, ctx)

    for i in range(pasos_num):
        # Lógica RK4
        k1 = f(t_actual, y_actual)
        k2 = f(t_actual + h/2, y_actual + (h/2)*k1)
        k3 = f(t_actual + h/2, y_actual + (h/2)*k2)
        k4 = f(t_actual + h, y_actual + h*k3)
        
        pendiente_ponderada = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        y_siguiente = y_actual + h * pendiente_ponderada
        
        texto = (
            f"Paso {i+1}:\n"
            f"   -> k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, k4={k4:.4f}\n"
            f"   -> Pendiente Ponderada = {pendiente_ponderada:.6f}\n"
            f"   -> Nuevo y = {y_actual:.6f} + {h} * {pendiente_ponderada:.6f} = {y_siguiente:.6f}"
        )
        pasos_explicativos.append(texto)
        
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