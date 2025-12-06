import numpy as np
import math
import pandas as pd
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