"""
Verificacion numerica de los integradores (no requiere matplotlib).

Comprueba que la implementacion de oscilador.py es correcta:
  1. Orden de convergencia: Verlet -> 2,  RK4 -> 4.
  2. Reversibilidad temporal de Verlet (propiedad de los metodos simplecticos).
  3. Energia inicial coherente.

Uso:
    python3 verificacion.py
"""

import numpy as np
import oscilador as o


def error_global(integrador, dt, T=10.0):
    """Error en la posicion a tiempo T respecto de la solucion exacta."""
    t, x, v = integrador(o.X0, o.V0, dt, T)
    return abs(x[-1] - o.exacta(t[-1]))


def orden_estimado(errores, pasos):
    """Pendiente local en escala log-log entre pasos consecutivos."""
    return [np.log(errores[i] / errores[i + 1]) / np.log(pasos[i] / pasos[i + 1])
            for i in range(len(errores) - 1)]


def test_convergencia():
    dts = [0.1, 0.05, 0.025, 0.0125]
    ev = [error_global(o.verlet, dt) for dt in dts]
    er = [error_global(o.rk4, dt) for dt in dts]

    p_verlet = orden_estimado(ev, dts)
    p_rk4 = orden_estimado(er, dts)

    print("=== Orden de convergencia (error global en t=10) ===")
    print(f"{'dt':>8} {'err Verlet':>14} {'err RK4':>14}")
    for dt, a, b in zip(dts, ev, er):
        print(f"{dt:8.4f} {a:14.3e} {b:14.3e}")
    print("  orden Verlet:", [f"{p:.2f}" for p in p_verlet], "(esperado ~2)")
    print("  orden RK4   :", [f"{p:.2f}" for p in p_rk4], "(esperado ~4)")

    # El ultimo tramo, ya en regimen asintotico, debe acercarse al orden teorico
    assert abs(p_verlet[-1] - 2.0) < 0.15, "Verlet deberia ser de orden 2"
    assert abs(p_rk4[-1] - 4.0) < 0.25, "RK4 deberia ser de orden 4"


def test_reversibilidad():
    # Integramos hacia adelante, invertimos la velocidad y volvemos:
    # un metodo reversible debe regresar al punto de partida.
    dt, T = 0.1, 5.0
    _, x, v = o.verlet(o.X0, o.V0, dt, T)
    _, xb, vb = o.verlet(x[-1], -v[-1], dt, T)
    err = abs(xb[-1] - o.X0)
    print("\n=== Reversibilidad temporal de Verlet ===")
    print(f"  |x_vuelta - x0| = {err:.2e} (esperado ~0, precision de maquina)")
    assert err < 1e-12, "Verlet deberia ser reversible en el tiempo"


def test_energia_inicial():
    E0 = o.energia(o.X0, o.V0)
    print("\n=== Energia inicial ===")
    print(f"  E(x0, v0) = {E0} (esperado 0.5)")
    assert np.isclose(E0, 0.5)


if __name__ == "__main__":
    test_convergencia()
    test_reversibilidad()
    test_energia_inicial()
    print("\nTodos los chequeos pasaron correctamente.")
