"""
Verlet vs Runge-Kutta: conservacion de energia en el oscilador armonico
=======================================================================

Objetivo didactico
-------------------
Mostrar, de la forma mas simple posible, por que un integrador SIMPLECTICO
(Verlet de velocidades) conserva la energia a tiempos largos, mientras que un
integrador clasico de alta precision como Runge-Kutta de orden 4 (RK4)
*deriva* lentamente: pierde (o gana) energia de forma sistematica.

El sistema fisico
-----------------
Oscilador armonico simple (una masa unida a un resorte ideal):

        d2x/dt2 = -(k/m) x

Con omega^2 = k/m. Tomamos m = k = 1  =>  omega = 1.

Lo reescribimos como un sistema de primer orden con posicion x y velocidad v:

        dx/dt = v
        dv/dt = -omega^2 * x

La energia total (cinetica + potencial), que deberia conservarse, es:

        E = 1/2 * m * v^2 + 1/2 * k * x^2

Solucion analitica (la "verdad" contra la que comparamos):

        x(t) = x0 cos(omega t) + (v0/omega) sin(omega t)

Autor: material para una clase de grado.
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Parametros fisicos y de simulacion
# ----------------------------------------------------------------------
OMEGA = 1.0          # frecuencia angular (omega^2 = k/m, con m=k=1)
X0, V0 = 1.0, 0.0    # condiciones iniciales: parte del reposo en x=1
DT = 0.5             # paso de tiempo (deliberadamente grande para ver la deriva)
T_FINAL = 1000.0     # tiempo total: muchos periodos para ver el efecto a largo plazo


def aceleracion(x):
    """Fuerza por unidad de masa: a = -omega^2 x."""
    return -OMEGA**2 * x


def energia(x, v):
    """Energia total del oscilador.

    E = 1/2 m v^2 + 1/2 k x^2. Como tomamos m = 1 y k = omega^2 (con m = 1),
    queda E = 1/2 v^2 + 1/2 omega^2 x^2. Si cambiaras m =/= 1 habria que
    reescribir esta formula con m y k explicitos.
    """
    return 0.5 * v**2 + 0.5 * OMEGA**2 * x**2


# ----------------------------------------------------------------------
# Integrador 1: VERLET DE VELOCIDADES (simplectico)
# ----------------------------------------------------------------------
#   x_{n+1} = x_n + v_n dt + 1/2 a_n dt^2
#   v_{n+1} = v_n + 1/2 (a_n + a_{n+1}) dt
#
# Es simplectico: preserva exactamente la estructura geometrica del espacio
# de fases. No conserva la energia exacta E, pero si una energia "sombra"
# muy proxima a ella, por eso E oscila acotada y NO deriva.
# ----------------------------------------------------------------------
def verlet(x0, v0, dt, t_final):
    n = int(t_final / dt)
    t = np.linspace(0.0, n * dt, n + 1)
    x = np.empty(n + 1)
    v = np.empty(n + 1)
    x[0], v[0] = x0, v0

    a = aceleracion(x[0])
    for i in range(n):
        x[i + 1] = x[i] + v[i] * dt + 0.5 * a * dt**2
        a_nuevo = aceleracion(x[i + 1])
        v[i + 1] = v[i] + 0.5 * (a + a_nuevo) * dt
        a = a_nuevo          # reutilizamos la aceleracion: 1 sola evaluacion por paso
    return t, x, v


# ----------------------------------------------------------------------
# Integrador 2: RUNGE-KUTTA 4 (NO simplectico)
# ----------------------------------------------------------------------
# Muy preciso localmente (error O(dt^5) por paso), pero al no respetar la
# estructura simplectica, el error de energia se acumula de forma sistematica
# y la energia deriva monotonamente con el tiempo.
# ----------------------------------------------------------------------
def derivadas(estado):
    """estado = [x, v]  ->  [dx/dt, dv/dt]."""
    x, v = estado
    return np.array([v, aceleracion(x)])


def rk4(x0, v0, dt, t_final):
    n = int(t_final / dt)
    t = np.linspace(0.0, n * dt, n + 1)
    x = np.empty(n + 1)
    v = np.empty(n + 1)
    x[0], v[0] = x0, v0

    s = np.array([x0, v0])
    for i in range(n):
        k1 = derivadas(s)
        k2 = derivadas(s + 0.5 * dt * k1)
        k3 = derivadas(s + 0.5 * dt * k2)
        k4 = derivadas(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i + 1], v[i + 1] = s
    return t, x, v


# ----------------------------------------------------------------------
# Solucion analitica exacta
# ----------------------------------------------------------------------
def exacta(t):
    return X0 * np.cos(OMEGA * t) + (V0 / OMEGA) * np.sin(OMEGA * t)


# ----------------------------------------------------------------------
# Ejecutamos las simulaciones y graficamos
# ----------------------------------------------------------------------
def main():
    t_v, x_v, v_v = verlet(X0, V0, DT, T_FINAL)
    t_r, x_r, v_r = rk4(X0, V0, DT, T_FINAL)

    E_v = energia(x_v, v_v)
    E_r = energia(x_r, v_r)
    E0 = energia(X0, V0)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Verlet (simplectico) vs RK4: conservacion de energia\n"
                 f"Oscilador armonico, dt = {DT}, omega = {OMEGA}",
                 fontsize=14, fontweight="bold")

    # (a) Energia vs tiempo -- el grafico clave
    ax = axs[0, 0]
    ax.plot(t_v, E_v, label="Verlet", color="tab:blue", lw=1.5)
    ax.plot(t_r, E_r, label="RK4", color="tab:red", lw=1.5)
    ax.axhline(E0, color="k", ls="--", lw=1, label="Energia exacta")
    ax.set_xlabel("tiempo t")
    ax.set_ylabel("Energia total E")
    ax.set_title("(a) Energia vs tiempo\nVerlet oscila acotado; RK4 deriva")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Error relativo de energia (escala log para apreciar la deriva)
    ax = axs[0, 1]
    ax.plot(t_v, np.abs(E_v - E0) / E0, label="Verlet", color="tab:blue", lw=1.5)
    ax.plot(t_r, np.abs(E_r - E0) / E0, label="RK4", color="tab:red", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("tiempo t")
    ax.set_ylabel("|E - E0| / E0")
    ax.set_title("(b) Error relativo de energia (escala log)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # (c) Posicion vs tiempo (ventana inicial: ambos siguen bien la solucion)
    ax = axs[1, 0]
    mask = t_v <= 20
    ax.plot(t_v[mask], exacta(t_v[mask]), "k--", lw=2, label="exacta")
    ax.plot(t_v[mask], x_v[mask], color="tab:blue", lw=1.2, label="Verlet")
    ax.plot(t_r[t_r <= 20], x_r[t_r <= 20], color="tab:red", lw=1.2, label="RK4")
    ax.set_xlabel("tiempo t")
    ax.set_ylabel("posicion x")
    ax.set_title("(c) Posicion (primeros periodos): ambos aciertan")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) Espacio de fases: Verlet = elipse cerrada; RK4 = espiral
    ax = axs[1, 1]
    ax.plot(x_v, v_v, color="tab:blue", lw=0.8, label="Verlet")
    ax.plot(x_r, v_r, color="tab:red", lw=0.8, label="RK4")
    ax.set_xlabel("posicion x")
    ax.set_ylabel("velocidad v")
    ax.set_title("(d) Espacio de fases\nVerlet cierra la orbita; RK4 espirala")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("energia_verlet_vs_rk4.png", dpi=140)
    print("Figura guardada en: energia_verlet_vs_rk4.png")

    # Resumen numerico en consola
    print("\n--- Resumen ---")
    print(f"Energia inicial E0            = {E0:.6f}")
    print(f"Verlet: deriva neta de E     = {E_v[-1] - E0:+.3e}")
    print(f"RK4   : deriva neta de E     = {E_r[-1] - E0:+.3e}")
    print(f"Verlet: error max |E-E0|/E0  = {np.max(np.abs(E_v - E0)) / E0:.3e}")
    print(f"RK4   : error max |E-E0|/E0  = {np.max(np.abs(E_r - E0)) / E0:.3e}")

    plt.show()


if __name__ == "__main__":
    main()
