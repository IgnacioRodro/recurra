"""Reference chaotic systems with known invariants.

These exist mainly as ground truth for tests: Lorenz has D2 ~ 2.06 and
lambda_1 ~ 0.906, which is how the dynamics module gets validated.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..config import SeedLike, resolve_rng
from ..exceptions import ParameterError


def _integrate(rhs, n_points, dt, y0, params, *, rtol=1e-9, atol=1e-11,
               transient=0.0):
    """Integrate with tight tolerances and verify success.

    The old draft used solve_ivp defaults (rtol=1e-3), which for a chaotic
    system means the trajectory is wrong well before the end of the record.
    """
    t_end = dt * (n_points - 1) + transient
    t_eval = np.linspace(transient, t_end, n_points)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, **params), (0.0, t_end), y0,
        t_eval=t_eval, rtol=rtol, atol=atol, method="RK45", dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    if sol.y.shape[1] != n_points:
        raise RuntimeError(
            f"integrator returned {sol.y.shape[1]} points, expected {n_points}"
        )
    return sol.y.T


def lorenz(n_points=10000, dt=0.01, sigma=10.0, rho=28.0, beta=8 / 3,
           y0=(1.0, 1.0, 1.0), transient=10.0, **kw):
    def rhs(t, s, sigma, rho, beta):
        x, y, z = s
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    return _integrate(rhs, n_points, dt, list(y0),
                      dict(sigma=sigma, rho=rho, beta=beta), transient=transient, **kw)


def rossler(n_points=10000, dt=0.05, a=0.2, b=0.2, c=5.7,
            y0=(1.0, 1.0, 1.0), transient=50.0, **kw):
    def rhs(t, s, a, b, c):
        x, y, z = s
        return [-y - z, x + a * y, b + z * (x - c)]

    return _integrate(rhs, n_points, dt, list(y0), dict(a=a, b=b, c=c),
                      transient=transient, **kw)


def van_der_pol(n_points=10000, dt=0.05, mu=2.0, y0=(1.0, 0.0), transient=20.0, **kw):
    def rhs(t, s, mu):
        x, y = s
        return [y, mu * (1 - x**2) * y - x]

    return _integrate(rhs, n_points, dt, list(y0), dict(mu=mu),
                      transient=transient, **kw)


def henon(n_points=10000, a=1.4, b=0.3, y0=(0.1, 0.3), transient=1000):
    x, y = y0
    out = np.empty((n_points, 2))
    for i in range(n_points + transient):
        x, y = 1 - a * x * x + y, b * x
        if i >= transient:
            out[i - transient] = (x, y)
    return out


def mackey_glass(n_points=10000, tau=17, beta=0.2, gamma=0.1, n=10,
                 dt=1.0, transient=1000, rng: SeedLike = None):
    rng = resolve_rng(rng)
    total = n_points + transient
    hist = int(tau / dt)
    x = np.empty(total + hist)
    x[:hist] = 1.2 + 0.01 * rng.standard_normal(hist)
    for i in range(hist, total + hist):
        xt = x[i - hist]
        x[i] = x[i - 1] + dt * (beta * xt / (1 + xt**n) - gamma * x[i - 1])
    return x[hist + transient:][:n_points].reshape(-1, 1)


SYSTEMS = {
    "lorenz": lorenz, "rossler": rossler, "van_der_pol": van_der_pol,
    "henon": henon, "mackey_glass": mackey_glass,
}


def generate_system(name: str, **kw) -> np.ndarray:
    if name not in SYSTEMS:
        raise ParameterError(f"unknown system {name!r}; have {sorted(SYSTEMS)}")
    return SYSTEMS[name](**kw)
