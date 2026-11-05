"""Scalar, native-rate one-RC EKF with temperature tables and optional adaptive R."""

import bisect
import math
import numpy as np
from data import validate_sensors


def lookup(z, x, y):
    """Linearly interpolate a parameter table and return its local derivative."""
    if z <= x[0]:
        return y[0], 0.0
    if z >= x[-1]:
        return y[-1], 0.0
    j = bisect.bisect_right(x, z) - 1
    s = (y[j + 1] - y[j]) / (x[j + 1] - x[j])
    return y[j] + s * (z - x[j]), s


class EKF:
    """Two-state SOC and polarization observer using the calibrated one-RC circuit."""

    def __init__(self, battery, config, initial_soc=1.0):
        if not 0 <= initial_soc <= 1:
            raise ValueError("initial SOC must be a fraction in [0,1]")
        self.b = battery
        self.c = config
        self.z = float(initial_soc)
        self.vp = 0.0
        self.p00 = 0.2**2
        self.p01 = 0.0
        self.p11 = 0.05**2
        self.t = None
        self.i = None
        self.rv = config["voltage_std_V"] ** 2
        self.P_min_eigenvalue = float("inf")

    def values(self, z, T):
        tables = self.b["tables"]
        temps = self.b["temperatures_C"]
        if len(tables) == 1:
            weights = [1.0]
        else:
            w = min(1.0, max(0.0, (T - temps[0]) / (temps[1] - temps[0])))
            weights = [1 - w, w]
        values = [0.0] * 8
        for w, b in zip(weights, tables):
            if w == 0:
                continue
            for k, (xn, yn) in enumerate(
                [
                    ("ocv_soc_knots", "ocv_voltage_knots"),
                    ("parameter_soc_knots", "R0_ohm"),
                    ("parameter_soc_knots", "R1_ohm"),
                    ("parameter_soc_knots", "tau_s"),
                ]
            ):
                val, der = lookup(z, b[xn], b[yn])
                values[2 * k] += w * val
                values[2 * k + 1] += w * der
        scale = self.c.get("resistance_scale", 1.0)
        values[2] *= scale
        values[3] *= scale
        values[4] *= scale
        values[5] *= scale
        return values

    def step(self, t, v, i, T):
        if not all(math.isfinite(x) for x in [t, v, i, T]):
            raise ValueError("Nonfinite sensor")
        dt = 0.0 if self.t is None else t - self.t
        if self.t is not None and (dt <= 0 or dt > 5.0):
            raise ValueError("Bad interval; reset at recording boundary")
        im = i if self.i is None else 0.5 * (i + self.i)
        _, _, _, _, r1, d1, tau, dtau = self.values(self.z, T)
        a = math.exp(-dt / tau)
        f10 = a * dt * dtau / (tau * tau) * (self.vp - r1 * im) + (1 - a) * d1 * im
        q = self.b["capacity_Ah"] * self.c.get("capacity_scale", 1.0)
        zp = min(1.0, max(0.0, self.z - im * dt / (3600 * q)))
        vpp = a * self.vp + (1 - a) * r1 * im
        p00 = self.p00 + self.c["soc_process_std_per_sqrt_s"] ** 2 * dt
        p01 = f10 * self.p00 + a * self.p01
        p11 = (
            f10 * f10 * self.p00
            + 2 * f10 * a * self.p01
            + a * a * self.p11
            + self.c.get("vp_process_std_per_sqrt_s", 0.001) ** 2 * dt
        )
        ocv, docv, r0, d0, _, _, _, _ = self.values(zp, T)
        pred = ocv - r0 * i - vpp
        nu = v - pred
        h = docv - i * d0
        projected = h * h * p00 - 2 * h * p01 + p11
        rv = self.rv
        s = max(projected + rv, 1e-12)
        k0 = (h * p00 - p01) / s
        k1 = (h * p01 - p11) / s
        self.z = min(1.0, max(0.0, zp + k0 * nu))
        self.vp = vpp + k1 * nu
        m00 = 1 - k0 * h
        m01 = k0
        m10 = -k1 * h
        m11 = 1 + k1
        self.p00 = max(
            1e-14,
            m00 * m00 * p00 + 2 * m00 * m01 * p01 + m01 * m01 * p11 + k0 * k0 * rv,
        )
        self.p01 = (
            m00 * m10 * p00
            + (m00 * m11 + m01 * m10) * p01
            + m01 * m11 * p11
            + k0 * k1 * rv
        )
        self.p11 = max(
            1e-14,
            m10 * m10 * p00 + 2 * m10 * m11 * p01 + m11 * m11 * p11 + k1 * k1 * rv,
        )
        if self.c.get("adaptive_R", False):
            base = self.c["voltage_std_V"] ** 2
            alpha = 1 - math.exp(-dt / 60.0)
            target = min(4 * base, max(base / 4, nu * nu - projected))
            self.rv = min(4 * base, max(base / 4, (1 - alpha) * rv + alpha * target))
        self.t = t
        self.i = i
        return self.z, nu, pred, self.vp, math.sqrt(self.p00), k0, h, nu * nu / s, rv


def replay(sensors, battery, config, initial_soc):
    """Apply the EKF to a recording and return its state and diagnostic outputs."""
    validate_sensors(sensors)
    f = EKF(battery, config, initial_soc)
    out = np.empty((len(sensors), 9))
    for k, (t, v, i, T) in enumerate(sensors):
        out[k] = f.step(float(t), float(v), float(i), float(T))
    return out
