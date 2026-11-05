"""Feature construction, resampling and the neural-corrected EKF."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PHYSICS_SOURCE = Path(__file__).resolve().parents[1] / "physics_backend" / "src"
if str(PHYSICS_SOURCE) not in sys.path:
    sys.path.insert(0, str(PHYSICS_SOURCE))
from data import read_sensor_csv, read_mat, load_record, validate_sensors
from physics import EKF as BaseEKF

N_FEATURES = 15
FEATURE_NAMES = [
    "current_A",
    "temp_C",
    "prev_voltage_V",
    "prev_current_A",
    "prev_temp_C",
    "ekf_soc_pre",
    "ekf_polarization_V",
    "prev_voltage_innovation_V",
    "ema_current_fast",
    "ema_current_slow",
    "ema_voltage_fast",
    "ema_voltage_slow",
    "delta_current_A",
    "sigma_soc_pre",
    "elapsed_s",
]


class CausalFeatures:
    """Maintain the current and past measurements used by the residual network."""

    def __init__(self):
        self.k = 0
        self.last_v = 3.8
        self.last_i = 0.0
        self.last_T = 25.0
        self.last_nu = 0.0
        self.i_f = 0.0
        self.i_s = 0.0
        self.v_f = 3.8
        self.v_s = 3.8

    def features(self, t, i, T, ekf):
        return np.array(
            [
                i,
                T,
                self.last_v,
                self.last_i,
                self.last_T,
                ekf.z,
                ekf.vp,
                self.last_nu,
                self.i_f,
                self.i_s,
                self.v_f,
                self.v_s,
                i - self.last_i,
                np.sqrt(max(ekf.p00, 0)),
                min(t, 10000.0),
            ],
            dtype=np.float32,
        )

    def update(self, t, v, i, T, nu):
        if self.k == 0:
            self.v_f = self.v_s = v
            self.i_f = self.i_s = i
        else:
            # Approximate causal one-second low-pass features (no future samples).
            self.i_f = 0.18 * i + 0.82 * self.i_f
            self.i_s = 0.015 * i + 0.985 * self.i_s
            self.v_f = 0.18 * v + 0.82 * self.v_f
            self.v_s = 0.015 * v + 0.985 * self.v_s
        self.last_v = float(v)
        self.last_i = float(i)
        self.last_T = float(T)
        self.last_nu = float(nu)
        self.k += 1


class CorrectedEKF(BaseEKF):
    """EKF measurement update with a bounded neural voltage offset.

    The voltage offset is decided from causal features *before* observing voltage_k.
    The SOC residual is applied separately by runner.run.
    """

    def step(self, t, v, i, T, offset_V=0.0):
        import math

        if not all(math.isfinite(x) for x in (t, v, i, T, offset_V)):
            raise ValueError("Nonfinite measurement")
        if abs(offset_V) > 0.03000001:
            raise ValueError("Offset over 30mV is disallowed")
        dt = 0.0 if self.t is None else t - self.t
        if self.t is not None and (dt <= 0 or dt > 5.0):
            raise ValueError("Bad interval")
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
        pred = ocv - r0 * i - vpp + offset_V
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


def one_hz(raw, reference=None):
    """Choose latest available sample before each integer second (no interpolation ahead)."""
    raw = np.asarray(raw, float)
    from data import validate_sensors

    validate_sensors(raw[:, :4])
    times = np.arange(0, np.floor(raw[-1, 0]) + 1)
    ids = np.searchsorted(raw[:, 0], times, side="right") - 1
    ids = np.maximum(ids, 0)
    # Use actual original timestamps, not synthetic 1s timestamps.
    ids = np.unique(ids)
    s = raw[ids, :4].copy()
    s[:, 0] -= s[0, 0]
    validate_sensors(s)
    if reference is not None:
        return s, np.asarray(reference)[ids]
    return s


def frozen_physics():
    """Load the selected circuit calibration and EKF configuration."""
    import json

    file = (
        Path(__file__).resolve().parents[1]
        / "physics_backend/models/physics_selection.json"
    )
    d = json.loads(file.read_text())
    return d["selected_calibration"], d["selected"]


def baseline_trace(sensors, init_soc):
    """Replay one recording to obtain causal features and training targets."""
    battery, config = frozen_physics()
    ekf = CorrectedEKF(battery, config, init_soc)
    mem = CausalFeatures()
    xs = []
    yV = []
    z = []
    pred = []
    for t, v, i, T in sensors:
        x = mem.features(t, i, T, ekf)
        o = ekf.step(float(t), float(v), float(i), float(T))
        xs.append(x)
        yV.append(o[1])
        z.append(o[0])
        pred.append(o[2])
        mem.update(t, v, i, T, o[1])
    return (
        np.asarray(xs, np.float32),
        np.asarray(yV, np.float32),
        np.asarray(z),
        np.asarray(pred),
    )


def metrics(reference, estimate):
    """Return RMSE, MAE and maximum absolute error in SOC percentage points."""
    e = (np.asarray(estimate) - np.asarray(reference)) * 100.0
    return {
        "RMSE_pp": round(float(np.sqrt(np.mean(e**2))), 6),
        "MAE_pp": round(float(np.mean(np.abs(e))), 6),
        "max_abs_pp": round(float(np.max(np.abs(e))), 6),
    }
