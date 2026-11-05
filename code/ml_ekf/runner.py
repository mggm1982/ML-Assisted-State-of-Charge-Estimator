"""Causal online replay for physics EKF and ML-modified observer."""

from __future__ import annotations
from collections import deque
import numpy as np
from .core import CausalFeatures, CorrectedEKF, frozen_physics


def run(
    sensors,
    initial_soc,
    artifact=None,
    alpha_voltage=0.0,
    beta_soc=0.0,
    warmup_s=60.0,
    temperature_range=(9.0, 31.0),
):
    # Without a trained network, both observers use the same physics model.
    """Estimate SOC in time order using a baseline EKF and an optional neural correction.

    Sensors contain time in seconds, voltage, discharge-positive current and
    temperature. The network uses a 64-step causal feature history. SOC output
    corrections are bounded and do not update the internal EKF state.
    """
    import torch

    model = scaler = None
    if artifact is not None:
        from .networks import load_artifact

        model, scaler, cfg = load_artifact(artifact)
        if alpha_voltage is None:
            alpha_voltage = float(cfg.get("alpha_voltage", 0.0))
        if beta_soc is None:
            beta_soc = float(cfg.get("beta_soc", 0.0))
        history = deque(maxlen=64)
    battery, config = frozen_physics()
    physics = CorrectedEKF(battery, config, initial_soc)
    hybrid = CorrectedEKF(battery, config, initial_soc)
    mem = CausalFeatures()
    d = {
        k: []
        for k in [
            "time_s",
            "soc_ekf",
            "soc_voltage_ml",
            "soc_hybrid",
            "delta_voltage_V",
            "delta_soc",
            "voltage_V",
            "current_discharge_A",
            "temperature_C",
        ]
    }
    for t, v, i, T in sensors:
        t = float(t)
        v = float(v)
        i = float(i)
        T = float(T)
        f = mem.features(t, i, T, hybrid)
        corr_v = corr_z = 0.0
        if model is not None:
            history.append(scaler.transform(f))
            if t >= warmup_s and temperature_range[0] <= T <= temperature_range[1]:
                h = np.asarray(list(history), np.float32)
                if len(h) < 64:
                    h = np.concatenate(
                        (np.repeat(h[:1], 64 - len(h), axis=0), h), axis=0
                    )
                with torch.inference_mode():
                    raw = model(torch.from_numpy(h[None, :, :])).numpy()[0]
                # Independent bounds on each learned quantity; settings chosen on validation only.
                corr_v = float(np.clip(alpha_voltage * raw[0] * 0.025, -0.025, 0.025))
                corr_z = float(np.clip(beta_soc * raw[1] * 0.04, -0.03, 0.03))
        base = physics.step(t, v, i, T)
        mixed = hybrid.step(t, v, i, T, corr_v)
        # Never feed corrected output into the internal EKF state.
        final_z = float(np.clip(mixed[0] + corr_z, 0.0, 1.0))
        mem.update(t, v, i, T, mixed[1])
        for key, value in [
            ("time_s", t),
            ("soc_ekf", base[0]),
            ("soc_voltage_ml", mixed[0]),
            ("soc_hybrid", final_z),
            ("delta_voltage_V", corr_v),
            ("delta_soc", corr_z),
            ("voltage_V", v),
            ("current_discharge_A", i),
            ("temperature_C", T),
        ]:
            d[key].append(value)
    return {k: np.array(v) for k, v in d.items()}
