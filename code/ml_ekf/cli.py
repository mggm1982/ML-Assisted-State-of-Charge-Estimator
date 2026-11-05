"""Predict SOC from a measurement CSV containing time_s, voltage_V, current_discharge_A, temperature_C."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .core import read_sensor_csv, one_hz
from .runner import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--initial-soc", type=float, required=True)
    p.add_argument("--output", default="results/my_predictions_ml_ekf.csv")
    p.add_argument(
        "--model",
        default=None,
        help="Path to trained .pt file; omit for physics-only smoke run",
    )
    p.add_argument("--alpha-voltage", type=float, default=None)
    p.add_argument("--beta-soc", type=float, default=None)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if not 0 <= a.initial_soc <= 1:
        p.error("Initial SOC must be between 0 and 1")
    s = one_hz(read_sensor_csv(a.input))
    av = a.alpha_voltage
    bz = a.beta_soc
    if a.model:
        if av is None or bz is None:
            from .networks import load_artifact

            _, _, cfg = load_artifact(a.model)
            if av is None:
                av = float(cfg.get("alpha_voltage", 0))
            if bz is None:
                bz = float(cfg.get("beta_soc", 0))
    else:
        av = bz = 0.0
        print("No network supplied. Running the calibrated EKF baseline.")
    if av < 0 or av > 1 or bz < 0 or bz > 1:
        p.error("Correction strengths must be in [0,1]")
    result = run(s, a.initial_soc, artifact=a.model, alpha_voltage=av, beta_soc=bz)
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(result.keys())
        w.writerows(zip(*result.values()))
    print(f"Saved {len(s)} one-second prediction ticks to {dest}")
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            result["time_s"] / 60,
            result["soc_ekf"] * 100,
            label="Physics EKF",
            linewidth=1.3,
        )
        if a.model and (av or bz):
            ax.plot(
                result["time_s"] / 60,
                result["soc_hybrid"] * 100,
                label="ML + EKF",
                linewidth=1.1,
            )
        ax.set(
            xlabel="Time (min)",
            ylabel="SOC (%)",
            title="Sensor-only SOC prediction (no ground-truth SOC in example)",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        f = dest.with_suffix(".png")
        fig.savefig(f, dpi=140)
        plt.close(fig)
        print("Saved plot:", f)


if __name__ == "__main__":
    main()
