"""Evaluate the fixed model on the ten previously examined benchmark recordings."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .dataset import resolve_root, groups
from .core import metrics
from .runner import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--output", default="results/benchmark_diagnostic.csv")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    selection = (
        json.loads((root / "artifacts/selected.json").read_text())
        if (root / "artifacts/selected.json").exists()
        else {}
    )
    if a.model is None:
        if not selection:
            raise RuntimeError(
                "Train a ML_EKF model before evaluating, or use --model FILE.pt"
            )
        a.model = str(root / "artifacts" / (selection["kind"] + ".pt"))
    av = selection.get("alpha_voltage", 0)
    bz = selection.get("beta_soc", 0)
    data_root = resolve_root(a.data)
    results = []
    for role in ("exploratory_25C", "prospective_10C"):
        for group in groups(data_root, role):
            o = run(
                group["sensors"], 0.8, artifact=a.model, alpha_voltage=av, beta_soc=bz
            )
            for m, c in [("EKF", o["soc_ekf"]), ("ML_EKF", o["soc_hybrid"])]:
                results.append(
                    {
                        "run": group["run"],
                        "role": role,
                        "method": m,
                        **metrics(group["reference"], c),
                    }
                )
            print("Scored", group["run"], flush=True)
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0])
        w.writeheader()
        w.writerows(results)
    print("Saved benchmark diagnostic to", dest)
    print(
        "CAUTION: These recordings were examined during development and are not an untouched test set."
    )


if __name__ == "__main__":
    main()
