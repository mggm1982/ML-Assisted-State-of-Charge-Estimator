"""Train compact voltage/SOC correction heads and select strengths on held-out Cycle 4 recordings."""

from __future__ import annotations
import argparse
import csv
import json
import random
import zipfile
from pathlib import Path
import numpy as np


def evaluate_one(data, model_path, av, bz):
    from .runner import run
    from .core import metrics

    o = run(data["sensors"], 0.8, artifact=model_path, alpha_voltage=av, beta_soc=bz)
    return {
        "run": data["run"],
        "EKF": metrics(data["reference"], o["soc_ekf"])["RMSE_pp"],
        "ML_EKF": metrics(data["reference"], o["soc_hybrid"])["RMSE_pp"],
        "max_abs": metrics(data["reference"], o["soc_hybrid"])["max_abs_pp"],
    }


def checkpoint(root, name):
    """Save model checkpoints and training records without copying raw measurements."""
    root = Path(root)
    dest = root / "artifacts" / (name + ".zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for section in (
            "ml_ekf",
            "artifacts",
            "results",
            "README.md",
            "PROGRESS.md",
            "requirements.txt",
        ):
            entry = root / section
            if entry.is_file():
                z.write(entry, entry.relative_to(root))
            elif entry.is_dir():
                for p in entry.rglob("*"):
                    if (
                        p.is_file()
                        and p != dest
                        and p.suffix != ".zip"
                        and "__pycache__" not in p.parts
                    ):
                        z.write(p, p.relative_to(root))
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:
            raise IOError("Checkpoint ZIP verification failed")
    print("Saved verified checkpoint:", dest)


def run_train(args):
    import torch
    from torch.utils.data import DataLoader
    from .dataset import resolve_root, train_validation
    from .networks import network, FeatureScaler, WindowDataset

    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts"
    out.mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    print("Locate and audit original MAT recordings...")
    data_root = resolve_root(args.data)
    train, val = train_validation(data_root)
    (root / "PROGRESS.md").write_text(
        "# ML_EKF progress\n\nTraining recordings loaded.\n\n"
    )
    scaler = FeatureScaler().fit(train)
    ds = WindowDataset(train, scaler, window=64, stride=args.stride, burn=60)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    dsv = WindowDataset(val, scaler, window=64, stride=args.stride, burn=60)
    dv = DataLoader(dsv, batch_size=args.batch, shuffle=False, num_workers=0)
    all_results = []
    model_paths = {}
    for kind in args.models:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        model = network(kind, train[0]["X"].shape[1], torch)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_fn = torch.nn.SmoothL1Loss(beta=0.2)
        best = float("inf")
        best_state = None
        stale = 0
        logs = []
        for epoch in range(args.epochs):
            model.train()
            train_loss = []
            for xb, yb in dl:
                optim.zero_grad()
                p = model(xb)
                loss = loss_fn(p, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                train_loss.append(float(loss))
            model.eval()
            valid = []
            with torch.no_grad():
                for xb, yb in dv:
                    valid.append(float(loss_fn(model(xb), yb)))
            loss_val = float(np.mean(valid))
            logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(train_loss)),
                    "validation_loss": loss_val,
                }
            )
            print(
                kind,
                "epoch",
                epoch + 1,
                "validation_loss",
                round(loss_val, 6),
                flush=True,
            )
            if loss_val < best - 1e-5:
                best = loss_val
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        model_path = out / (kind + ".pt")
        torch.save(model.state_dict(), model_path)
        cfg = dict(
            kind=kind,
            features=train[0]["X"].shape[1],
            scaler=scaler.to_dict(),
            target_scales=[0.025, 0.04],
            data_split="Recording split: 4 training, 2 validation",
            reference_capacity_Ah=2.79826,
            seed=args.seed,
            init_soc=0.8,
            alpha_voltage=0.0,
            beta_soc=0.0,
            note="Model trained; correction strengths selected subsequently",
        )
        model_path.with_suffix(".json").write_text(json.dumps(cfg, indent=2) + "\n")
        with (root / "results" / (kind + "_learning_curve.csv")).open(
            "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=logs[0])
            w.writeheader()
            w.writerows(logs)
        model_paths[kind] = model_path
        checkpoint(root, "ML_EKF_after_training_" + kind)
    print(
        "Evaluate candidate strengths on independent development-validation recordings..."
    )
    # Selection may be expensive because each combination is a full sequential EKF replay.
    for kind, model_path in model_paths.items():
        for av, bz in [
            (0, 0),
            (0.25, 0),
            (0.5, 0),
            (1, 0),
            (0, 0.25),
            (0, 0.5),
            (0.25, 0.25),
            (0.5, 0.25),
            (0.25, 0.5),
            (0.5, 0.5),
        ]:
            rows = [evaluate_one(d, model_path, av, bz) for d in val]
            avg = float(np.mean([r["ML_EKF"] for r in rows]))
            base = float(np.mean([r["EKF"] for r in rows]))
            harm = max(0.0, max(r["ML_EKF"] - r["EKF"] for r in rows))
            rec = {
                "kind": kind,
                "alpha_voltage": av,
                "beta_soc": bz,
                "mean_RMSE_pp": avg,
                "mean_EKF_RMSE_pp": base,
                "worst_case_harm_pp": harm,
                "selection_score": avg + 0.35 * harm,
            }
            all_results.append(rec)
            print("validation:", rec, flush=True)
        checkpoint(root, "ML_EKF_after_selection_scan_" + kind)
    best = min(all_results, key=lambda x: x["selection_score"])

    with (root / "results" / "development_selection.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0]))
        w.writeheader()
        w.writerows(all_results)
    (out / "selected.json").write_text(json.dumps(best, indent=2) + "\n")
    p = model_paths[best["kind"]]
    cfg = json.loads(p.with_suffix(".json").read_text())
    cfg.update(alpha_voltage=best["alpha_voltage"], beta_soc=best["beta_soc"])
    p.with_suffix(".json").write_text(json.dumps(cfg, indent=2) + "\n")
    (root / "PROGRESS.md").write_text(
        "# ML_EKF progress\n\nTraining and development validation completed.\n\n"
        + "Selection: `artifacts/selected.json`.\n\n"
        + json.dumps(best, indent=2)
        + "\n\n"
        + "Training and validation are complete. Benchmark evaluation is a separate step.\n"
    )
    checkpoint(root, "ML_EKF_selected_development_model")
    print("TRAINING FINISHED. Selected:", best)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", required=True, help="Extracted Panasonic 18650PF MAT data root"
    )
    p.add_argument("--models", nargs="+", choices=["fnn"], default=["fnn"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=351)
    run_train(p.parse_args())


if __name__ == "__main__":
    main()
