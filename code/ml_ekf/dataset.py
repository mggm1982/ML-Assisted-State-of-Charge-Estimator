"""Load recording groups from the Panasonic data manifest."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .core import baseline_trace, one_hz, load_record

EXCLUDED_BAD = {"25C_Cycle_1", "10C_Cycle_2", "10C_LA92", "0C_Cycle_2"}


def resolve_root(raw):
    """Locate the dataset root while preserving the manifest-relative paths."""
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Panasonic folder not found: {p}. Supply --data with the extracted MAT directory."
        )
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "physics_backend/docs/data_manifest.json"
        ).read_text()
    )
    row = next(
        r for r in manifest if r["role"] == "train" and r["run"] not in EXCLUDED_BAD
    )
    if (p / row["path"]).is_file():
        return p
    for f in p.rglob(Path(row["path"]).name):
        root = f
        for _ in Path(row["path"]).parts:
            root = root.parent
        if (root / row["path"]).is_file():
            return root
    raise FileNotFoundError(
        f'Could not locate expected MAT file {row["path"]} under {p}. Check extraction.'
    )


def groups(root, role, init_soc=0.8, limit=None):
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "physics_backend/docs/data_manifest.json"
        ).read_text()
    )
    rows = [r for r in manifest if r["role"] == role and r["run"] not in EXCLUDED_BAD]
    if limit:
        rows = rows[:limit]
    for r in rows:
        if r.get("wallclock_discontinuity_count", 0):
            continue
        a = load_record(root, r)
        s, z = one_hz(a, 1.0 + a[:, 4] / 2.79826)
        x, yV, zEKF, _ = baseline_trace(s, init_soc)
        target = np.column_stack(
            [np.clip(yV / 0.025, -1, 1), np.clip((z - zEKF) / 0.04, -1, 1)]
        ).astype(np.float32)
        yield {
            "run": r["run"],
            "acquisition_group": r["acquisition_group"],
            "sensors": s,
            "reference": z,
            "X": x,
            "Y": target,
            "zEKF": zEKF,
        }


def train_validation(root):
    """Load the development recordings and check acquisition-group separation."""
    train = list(groups(root, "train"))
    validation = list(groups(root, "validation"))
    if len(train) < 3 or len(validation) < 2:
        raise ValueError(
            f"Insufficient usable independent recordings: train={len(train)} validation={len(validation)}"
        )
    a = {g["acquisition_group"] for g in train}
    b = {g["acquisition_group"] for g in validation}
    if a & b:
        raise ValueError("Acquisition-group leakage between train and validation")
    return train, validation
