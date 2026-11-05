"""Read original MAT measurements without executing the supplied MATLAB code."""

from __future__ import annotations
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat

Q_REF = 2.79826
SENSOR_NAMES = ("time_s", "voltage_V", "current_discharge_A", "temperature_C")


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read_mat(path):
    m = loadmat(path, simplify_cells=True)["meas"]
    a = np.column_stack(
        [
            np.asarray(m[k], float).reshape(-1)
            for k in ["Time", "Voltage", "Current", "Battery_Temp_degC", "Ah"]
        ]
    )
    a[:, 2] *= -1
    return a, m


def clean(a, max_gap=None):
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite measurements: inspect source")
    if np.any(np.diff(a[:, 0]) < 0):
        raise ValueError("Reversed time: split/review source")
    # Last observation at a duplicated instant is available at that instant.
    keep = np.r_[np.diff(a[:, 0]) != 0, True]
    a = a[keep].copy()
    a[:, 0] -= a[0, 0]
    if max_gap is not None and np.max(np.diff(a[:, 0]), initial=0) > max_gap:
        raise ValueError("Gap exceeds allowed contiguous-drive limit")
    return a


def reference(a, q=Q_REF, z0=1.0):
    """Column 4 is counter change from ORIGINAL full-record start, even after slicing."""
    return z0 + a[:, 4] / q


def read_sensor_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        reader = csv.DictReader(f)
        if not set(SENSOR_NAMES).issubset(reader.fieldnames or []):
            raise ValueError("Require " + ",".join(SENSOR_NAMES))
        a = np.array([[float(r[k]) for k in SENSOR_NAMES] for r in reader])
    validate_sensors(a)
    return a


def validate_sensors(a):
    if a.ndim != 2 or a.shape[1] != 4 or len(a) < 1 or not np.isfinite(a).all():
        raise ValueError("Expected finite Nx4 sensor measurements")
    if abs(a[0, 0]) > 1e-7 or (np.diff(a[:, 0]) <= 0).any():
        raise ValueError(
            "Start at zero; strictly increasing timestamps; reset per recording"
        )
    if np.max(np.diff(a[:, 0]), initial=0) > 5:
        raise ValueError("Unlogged interval >5 s: do not integrate unknown current")


def recording_info(path, root):
    rel = path.relative_to(root).as_posix()
    nominal = next(
        (
            s
            for s in path.parts
            if re.fullmatch(r"-?\d+degC(?: Trise(?: with pause)?)?", s)
        ),
        "",
    )
    match = re.search(r"(?:n?\d+degC)_(.*?)_Pan18650PF", path.name)
    name = match.group(1) if match else path.stem[15:]
    drive = bool(match and "pulse" not in name.lower())
    combined = drive and (
        (sum(k in name for k in ["US06", "HWFET", "UDDS", "LA92", "NN"]) > 1)
        or "to4" in name
    )
    temp = int(re.search(r"-?\d+", nominal).group()) if nominal else None
    rid = (nominal.replace("degC", "C").replace(" ", "_") + "_" + name).replace(
        "-", "m"
    )
    if drive and not combined and "Trise" not in nominal:
        if name in ["Cycle_1", "Cycle_2", "Cycle_3"] and temp in [25, 10]:
            split = "train"
        elif name == "Cycle_4" and temp in [25, 10]:
            split = "validation"
        elif temp == 25:
            split = "exploratory_25C"
        elif temp == 10:
            split = "prospective_10C"
        else:
            split = "external_cold_diagnostic"
    elif drive and "Trise" in nominal:
        split = "excluded_late_thermal_ageing"
    elif combined:
        split = "excluded_overlap"
    else:
        split = "characterization_or_charge"
    # All individually exported standard profiles from a combined source share a group.
    group = (
        (nominal + "_standard_profiles")
        if drive and name not in ["Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4"]
        else rid
    )
    return dict(
        run=rid,
        path=rel,
        nominal_C=temp,
        profile=name,
        role=split,
        acquisition_group=group,
        drive=drive,
        combined=combined,
    )


def load_record(root, row):
    a, _ = read_mat(Path(root) / row["path"])
    a = clean(a)
    a[:, 4] -= a[0, 4]
    # A few split exports retain the preceding, low-rate zero-current pause.
    # Drop that prefix only; do not bridge a long unlogged interval under load.
    gaps = np.flatnonzero(np.diff(a[:, 0]) > 5)
    if len(gaps):
        stop = gaps[-1] + 1
        if np.max(abs(a[: stop + 1, 2])) > 0.05 or np.ptp(a[: stop + 1, 4]) > 0.0002:
            raise ValueError("Long gap is not a verified zero-load prefix")
        a = a[stop:].copy()
        a[:, 0] -= a[0, 0]
    validate_sensors(a[:, :4])
    return a
