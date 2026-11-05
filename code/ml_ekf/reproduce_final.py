"""Replay the trained ML + EKF model on the Panasonic benchmark recordings.

Inputs are raw measurements, trained network weights and the EKF calibration.
The Ah-derived SOC reference is used to score predictions after inference.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import datetime
import platform
import zipfile
from pathlib import Path
import numpy as np
from .core import one_hz, load_record, metrics
from .runner import run
from .dataset import resolve_root

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("exploratory_25C", "prospective_10C")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_all(rows, traces, figdir, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["agg.path.chunksize"] = 10000
    from matplotlib.ticker import MultipleLocator

    figdir.mkdir(parents=True, exist_ok=True)
    print(
        "Rendering summary plots and full-recording time-series (up to 4000 plotted points each)...",
        flush=True,
    )
    good = "#13865a"
    bad = "#bb414c"
    base = "#364750"
    ml = "#007f90"
    names = [r["run"] for r in rows]
    yy = np.arange(len(rows))
    av = np.array([r["EKF_RMSE_pp"] for r in rows])
    bv = np.array([r["ML_EKF_RMSE_pp"] for r in rows])
    wins = [r["improved"] for r in rows]

    fig, ax = plt.subplots(figsize=(11.8, 6.5))
    h = 0.36
    ax.barh(yy - h / 2, av, h, label="Calibrated physics-only EKF", color=base)
    ax.barh(
        yy + h / 2, bv, h, label="ML + EKF", color=[good if w else bad for w in wins]
    )
    for i, r in enumerate(rows):
        ax.text(
            max(av[i], bv[i]) + 0.025,
            i,
            ("IMPROVED " if r["improved"] else "WORSE ")
            + f'{r["relative_gain_pct"]:+.1f}%',
            va="center",
            fontsize=9,
            color=good if r["improved"] else bad,
            fontweight="bold",
        )
    ax.set_yticks(yy, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(av), max(bv)) * 1.38)
    ax.set_xlabel("SOC RMSE (percentage points); lower is better")
    ax.set_title(
        f"ML + EKF benchmark: {sum(wins)}/{len(wins)} recordings improved",
        fontsize=15,
        pad=15,
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figdir / "01_EKF_vs_ML_EKF_RMSE.png", dpi=165)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 6))
    order = np.argsort([-r["relative_gain_pct"] for r in rows])
    rr = [rows[i] for i in order]
    xx = np.arange(len(rr))
    benefit = [r["relative_gain_pct"] for r in rr]
    bar = ax.bar(
        xx, benefit, color=[good if x > 0 else bad for x in benefit], width=0.72
    )
    for b, r in zip(bar, rr):
        z = b.get_height()
        ax.annotate(
            f"{z:+.1f}%",
            (b.get_x() + b.get_width() / 2, z),
            textcoords="offset points",
            xytext=(0, 5 if z >= 0 else -14),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=good if z > 0 else bad,
        )
    ax.axhline(0, color="#454545", linewidth=0.9)
    ax.set_xticks(xx, [r["run"].replace("_", "\n") for r in rr], fontsize=9)
    ax.set_ylabel("RMSE reduction relative to EKF (%)")
    ax.set_title("RMSE reduction by recording", fontsize=13)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figdir / "02_ML_EKF_Improvement_By_Driving_Cycle.png", dpi=165)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    temps = ["25C", "10C"]
    by = {t: [r for r in rows if r["run"].startswith(t)] for t in temps}
    x = np.arange(2)
    width = 0.32
    a = [np.mean([r["EKF_RMSE_pp"] for r in by[t]]) for t in temps]
    b = [np.mean([r["ML_EKF_RMSE_pp"] for r in by[t]]) for t in temps]
    ra = ax.bar(x - width / 2, a, width, color=base, label="EKF")
    rb = ax.bar(x + width / 2, b, width, color=good, label="ML + EKF")
    for bars in [ra, rb]:
        for r in bars:
            ax.text(
                r.get_x() + r.get_width() / 2,
                r.get_height() + 0.016,
                f"{r.get_height():.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_xticks(x, ["25 C (6 recordings)", "10 C (4 recordings)"])
    ax.set_ylabel("Mean per-recording SOC RMSE (pp)")
    ax.set_ylim(0, max(a + b) * 1.22)
    ax.set_title("Mean RMSE by temperature")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figdir / "03_ML_EKF_Temperature_Summary.png", dpi=165)
    plt.close(fig)

    featured = {"10C_HWFET", "10C_NN", "25C_US06", "10C_UDDS"}
    for r in rows:
        if r["run"] not in featured:
            continue
        name = r["run"]
        print("  Drawing featured time-series", name, flush=True)
        q_full = traces[name]
        # Plot a subset of samples for readability; metrics use the full one-hertz replay.
        stride = max(1, int(np.ceil(len(q_full["time_s"]) / 4000)))
        q = {k: v[::stride] for k, v in q_full.items()}
        time_minutes = q["time_s"] / 60.0
        fig, (a1, a2) = plt.subplots(
            2,
            1,
            figsize=(11.5, 6.9),
            sharex=True,
            gridspec_kw={"height_ratios": [1.28, 1]},
            layout="constrained",
        )
        a1.plot(
            time_minutes,
            100 * q["reference"],
            linestyle="--",
            color="#222222",
            linewidth=1.5,
            label="Derived reference SOC",
        )
        a1.plot(
            time_minutes,
            100 * q["soc_ekf"],
            color=base,
            linewidth=1.1,
            label="Physics EKF",
        )
        a1.plot(
            time_minutes,
            100 * q["soc_hybrid"],
            color=good if r["improved"] else bad,
            linewidth=1.15,
            label="ML + EKF",
        )
        a1.set_ylabel("SOC (%)")
        a1.legend(loc="best", fontsize=9)
        a1.grid(alpha=0.16)
        a2.axhline(0, color="#4f4f4f", linewidth=0.8)
        a2.plot(
            time_minutes,
            100 * (q["soc_ekf"] - q["reference"]),
            color=base,
            linewidth=1.1,
            label="EKF error",
        )
        a2.plot(
            time_minutes,
            100 * (q["soc_hybrid"] - q["reference"]),
            color=good if r["improved"] else bad,
            linewidth=1.1,
            label="ML + EKF error",
        )
        a2.set_ylabel("Estimate - reference (SOC pp)")
        a2.set_xlabel("Time from start (minutes)")
        a2.grid(alpha=0.16)
        a2.legend(loc="best", fontsize=9)
        status = "IMPROVED" if r["improved"] else "WORSE"
        fig.suptitle(
            f'{name}: {status} | EKF {r["EKF_RMSE_pp"]:.3f} pp to ML + EKF {r["ML_EKF_RMSE_pp"]:.3f} pp ({r["relative_gain_pct"]:+.1f}% gain)',
            fontsize=12,
            fontweight="bold",
        )
        fig.savefig(
            figdir / f'TRACE_{name}_{"SUCCESS" if r["improved"] else "FAILURE"}.png',
            dpi=135,
        )
        plt.close(fig)
    print(
        "Drew",
        3 + len(featured),
        "PNG plots from newly computed predictions",
        flush=True,
    )


def fresh_pdf(rows, figdir, out):
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
        KeepTogether,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallLocal",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            spaceAfter=9,
        )
    )
    story = [
        Paragraph("ML + EKF for Battery State of Charge Estimation", styles["Title"]),
        Paragraph("Evaluation report", styles["Heading2"]),
    ]
    story.append(
        Paragraph(
            "The trained feedforward network assists a calibrated extended Kalman filter. This report evaluates both estimators on ten Panasonic recordings using the same measurements and reference.",
            styles["SmallLocal"],
        )
    )
    ev = [r["EKF_RMSE_pp"] for r in rows]
    mv = [r["ML_EKF_RMSE_pp"] for r in rows]
    n = sum(r["improved"] for r in rows)
    story.append(
        Paragraph(
            f"<b>Results:</b> {n} of {len(rows)} recordings improved. Mean recording RMSE: EKF {np.mean(ev):.4f} pp versus ML + EKF {np.mean(mv):.4f} pp, relative reduction {(1-np.mean(mv)/np.mean(ev))*100:.1f}%.",
            styles["SmallLocal"],
        )
    )
    data = [["Recording", "EKF RMSE", "ML + EKF RMSE", "Gain vs EKF"]] + [
        [
            r["run"],
            f'{r["EKF_RMSE_pp"]:.3f} pp',
            f'{r["ML_EKF_RMSE_pp"]:.3f} pp',
            f'{r["relative_gain_pct"]:+.1f}%',
        ]
        for r in rows
    ]
    t = Table(data, colWidths=[144, 93, 93, 110], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20374b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f1f5f8")],
                ),
                ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#a1adba")),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 9))
    story.append(
        Paragraph(
            "The two UDDS cases become worse. Results come from one cell, previously examined recordings and an Ah-derived reference using assumed 2.79826 Ah; improvements are not independent cross-cell proof.",
            styles["SmallLocal"],
        )
    )
    for img, title in [
        ("01_EKF_vs_ML_EKF_RMSE.png", "Per-recording performance"),
        ("03_ML_EKF_Temperature_Summary.png", "Results by temperature"),
        (
            "02_ML_EKF_Improvement_By_Driving_Cycle.png",
            "Eight successes and two failures",
        ),
    ]:
        story.append(PageBreak())
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Image(str(figdir / img), width=465, height=290, hAlign="CENTER"))
    examples = [
        ("10C_HWFET", "Highway driving / cooler battery - useful reduction"),
        ("10C_NN", "Mixed driving - largest relative improvement"),
        ("25C_US06", "Aggressive acceleration - improved"),
        ("10C_UDDS", "City-driving UDDS - correction unnecessarily worsens accuracy"),
    ]
    for name, title in examples:
        match = next(figdir.glob("TRACE_" + name + "_*.png"))
        story.append(PageBreak())
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Image(str(match), width=475, height=285, hAlign="CENTER"))
        story.append(
            Paragraph(
                "Top: reference and both SOC estimates. Bottom: signed SOC estimation error. Reference SOC is constructed from tester Ah counter and assumed capacity. Real-world benefits such as better remaining-range estimation are potential applications, not demonstrated on a vehicle.",
                styles["SmallLocal"],
            )
        )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(
            40, 28, "Reproduced from raw MAT measurements + frozen trained model"
        )
        canvas.drawRightString(558, 28, str(doc.page))

    doc = SimpleDocTemplate(
        str(out),
        pagesize=(612, 792),
        leftMargin=44,
        rightMargin=44,
        topMargin=40,
        bottomMargin=50,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        default=str(ROOT.parent / "data" / "Panasonic 18650PF Data"),
        help="Extracted Panasonic dataset root",
    )
    ap.add_argument("--model", default=str(ROOT / "artifacts/fnn.pt"))
    ap.add_argument("--output-dir", default=str(ROOT / "results/generated"))
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()
    import torch

    torch.set_num_threads(1)
    from .networks import load_artifact

    mp = Path(args.model)
    hashes = json.loads((ROOT / "FROZEN_MODEL_SHA256.json").read_text())
    for relative, digest in hashes.items():
        if sha(ROOT / relative) != digest:
            raise ValueError(
                "Frozen trained weights/config/calibration modified: " + relative
            )
    model, scaler, cfg = load_artifact(mp)
    if cfg.get("kind") != "fnn":
        raise ValueError("Final frozen model must be FNN")
    av = float(cfg["alpha_voltage"])
    bz = float(cfg["beta_soc"])
    if not (0 <= av <= 1 and 0 <= bz <= 1):
        raise ValueError("Unexpected correction multipliers")
    rawroot = resolve_root(args.data)
    manifest = json.loads(
        (ROOT / "physics_backend/docs/data_manifest.json").read_text()
    )
    selected = [r for r in manifest if r["role"] in ROLES]
    if len(selected) != 10:
        raise RuntimeError("Expected exactly ten frozen evaluation recordings")
    verified = []
    file_checks = {}
    for r in selected:
        src = rawroot / r["path"]
        if not src.is_file():
            raise FileNotFoundError(f"Cannot run without ORIGINAL MAT file: {src}")
        digest = sha(src)
        known = json.loads((ROOT / "data_files_sha256.json").read_text())
        if digest != known[r["run"]]["sha256"]:
            raise ValueError(f'Original MAT hash mismatch for {r["run"]}')
        file_checks[r["run"]] = {
            "file": str(src.relative_to(rawroot)),
            "sha256": digest,
        }
    print("Raw recordings verified. Starting benchmark evaluation.", flush=True)
    out = Path(args.output_dir).resolve()
    if out.exists():
        # Do not leave stale graphs/metrics behind after a failed new run.
        import shutil

        if out == ROOT.resolve() or ROOT.resolve() not in out.parents:
            raise ValueError("Output directory must be inside the project folder")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    traces = {}
    rows = []
    for r in selected:
        print("Fresh EKF + FNN replay:", r["run"], flush=True)
        start = time.perf_counter()
        a = load_record(rawroot, r)
        s, ref = one_hz(a, 1.0 + a[:, 4] / 2.79826)
        pred = run(s, 0.8, artifact=mp, alpha_voltage=av, beta_soc=bz)
        if len(ref) != len(pred["soc_ekf"]):
            raise ValueError("Prediction/reference timestamp misalignment")
        e = metrics(ref, pred["soc_ekf"])
        v = metrics(ref, pred["soc_hybrid"])
        gain = float(100 * (1 - v["RMSE_pp"] / e["RMSE_pp"]))
        item = {
            "run": r["run"],
            "role": r["role"],
            "sample_count": len(ref),
            "EKF_RMSE_pp": e["RMSE_pp"],
            "ML_EKF_RMSE_pp": v["RMSE_pp"],
            "EKF_MAE_pp": e["MAE_pp"],
            "ML_EKF_MAE_pp": v["MAE_pp"],
            "EKF_max_abs_pp": e["max_abs_pp"],
            "ML_EKF_max_abs_pp": v["max_abs_pp"],
            "absolute_RMSE_gain_pp": round(e["RMSE_pp"] - v["RMSE_pp"], 6),
            "relative_gain_pct": round(gain, 5),
            "improved": v["RMSE_pp"] < e["RMSE_pp"],
        }
        rows.append(item)
        traces[r["run"]] = {
            "time_s": pred["time_s"],
            "reference": ref,
            "soc_ekf": pred["soc_ekf"],
            "soc_hybrid": pred["soc_hybrid"],
        }
        np.savez_compressed(out / ("replay_" + r["run"] + ".npz"), **traces[r["run"]])
        print(
            f'  EKF={e["RMSE_pp"]:.4f} pp | ML + EKF={v["RMSE_pp"]:.4f} pp | {gain:+.1f}% | {time.perf_counter()-start:.1f} s',
            flush=True,
        )
    target = out / "fresh_evaluation_metrics.csv"
    with target.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    draw_all(rows, traces, figdir, None)
    if not args.skip_pdf:
        fresh_pdf(rows, figdir, out / "ML_EKF_Evaluation_Report.pdf")
    import numpy
    import torch
    import scipy

    meta = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frozen_model_sha256": sha(mp),
        "metadata_sha256": sha(mp.with_suffix(".json")),
        "original_mat_sha256": file_checks,
        "python_version": sys.version,
        "numpy_version": numpy.__version__,
        "torch_version": torch.__version__,
        "scipy_version": scipy.__version__,
        "correction_alpha_voltage": av,
        "correction_beta_soc": bz,
        "soc_reference": "derived from tester Ah counter and Q=2.79826 Ah, initialized at 100%",
        "evaluation": "10 previously observed benchmarks, one Panasonic cell; not independent cross-cell proof",
        "note": "Metrics and figures were calculated from the predictions produced by this execution.",
    }
    (out / "REPRODUCTION_PROVENANCE.json").write_text(json.dumps(meta, indent=2) + "\n")
    ekf = np.mean([r["EKF_RMSE_pp"] for r in rows])
    ml_ekf = np.mean([r["ML_EKF_RMSE_pp"] for r in rows])
    win = sum(x["improved"] for x in rows)
    print("\n" + ("=" * 65))
    print(f"RUN COMPLETED: {win}/10 recordings improved")
    print(f"EKF mean RMSE: {ekf:.6f} pp")
    print(f"ML + EKF mean RMSE: {ml_ekf:.6f} pp")
    print(f"Relative reduction: {(1-ml_ekf/ekf)*100:.2f}%")
    print("Results from fresh raw-data replay at", out)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
