"""Verify the inputs, run the ML + EKF evaluation and save its results."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
RESULTS = PROJECT / "Results"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def prepare_data(raw_path, workspace):
    """Use the complete dataset when supplied, otherwise use the split folders."""
    from ml_ekf.dataset import resolve_root

    assignments = json.loads((CODE / "data_assignments.json").read_text())
    default = PROJECT / "data" / "Panasonic 18650PF Data"
    use_complete = raw_path is not None or any(default.rglob("*.mat"))
    raw_root = resolve_root(raw_path or default) if use_complete else None
    verified = []
    for entry in assignments:
        relative = entry["original_relative_path"]
        if raw_root is not None:
            source = raw_root / relative
        else:
            source = PROJECT / entry["drive_folder"] / entry["filename"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing recording: {entry['filename']}")
        actual = sha256(source)
        if actual != entry["sha256"]:
            raise ValueError(f"Recording checksum differs: {entry['filename']}")
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
        verified.append({"run": entry["run"], "path": relative, "sha256": actual})
    return verified, "complete_dataset" if use_complete else "split_folders"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", help="Optional path to the extracted Panasonic dataset"
    )
    args = parser.parse_args()
    record_dir = RESULTS / "reproduction"
    record_dir.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "started_utc": timestamp()}
    status_file = record_dir / "run_status.json"
    status_file.write_text(json.dumps(status, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MPLCONFIGDIR=str(CODE / ".mpl-cache"),
        PYTHONDONTWRITEBYTECODE="1",
    )

    with (record_dir / "execution.log").open("w") as log:

        def record(message):
            message = message.replace(str(PROJECT), "<project>")
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        def execute(arguments, separate_log=None):
            process = subprocess.Popen(
                [sys.executable, *arguments],
                cwd=CODE,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            lines = []
            for line in process.stdout:
                line = line.rstrip()
                record(line)
                lines.append(line)
            return_code = process.wait()
            if separate_log:
                separate_log.write_text("\n".join(lines) + "\n")
            if return_code:
                raise RuntimeError(f"Command failed with exit code {return_code}")

        try:
            record("ML + EKF evaluation started: " + status["started_utc"])
            expected = json.loads((CODE / "FROZEN_MODEL_SHA256.json").read_text())
            for relative, digest in expected.items():
                if sha256(CODE / relative) != digest:
                    raise ValueError("Model file checksum differs: " + relative)
            with tempfile.TemporaryDirectory(
                prefix=".evaluation-", dir=CODE
            ) as temporary:
                work = Path(temporary)
                verified, mode = prepare_data(args.data, work / "raw")
                record(
                    f"Verified {len(verified)} recordings and {len(expected)} model files."
                )
                record("Dataset source: " + mode)
                execute(
                    ["-m", "unittest", "discover", "-s", "tests", "-v"],
                    record_dir / "unit_tests.log",
                )
                output = work / "results"
                execute(
                    [
                        "-m",
                        "ml_ekf.reproduce_final",
                        "--data",
                        str(work / "raw"),
                        "--model",
                        str(CODE / "artifacts/fnn.pt"),
                        "--output-dir",
                        str(output),
                    ]
                )
                with (output / "fresh_evaluation_metrics.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                if (
                    len(rows) != 10
                    or len(list((output / "figures").glob("*.png"))) != 7
                ):
                    raise RuntimeError("Evaluation output is incomplete.")
                if len(list(output.glob("replay_*.npz"))) != 10:
                    raise RuntimeError("Prediction files are incomplete.")
                if not (output / "ML_EKF_Evaluation_Report.pdf").is_file():
                    raise RuntimeError("Evaluation report is missing.")
                provenance_file = output / "REPRODUCTION_PROVENANCE.json"
                provenance = json.loads(provenance_file.read_text())
                provenance.update(
                    model_name="ML + EKF",
                    verified_inputs=verified,
                    dataset_source=mode,
                    evaluation_sampling="Causal one-hertz sampling",
                )
                provenance["source_sha256"] = {
                    p.relative_to(CODE).as_posix(): sha256(p)
                    for p in sorted(CODE.rglob("*.py"))
                    if not any(
                        part.startswith(".") for part in p.relative_to(CODE).parts
                    )
                }
                provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")
                for source in sorted(output.rglob("*")):
                    if not source.is_file():
                        continue
                    relative = source.relative_to(output)
                    if source.suffix == ".npz":
                        relative = Path("predictions") / source.name
                    destination = RESULTS / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                mean_ekf = sum(float(r["EKF_RMSE_pp"]) for r in rows) / len(rows)
                mean_hybrid = sum(float(r["ML_EKF_RMSE_pp"]) for r in rows) / len(rows)
                wins = sum(r["improved"] == "True" for r in rows)
                summary = (
                    "ML + EKF evaluation\n\n"
                    f"EKF mean RMSE: {mean_ekf:.7f} SOC percentage points\n"
                    f"ML + EKF mean RMSE: {mean_hybrid:.7f} SOC percentage points\n"
                    f"Relative reduction: {100 * (1 - mean_hybrid / mean_ekf):.2f}%\n"
                    f"Recordings with lower RMSE: {wins}/{len(rows)}\n\n"
                    "Both UDDS recordings have higher error with the neural correction.\n"
                    "The benchmark comprises previously examined single-cell recordings.\n"
                    "Reference SOC is derived from the Ah counter using 2.79826 Ah.\n"
                )
                (RESULTS / "RESULTS_SUMMARY.txt").write_text(summary)
                status.update(
                    status="completed",
                    finished_utc=timestamp(),
                    recording_count=len(rows),
                    improved_count=wins,
                    plot_count=7,
                )
                record(
                    "Completed. Metrics, figures and report are available in Results/."
                )
        except Exception as error:
            status.update(status="failed", finished_utc=timestamp(), error=str(error))
            record("Evaluation failed: " + str(error))
            record("Any existing result files belong to an earlier execution.")
            raise
        finally:
            status_file.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
