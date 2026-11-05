"""Verify the delivered snapshot with Python's standard library (no dataset needed)."""

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "libraries/SUBMISSION_SHA256.json"
    if not manifest.is_file():
        raise SystemExit("Submission checksum manifest is missing.")
    bad = []
    expected = json.loads(manifest.read_text())
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file():
            bad.append(relative + ": missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(relative + ": checksum changed")
    if bad:
        print("\n".join(bad))
        print(
            "A rerun legitimately changes generated results and timestamps; this checks the delivered snapshot."
        )
        return 1
    print(f"PASS: all {len(expected)} delivered files match SHA-256 checksums.")
    print(
        "This checks archive integrity. Run RUN_ML_EKF.sh with the original dataset to reproduce inference."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
