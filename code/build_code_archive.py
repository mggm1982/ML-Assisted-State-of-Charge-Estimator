"""Package the source, trained model and dependency instructions as code.zip."""

from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def main():
    destination = PROJECT / "code/code.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for section in ("code", "libraries"):
            for path in sorted((PROJECT / section).rglob("*")):
                relative = path.relative_to(PROJECT)
                if not path.is_file() or path == destination:
                    continue
                if any(
                    part.startswith(".") or part == "__pycache__"
                    for part in relative.parts
                ):
                    continue
                if (
                    path.suffix in (".pyc", ".zip")
                    or path.name == "SUBMISSION_SHA256.json"
                ):
                    continue
                archive.write(path, relative)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Source archive integrity check failed.")
    print("Created code/code.zip. The dataset is distributed separately.")


if __name__ == "__main__":
    main()
