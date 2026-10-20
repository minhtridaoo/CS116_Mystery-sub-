from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    source_root = root / "baseline" / "baseline"
    destination = root / "submission.zip"

    if destination.exists():
        destination.unlink()

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_root.rglob("*"):
            if path.is_file() and "__MACOSX" not in path.parts:
                archive.write(path, arcname=path.relative_to(source_root.parent))


if __name__ == "__main__":
    main()
