import argparse
import sys
import shutil
from pathlib import Path
from typing import Optional



SEPARATOR = "___"  # PlantVillage naming convention crop___disease


def run_kaggle_download(slug: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading dataset '{slug}' to '{output}' via Kaggle Python API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print("[ERROR] Failed to import Kaggle API. Ensure 'kaggle' is installed in this environment.")
        print("        Try: pip install kaggle")
        print(f"        Import error: {e}")
        sys.exit(1)

    try:
        api = KaggleApi()
        api.authenticate()
        # Show dataset URL to help with terms acceptance
        print(f"Dataset URL: https://www.kaggle.com/datasets/{slug}")
        api.dataset_download_files(dataset=slug, path=str(output), unzip=True, quiet=False)
    except Exception as e:
        status: Optional[int] = None
        try:
            # requests.exceptions.HTTPError typically has a response with status_code
            status = getattr(getattr(e, "response", None), "status_code", None)
        except Exception:
            pass
        if status == 403:
            print("[HINT] Got 403 Forbidden. This usually means:")
            print("  - You need to visit the dataset URL in a browser while logged into Kaggle and accept the dataset terms; then retry.")
            print("  - OR the dataset requires special access. Try an alternative public slug like 'emmarex/plantdisease' or 'arjuntejaswi/plant-village'.")
        print(f"[ERROR] Kaggle API download failed: {e}")
        sys.exit(1)


def detect_loose_images(output: Path) -> bool:
    # Heuristic: jpg/png files directly under output (not in subfolders)
    for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG"):
        if any(output.glob(ext)):
            # ensure there are no existing class folders with many images already
            return True
    return False


def extract_class_name(filename: str) -> str:
    stem = Path(filename).stem
    # For names like Apple___Black_rot or Tomato___Late_blight_123
    if SEPARATOR in stem:
        parts = stem.split(SEPARATOR)
        # Remove trailing numeric tokens from last part (e.g., Late_blight_123)
        disease_tokens = parts[-1].split("_")
        # If last token numeric, drop it
        if disease_tokens and disease_tokens[-1].isdigit():
            disease_tokens = disease_tokens[:-1]
        parts[-1] = "_".join(disease_tokens)
        return SEPARATOR.join(parts[:2]) if len(parts) >= 2 else parts[0]
    return stem  # fallback


def reorganize_loose_files(output: Path) -> None:
    print("[INFO] Reorganizing loose image files into class folders...")
    image_files = []
    for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG"):
        image_files.extend(output.glob(ext))
    if not image_files:
        print("[WARN] No loose image files found; skipping reorganization.")
        return

    moved = 0
    for img in image_files:
        cls = extract_class_name(img.name)
        cls_dir = output / cls
        cls_dir.mkdir(exist_ok=True)
        dest = cls_dir / img.name
        # Skip if already inside correct folder
        if img.parent == cls_dir:
            continue
        # If destination exists, assume already moved
        if dest.exists():
            continue
        shutil.move(str(img), str(dest))
        moved += 1
    print(f"[INFO] Moved {moved} files into class folders.")


def summarize(output: Path) -> None:
    print("[SUMMARY] Class folder counts:")
    for d in sorted([p for p in output.iterdir() if p.is_dir()]):
        count = sum(1 for _ in d.glob("*.*"))
        print(f"  {d.name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Download and optionally reorganize PlantVillage dataset")
    parser.add_argument("--slug", required=True, help="Kaggle dataset slug (e.g., emmarex/plantdisease)")
    parser.add_argument("--output", default="data/raw", help="Download destination directory")
    parser.add_argument("--reorganize", action="store_true", help="Reorganize loose files into class folders if needed")
    args = parser.parse_args()

    output_path = Path(args.output)
    run_kaggle_download(args.slug, output_path)

    if args.reorganize and detect_loose_images(output_path):
        reorganize_loose_files(output_path)
    else:
        print("[INFO] Reorganization not requested or not needed.")

    summarize(output_path)
    print("[DONE] Dataset preparation complete.")


if __name__ == "__main__":
    main()
