"""
HAM10000 Dataset Downloader and Validator using KaggleHub.
Dataset: kmader/skin-cancer-mnist-ham10000
Target Location: data/HAM10000
"""

import os
import sys
import glob
import shutil
from typing import Dict, List, Tuple
import pandas as pd

# Canonical 7 classes of HAM10000
CANONICAL_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
TARGET_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "HAM10000")


def check_existing_dataset(data_dir: str) -> Tuple[bool, Dict]:
    """
    Checks if the dataset is already downloaded and valid in the specified folder.
    Avoids re-downloading if already present.
    """
    summary = {
        "is_valid": False,
        "metadata_csv": None,
        "total_images": 0,
        "matched_images": 0,
        "classes": [],
        "missing_classes": [],
        "message": ""
    }

    if not os.path.exists(data_dir):
        summary["message"] = f"Directory {data_dir} does not exist."
        return False, summary

    # Search for metadata CSV
    csv_candidates = glob.glob(os.path.join(data_dir, "**", "*metadata*.csv"), recursive=True)
    if not csv_candidates:
        summary["message"] = "No metadata CSV found."
        return False, summary

    metadata_path = csv_candidates[0]
    summary["metadata_csv"] = metadata_path

    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        summary["message"] = f"Error reading CSV {metadata_path}: {e}"
        return False, summary

    if "image_id" not in df.columns or "dx" not in df.columns:
        summary["message"] = f"Missing required columns in {metadata_path}. Found: {list(df.columns)}"
        return False, summary

    # Collect images
    image_paths = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for p in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
            img_id = os.path.splitext(os.path.basename(p))[0]
            image_paths[img_id] = p

    summary["total_images"] = len(image_paths)
    if len(image_paths) == 0:
        summary["message"] = "No image files found."
        return False, summary

    df["path"] = df["image_id"].map(image_paths)
    matched_df = df.dropna(subset=["path", "dx"])
    summary["matched_images"] = len(matched_df)

    classes_found = sorted(matched_df["dx"].unique().tolist())
    summary["classes"] = classes_found
    missing = [c for c in CANONICAL_CLASSES if c not in classes_found]
    summary["missing_classes"] = missing

    # Dataset is considered valid if images are matched and all 7 classes are present
    if len(matched_df) >= 1000 and len(missing) == 0:
        summary["is_valid"] = True
        summary["message"] = f"Dataset is already present and verified ({len(matched_df)} matched images, all 7 classes)."
        return True, summary
    else:
        summary["message"] = f"Dataset present but incomplete (matched: {len(matched_df)}, missing classes: {missing})"
        return False, summary


def download_ham10000(output_dir: str = TARGET_DATA_DIR) -> str:
    """
    Downloads the HAM10000 dataset using kagglehub with authentication checks.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is not installed in the active Python environment. "
            "Please install it using: pip install kagglehub"
        )

    os.makedirs(output_dir, exist_ok=True)
    handle = "kmader/skin-cancer-mnist-ham10000"
    print(f"[*] Requesting dataset '{handle}' via kagglehub...")

    try:
        # First attempt with output_dir keyword argument as specified by user
        try:
            download_path = kagglehub.dataset_download(handle, output_dir=output_dir)
        except TypeError:
            # Fallback for kagglehub versions supporting path keyword or default cache
            try:
                download_path = kagglehub.dataset_download(handle, path=output_dir)
            except TypeError:
                download_path = kagglehub.dataset_download(handle)
                # If downloaded to default cache, copy or link to output_dir
                if download_path and download_path != output_dir:
                    print(f"[*] Downloaded to cache: {download_path}. Copying/organizing to {output_dir}...")
                    for item in os.listdir(download_path):
                        s = os.path.join(download_path, item)
                        d = os.path.join(output_dir, item)
                        if not os.path.exists(d):
                            if os.path.isdir(s):
                                shutil.copytree(s, d)
                            else:
                                shutil.copy2(s, d)
                    download_path = output_dir

        print(f"[+] Dataset downloaded to: {download_path}")
        return download_path

    except Exception as e:
        err_msg = str(e)
        print(f"[-] KaggleHub download encountered an error: {err_msg}", file=sys.stderr)
        
        # Check for authentication / credentials / 401 / 403 / consent requirements
        if "401" in err_msg or "Unauthorized" in err_msg or "credentials" in err_msg.lower() or "kaggle.json" in err_msg.lower():
            print("\n" + "="*70)
            print("ACTION REQUIRED: KAGGLE AUTHENTICATION / API TOKEN NEEDED")
            print("="*70)
            print("KaggleHub requires Kaggle API credentials to download this dataset.")
            print("Please perform the following steps:")
            print("1. Log in to your Kaggle account: https://www.kaggle.com")
            print("2. Navigate to: Account -> Settings -> API -> 'Create New Token'.")
            print("3. A file named 'kaggle.json' will be downloaded to your computer.")
            print("4. Place 'kaggle.json' in your Windows Kaggle directory:")
            print(f"   {os.path.expanduser('~')}\\.kaggle\\kaggle.json")
            print("5. Re-run: python download_dataset.py")
            print("="*70 + "\n")
        elif "consent" in err_msg.lower() or "terms" in err_msg.lower() or "403" in err_msg:
            print("\n" + "="*70)
            print("ACTION REQUIRED: DATASET TERMS ACCEPTANCE NEEDED")
            print("="*70)
            print("This dataset requires accepting the terms/competition rules on Kaggle.")
            print("1. Open: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
            print("2. Click 'Download' or accept the dataset terms on the web page.")
            print("3. Re-run: python download_dataset.py")
            print("="*70 + "\n")
            
        raise e


def validate_downloaded_dataset(data_dir: str = TARGET_DATA_DIR) -> Dict:
    """
    Recursively inspects downloaded files, verifies metadata columns, image counts,
    and supports all 7 canonical labels.
    """
    print(f"\n[*] Inspecting downloaded dataset in: {data_dir}")
    report = {
        "data_dir": data_dir,
        "metadata_path": None,
        "total_records": 0,
        "image_count": 0,
        "matched_count": 0,
        "columns": [],
        "classes_found": [],
        "missing_classes": [],
        "class_counts": {},
        "is_valid": False,
        "validation_errors": []
    }

    if not os.path.exists(data_dir):
        report["validation_errors"].append(f"Directory {data_dir} does not exist.")
        return report

    # 1. Locate HAM10000_metadata.csv
    csv_candidates = (
        glob.glob(os.path.join(data_dir, "HAM10000_metadata.csv")) +
        glob.glob(os.path.join(data_dir, "**", "*metadata*.csv"), recursive=True)
    )

    if not csv_candidates:
        report["validation_errors"].append("HAM10000_metadata.csv not found in downloaded files.")
        return report

    meta_path = os.path.abspath(csv_candidates[0])
    report["metadata_path"] = meta_path
    print(f"[+] Found metadata file: {meta_path}")

    # 2. Read and verify columns
    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        report["validation_errors"].append(f"Failed to read metadata CSV: {e}")
        return report

    report["total_records"] = len(df)
    report["columns"] = list(df.columns)
    print(f"[+] Total metadata rows: {len(df)}")
    print(f"[+] Columns: {report['columns']}")

    expected_cols = {"image_id", "dx"}
    if not expected_cols.issubset(df.columns):
        report["validation_errors"].append(f"Metadata missing expected columns {expected_cols}. Found: {report['columns']}")
        return report

    # 3. Recursively locate all image directories and files
    img_map = {}
    image_dirs = set()
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for p in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
            img_id = os.path.splitext(os.path.basename(p))[0]
            img_map[img_id] = os.path.abspath(p)
            image_dirs.add(os.path.dirname(os.path.abspath(p)))

    report["image_count"] = len(img_map)
    print(f"[+] Found {len(img_map)} image files across {len(image_dirs)} directories:")
    for d in sorted(image_dirs):
        print(f"    - {d}")

    # 4. Image-to-metadata matching
    df["path"] = df["image_id"].map(img_map)
    matched = df.dropna(subset=["path", "dx"]).copy()
    report["matched_count"] = len(matched)
    print(f"[+] Successfully matched {len(matched)} / {len(df)} metadata records to image files.")

    if len(matched) == 0:
        report["validation_errors"].append("Zero images matched metadata 'image_id' values.")
        return report

    # 5. Class validation
    classes_present = sorted(matched["dx"].unique().tolist())
    report["classes_found"] = classes_present
    report["class_counts"] = matched["dx"].value_counts().to_dict()
    missing_classes = [c for c in CANONICAL_CLASSES if c not in classes_present]
    report["missing_classes"] = missing_classes

    print("\n[+] Diagnostic Class Distribution:")
    for cls in CANONICAL_CLASSES:
        cnt = report["class_counts"].get(cls, 0)
        status = "PRESENT" if cnt > 0 else "MISSING"
        print(f"    - {cls:6s}: {cnt:5d} images [{status}]")

    if missing_classes:
        report["validation_errors"].append(f"Missing canonical classes: {missing_classes}")
        return report

    report["is_valid"] = True
    print("\n[SUCCESS] Dataset validation passed: All 7 canonical classes verified and matched.")
    return report


def main():
    print("="*70)
    print("HAM10000 DATASET DOWNLOAD & VALIDATION")
    print("="*70)

    # 1. Check if dataset already present and valid
    is_valid, summary = check_existing_dataset(TARGET_DATA_DIR)
    if is_valid:
        print(f"[+] {summary['message']}")
        report = validate_downloaded_dataset(TARGET_DATA_DIR)
        return report

    # 2. Download via kagglehub
    try:
        download_ham10000(TARGET_DATA_DIR)
    except Exception as e:
        print(f"\n[!] Download could not proceed: {e}", file=sys.stderr)
        return None

    # 3. Validate downloaded files
    report = validate_downloaded_dataset(TARGET_DATA_DIR)
    return report


if __name__ == "__main__":
    main()
