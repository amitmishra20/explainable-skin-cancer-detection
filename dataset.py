"""
Dataset discovery, validation, preprocessing, and metadata utilities for HAM10000.
Strictly adheres to the canonical 7-class HAM10000 taxonomy.
"""

import os
import glob
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image

# Canonical 7 classes of HAM10000 in alphabetical order
CANONICAL_CLASSES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Detailed human-readable labels and descriptions for clinical education
CLASS_METADATA: Dict[str, Dict[str, str]] = {
    "akiec": {
        "code": "akiec",
        "name": "Actinic Keratosis / Bowen's Disease",
        "category": "Precancerous / Early-stage Carcinoma",
        "risk_level": "Moderate",
        "description": "Sun-induced pre-malignant cutaneous lesion that can develop into invasive squamous cell carcinoma."
    },
    "bcc": {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "category": "Malignant",
        "risk_level": "High (Locally invasive)",
        "description": "The most prevalent malignant skin neoplasm; slow-growing and destructive locally, rarely metastasizes."
    },
    "bkl": {
        "code": "bkl",
        "name": "Benign Keratosis-like Lesions",
        "category": "Benign",
        "risk_level": "Low (Harmless)",
        "description": "Harmless skin growths including seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses."
    },
    "df": {
        "code": "df",
        "name": "Dermatofibroma",
        "category": "Benign",
        "risk_level": "Low (Harmless)",
        "description": "Benign fibrous histiocytoma presenting as a firm, hyperpigmented dermal nodule, typically on limbs."
    },
    "mel": {
        "code": "mel",
        "name": "Melanoma",
        "category": "Malignant",
        "risk_level": "Critical (High metastasis potential)",
        "description": "Highly aggressive malignancy arising from melanocytes; rapid diagnosis and surgical excision are imperative."
    },
    "nv": {
        "code": "nv",
        "name": "Melanocytic Nevus",
        "category": "Benign",
        "risk_level": "Low (Common mole)",
        "description": "Common benign proliferations of pigment-producing melanocytes; generally benign but monitored for changes."
    },
    "vasc": {
        "code": "vasc",
        "name": "Vascular Lesion",
        "category": "Benign",
        "risk_level": "Low (Harmless)",
        "description": "Benign cutaneous vascular proliferations including cherry angiomas, angiokeratomas, and pyogenic granulomas."
    }
}

CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for idx, cls in enumerate(CANONICAL_CLASSES)}


def get_canonical_classes() -> List[str]:
    """Returns a copy of the canonical 7-class list."""
    return list(CANONICAL_CLASSES)


def validate_metadata(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validates that a dataframe adheres to expected HAM10000 metadata standards.
    """
    required_cols = {"image_id", "dx"}
    if not required_cols.issubset(df.columns):
        return False, f"Missing required columns. Expected {required_cols}, found {list(df.columns)}"
    
    unique_dx = set(df["dx"].dropna().unique())
    unknown_dx = unique_dx - set(CANONICAL_CLASSES)
    if unknown_dx:
        return False, f"Found unrecognized diagnostic labels: {unknown_dx}"
    
    return True, "Metadata format is valid."


def discover_dataset(data_dir: str = "data/HAM10000") -> Dict:
    """
    Recursively discovers HAM10000 metadata and corresponding images.
    Returns a dictionary summarizing dataset status, dataframe, and validation details.
    """
    summary = {
        "data_dir": data_dir,
        "is_available": False,
        "metadata_found": False,
        "metadata_path": None,
        "total_metadata_records": 0,
        "matched_images": 0,
        "missing_images": 0,
        "classes_present": [],
        "missing_classes": [],
        "class_counts": {},
        "df": None,
        "issues": []
    }

    if not os.path.exists(data_dir):
        summary["issues"].append(f"Data directory '{data_dir}' does not exist.")
        return summary

    # Search for metadata CSV
    csv_candidates = [
        os.path.join(data_dir, "HAM10000_metadata.csv"),
        os.path.join(data_dir, "metadata.csv"),
    ] + glob.glob(os.path.join(data_dir, "**", "*metadata*.csv"), recursive=True)

    csv_path = None
    for cand in csv_candidates:
        if os.path.exists(cand):
            csv_path = cand
            break

    if not csv_path:
        summary["issues"].append("HAM10000_metadata.csv was not found in dataset folder.")
        return summary

    summary["metadata_found"] = True
    summary["metadata_path"] = csv_path

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        summary["issues"].append(f"Failed to read metadata CSV: {e}")
        return summary

    valid, msg = validate_metadata(df)
    if not valid:
        summary["issues"].append(msg)
        return summary

    summary["total_metadata_records"] = len(df)

    # Index all available images
    image_map = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
            file_id = os.path.splitext(os.path.basename(path))[0]
            image_map[file_id] = os.path.abspath(path)

    df["path"] = df["image_id"].map(image_map)
    matched_df = df.dropna(subset=["path", "dx"]).copy().reset_index(drop=True)

    summary["matched_images"] = len(matched_df)
    summary["missing_images"] = len(df) - len(matched_df)

    if len(matched_df) == 0:
        summary["issues"].append("No images matched the 'image_id' entries in the metadata.")
        return summary

    present_classes = sorted(matched_df["dx"].unique().tolist())
    summary["classes_present"] = present_classes
    summary["missing_classes"] = [c for c in CANONICAL_CLASSES if c not in present_classes]
    summary["class_counts"] = matched_df["dx"].value_counts().to_dict()

    if summary["missing_classes"]:
        summary["issues"].append(
            f"Dataset is missing {len(summary['missing_classes'])} canonical class(es): {summary['missing_classes']}"
        )
    else:
        summary["is_available"] = True

    # Map labels to canonical indices
    matched_df["label"] = matched_df["dx"].map(CLASS_TO_IDX).astype(int)
    summary["df"] = matched_df
    return summary


def create_demo_samples(output_dir: str = "data/sample_images", data_dir: str = "data/HAM10000") -> List[str]:
    """
    Populates sample images for UI/Grad-CAM demonstration.
    Prefers real HAM10000 benchmark images if available; otherwise falls back to synthetic samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_files = []

    # Authentic candidate images across the 7 HAM10000 classes
    authentic_samples = {
        "sample_akiec_ISIC_0029041.jpg": "ISIC_0029041.jpg",
        "sample_bcc_ISIC_0026760.jpg": "ISIC_0026760.jpg",
        "sample_bkl_ISIC_0029176.jpg": "ISIC_0029176.jpg",
        "sample_df_ISIC_0032642.jpg": "ISIC_0032642.jpg",
        "sample_mel_ISIC_0030417.jpg": "ISIC_0030417.jpg",
        "sample_nv_ISIC_0025843.jpg": "ISIC_0025843.jpg",
        "sample_vasc_ISIC_0026092.jpg": "ISIC_0026092.jpg",
    }

    found_authentic = False
    if os.path.exists(data_dir):
        import shutil
        for dest_name, src_file in authentic_samples.items():
            dest_path = os.path.join(output_dir, dest_name)
            if not os.path.exists(dest_path):
                # Search across part_1 and part_2
                matches = glob.glob(os.path.join(data_dir, "**", src_file), recursive=True)
                if matches:
                    shutil.copyfile(matches[0], dest_path)
                    sample_files.append(dest_path)
                    found_authentic = True
            else:
                sample_files.append(dest_path)
                found_authentic = True

    if found_authentic and len(sample_files) > 0:
        return sorted(sample_files)

    # Fallback synthetic patterns if dataset directory is not present
    patterns = {
        "melanoma_sample_mel.jpg": (120, 45, 30),
        "nevus_sample_nv.jpg": (160, 110, 80),
        "basal_cell_sample_bcc.jpg": (190, 130, 130),
        "keratosis_sample_bkl.jpg": (140, 100, 60),
    }

    for fname, base_rgb in patterns.items():
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            arr[:, :] = [225, 195, 175]
            y, x = np.ogrid[:224, :224]
            cy, cx = 112, 112
            dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            lesion_mask = dist_from_center <= 60
            
            noise = np.random.RandomState(42).randint(-20, 20, size=(224, 224, 3))
            lesion_color = np.clip(np.array(base_rgb) + noise, 0, 255).astype(np.uint8)
            arr[lesion_mask] = lesion_color[lesion_mask]
            
            img = Image.fromarray(arr)
            img.save(fpath, quality=92)
        sample_files.append(fpath)

    return sorted(sample_files)

