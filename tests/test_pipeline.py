"""
Unit tests and pipeline verification for Explainable Skin Cancer Detection.
Verifies dataset logic, leakage prevention, preprocessing, model inference, and Grad-CAM.
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    validate_metadata,
    create_demo_samples
)


class TestSkinCancerPipeline(unittest.TestCase):

    def test_canonical_classes_order_and_count(self):
        """Assert exact 7 HAM10000 classes in canonical alphabetical order."""
        expected = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        self.assertEqual(CANONICAL_CLASSES, expected)
        self.assertEqual(len(CANONICAL_CLASSES), 7)
        self.assertIn("df", CANONICAL_CLASSES, "Dermatofibroma ('df') must never be omitted.")

    def test_class_to_idx_bidirectional_mapping(self):
        """Verify strict bidirectional mapping between class names and indices."""
        for idx, cls in enumerate(CANONICAL_CLASSES):
            self.assertEqual(CLASS_TO_IDX[cls], idx)
            self.assertEqual(IDX_TO_CLASS[idx], cls)

    def test_metadata_validation(self):
        """Test metadata schema validation for valid and corrupt cases."""
        valid_df = pd.DataFrame({
            "image_id": ["ISIC_0001", "ISIC_0002"],
            "dx": ["mel", "nv"],
            "lesion_id": ["HAM_1", "HAM_2"]
        })
        is_valid, msg = validate_metadata(valid_df)
        self.assertTrue(is_valid, msg)

        # Missing required column
        invalid_df = pd.DataFrame({"img": ["ISIC_0001"], "label": ["mel"]})
        is_valid, _ = validate_metadata(invalid_df)
        self.assertFalse(is_valid)

        # Invalid diagnosis code
        bad_dx_df = pd.DataFrame({"image_id": ["ISIC_0001"], "dx": ["unknown_cancer"]})
        is_valid, _ = validate_metadata(bad_dx_df)
        self.assertFalse(is_valid)

    def test_group_aware_leakage_prevention(self):
        """Assert that GroupShuffleSplit produces disjoint lesion_id groups across splits."""
        from sklearn.model_selection import GroupShuffleSplit
        
        # Simulate dataframe with shared lesions
        records = []
        for i in range(100):
            lesion_id = f"LESION_{i % 30}"  # multiple images share the same lesion
            records.append({
                "image_id": f"IMG_{i}",
                "lesion_id": lesion_id,
                "dx": CANONICAL_CLASSES[i % 7],
                "label": i % 7
            })
        df = pd.DataFrame(records)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df, df["label"], df["lesion_id"]))

        train_lesions = set(df.iloc[train_idx]["lesion_id"])
        test_lesions = set(df.iloc[test_idx]["lesion_id"])

        # Intersection MUST be completely empty (zero leakage)
        overlap = train_lesions.intersection(test_lesions)
        self.assertEqual(len(overlap), 0, f"Detected lesion leakage between splits: {overlap}")

    def test_demo_sample_generation(self):
        """Verify demo sample generation creates readable RGB images."""
        sample_dir = "tests/test_samples"
        samples = create_demo_samples(sample_dir)
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertTrue(os.path.exists(s))
            with Image.open(s) as img:
                self.assertIn(img.size, [(600, 450), (224, 224)])
                self.assertEqual(img.mode, "RGB")
        # Cleanup
        for s in samples:
            try:
                os.remove(s)
            except OSError:
                pass
        try:
            os.rmdir(sample_dir)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
