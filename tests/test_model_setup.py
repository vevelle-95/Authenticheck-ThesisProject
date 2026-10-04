import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("setup_models", ROOT / "scripts" / "setup_models.py")
SETUP_MODELS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP_MODELS)


class ModelSetupTests(unittest.TestCase):
    def test_manifest_has_six_features_and_four_classes(self):
        manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["featureOrder"]), 6)
        self.assertEqual(manifest["classOrder"], ["authentic", "deceptive", "liv", "irrelevant"])

    def test_verifier_reports_absent_weights(self):
        manifest = SETUP_MODELS.read_manifest()
        missing = SETUP_MODELS.verify(manifest)
        self.assertTrue(any("xgboost_meta_classifier.json" in item for item in missing))


if __name__ == "__main__":
    unittest.main()
