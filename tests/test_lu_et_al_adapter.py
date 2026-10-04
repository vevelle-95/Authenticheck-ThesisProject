import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from backend.services.lu_et_al_adapter import (
    BaselineExecutionError, BaselineUnavailable, project_review, run_baseline,
)


class AdapterTests(unittest.TestCase):
    def test_holistic_projection_marks_provenance_and_fulfillment(self):
        review = {"id": "r1", "text": "Great sound but late delivery"}
        result = project_review(review, {"sentiment": "negative"})
        self.assertEqual([a["category"] for a in result["aspects"]], ["product_quality", "seller_service"])
        self.assertTrue(all(a["sentiment_source"] == "holistic_projection" for a in result["aspects"]))

    def test_segment_sentiment_is_preserved(self):
        review = {"id": "r1", "text": "Great sound but late delivery"}
        result = project_review(review, {"sentiment": "negative", "segments": [{"text": "Great sound", "sentiment": "positive"}]})
        self.assertEqual(result["aspects"][0]["sentiment"], "positive")
        self.assertEqual(result["aspects"][0]["sentiment_source"], "baseline_segment")

    def test_external_process_contract_and_error(self):
        directory = Path(__file__).resolve().parent / "fixtures"
        review = {"id": "r1", "text": "Great sound", "star_rating": 5, "image_url": ""}
        with patch.dict(os.environ, {"LU_ET_AL_REPO": str(directory), "LU_ET_AL_COMMAND": f'"{sys.executable}" "{directory / "runner.py"}"'}):
            self.assertEqual(run_baseline([review])[0]["holistic_sentiment"], "neutral")
        with patch.dict(os.environ, {"LU_ET_AL_REPO": str(directory), "LU_ET_AL_COMMAND": f'"{sys.executable}" "{directory / "bad_runner.py"}"'}):
            with self.assertRaises(BaselineExecutionError):
                run_baseline([review])

    def test_python_entrypoint_uses_isolated_interpreter(self):
        directory = Path(__file__).resolve().parent / "fixtures"
        review = {"id": "r1", "text": "Great sound", "star_rating": 5, "image_url": ""}
        with patch.dict(os.environ, {
            "LU_ET_AL_REPO": str(directory), "LU_ET_AL_COMMAND": "",
            "LU_ET_AL_PYTHON": sys.executable, "LU_ET_AL_ENTRYPOINT": "model_stub:predict",
        }):
            self.assertEqual(run_baseline([review])[0]["holistic_sentiment"], "neutral")

    def test_missing_configuration(self):
        with patch.dict(os.environ, {"LU_ET_AL_REPO": "does-not-exist", "LU_ET_AL_COMMAND": ""}):
            with self.assertRaises(BaselineUnavailable):
                run_baseline([])


if __name__ == "__main__":
    unittest.main()
