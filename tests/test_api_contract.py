import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMPORT_ERROR = None
try:
    import main as backend
except ModuleNotFoundError as error:  # Dependencies are installed through requirements.txt.
    backend = None
    IMPORT_ERROR = error


@unittest.skipIf(backend is None, f"Backend dependencies unavailable: {IMPORT_ERROR}")
class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.original_pipeline = backend.pipeline
        self.original_ready = backend.app.state.model_ready

        class FakePipeline:
            def check_artifacts(self):
                return None

            def run(self, frame):
                reviews = [
                    {
                        "id": row.review_id,
                        "text": row.text,
                        "starRating": row.star_rating,
                        "label": "authentic",
                        "confidence": 0.9,
                        "probabilities": {"authentic": 0.9, "deceptive": 0.03, "liv": 0.04, "irrelevant": 0.03},
                        "signals": ["test model"],
                        "aspectSentiment": [],
                    }
                    for row in frame.itertuples()
                ]
                return {
                    "reviewCount": len(reviews),
                    "authenticShare": 100.0,
                    "verifiedRating": 5.0,
                    "counts": {"authentic": len(reviews), "deceptive": 0, "liv": 0, "irrelevant": 0},
                    "sentimentCounts": {"positive": 0, "neutral": 0, "negative": 0},
                    "aspects": [],
                    "reviews": reviews,
                }

        backend.pipeline = FakePipeline()
        backend.app.state.model_ready = True

    def tearDown(self):
        backend.pipeline = self.original_pipeline
        backend.app.state.model_ready = self.original_ready

    def test_sample_request_reaches_response_contract(self):
        payload = json.loads((ROOT / "data" / "samples" / "extension_api_payload.json").read_text(encoding="utf-8"))
        request = backend.AnalyzeRequest.model_validate(payload)
        response = backend.analyze(request)
        self.assertEqual(response["schemaVersion"], "1.0")
        self.assertEqual(response["reviewCount"], len(payload["reviews"]))
        self.assertEqual(response["reviews"][0]["id"], payload["reviews"][0]["id"])

    def test_request_rejects_more_than_twenty_reviews(self):
        payload = json.loads((ROOT / "data" / "samples" / "extension_api_payload.json").read_text(encoding="utf-8"))
        payload["reviews"] = [payload["reviews"][0] | {"id": f"review-{index}"} for index in range(21)]
        with self.assertRaises(ValueError):
            backend.AnalyzeRequest.model_validate(payload)


    def test_request_requires_integer_stars(self):
        payload = json.loads((ROOT / "data" / "samples" / "extension_api_payload.json").read_text(encoding="utf-8"))
        for rating in (None, 0, 6, 4.5):
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                backend.AnalyzeRequest.model_validate(payload | {"reviews": [payload["reviews"][0] | {"rating": rating}]})
        review = payload["reviews"][0].copy()
        del review["rating"]
        with self.assertRaises(ValueError):
            backend.AnalyzeRequest.model_validate(payload | {"reviews": [review]})


if __name__ == "__main__":
    unittest.main()
