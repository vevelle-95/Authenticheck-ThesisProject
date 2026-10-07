"""HTTP contracts and failure handling without downloading or training models."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api


class FakePredictor:
    def __init__(self):
        self.frames = []

    def validate_cache(self):
        return None

    def predict(self, frame, prepare_cache=False):
        if not prepare_cache:
            raise AssertionError("New frontend reviews need cache preparation")
        self.frames.append(frame.copy())
        return {"reviews": [
            {"id": row["review_id"], "aspects": [
                {"category": "product_quality", "text": row["review_text"], "sentiment": "positive"}
            ]} for row in frame.to_dict("records")
        ]}, None, None


def payload():
    return {"schemaVersion": "1.0", "reviews": [
        {"id": "review-a", "text": "Matibay ang material.", "imageUrls": []}
    ]}


class ApiTests(unittest.TestCase):
    def test_prediction_preserves_ids_and_forwards_only_review_inputs(self):
        predictor = FakePredictor()
        request = payload()
        request["platform"] = "shopee"
        request["reviews"][0].update(text="  Matibay\n ang material.  ", rating=5)
        request["reviews"].append({"id": "review-b", "text": "Sulit.",
                                   "imageUrls": ["https://example.com/buyer.jpg"]})
        with patch("api.BaselinePredictor", return_value=predictor):
            with TestClient(api.app) as client:
                self.assertTrue(client.get("/ready").json()["ready"])
                response = client.post("/predict-aspects", json=request)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["schemaVersion"], "1.0")
        self.assertEqual(result["modelVersion"], api.MODEL_VERSION)
        self.assertEqual([row["id"] for row in result["reviews"]], ["review-a", "review-b"])
        self.assertEqual(result["reviews"][0]["aspects"][0], {
            "category": "product_quality", "text": "Matibay ang material.", "sentiment": "positive"
        })
        frame = predictor.frames[0]
        self.assertEqual(set(frame.columns), {"review_id", "review_text", "image_urls"})
        self.assertEqual(frame.iloc[0].image_urls, [])
        self.assertEqual(frame.iloc[1].image_urls, ["https://example.com/buyer.jpg"])

    def test_repeated_requests_reuse_the_predictor(self):
        predictor = FakePredictor()
        with patch("api.BaselinePredictor", return_value=predictor) as factory:
            with TestClient(api.app) as client:
                self.assertEqual(client.post("/predict-aspects", json=payload()).status_code, 200)
                self.assertEqual(client.post("/predict-aspects", json=payload()).status_code, 200)
            factory.assert_called_once()
        self.assertEqual(len(predictor.frames), 2)

    def test_invalid_requests_are_rejected_before_inference(self):
        invalid = [
            {"reviews": []},
            {"schemaVersion": "2.0", "reviews": payload()["reviews"]},
            {"reviews": [{"id": "blank", "text": " \n "}]},
            {"reviews": [{"id": "", "text": "Sulit."}]},
            {"reviews": payload()["reviews"] * 2},
            {"reviews": [{"id": "image", "text": "Sulit.", "imageUrls": ["file:///C:/photo.jpg"]}]},
            {"reviews": [{"id": "images", "text": "Sulit.", "imageUrls": ["https://example.com/a.jpg"] * 6}]},
            {"reviews": [{"id": str(i), "text": "Sulit."} for i in range(21)]},
        ]
        predictor = FakePredictor()
        with patch("api.BaselinePredictor", return_value=predictor):
            with TestClient(api.app) as client:
                for request in invalid:
                    with self.subTest(request=request):
                        self.assertEqual(client.post("/predict-aspects", json=request).status_code, 422)
        self.assertEqual(predictor.frames, [])

    def test_missing_artifacts_keep_health_available_and_prediction_unavailable(self):
        with patch("api.BaselinePredictor", side_effect=FileNotFoundError("Missing best.pt")):
            with TestClient(api.app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertFalse(client.get("/ready").json()["ready"])
                response = client.post("/predict-aspects", json=payload())
        self.assertEqual(response.status_code, 503)
        self.assertIn("best.pt", response.json()["detail"])

    def test_incompatible_cache_is_reported_as_unavailable(self):
        predictor = FakePredictor()
        with patch("api.BaselinePredictor", return_value=predictor), \
             patch.object(predictor, "validate_cache", side_effect=ValueError("CLIP revision mismatch")):
            with TestClient(api.app) as client:
                self.assertFalse(client.get("/ready").json()["ready"])
                self.assertEqual(client.post("/predict-aspects", json=payload()).status_code, 503)

    def test_prediction_cache_failure_returns_503(self):
        predictor = FakePredictor()
        with patch("api.BaselinePredictor", return_value=predictor), \
             patch.object(predictor, "predict", side_effect=FileNotFoundError("Cached image bytes missing")):
            with TestClient(api.app) as client:
                response = client.post("/predict-aspects", json=payload())
        self.assertEqual(response.status_code, 503)
        self.assertIn("Cached image", response.json()["detail"])

    def test_prediction_failure_returns_an_error_instead_of_predictions(self):
        predictor = FakePredictor()
        with patch("api.BaselinePredictor", return_value=predictor), \
             patch.object(predictor, "predict", side_effect=RuntimeError("Device failure")):
            with TestClient(api.app) as client:
                with self.assertLogs("api", level="ERROR"):
                    response = client.post("/predict-aspects", json=payload())
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("reviews", response.json())

    def test_cors_allows_local_frontends_and_extensions(self):
        with patch("api.BaselinePredictor", return_value=FakePredictor()):
            with TestClient(api.app) as client:
                for origin in ("http://localhost:5500", "http://127.0.0.1:5500", "chrome-extension://" + "a" * 32):
                    response = client.options("/predict-aspects", headers={
                        "Origin": origin, "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    })
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["access-control-allow-origin"], origin)
                denied = client.options("/predict-aspects", headers={
                    "Origin": "https://example.com", "Access-Control-Request-Method": "POST",
                })
                self.assertEqual(denied.status_code, 400)


if __name__ == "__main__":
    unittest.main()
