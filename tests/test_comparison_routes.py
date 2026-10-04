import unittest
from unittest.mock import patch

from backend.routes.comparison_routes import ComparisonRequest, compare_lu_et_al


class ComparisonRouteTests(unittest.TestCase):
    def test_description_and_discrepancy_are_reported(self):
        request = ComparisonRequest(product_description="Wireless headphones", reviews=[
            {"id": "r1", "text": "Great sound but late delivery", "star_rating": 3}
        ])
        ours = [{"id": "r1", "classification": "authentic", "aspects": [
            {"category": "seller_service", "sentiment": "negative", "evidence": "late delivery", "sentiment_source": "authenticheck_absa"}
        ]}]
        baseline = [{"id": "r1", "holistic_sentiment": "positive", "aspects": [
            {"category": "seller_service", "sentiment": "positive", "evidence": "late delivery", "sentiment_source": "holistic_projection"}
        ]}]
        with patch("backend.routes.comparison_routes.run_authenticheck", return_value=ours) as auth_run:
            with patch("backend.routes.comparison_routes.run_baseline", return_value=baseline):
                result = compare_lu_et_al(request)
        self.assertEqual(auth_run.call_args.args[1], "Wireless headphones")
        self.assertEqual(result["metrics"]["divergences"], 1)
        self.assertEqual(result["metrics"]["rows"][0]["category"], "seller_service")


if __name__ == "__main__":
    unittest.main()
