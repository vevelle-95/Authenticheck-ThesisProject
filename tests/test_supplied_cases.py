"""Exercise user-supplied case transport and diagnostic scoring, without model weights."""
import unittest
from scripts.run_supplied_test_cases import load_cases, prepare_case, score_case


class SuppliedCaseTests(unittest.TestCase):
    def test_all_cases_match_api_validation_expectations(self):
        from main import AnalyzeRequest
        cases = load_cases()
        self.assertEqual(len(cases), 64)
        for case in cases:
            payload, expected = prepare_case(case)
            with self.subTest(case=case['review_id']):
                if expected['expectedHttpStatus'] == 422:
                    with self.assertRaises(ValueError):
                        AnalyzeRequest.model_validate(payload)
                else:
                    request = AnalyzeRequest.model_validate(payload)
                    self.assertEqual(request.reviews[0].id, case['review_id'])
                    self.assertEqual(request.reviews[0].rating, case['star_rating'])

    def test_scoring_reports_mismatches_and_keeps_all_polarities(self):
        case = {'review_id': 'mixed', 'ground_truth': 'Authentic',
                'aspects': '["performance", "performance"]', 'sentiment': '["positive", "negative"]'}
        response = {'reviews': [{'id': 'mixed', 'label': 'liv', 'aspectSentiment': [
            {'category': 'performance', 'sentiment': 'Positive'}]}]}
        result = score_case(case, response)
        self.assertFalse(result['qualityMatch'])
        self.assertFalse(result['aspectSentimentExactMatch'])
        response['reviews'][0]['aspectSentiment'].append({'category': 'performance', 'sentiment': 'Negative'})
        self.assertTrue(score_case(case, response)['aspectSentimentExactMatch'])

    def test_scoring_rejects_wrong_review_id(self):
        with self.assertRaises(ValueError):
            score_case({'review_id': 'expected'}, {'reviews': [{'id': 'wrong'}]})


if __name__ == '__main__':
    unittest.main()
