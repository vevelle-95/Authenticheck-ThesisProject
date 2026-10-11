"""Run supplied diagnostic cases against the local API, without training models."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / 'data/samples/authenticheck-test-cases.json'
LABELS = {'authentic', 'deceptive', 'liv', 'irrelevant'}
ASPECTS = {'product_quality', 'functionality', 'performance', 'design', 'aesthetics',
           'sensory_experience', 'value', 'packaging', 'seller_service', 'accuracy_of_description'}


def load_cases(path=DEFAULT_CASES):
    cases = json.loads(Path(path).read_text(encoding='utf-8'))['test_cases']
    ids = set()
    for case in cases:
        if not case['review_id'] or case['review_id'] in ids:
            raise ValueError('Case IDs must be unique and nonempty')
        ids.add(case['review_id'])
        if case['ground_truth'].lower() not in LABELS:
            raise ValueError('Unknown expected quality label')
        aspects, sentiments = json.loads(case['aspects']), json.loads(case['sentiment'])
        if len(aspects) != len(sentiments) or not set(aspects) <= ASPECTS:
            raise ValueError('Invalid aspect expectations')
        if not set(sentiments) <= {'positive', 'negative', 'neutral'}:
            raise ValueError('Invalid sentiment expectations')
    return cases


def prepare_case(case):
    rating = case['star_rating']
    valid = (isinstance(rating, (int, float)) and not isinstance(rating, bool)
             and math.isfinite(rating) and float(rating).is_integer() and 1 <= rating <= 5)
    images = [url.strip() for url in (case.get('review_image_urls') or '').split('|') if url.strip()]
    text = case.get('review_text')
    valid = valid and isinstance(text, str) and bool(text.strip()) and len(text) <= 2000
    valid = valid and len(images) <= 5 and all(urlparse(u).scheme in {'http', 'https'} and urlparse(u).netloc for u in images)
    description = case.get('product_description', '')
    # API accepts at most 2,000 description characters. Keep original fixture intact.
    payload = {'schemaVersion': '1.0', 'platform': case['platform_source'],
        'url': f"https://{case['platform_source']}.ph/", 'productTitle': case['product_title'],
        'productDescription': description[:2000], 'reviews': [{
            'id': case['review_id'], 'text': case['review_text'],
            'rating': int(rating) if isinstance(rating, (int, float)) and math.isfinite(rating) else None, 'hasImage': bool(images), 'imageUrls': images}]}
    return payload, {'expectedHttpStatus': 200 if valid else 422,
                     'descriptionTruncated': len(description) > 2000}


def score_case(case, response):
    reviews = response.get('reviews', [])
    if len(reviews) != 1 or reviews[0].get('id') != case['review_id']:
        raise ValueError('Response review ID/coverage mismatch')
    row = reviews[0]
    actual = Counter((a['category'], a['sentiment'].lower()) for a in row.get('aspectSentiment', []))
    expected = Counter(zip(json.loads(case['aspects']), json.loads(case['sentiment'])))
    return {'actualLabel': row['label'], 'qualityMatch': row['label'] == case['ground_truth'].lower(),
            'aspectSentimentExactMatch': actual == expected,
            'predictedAspects': row.get('aspectSentiment', [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default=str(DEFAULT_CASES))
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000/analyze')
    parser.add_argument('--output', default='reports/generated/supplied_test_cases_results.json')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.validate_only:
        print(json.dumps({'cases': len(cases), 'labels': dict(Counter(c['ground_truth'].lower() for c in cases)),
            'expected_rejections': sum(prepare_case(c)[1]['expectedHttpStatus'] != 200 for c in cases)}))
        return
    rows = []
    for index, case in enumerate(cases, 1):
        payload, expectations = prepare_case(case)
        row = {'id': case['review_id'], 'type': case['test_case_type'],
            'confidence': case['confidence'], 'notes': case.get('notes'),
            'expectedLabel': case['ground_truth'].lower(), **expectations}
        try:
            request = Request(args.endpoint, data=json.dumps(payload, allow_nan=False).encode('utf-8'),
                              headers={'Content-Type': 'application/json'})
            try:
                with urlopen(request, timeout=args.timeout) as response:
                    status, result = response.status, json.load(response)
            except HTTPError as error:
                status, result = error.code, json.loads(error.read().decode('utf-8'))
            row.update(httpStatus=status, httpStatusMatch=status == expectations['expectedHttpStatus'])
            if status == 200:
                row.update(modelVersion=result.get('modelVersion'), **score_case(case, result))
            else:
                row['detail'] = result.get('detail')
        except (URLError, TimeoutError, ValueError, KeyError) as error:
            row.update(error=str(error), httpStatusMatch=False)
        rows.append(row)
        print(f"{index}/{len(cases)} {case['review_id']}: HTTP={row.get('httpStatus', 'ERROR')}, quality match={row.get('qualityMatch', 'N/A')}", flush=True)
    successful = [r for r in rows if r.get('httpStatus') == 200 and 'qualityMatch' in r]
    report = {'endpoint': args.endpoint, 'purpose': 'User-supplied diagnostic expectations; not a held-out thesis benchmark.',
        'caseCount': len(rows), 'predictionCount': len(successful),
        'httpStatusMatches': sum(r['httpStatusMatch'] for r in rows),
        'qualityMatches': sum(r['qualityMatch'] for r in successful),
        'aspectSentimentExactMatches': sum(r['aspectSentimentExactMatch'] for r in successful), 'cases': rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Report saved: {output}', flush=True)


if __name__ == '__main__':
    main()
