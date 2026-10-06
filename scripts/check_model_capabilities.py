"""Run diagnostic review challenges; report quality gate and isolated ABSA separately."""
import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/analyze")
    args = parser.parse_args()
    source = json.loads((ROOT / "data/samples/capability_challenge.json").read_text(encoding="utf-8"))
    cases = source["cases"]
    # Ensure the diagnostic texts do not duplicate practice-training records.
    import torch
    from model_data import load_reviews
    from stage2.absa_model import ABSAHeadModel
    from transformers import AutoTokenizer
    training = load_reviews(ROOT / "data/test_reviews.csv", require_annotations=True)
    existing = set(training.review_text.str.casefold())
    assert not any(case["text"].casefold() in existing for case in cases)
    torch.set_num_threads(2)
    bundle = ROOT / "models/development_v2/absa_model"
    model = ABSAHeadModel.from_pretrained(bundle).eval()
    tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_dir, local_files_only=True)
    rows = []
    for case in cases:
        payload = {"schemaVersion":"1.0", "platform":"shopee", "url":"https://shopee.ph/product/123/456",
            "productTitle":"Diagnostic headphones", "reviews":[{
                "id":case["id"], "text":case["text"], "rating":case["rating"], "hasImage":False,"imageUrls":[]}]}
        try:
            with urlopen(Request(args.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}), timeout=60) as response:
                result = json.load(response)
            review = result["reviews"][0]
            actual = review["label"]
            error = None
        except HTTPError as failure:
            actual = None
            error = f"HTTP {failure.code}: {failure.read().decode()}"
        row = {**case, "actualLabel":actual, "qualityMatch":actual == case["expectedLabel"], "error":error}
        if case["expectedLabel"] == "authentic":
            # Diagnostic bypass only: distinguish ABSA behavior from quality-filter failures.
            predicted = model.predict([case["text"]], tokenizer)[0]
            actual_aspects = {item["category"]: item["sentiment"].lower() for item in predicted}
            expected = case["expectedAspects"]
            row["isolatedAbsa"] = actual_aspects
            row["expectedAspectMatches"] = {key:actual_aspects.get(key) == value for key,value in expected.items()}
            row["extraAspects"] = sorted(set(actual_aspects) - set(expected))
        rows.append(row)
    output = ROOT / "reports/generated"
    output.mkdir(parents=True, exist_ok=True)
    report = {"endpoint":args.endpoint, "note":source["purpose"], "no_exact_training_text_overlap":True,
        "quality_matches":sum(row["qualityMatch"] for row in rows), "case_count":len(rows), "cases":rows}
    (output / "capability_challenge_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Development model capability check", "", source["purpose"], "",
        f"Quality labels matched {report['quality_matches']} of {len(rows)} provisional expectations.", "",
        "| Case | Capability | Expected | Actual | Match |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['capability']} | {row['expectedLabel']} | {row['actualLabel'] or 'ERROR'} | {'Yes' if row['qualityMatch'] else 'No'} |")
    lines += ["", "## Isolated ABSA diagnostic", "", "These calls bypass the quality gate for diagnosis only. Normal API behavior still applies the gate.", ""]
    for row in rows:
        if "isolatedAbsa" in row:
            lines += [f"### {row['id']}", "", f"Expected: `{json.dumps(row['expectedAspects'])}`", "",
                f"Predicted: `{json.dumps(row['isolatedAbsa'])}`", ""]
        if row["error"]:
            lines += [f"Error for {row['id']}: {row['error']}", ""]
    lines += ["This compact model was randomly initialized and trained for one epoch on draft/synthetic data. Image capability is not tested. These results are not thesis performance evidence."]
    (output / "capability_challenge_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"quality_matches":report["quality_matches"],"cases":len(rows),"errors":sum(bool(row["error"]) for row in rows),"report":str(output / "capability_challenge_results.md")}))


if __name__ == "__main__":
    main()
