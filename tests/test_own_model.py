"""Offline contract, leakage, category, and aggregation regressions for our model."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BatchEncoding

from model_contract import (
    ASPECTS, CLASS_NAMES, FEATURE_COLUMNS, INPUT_VERSION, map_quality, normalize_rating,
    parse_annotations, parse_image_urls,
)
from model_data import fingerprint, load_reviews, prepare_splits, read_splits
from stage1.generate_oof_features import generate_oof
from stage1.features import FeatureExtractor
from stage2.absa_model import ABSAHeadModel, tokenize_sentiment
from stage2.fine_tune_absa import build_targets, detection_metrics, select_threshold
from stage2.online_inference import OnlineInference, aggregate_products, build_summary
from stage2.train_xgboost import validate_features


def fixture_frame(products=20):
    rows = []
    for product in range(products):
        for label in range(4):
            text = f"product {product} class {label} quality value"
            rows.append({
                "review_id": f"r{product}-{label}", "product_id": f"p{product}",
                "review_text": text, "review_image_urls": "https://example.invalid/photo.jpg",
                "star_rating": 5, "ground_truth": CLASS_NAMES[label], "product_category": "general",
                "aspect_annotations": json.dumps([{
                    "category": "product_quality", "text": "quality", "sentiment": product % 3,
                }]) if label == 0 else "[]",
            })
    return pd.DataFrame(rows)


def validated_fixture(directory, products=20):
    path = Path(directory) / "reviews.csv"
    fixture_frame(products).to_csv(path, index=False)
    return load_reviews(path, require_annotations=True)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, first, second=None, **kwargs):
        self.calls.append((first, second, kwargs))
        ids = [ASPECTS.index(text.split(":")[0]) + 2 if second is not None else 1 for text in first]
        return BatchEncoding({
            "input_ids": torch.tensor([[value, 1, 0] for value in ids]),
            "attention_mask": torch.tensor([[1, 1, 0] for _ in ids]),
        })


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(20, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ContractTests(unittest.TestCase):
    def test_identifiers_preserve_leading_zeroes(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture_frame(1)
            frame["review_id"] = ["0001", "0002", "0003", "0004"]
            frame["product_id"] = "001"
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            loaded = load_reviews(path)
            self.assertEqual(loaded.review_id.tolist(), ["0001", "0002", "0003", "0004"])
            self.assertEqual(set(loaded.product_id), {"001"})

    def test_taxonomy_order_and_strict_labels(self):
        self.assertEqual(len(ASPECTS), 10)
        self.assertEqual(ASPECTS[5], "sensory_experience")
        self.assertEqual(map_quality("vague"), 2)
        for value in (1.5, "nan", None, -1):
            with self.assertRaises(ValueError):
                map_quality(value)

    def test_rating_does_not_impute_or_clamp(self):
        self.assertEqual(normalize_rating(1), 0)
        self.assertEqual(normalize_rating(5), 1)
        for value in (None, "", float("nan"), 0, 6, 2.5):
            with self.assertRaises(ValueError):
                normalize_rating(value)

    def test_image_formats_preserve_all_matched_urls(self):
        self.assertEqual(parse_image_urls('["a", "b", "a"]'), ["a", "b"])
        self.assertEqual(parse_image_urls("a | b"), ["a", "b"])
        with self.assertRaises(ValueError):
            parse_image_urls("[3]")

    def test_annotations_keep_category_evidence_and_reject_conflicts(self):
        annotations = [
            {"category": "performance", "text": "sound", "sentiment": "positive"},
            {"category": "performance", "text": "battery", "sentiment": "positive"},
        ]
        result = parse_annotations(annotations, "sound and battery")
        self.assertEqual(result[0]["evidence"], ["sound", "battery"])
        annotations[1]["sentiment"] = "negative"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            parse_annotations(annotations, "sound and battery")

    def test_sensory_restriction_and_unknown_category(self):
        annotation = [{"category": "sensory_experience", "text": "smooth", "sentiment": "neutral"}]
        with self.assertRaisesRegex(ValueError, "product_category"):
            parse_annotations(annotation, "smooth", "electronics")
        self.assertEqual(len(parse_annotations(annotation, "smooth", "skincare")), 1)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            parse_annotations([{"category": "delivery", "text": "fast", "sentiment": 2}], "fast")

    def test_dataset_validator_requires_ids_but_not_a_large_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture_frame(1)
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            self.assertEqual(len(load_reviews(path, require_annotations=True)), 4)
            frame.drop(columns="review_id").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "review_id"):
                load_reviews(path)


class SplitAndOofTests(unittest.TestCase):
    def test_persisted_partitions_and_folds_are_product_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = validated_fixture(directory)
            path = Path(directory) / "splits.json"
            assigned = prepare_splits(frame, path)
            self.assertEqual(assigned.partition.value_counts().to_dict(), {"train": 56, "test": 12, "validation": 12})
            self.assertEqual(assigned.groupby("product_id").partition.nunique().max(), 1)
            train = assigned[assigned.partition.eq("train")]
            self.assertEqual(train.groupby("product_id").fold.nunique().max(), 1)
            pd.testing.assert_frame_equal(
                assigned.sort_values("review_id").reset_index(drop=True),
                read_splits(frame.sample(frac=1), path).sort_values("review_id").reset_index(drop=True),
            )
            self.assertEqual(fingerprint(frame), fingerprint(frame.sample(frac=1)))
            self.assertEqual(len(read_splits(frame, path)), len(frame))
            frame.loc[0, "review_text"] += " changed"
            with self.assertRaisesRegex(ValueError, "match"):
                read_splits(frame, path)

    def test_oof_cross_fits_only_training_records(self):
        with tempfile.TemporaryDirectory() as directory:
            assigned = prepare_splits(validated_fixture(directory), Path(directory) / "split.json")
            calls = []
            def fit(frame, validation, output, **kwargs):
                self.assertIsNone(validation)
                self.assertTrue(frame.partition.eq("train").all())
                calls.append(set(frame.review_id))
                return {"ids": set(frame.review_id), "products": set(frame.product_id)}, None
            def predict(model, tokenizer, held):
                self.assertFalse(model["ids"] & set(held.review_id))
                self.assertFalse(model["products"] & set(held.product_id))
                return np.tile([0.4, 0.3, 0.2, 0.1], (len(held), 1))
            result = generate_oof(assigned, Path(directory) / "oof.csv", Path(directory) / "folds", fit=fit, predict=predict)
            self.assertEqual(len(calls), 5)
            self.assertEqual(set(result.review_id), set(assigned[assigned.partition.eq("train")].review_id))
            self.assertEqual(result.review_id.nunique(), len(result))

    def test_xgboost_rejects_in_sample_or_test_features(self):
        with tempfile.TemporaryDirectory() as directory:
            assigned = prepare_splits(validated_fixture(directory), Path(directory) / "split.json")
            features = assigned[assigned.partition.ne("test")][["review_id", "product_id", "partition", "fold"]].copy()
            features["ground_truth"] = assigned.loc[features.index, "label"]
            features["probability_source"] = features.apply(
                lambda row: f"oof_fold_{row.fold}" if row.partition == "train" else "final_roberta", axis=1,
            )
            for name, value in zip(FEATURE_COLUMNS, [0.4, 0.3, 0.2, 0.1, 0.5, 1]):
                features[name] = value
            self.assertEqual(len(validate_features(features, assigned)), len(features))
            features.loc[features.partition.eq("train"), "probability_source"] = "final_roberta"
            with self.assertRaisesRegex(ValueError, "OOF"):
                validate_features(features, assigned)

    def test_invalid_image_cannot_be_replaced_with_zero_similarity(self):
        extractor = FeatureExtractor("unused", image_loader=lambda url: None)
        with self.assertRaisesRegex(ValueError, "No accessible"):
            extractor.extract("valid review", ["https://example.invalid/a"], 5, [0.4, 0.3, 0.2, 0.1])


class AbsaTests(unittest.TestCase):
    def test_ten_outputs_and_product_context_gate(self):
        model = ABSAHeadModel(encoder=FakeEncoder(), dropout=0)
        model.aspect_head.weight.data.zero_()
        model.aspect_head.bias.data.fill_(10)
        results = model.predict(["review", "review"], FakeTokenizer(), ["electronics", "skincare"])
        self.assertEqual(len(results[0]), 9)
        self.assertEqual(len(results[1]), 10)
        self.assertNotIn("sensory_experience", {row["category"] for row in results[0]})
        self.assertTrue(all(row["category"] in ASPECTS for row in results[1]))

    def test_sentiment_encoding_explicitly_receives_category_and_review(self):
        tokenizer = FakeTokenizer()
        tokenize_sentiment(tokenizer, ["good sound", "comfortable"], ["performance", "design"], 128)
        first, second, kwargs = tokenizer.calls[-1]
        self.assertTrue(first[0].startswith("performance:"))
        self.assertTrue(first[1].startswith("design:"))
        self.assertEqual(second, ["good sound", "comfortable"])
        self.assertEqual(kwargs["truncation"], "only_second")

    def test_targets_use_categories_not_phrase_lengths(self):
        frame = pd.DataFrame([{
            "review_text": "clear sound and battery lasts all day", "product_category": "electronics",
            "annotations": [{"category": "performance", "sentiment": 2, "evidence": ["clear sound", "battery lasts all day"]}],
        }])
        dataset = build_targets(frame, FakeTokenizer(), 128)
        targets, sentiments = dataset.tensors[2], dataset.tensors[5]
        self.assertEqual(targets.shape, (1, 10))
        self.assertEqual(int(targets.sum()), 1)
        self.assertEqual(sentiments[0, ASPECTS.index("performance")], 2)
        self.assertEqual(int(sentiments.ne(-100).sum()), 1)

    def test_validation_threshold_is_selected_from_probabilities(self):
        targets = np.zeros((2, 10))
        targets[0, 0] = 1
        probabilities = np.full((2, 10), 0.1)
        probabilities[0, 0] = 0.4
        eligible = np.ones((2, 10))
        selected = select_threshold(targets, probabilities, eligible)
        self.assertLessEqual(selected, 0.4)
        self.assertGreater(detection_metrics(targets, probabilities, eligible, selected)["macro_f1"],
                           detection_metrics(targets, probabilities, eligible, 0.5)["macro_f1"])


class InferenceTests(unittest.TestCase):
    def test_only_predicted_authentic_reviews_reach_absa(self):
        class StubPipeline(OnlineInference):
            def check_artifacts(self):
                pass
            def extract_features(self, description, text, images, rating):
                return [0.4, 0.3, 0.2, 0.1, 0.5, 1], [0.4, 0.3, 0.2, 0.1], 0.5, 1, "image"
            def classify(self, features):
                value = next(self.verdicts)
                return value, [1 if i == value else 0 for i in range(4)]
            def aspect_sentiment(self, text, product_category=""):
                self.calls.append(text)
                return [{"category": "performance", "sentiment": "Positive"}]
        pipe = StubPipeline("unused", "unused", "unused")
        pipe.verdicts = iter([0, 1, 2, 3])
        pipe.calls = []
        frame = pd.DataFrame({"review_id": ["a", "b", "c", "d"], "product_id": ["p"] * 4,
                              "text": ["first", "second", "third", "fourth"], "star_rating": [5] * 4})
        result = pipe.run(frame)
        self.assertEqual(pipe.calls, ["first"])
        self.assertEqual(result["authenticShare"], 25)
        self.assertEqual(result["aggregateSentiment"], 1)
        self.assertEqual([row["id"] for row in result["reviews"]], ["a", "b", "c", "d"])

    def test_aggregation_uses_polarity_and_preserves_undefined_products(self):
        rows = [
            {"product_id": "a", "label": "authentic", "starRating": 5,
             "aspectSentiment": [{"category": "performance", "sentiment": "Negative"}]},
            {"product_id": "a", "label": "authentic", "starRating": 5,
             "aspectSentiment": [{"category": "design", "sentiment": "Neutral"}]},
            {"product_id": "b", "label": "deceptive", "starRating": 5, "aspectSentiment": []},
        ]
        result = aggregate_products(rows)
        self.assertEqual(result[0]["aggregateSentiment"], -0.5)
        self.assertIsNone(result[1]["aggregateSentiment"])
        summary = build_summary(pd.DataFrame(rows), ["authentic", "authentic", "deceptive"], rows)
        self.assertEqual(summary["verifiedRating"], 5)
        self.assertEqual(summary["aggregateSentiment"], -0.5)


if __name__ == "__main__":
    unittest.main()
