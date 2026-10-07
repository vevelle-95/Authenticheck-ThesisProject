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
    parse_annotations, parse_image_urls, quality_input_text,
)
from model_data import fingerprint, load_reviews, prepare_splits, read_splits
from stage1.generate_oof_features import generate_oof
from stage1.train_roberta import predict_probabilities
from stage1.features import FeatureExtractor
from stage2.absa_model import ABSAHeadModel, split_evidence_segments, tokenize_sentiment
from stage2.fine_tune_absa import batch_forward, build_targets, detection_metrics, select_threshold
from stage2.online_inference import OnlineInference, aggregate_products, build_summary
from stage2.train_xgboost import validate_features


def fixture_frame(products=20):
    rows = []
    for product in range(products):
        for label in range(4):
            text = f"product {product} class {label} quality value"
            rows.append({
                "review_id": f"r{product}-{label}", "product_id": f"p{product}",
                "product_title": f"Product {product}", "product_description": "Advertised quality and value",
                "review_text": text, "review_image_urls": "https://example.invalid/photo.jpg",
                "star_rating": 5, "ground_truth": CLASS_NAMES[label],
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
    def test_quality_context_uses_listing_text_but_never_targets(self):
        row = {
            "product_title": "  Phone\n case  ", "product_description": "Matte   finish",
            "review_text": "  Matibay\tang casing  ", "ground_truth": "deceptive",
            "text_label": "irrelevant", "aspect_annotations": "positive", "star_rating": 5,
        }
        self.assertEqual(quality_input_text(row), "Phone case Matte finish Matibay ang casing")
        self.assertEqual(quality_input_text({"text": "Matibay."}), "Matibay.")
        self.assertEqual(quality_input_text({"prod_title": "Case", "prod_description": "", "text": "Matibay."}),
                         "Case Matibay.")
        self.assertEqual(quality_input_text({"product_title": float("nan"), "product_description": None,
                                            "review_text": "Matibay."}), "Matibay.")
        with self.assertRaisesRegex(ValueError, "Review text"):
            quality_input_text({"product_title": "A title", "review_text": " "})

    def test_quality_prediction_batches_include_product_context(self):
        class QualityEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(review_max_length=128)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=torch.zeros(len(input_ids), 4))

        tokenizer = FakeTokenizer()
        frame = fixture_frame(1)
        probabilities = predict_probabilities(QualityEncoder(), tokenizer, frame, batch_size=2)
        encoded = [text for first, _, _ in tokenizer.calls for text in first]
        self.assertEqual(encoded, [quality_input_text(row) for row in frame.to_dict("records")])
        self.assertTrue(all(text.startswith("Product 0 Advertised quality and value ") for text in encoded))
        self.assertEqual(probabilities.shape, (4, 4))

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

    def test_annotations_preserve_same_category_mixed_sentiments(self):
        annotations = [
            {"category": "performance", "text": "sound", "sentiment": "positive"},
            {"category": "performance", "text": "battery", "sentiment": "positive"},
        ]
        result = parse_annotations(annotations, "sound and battery")
        self.assertEqual(result[0]["evidence"], ["sound", "battery"])
        annotations[1]["sentiment"] = "negative"
        result = parse_annotations(annotations, "sound and battery")
        self.assertEqual(len(result), 2)
        self.assertEqual({entry["sentiment"] for entry in result}, {0, 2})
        self.assertEqual({text for entry in result for text in entry["evidence"]}, {"sound", "battery"})

    def test_evidence_can_paraphrase_review_text(self):
        annotation = [{"category": "performance", "text": "scroll wheel does not work", "sentiment": "negative"}]
        result = parse_annotations(annotation, "the middle is kinda like ain't working")
        self.assertEqual(result[0]["evidence"], ["scroll wheel does not work"])

    def test_sensory_annotations_need_no_product_category(self):
        annotation = [{"category": "sensory_experience", "text": "smooth", "sentiment": "neutral"}]
        expected = parse_annotations(annotation, "smooth")
        self.assertEqual(expected[0]["category"], "sensory_experience")
        for legacy_category in ("", "electronics", "skincare"):
            self.assertEqual(parse_annotations(annotation, "smooth", legacy_category), expected)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            parse_annotations([{"category": "delivery", "text": "fast", "sentiment": 2}], "fast")

    def test_csv_with_sensory_annotations_loads_without_product_category(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture_frame(1)
            frame.loc[0, "aspect_annotations"] = json.dumps([
                {"category": "sensory_experience", "text": "quality", "sentiment": "positive"},
            ])
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            loaded = load_reviews(path, require_annotations=True)
            self.assertNotIn("product_category", loaded.columns)
            self.assertEqual(loaded.loc[0, "annotations"][0]["category"], "sensory_experience")

    def test_dataset_validator_requires_ids_but_not_a_large_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture_frame(1)
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            self.assertEqual(len(load_reviews(path, require_annotations=True)), 4)
            frame.drop(columns="review_id").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "review_id"):
                load_reviews(path)

    def test_reviews_without_images_remain_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture_frame(1)
            frame.loc[0, "review_image_urls"] = ""
            frame.loc[1, "review_image_urls"] = "[]"
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            loaded = load_reviews(path, require_annotations=True)
            self.assertEqual(loaded.loc[0, "image_urls"], [])
            self.assertEqual(loaded.loc[1, "image_urls"], [])

    def test_duplicate_text_keeps_records_but_ids_must_still_be_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture_frame(2)
            frame.loc[4, "review_text"] = frame.loc[0, "review_text"]
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            loaded = load_reviews(path, require_annotations=True)
            self.assertEqual(len(loaded), len(frame))
            self.assertEqual(loaded.loc[0, "review_text"], loaded.loc[4, "review_text"])
            frame.loc[4, "review_id"] = frame.loc[0, "review_id"]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "uniquely"):
                load_reviews(path)


class SplitAndOofTests(unittest.TestCase):
    def test_duplicate_links_and_transitive_products_stay_together(self):
        with tempfile.TemporaryDirectory() as directory:
            source = fixture_frame(30)
            for product in range(0, 30, 2):
                source.loc[product * 4, "review_text"] = f"Shared review {product}"
                source.loc[(product + 1) * 4, "review_text"] = f"SHARED  review\n{product}"
            # Products 0/1 and 2/3 are also linked through a different review.
            source.loc[5, "review_text"] = "another shared opinion"
            source.loc[9, "review_text"] = "Another\tshared opinion"
            csv_path = Path(directory) / "reviews.csv"
            source.to_csv(csv_path, index=False)
            frame = load_reviews(csv_path, require_annotations=True)
            path = Path(directory) / "splits.json"
            assigned = prepare_splits(frame, path)
            self.assertEqual(len(assigned), len(source))
            linked = assigned[assigned.product_id.isin(["p0", "p1", "p2", "p3"])]
            self.assertEqual(linked.partition.nunique(), 1)
            self.assertEqual(linked.fold.nunique(), 1)
            keys = assigned.review_text.map(lambda text: " ".join(text.split()).casefold())
            self.assertEqual(assigned.groupby(keys).partition.nunique().max(), 1)
            training = assigned[assigned.partition.eq("train")]
            self.assertTrue(keys.loc[training.index].duplicated().any())
            self.assertEqual(training.groupby(keys.loc[training.index]).fold.nunique().max(), 1)
            pd.testing.assert_frame_equal(
                assigned.sort_values("review_id").reset_index(drop=True),
                read_splits(frame.sample(frac=1), path).sort_values("review_id").reset_index(drop=True),
            )

            def fit(fitting, validation, output, **kwargs):
                return set(" ".join(text.split()).casefold() for text in fitting.review_text), None

            def predict(seen_text, tokenizer, held):
                held_text = {" ".join(text.split()).casefold() for text in held.review_text}
                self.assertFalse(seen_text & held_text)
                return np.tile([0.4, 0.3, 0.2, 0.1], (len(held), 1))

            result = generate_oof(assigned, Path(directory) / "oof.csv", Path(directory) / "folds",
                                  fit=fit, predict=predict)
            self.assertEqual(set(result.review_id), set(training.review_id))

    def test_saved_splits_reject_duplicate_text_across_partitions_or_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            base = validated_fixture(directory)
            initial = prepare_splits(base, Path(directory) / "initial.json")
            for leak in ("partitions", "OOF folds"):
                with self.subTest(leak=leak):
                    assigned = initial.copy()
                    training = assigned[assigned.partition.eq("train")]
                    first = training.index[0]
                    if leak == "partitions":
                        second = assigned.index[assigned.partition.eq("validation")][0]
                    else:
                        second = training.index[training.fold.ne(assigned.loc[first, "fold"])][0]
                    assigned.loc[second, "review_text"] = "  " + assigned.loc[first, "review_text"].upper() + "  "
                    payload = {
                        "version": "product-70-15-15-oof5-v1", "dataset_sha256": fingerprint(assigned),
                        "records": assigned[["review_id", "product_id", "partition", "fold"]].to_dict("records"),
                    }
                    path = Path(directory) / "leaking.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Duplicate review text leakage between " + leak):
                        read_splits(assigned.drop(columns=["partition", "fold"]), path)

    def test_duplicate_links_cannot_fall_back_to_row_splitting(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = validated_fixture(directory)
            frame.loc[frame.label.eq(0), "review_text"] = "same opinion across every product"
            path = Path(directory) / "invalid.json"
            with self.assertRaisesRegex(ValueError, "independent product/duplicate-text groups"):
                prepare_splits(frame, path)
            self.assertFalse(path.exists())

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

    def test_missing_or_unusable_images_produce_zero_without_loading_clip(self):
        extractor = FeatureExtractor("unused", image_loader=lambda url: None)
        for images in ([], "", None, ["https://example.invalid/a"]):
            features, best, scores = extractor.extract("valid review", images, 5, [0.4, 0.3, 0.2, 0.1])
            self.assertEqual(features, [0.4, 0.3, 0.2, 0.1, 0.0, 1.0])
            self.assertIsNone(best)
            self.assertEqual(scores, [])
            self.assertIsNone(extractor.clip_text)


class AbsaTests(unittest.TestCase):
    def test_all_ten_outputs_are_available_without_product_metadata(self):
        model = ABSAHeadModel(encoder=FakeEncoder(), dropout=0)
        model.aspect_head.weight.data.zero_()
        model.aspect_head.bias.data.fill_(10)
        results = model.predict(["review", "review"], FakeTokenizer())
        self.assertTrue(all(len(row) == 10 for row in results))
        self.assertIn("sensory_experience", {row["category"] for row in results[0]})
        self.assertTrue(all(row["category"] in ASPECTS for row in results[1]))
        legacy = model.predict(["review", "review"], FakeTokenizer(), ["electronics", "skincare"])
        self.assertEqual(legacy, results)

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
            "review_text": "clear sound and battery lasts all day",
            "annotations": [{"category": "performance", "sentiment": 2, "evidence": ["clear sound", "battery lasts all day"]}],
        }])
        dataset = build_targets(frame, FakeTokenizer(), 128)
        targets, sentiments = dataset.tensors[2], dataset.tensors[5]
        self.assertEqual(targets.shape, (1, 10))
        self.assertEqual(int(targets.sum()), 1)
        self.assertEqual(sentiments[0, ASPECTS.index("performance"), 2], 1)
        self.assertEqual(int(sentiments.gt(0).sum()), 1)

    def test_sensory_targets_are_included_without_category_metadata(self):
        frame = pd.DataFrame([{
            "review_text": "smooth ang texture",
            "annotations": [{"category": "sensory_experience", "sentiment": 2, "evidence": ["smooth ang texture"]}],
        }])
        dataset = build_targets(frame, FakeTokenizer(), 128)
        sensory = ASPECTS.index("sensory_experience")
        self.assertEqual(dataset.tensors[2][0, sensory], 1)
        self.assertEqual(dataset.tensors[5][0, sensory, 2], 1)
        self.assertTrue(torch.equal(dataset.tensors[6], torch.ones(1, 10)))

    def test_mixed_category_targets_contribute_both_polarities_to_loss(self):
        annotations = parse_annotations([
            {"category": "sensory_experience", "text": "mabango", "sentiment": "positive"},
            {"category": "sensory_experience", "text": "watery", "sentiment": "negative"},
        ], "mabango pero watery")
        frame = pd.DataFrame([{"review_text": "mabango pero watery", "annotations": annotations}])
        dataset = build_targets(frame, FakeTokenizer(), 128)
        model = ABSAHeadModel(encoder=FakeEncoder(), dropout=0)
        _, _, _, polarities, _ = batch_forward(model, dataset.tensors, torch.device("cpu"))
        self.assertTrue(torch.equal(polarities, torch.tensor([[1.0, 0.0, 1.0]])))
        logits = torch.tensor([[1.0, -2.0, 0.0]], requires_grad=True)
        loss = model.compute_loss(torch.zeros(1, 10), logits, dataset.tensors[2], polarities, dataset.tensors[6])
        expected = np.log(2) - torch.log_softmax(logits, -1)[0, [0, 2]].mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertGreater(float(logits.grad[0, 1]), 0)

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


    def test_evidence_text_is_a_matching_review_span(self):
        model = ABSAHeadModel(encoder=FakeEncoder(), dropout=0)
        model.aspect_head.weight.data.zero_()
        model.aspect_head.bias.data.fill_(10)
        model.sentiment_head.weight.data.zero_()
        model.sentiment_head.bias.data.copy_(torch.tensor([3.0, 0.0, -3.0]))
        review = "Matibay ang build quality at maganda ang finish. Pero mabilis ma-drain ang battery."
        results = model.predict([review], FakeTokenizer())
        self.assertTrue(results[0])
        for item in results[0]:
            self.assertIn(item["text"], review)
            self.assertLess(len(item["text"]), len(review))
            self.assertEqual(item["sentiment"], "Negative")

    def test_evidence_span_follows_the_matching_category_clause(self):
        review = "Maganda ang design ng casing. Sira agad ang battery after one week."
        model = ABSAHeadModel(encoder=FakeEncoder(), dropout=0)

        def fake_scores(texts, tokenizer, batch_size=16):
            scores = torch.full((len(texts), len(ASPECTS)), 0.1)
            for index, text in enumerate(texts):
                if "battery" in text:
                    scores[index, ASPECTS.index("performance")] = 0.9
                if "design" in text:
                    scores[index, ASPECTS.index("design")] = 0.8
            return scores

        def fake_sentiment(texts, categories, tokenizer, batch_size=16):
            return [[0.9, 0.05, 0.05] if "battery" in text else [0.05, 0.05, 0.9] for text in texts]

        model.aspect_scores = fake_scores
        model.predict_sentiment = fake_sentiment
        results = model.predict([review], FakeTokenizer())
        by_category = {item["category"]: item for item in results[0]}
        self.assertEqual(by_category["design"]["text"], "Maganda ang design ng casing.")
        self.assertEqual(by_category["performance"]["text"], "Sira agad ang battery after one week.")
        self.assertEqual(by_category["performance"]["sentiment"], "Negative")
        self.assertTrue(all(item["text"] in review for item in results[0]))

    def test_run_on_review_splits_into_clause_segments(self):
        review = "Mura ang presyo pero mabilis masira ang produkto; hindi sulit ang bayad"
        segments = split_evidence_segments(review)
        self.assertEqual(len(segments), 3)
        self.assertTrue(all(segment in review for segment in segments))
        self.assertTrue(all(segment != review for segment in segments))

    def test_long_run_on_review_splits_on_commas(self):
        review = ("sobrang ganda ng produkto, mura pa siya kumpara sa ibang tindahan, "
                  "matibay ang ginawa kahit matagal na gamit, at nagamit ko na ito ng ilang buwan na")
        segments = split_evidence_segments(review)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(segment in review for segment in segments))

    def test_review_without_boundaries_keeps_the_whole_text(self):
        self.assertEqual(split_evidence_segments("  sobrang ganda at mura pa  "), ["sobrang ganda at mura pa"])


class InferenceTests(unittest.TestCase):
    def test_only_predicted_authentic_reviews_reach_absa(self):
        class StubPipeline(OnlineInference):
            def check_artifacts(self):
                pass
            def extract_features(self, text, images, rating):
                self.feature_texts.append(text)
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
        pipe.feature_texts = []
        frame = pd.DataFrame({"review_id": ["a", "b", "c", "d"], "product_id": ["p"] * 4,
                              "text": ["first", "second", "third", "fourth"], "star_rating": [5] * 4,
                              "product_title": ["Advertised product"] * 4,
                              "product_description": ["Fast and durable"] * 4})
        result = pipe.run(frame)
        self.assertEqual(pipe.calls, ["first"])
        self.assertEqual(pipe.feature_texts, [quality_input_text(row) for row in frame.to_dict("records")])
        self.assertEqual([row["text"] for row in result["reviews"]], frame.text.tolist())
        self.assertEqual(result["authenticShare"], 25)
        self.assertEqual(result["aggregateSentiment"], 1)
        self.assertEqual([row["id"] for row in result["reviews"]], ["a", "b", "c", "d"])
        pipe.development_mode = True
        pipe.verdicts = iter([0, 1, 2, 3])
        development = pipe.run(frame)
        self.assertIn("Compact development encoder", development["reviews"][0]["signals"][0])
        self.assertIn("images omitted", development["reviews"][0]["signals"][1])

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
