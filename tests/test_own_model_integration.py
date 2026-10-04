"""Offline miniature training integration: real encoders/XGBoost, mocked image embeddings."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from model_contract import ASPECTS, SENSORY_POLICY, SENTIMENT_TARGET_POLICY, aspect_prompt, parse_annotations
from model_data import prepare_splits
from stage1.train_roberta import fit_roberta
from stage1.generate_oof_features import generate_oof
from stage1.extract_6d_features import build_features
from stage1.features import FeatureExtractor
from stage2.absa_model import ABSAHeadModel
from stage2.fine_tune_absa import train_absa
from stage2.train_xgboost import train_classifier
from stage2.online_inference import OnlineInference
from stage2.evaluate_models import check_training_provenance, evaluate_own_models

from test_own_model import validated_fixture


def create_tiny_encoder(directory):
    words = set("product class quality value".split())
    for category in ASPECTS:
        words.update(aspect_prompt(category).split())
    vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocabulary.update({word: index + 4 for index, word in enumerate(sorted(words))})
    token = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    token.pre_tokenizer = WhitespaceSplit()
    token.post_processor = TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=token, pad_token="<pad>", bos_token="<s>",
                                       eos_token="</s>", unk_token="<unk>", model_max_length=128)
    tokenizer.save_pretrained(directory)
    config = RobertaConfig(
        vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=256, type_vocab_size=1,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    RobertaModel(config).save_pretrained(directory)
    return tokenizer


class OfflineTrainingIntegrationTests(unittest.TestCase):
    def test_training_round_trip_and_held_out_evaluation(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    base, bundle = root / "base", root / "bundle"
                    tokenizer = create_tiny_encoder(base)
                    frame = validated_fixture(root)
                    mixed = [
                        {"category": "sensory_experience", "text": "quality", "sentiment": "positive"},
                        {"category": "sensory_experience", "text": "value", "sentiment": "negative"},
                    ]
                    frame.at[0, "aspect_annotations"] = json.dumps(mixed)
                    frame.at[0, "annotations"] = parse_annotations(mixed, frame.at[0, "review_text"])
                    frame.at[1, "review_image_urls"] = ""
                    frame.at[1, "image_urls"] = []
                    assigned = prepare_splits(frame, root / "splits.json")
                    oof_path = root / "oof.csv"
                    generate_oof(assigned, oof_path, root / "folds", base_model=str(base), epochs=1, batch_size=16)
                    fit_roberta(
                        assigned[assigned.partition.eq("train")], assigned[assigned.partition.eq("validation")],
                        bundle / "dost_roberta", base_model=str(base), epochs=1, batch_size=16,
                    )

                    class OfflineFeatures(FeatureExtractor):
                        def visual_features(self, text, urls):
                            if not urls:
                                return super().visual_features(text, urls)
                            return 0.25, urls[0], [(0.25, urls[0])]

                    extractor = OfflineFeatures(bundle / "dost_roberta")
                    features = build_features(assigned, oof_path, root / "features.csv",
                                              bundle / "dost_roberta", extractor=extractor)
                    self.assertNotIn("test", set(features.partition))
                    train_classifier(features, assigned, bundle / "xgboost_meta_classifier.json")
                    train_absa(assigned, bundle / "absa_model", encoder=str(base), epochs=1, batch_size=4)
                    with patch("stage2.absa_model.AutoModel.from_pretrained", side_effect=AssertionError("Saved ABSA must not fetch an external encoder")):
                        reloaded = ABSAHeadModel.from_pretrained(bundle / "absa_model")
                    self.assertEqual(reloaded.aspect_head.out_features, 10)
                    self.assertTrue(0 < reloaded.threshold < 1)

                    from transformers import AutoTokenizer
                    saved_tokenizer = AutoTokenizer.from_pretrained(reloaded.tokenizer_dir, local_files_only=True)
                    predictions = reloaded.predict(["quality value"], saved_tokenizer)
                    self.assertTrue(all(row["category"] in ASPECTS for row in predictions[0]))
                    config_path = bundle / "absa_model" / "absa_config.json"
                    saved_config = json.loads(config_path.read_text())
                    self.assertEqual(saved_config["sensory_domain_policy"], SENSORY_POLICY)
                    self.assertEqual(saved_config["sentiment_target_policy"], SENTIMENT_TARGET_POLICY)
                    old_config = {key: value for key, value in saved_config.items() if key != "sensory_domain_policy"}
                    config_path.write_text(json.dumps(old_config))
                    with self.assertRaisesRegex(ValueError, "sensory policy"):
                        ABSAHeadModel.from_pretrained(bundle / "absa_model")
                    config_path.write_text(json.dumps(saved_config))
                    long_features, _, _ = extractor.extract("quality " * 500, ["unused"], 5)
                    self.assertEqual(len(long_features), 6)

                    check_training_provenance(assigned, bundle)
                    pipe = OnlineInference(bundle / "dost_roberta", bundle / "xgboost_meta_classifier.json", bundle / "absa_model")
                    pipe.extractor = extractor
                    test = assigned[assigned.partition.eq("test")]
                    output = evaluate_own_models(test, pipe)
                    self.assertEqual(output["review_count"], 12)
                    self.assertEqual(output["partition"], "test")
                    self.assertEqual(len(output["filtering_ablation"]), 3)
                    self.assertEqual(len(output["quality_xgboost"]["support"]), 4)
                    self.assertEqual(len(output["gold_category_sentiment"]["support"]), 3)
                    json.dumps(output, allow_nan=False)
                    pipeline_output = pipe.run(test)
                    self.assertEqual(pipeline_output["reviewCount"], len(test))
                    self.assertTrue(np.isfinite(pipeline_output["authenticShare"]))
                    missing_image = pipe.run(frame.iloc[[1]])["reviews"][0]
                    self.assertEqual(missing_image["features"]["s_clip"], 0)
                    self.assertIsNone(missing_image["bestImageUrl"])

                    # Reusing OOF probabilities from in-sample predictions must fail,
                    # even if the feature table itself still has the right row count.
                    metadata_path = oof_path.with_suffix(".metadata.json")
                    metadata = json.loads(metadata_path.read_text())
                    metadata["folds"][0]["fit_review_ids"].extend(metadata["folds"][0]["predicted_review_ids"])
                    metadata_path.write_text(json.dumps(metadata))
                    with self.assertRaisesRegex(ValueError, "provenance"):
                        build_features(assigned, oof_path, root / "bad.csv", bundle / "dost_roberta", extractor=extractor)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
