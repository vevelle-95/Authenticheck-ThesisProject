"""Train an offline, compact smoke-test bundle; never use it for thesis accuracy."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel, set_seed

from model_contract import ASPECTS, INPUT_VERSION, SEED, aspect_prompt, quality_input_text
from model_data import load_reviews, prepare_splits
from stage1.generate_oof_features import generate_oof
from stage1.train_roberta import fit_roberta
from stage1.extract_6d_features import build_features
from stage1.features import FeatureExtractor
from stage2.fine_tune_absa import train_absa
from stage2.train_xgboost import train_classifier
from stage2.online_inference import OnlineInference


def main():
    torch.set_num_threads(2)
    set_seed(SEED)
    work = ROOT / "data" / "development_v3"
    bundle = ROOT / "models" / "development_v3"
    if (bundle / "development_manifest.json").exists():
        raise SystemExit("Development bundle already exists; refusing to overwrite it.")
    work.mkdir(parents=True, exist_ok=True)
    frame = load_reviews(ROOT / "data" / "test_reviews.csv", require_annotations=True)
    # Explicitly omit media for this offline software test.
    frame["review_image_urls"] = ""
    frame["image_urls"] = [[] for _ in range(len(frame))]
    assigned = prepare_splits(frame, work / "splits.json")
    print(assigned.partition.value_counts().to_string(), flush=True)
    training = assigned[assigned.partition.eq("train")]
    validation = assigned[assigned.partition.eq("validation")]
    words = set()
    for text in [*(quality_input_text(row) for row in training.to_dict("records")),
                 *(aspect_prompt(category) for category in ASPECTS)]:
        words.update(re.findall(r"\S+", text))
    vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocabulary.update({word: index + 4 for index, word in enumerate(sorted(words - set(vocabulary)))})
    base = work / "compact_encoder"
    tokenizer_impl = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = WhitespaceSplit()
    tokenizer_impl.post_processor = TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, pad_token="<pad>",
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", model_max_length=128)
    tokenizer.save_pretrained(base)
    RobertaModel(RobertaConfig(vocab_size=len(vocabulary), hidden_size=32,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=256, type_vocab_size=1,
        pad_token_id=0, bos_token_id=1, eos_token_id=2)).save_pretrained(base)
    oof = work / "oof.csv"
    generate_oof(assigned, oof, bundle / "oof", base_model=str(base), epochs=1, batch_size=16)
    fit_roberta(training, validation, bundle / "dost_roberta", base_model=str(base), epochs=1, batch_size=16)
    features = build_features(assigned, oof, work / "features.csv", bundle / "dost_roberta")
    train_classifier(features, assigned, bundle / "xgboost_meta_classifier.json")
    train_absa(assigned, bundle / "absa_model", encoder=str(base), epochs=1, batch_size=16)
    pipe = OnlineInference(bundle / "dost_roberta", bundle / "xgboost_meta_classifier.json", bundle / "absa_model", development_mode=True)
    pipe.extractor = FeatureExtractor(bundle / "dost_roberta", image_loader=lambda _url: None)
    pipe.check_artifacts()
    smoke = pipe.run(validation.head(4))
    report = {
        "model_version": "development-compact-v3",
        "input_version": INPUT_VERSION,
        "quality_text_input": "product_title + product_description + review_text",
        "absa_text_input": "review_text",
        "purpose": "Software smoke test only; not a pretrained DOST model or thesis result",
        "source": "data/test_reviews.csv (draft and synthetic annotations)",
        "image_inputs": "omitted; visual similarity is zero",
        "epochs": 1, "hidden_size": 32,
        "partitions": assigned.partition.value_counts().to_dict(),
        "test_partition_evaluated": False,
        "smoke_response": smoke,
    }
    (bundle / "development_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Development bundle trained and inference checked: {bundle}", flush=True)


if __name__ == "__main__":
    main()
