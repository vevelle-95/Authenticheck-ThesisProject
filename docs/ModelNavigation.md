# AuthentiCheck: start here

This is the main guide for our model. You do not need to run every Python file.

**Right now:** finish collecting and annotating your dataset. You do not need
to add a product category column.

## 1. What the model does

At prediction time:

1. RoBERTa reads the review and produces four review-quality probabilities.
2. CLIP calculates similarity between the review and its buyer photos.
3. The star rating is converted to a number from 0 to 1.
4. XGBoost uses those six numbers to predict Authentic, Deceptive, LIV, or Irrelevant.
5. Reviews predicted Authentic go to ABSA, which predicts aspect categories and sentiments.
6. The aspect sentiments are averaged per product: Positive = +1, Neutral = 0, Negative = -1.

ABSA means **aspect-based sentiment analysis**. It uses the
[ten agreed aspect categories](ASPECT_TAXONOMY.md).

XGBoost is part of the Stage 1 quality filter, although its training file is in
the `stage2/` folder. ABSA is trained independently from the pretrained DOST
encoder; it does not reuse the quality classifier's fine-tuned weights.

## 2. Which files matter?

| File | Job |
| --- | --- |
| [pipeline.py](../pipeline.py) | Runs the complete training and validation workflow. |
| [model_contract.py](../model_contract.py) | Defines shared labels, aspect categories, feature order, and accepted formats. |
| [model_data.py](../model_data.py) | Reads/checks the CSV and implements saved product-based splits. |
| [model_metrics.py](../model_metrics.py) | Calculates accuracy, precision, recall, F1, and confusion matrices. |
| [stage1/prepare_splits.py](../stage1/prepare_splits.py) | Command to check the dataset and create/read splits using `model_data.py`. |
| [stage1/generate_oof_features.py](../stage1/generate_oof_features.py) | Trains five fold models to create probabilities for unseen training reviews. |
| [stage1/train_roberta.py](../stage1/train_roberta.py) | Trains the final review-quality text model. |
| [stage1/features.py](../stage1/features.py) | Builds the six features in memory, using the same logic during training and prediction. |
| [stage1/extract_6d_features.py](../stage1/extract_6d_features.py) | Writes the training and validation feature CSV. |
| [stage2/train_xgboost.py](../stage2/train_xgboost.py) | Trains the final quality filter. |
| [stage2/absa_model.py](../stage2/absa_model.py) | Defines the aspect detector and category-conditioned sentiment model. |
| [stage2/fine_tune_absa.py](../stage2/fine_tune_absa.py) | Trains ABSA using human-labeled Authentic reviews. |
| [stage2/online_inference.py](../stage2/online_inference.py) | Loads trained models and predicts new reviews. |
| [stage2/predict_absa.py](../stage2/predict_absa.py) | Runs ABSA alone for diagnosis; skips the full quality filter. |
| [stage2/evaluate_models.py](../stage2/evaluate_models.py) | Measures the selected models on the held-out test partition. |

`model_data.py` contains the splitting logic; `prepare_splits.py` is one way to
run it. The pipeline also calls that logic directly.

## 3. What belongs in the dataset?

Keep one row per review.

| Column | Meaning |
| --- | --- |
| `review_id` | Unique, permanent local review ID. |
| `product_id` | Same ID for reviews from the same actual product listing. |
| `review_text` | Buyer-written review. |
| `review_image_urls` | Buyer-photo URLs as a JSON list or separated by `|`; leave empty if none. Missing/unusable images give a CLIP score of 0. |
| `star_rating` | Integer from 1 to 5. |
| `ground_truth` | Final human quality label: `authentic`, `deceptive`, `liv`, or `irrelevant`. |
| `aspect_annotations` | JSON list linking each category, evidence phrase, and sentiment. |

`product_title`, `product_description`, and `text_label` may stay as context/source
metadata. They are not learned inputs or training targets in the current model.
Separate `aspects`, `aspect_text`, and `sentiment` columns are not read by training.

For "Matibay ang casing pero mahal.", the annotation cell could contain:

```json
[
  {"category": "product_quality", "text": "Matibay ang casing", "sentiment": "Positive"},
  {"category": "value", "text": "mahal", "sentiment": "Negative"}
]
```

Use `[]` for a completed annotation with no applicable aspects. Evidence text may
paraphrase the review; exact character matching is not required. One category can
have multiple sentiments, such as positive fragrance and negative texture under
`sensory_experience`. The parser preserves each category/polarity group. Training
uses an equal-weight distribution over that category's distinct sentiments;
prediction still returns one sentiment per detected category.

The current CSV contains assistant draft annotations added for format testing.
Their provenance is in `reports/generated/csv_annotation_fill.json`; have human
annotators review them before using them for thesis evaluation.

For practice training, the file also includes 120 synthetic reviews for ten new
products. Added IDs start with `synthetic_review_` and `synthetic_prod_`; their
titles start with `[SYNTHETIC TEST]`. The combined file has 320 reviews across
14 products and saved five-fold assignments in `data/splits.json`. Generation
details are in `reports/generated/synthetic_reviews_added.json`. Results from
this mixed practice dataset are development checks.

`product_category` is not required or used by the model. All ten aspect categories,
including `sensory_experience`, are available during training and prediction.
Annotators still apply its beauty/skincare/personal-care definition using product
context. There is no automatic product-domain filter on predictions.

`data/test_reviews.csv` is a dataset-structure fixture. Its name does not make
it the held-out test partition; assignments come from `splits.json`.

## 4. Where do outputs go?

These are the defaults when using `pipeline.py`. Generated directories appear
when the relevant step runs.

```text
data/
  splits.json                         Review/product partition and fold assignments
  training_v2/
    oof_probabilities.csv             Four OOF probabilities per training review
    6d_features.csv                   Six features for training and validation

models/own_model_v2/
  oof/fold_0/ ... fold_4/              Five models used to generate OOF features
  dost_roberta/                       Final RoBERTa model and tokenizer
  xgboost_meta_classifier.json         Final XGBoost model
  absa_model/                         Final ABSA model, configuration, and tokenizer
  manifest.json                       Model bundle information

results/
  own_model_test.json                 Held-out performance report
```

Metadata files beside the outputs record training IDs, validation scores, and
settings. Normal prediction uses the three final models; it does not use the
five OOF models. Old weights outside `own_model_v2/` belong to the previous
architecture. See [models/README.md](../models/README.md) for artifact details.

Individual scripts can have different intermediate-output defaults. Use the
pipeline to keep the layout above.

## 5. What do I run when annotations are ready?

Open a terminal in `Authenticheck-ThesisProject` and activate the existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

In these examples, `data/reviews.csv` means your finalized dataset file. Replace
it with your actual filename.

**Check the dataset:**

```powershell
python pipeline.py --data data/reviews.csv --validate-only
```

**Train and validate:**

```powershell
python pipeline.py --data data/reviews.csv
```

This command saves product-based splits, generates OOF probabilities, trains
final RoBERTa, builds six features, trains XGBoost, and trains ABSA.

The split is approximately 70% training, 15% validation, and 15% testing.
Training teaches the models; validation selects checkpoints/settings; testing
measures the selected models on unseen products.

OOF means **out-of-fold**. Each training review gets probabilities from a
RoBERTa model trained on the other four folds. This prevents XGBoost from
learning from probabilities produced on reviews that RoBERTa already trained on.

**Evaluate the selected models:**

```powershell
python stage2/evaluate_models.py --data data/reviews.csv
```

This writes `results/own_model_test.json`. It reports quality classification,
aspect detection, sentiment classification, and product sentiment errors with
and without the quality filter. Final testing is separate from training.

**Predict new reviews:**

```powershell
python stage2/online_inference.py --data data/new_reviews.csv --output results/new_review_predictions.json
```

New reviews need text, rating, and IDs for identification and product grouping.
Images are optional; missing/unusable images produce a CLIP score of 0. Human
labels are not needed for prediction.

## 6. Do I need the tests folder?

Keep it, but you can ignore it during annotation and normal training.
It checks software behavior; it is separate from the thesis test partition.

- `test_own_model.py` checks labels, splits, features, ABSA, and aggregation.
- `test_own_model_integration.py` runs a small temporary training/prediction workflow.
- Other tests cover the backend, extension, or deferred comparison functionality.

To run only our model checks:

```powershell
python -m unittest discover -s tests -p "test_own_model*.py" -v
```

The integration test uses synthetic records, a tiny local random encoder, and
mocked image embeddings. Its scores are software checks, not thesis results.

For implementation constraints and experiment settings, use
[the technical notes](OWN_MODEL_CHANGES.md). Browser extraction details
are in [EXTRACTION_CONTRACT.md](EXTRACTION_CONTRACT.md).
