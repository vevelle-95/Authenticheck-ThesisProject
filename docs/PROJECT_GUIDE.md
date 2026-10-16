# AuthentiCheck project guide

This is the guide for the dataset, AuthentiCheck model, CLIP-CA-CG baseline,
APIs, browser extension, and frontend comparison. Commands use Windows
PowerShell. **Project root** means the `Authenticheck-ThesisProject` folder.

The dataset is still being collected and annotated. Both APIs are implemented,
and both models have saved local practice checkpoints. AuthentiCheck's earlier
review-only quality checkpoints need retraining for the current combined product
context input. The frontend connects to AuthentiCheck; the CLIP-CA-CG comparison
call and display remain to be connected. Practice results include draft/synthetic
data and are development checks.

**For groupmates preparing for the tool defense:** read
[what runs](#1-what-runs), the [project map](#8-file-map-tests-and-troubleshooting),
and the [API/frontend flow](#6-run-the-apis-and-connect-the-frontend) first.
The command sections are for the person setting up or training the models.
You do not need to open every Python file to explain the tool.

- [1. What runs](#1-what-runs)
- [2. Set up the environments](#2-set-up-the-environments)
- [3. Dataset and annotation rules](#3-dataset-and-annotation-rules)
- [4. Train, validate, and test](#4-train-validate-and-test)
- [5. Saved files and new-review prediction](#5-saved-files-and-new-review-prediction)
- [6. Run the APIs and connect the frontend](#6-run-the-apis-and-connect-the-frontend)
- [7. Browser extraction and interface behavior](#7-browser-extraction-and-interface-behavior)
- [8. File map, tests, and troubleshooting](#8-file-map-tests-and-troubleshooting)
- [9. Thesis scope and baseline sources](#9-thesis-scope-and-baseline-sources)

## 1. What runs

| Component | Inputs | Output |
| --- | --- | --- |
| AuthentiCheck quality filter | Product title + description + review text, buyer photos, and stars | `authentic`, `deceptive`, `liv`, or `irrelevant` |
| AuthentiCheck ABSA | Text of reviews predicted Authentic | Ten aspect categories and their sentiments |
| CLIP-CA-CG baseline | Review text and optional buyer photos | Six aspect categories and their sentiments |
| Extension | Buyer reviews already loaded on Shopee/Lazada | Review evidence, coverage, and local/API results |
| Prototype | Illustrative demo data | Interface demonstration |

`liv` means **low informational value** and appears as **Low-value** in the
interface. The four quality classes and the positive/neutral/negative aspect
sentiments are separate labels.

ABSA means **aspect-based sentiment analysis**. AuthentiCheck first applies this
quality filter:

```text
Review + bounded title/description context -> DOST RoBERTa -> four quality probabilities
The same review-first context policy + usable buyer photos -> multilingual CLIP similarity
Stars -> (stars - 1) / 4
Those six values -> XGBoost -> final quality label
Authentic reviews -> independent DOST ABSA -> aspects and sentiments
```

Both AuthentiCheck encoders start from `dost-asti/RoBERTa-tl-cased`. ABSA is
trained independently; it does not reuse the quality classifier's fine-tuned
weights. XGBoost belongs to the logical quality stage even though its script
lives in `stage2/`. CLIP is pretrained and is not fine-tuned by this pipeline.
The maximum combined-text/photo cosine similarity is used; no usable photos means
`0`. The shared `quality_input_text()` helper in `model_contract.py` preserves
review/title/description field boundaries and normalizes whitespace.
`quality_input_pairs()` tokenizes the review separately from a bounded listing
context. Quality training, OOF predictions, feature generation, final evaluation,
and the API all use this policy.
Missing listing context is treated as empty; a written buyer review is required.
ABSA receives the buyer review alone, matching its opinion annotations, and API
review text remains the buyer's text. Product category is not a model input.

RoBERTa has a configurable maximum input length (default 128 tokens). With a
long listing and review, the default paired input is 96 review tokens, 10 title
tokens, 18 description tokens and four special tokens. Context is capped at
28 tokens; shorter title/context frees room for description/review respectively.
Small `--max-length` values proportionally reduce the context allocation.
The review is always the first sequence; only that sequence can be shortened
by the final tokenizer call. Long reviews can still be truncated, but a long
description cannot remove the review. Full source fields remain in the CSV.
M-CLIP uses the same bounded-pair helper with its own tokenizer, special tokens
and encoder limit, independent of the RoBERTa setting. It stays pretrained.

Quality artifacts now use `product-context-six-features-v5-text-target`.
Existing v3 quality weights and feature CSVs are incompatible: changing their
version tags does not repair the training. ABSA now uses
`category-polarity-multilabel-v2` and needs fresh training too; old softmax
checkpoints cannot be converted by editing their version tags.

CLIP-CA-CG uses RoBERTa with a Bi-GRU, frozen ResNet image regions, cached frozen
CLIP embeddings, cross-attention, gating, and aspect/sentiment heads. It has no
quality classifier. Missing photos use zero CLIP image features and a masked
visual path. Both models use the same ten categories and can return multiple
sentiments for one detected category. AuthentiCheck selects review clauses as
evidence; the baseline returns full review context, not learned opinion spans.

### Explain the tool in everyday language

1. The extension collects the buyer reviews currently visible on a product page,
   including each review's written text, stars, and available buyer photos.
2. With **Use model API** enabled, it sends those reviews to AuthentiCheck's
   local API. The API is the connection between the interface and the trained models.
3. The quality model reads the product title, description, and buyer review
   together and produces four quality probabilities. CLIP compares that same
   combined text with buyer photos, and the stars supply one more number.
4. XGBoost uses those six numbers to predict the review's quality class.
5. For reviews predicted Authentic, a separately trained text model detects
   product aspects and predicts positive, neutral, or negative sentiment for each.
6. The API returns individual predictions and summary values. The extension
   displays them as review evidence, aspect summaries, Authentic share, and an
   authenticity-adjusted rating.

For example, "Matibay ang casing pero mahal" can express positive
`product_quality` and negative `value`. Whether that review passes the quality
filter is a separate prediction. An Authentic prediction is a review-quality
class; evaluation measures whether it agrees with a human label.

The baseline API predicts aspects directly from text and optional photos. Its
frontend comparison display is still pending. During a defense, describe that
connection as remaining work until the comparison call and display are implemented.

### Terms the team should know

| Term | Meaning in this project |
| --- | --- |
| Annotation / ground truth | A human's reference answer, such as a quality label or an aspect's sentiment. |
| Encoder | A pretrained component that turns text or a photo into numbers the model can use. AuthentiCheck uses DOST RoBERTa for text. |
| Frozen encoder | An encoder whose learned parameters are kept unchanged while other model components are trained. |
| Feature | A number supplied to another model. AuthentiCheck supplies six features to XGBoost. |
| Training | Learning from labeled training reviews and saving the learned parameters. |
| Epoch | One pass through a model's training examples. More epochs must be judged using validation results. |
| Validation | Checking separate products while choosing the model and its settings. |
| Final test | Measuring the selected model on reserved products after training choices are fixed. |
| OOF | Out-of-fold probabilities: each training review is scored by a quality model trained on other products. |
| Checkpoint / weights | Saved learned parameters. Loading them lets the model predict without training again. |
| Detection threshold | The minimum aspect-presence probability needed to include that category in the output. It is chosen using validation data. |
| Accuracy / F1 | Scores measured against human labels. Accuracy counts correct answers; F1 accounts for missed targets and incorrect predictions. Macro-F1 gives each class/category equal weight. |
| Cache | Previously computed features or downloaded files kept to avoid repeating work. |
| API | A service that accepts review inputs and sends predictions back to the interface. |
| Baseline | The second model used for comparison on the same data and shared aspect categories. |

## 2. Set up the environments

Use Python 3.11. Git contains the source, but `.venv`, trained weights, downloaded
encoders, feature caches, and generated reports are ignored. On another machine,
set up the environments and train the models before serving predictions.

**AuthentiCheck environment: run from the project root.**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the environment already exists, install dependencies with its interpreter.
Activation is optional; `.\.venv\Scripts\Activate.ps1` enables the shorter
`python` command. Examples here use explicit interpreter paths.

**Baseline environment: open another terminal at the project root.**

```powershell
cd baselines\clip-ca-cg
py -3.11 -m venv .venv
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_environment.ps1
```

For a machine without a supported NVIDIA GPU, add `-CpuOnly` to the setup
command. The baseline has its own environment for its CUDA/Torchvision and
Transformers dependencies. If it was set up before the API was added, run
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` in this folder.

Compatible trained files can also be restored from an existing experiment.
Keep each bundle's tokenizer, configuration, and metadata with its weights;
see [saved files](#5-saved-files-and-new-review-prediction). Moving a compatible
bundle to another folder does not require retraining.

## 3. Dataset and annotation rules

Keep **one row per review** in `data/test_reviews.csv`. Its filename does not
make every row a test example; the saved split manifest determines partitions.

| Column | Use |
| --- | --- |
| `review_id` | Unique, permanent local review identifier |
| `product_id` | Groups reviews from the same actual product listing |
| `review_text` | Buyer-authored text |
| `review_image_urls` | JSON list or pipe-separated buyer-photo URLs; blank if none |
| `star_rating` | Integer from 1 to 5 |
| `ground_truth` | Final review-quality target for XGBoost and Authentic-only ABSA selection: `authentic`, `deceptive`, `liv`, or `irrelevant` |
| `aspect_annotations` | JSON list of category, evidence text, and sentiment |
| `product_title`, `product_description` | Listing context included in AuthentiCheck quality/CLIP text; not opinion evidence for ABSA |
| `text_label` | Human text-quality label used to train and validate RoBERTa, including OOF models |
| `aspects`, `aspect_text`, `sentiment` | May remain in the CSV, but training reads `aspect_annotations` |

Both real and augmented reviews carry the same two label columns. RoBERTa
learns `text_label` from review text with bounded listing context. XGBoost
learns `ground_truth` from RoBERTa probabilities, image similarity and stars.
For example, detailed relevant text can be text-Authentic but finally Deceptive
when its image evidence contradicts the review. Assign these labels independently;
equal labels are valid when the evidence supports the same judgment.

Augmented paraphrases inherit each source label separately. The loader verifies
both labels and never replaces a missing text label with the final label.
The v5 text-target contract requires new OOF predictions, a newly trained
RoBERTa, new six-dimensional features and a newly trained XGBoost. Keep the
existing real split manifest; adding a derived numeric text target does not
change the split fingerprint. ABSA still selects ground-truth Authentic reviews.

Evaluation reports `quality_roberta_text_target` against `text_label`.
`quality_xgboost` and `quality_text_only` both use `ground_truth`; the latter
measures how RoBERTa alone performs at the final classification task. Compare
models using the same target and held-out reviews.

Legitimate repeated review text is allowed; every row still needs a unique
`review_id`. For splitting, text is compared ignoring case and whitespace
differences. Products connected by matching review text stay together in one
partition and, when used for training, one OOF fold. Connections are transitive:
if products A/B share one review and B/C share another, all three stay together.
The source reviews and their labels are preserved. Saved splits that separate
matching text are rejected by both models. This checks repeated text, not
semantic similarity or paraphrases.

`product_category` is not required. Annotators use product context to apply the
sensory definition; the model does not automatically suppress sensory predictions
for non-beauty products. Unknown or missing photos produce a CLIP score of `0`.

For "Matibay ang casing pero mahal.", an annotation cell can contain:

```json
[
  {"category": "product_quality", "text": "Matibay ang casing", "sentiment": "positive"},
  {"category": "value", "text": "mahal", "sentiment": "negative"}
]
```

Use `[]` when annotation is complete and no aspects apply. Evidence may
paraphrase the review; exact character matching is not required. An absent
aspect is not a neutral sentiment. Multiple sentiments for the same category
are allowed: preserve positive fragrance and negative texture, for example.
Repeated evidence with the same category/polarity is grouped once. Training
uses a multi-hot vector over distinct polarities, independent sigmoid scores,
and binary cross-entropy. Inference keeps each sentiment above a threshold
selected using validation data. If none passes for a detected aspect, the
highest-scoring sentiment is kept. Both models use this same rule. Detection
and sentiment have separate saved validation-selected thresholds.

The reusable [LLM ABSA annotation prompt](../scripts/prompts/absa_annotation.txt)
preserves separate sentiments for the same category and asks for exact evidence
phrases in newly annotated reviews. Supply product title, product description,
and buyer review separately: listing text helps identify product type and choose
the correct taxonomy category, while opinions, sentiments, and evidence phrases
must come from the buyer review. A listing feature alone must not create an
aspect annotation. This does not require a product-category column. The deployed
ABSA encoder still receives the buyer review only; annotation context does not
change its training or prediction inputs. Existing data validation still permits
paraphrased evidence; review those cases if adopting the prompt's stricter span
rule throughout the dataset. This prompt annotates review text; it does not
generate reviews or assign quality labels. Human reviewers must approve LLM
annotations before treating them as training targets.

Earlier practice CSV versions included assistant-filled annotations and synthetic
reviews recorded in `reports/generated/csv_annotation_fill.json` and
`reports/generated/synthetic_reviews_added.json`. Those logs describe historical
edits, not the contents of every replacement CSV. New augmentation drafts live
separately in [data/augmented_reviews.csv](../data/augmented_reviews.csv).
The training pipeline loads approved drafts; it does not call an LLM itself.

### Fixed taxonomy

The table preserves AuthentiCheck's category order. Code and saved label mappings
use the identifiers defined in [model_contract.py](../model_contract.py).

| Category | Definition |
| --- | --- |
| `product_quality` | Overall build, material quality, durability, or craftsmanship. |
| `functionality` | Whether the product has the expected features or capabilities. |
| `performance` | How well functions work: speed, responsiveness, lag, accuracy, sound, battery performance, results, effectiveness, or reliability. |
| `design` | Ergonomics, physical form, layout, fit, comfort, feel in the hand, non-beauty texture, or ease of use. |
| `aesthetics` | Purely visual appeal: color, appearance, style, or look. |
| `sensory_experience` | Beauty, skincare, and personal care only: smell, fragrance, texture, consistency, or feel on the skin; for example, "mabango", "walang amoy", "creamy", or "foamy". |
| `value` | Whether the product is worth its price: sulit, affordable, expensive, or overpriced. |
| `packaging` | Box condition, wrapping, protection, presentation, or secure packing. |
| `seller_service` | Concrete seller responsiveness, communication, helpfulness, problem resolution, support, or assistance; general thanks/praise alone does not qualify. |
| `accuracy_of_description` | Reviewer-stated agreement or mismatch with advertised size, color, variation, specifications, or features. |

Bluetooth availability is functionality; Bluetooth lag is performance. Comfort
is design; appearance is aesthetics. A listing comparison is
accuracy_of_description, while a standalone color preference is aesthetics.
Delivery speed/courier behavior has no category here and should not automatically
be assigned to seller service or packaging. Mentioning a property alone does not
establish an evaluative opinion. Aspect annotation and review-quality labeling
are separate decisions; being outside this taxonomy does not determine quality.

Both models import the same ten-category order from `model_contract.py`:
`product_quality`, `functionality`, `performance`, `design`, `aesthetics`,
`sensory_experience`, `value`, `packaging`, `seller_service`,
`accuracy_of_description`. The baseline now learns all ten; it no longer drops
four annotation categories. Describe it in the thesis as an aspect-conditioned
ten-aspect multi-label adaptation of CLIP-CA-CG.

## 4. Train, validate, and test

Training teaches the models. Validation selects settings/checkpoints. Final
testing measures the selected models on products they have not trained on.
Use the same dataset snapshot and saved split manifest for both models.

### Prepare shared product splits

Run from the project root:

```powershell
.\.venv\Scripts\python.exe pipeline.py --data data/test_reviews.csv --validate-only
.\.venv\Scripts\python.exe stage1/prepare_splits.py --data data/test_reviews.csv --splits data/splits_real_v2.json
```

These commands validate the CSV and create/reuse `data/splits_real_v2.json`; they do not
train weights. The baseline environment can run the same preparation script
from the root using `baselines\clip-ca-cg\.venv\Scripts\python.exe`.

The approximate split is 70% train, 15% validation, and 15% test, with each product
and its connected duplicate-text products entirely in one partition. Training
also has five OOF folds that keep those groups together. The splitter requires
all four quality classes in each partition, at least five independent training
groups per class, and all fitting classes in every OOF run. Duplicate links can
reduce the number of independent groups and make proportions less exact. If
suitable groups cannot be formed, it fails instead of allowing product or text
leakage. AuthentiCheck ABSA also requires all three sentiments among Authentic
training reviews.

The active shared manifest is `data/splits_real_v2.json`. All AuthentiCheck
training/evaluation commands use `DEFAULT_SPLITS_PATH` in
[model_contract.py](../model_contract.py).
Baseline training, caching, CSV prediction, and evaluation use `data.splits` in
[configs/config.yaml](../baselines/clip-ca-cg/configs/config.yaml). Both defaults
select the same file. An explicit `--splits` overrides the default for a different
experiment; renaming a manifest alone does not update command defaults.

For a changed real-review dataset, preserve the old manifest and prepare a new
one, for example `--splits data/splits_real_v3.json`. Use that same manifest for
both models. Establish the real partitions before LLM augmentation; generated
variants must remain in training and inherit their source product's OOF fold.
Pass `--augmentations data/augmented_reviews.csv` to attach approved variants after
the real manifest has been validated. The shared loader assigns source lineage,
product and fold; adding synthetic rows directly to the real CSV is unnecessary.

### Review and use augmentation

[data/augmented_reviews.csv](../data/augmented_reviews.csv) started with a pilot
of 12 assistant-generated paraphrases: four deceptive, four irrelevant, and four
Authentic. Balance batch 01 adds 20 pending drafts (eight deceptive, six irrelevant,
six Authentic). Those 32 drafts are preserved. Full balancing adds 1,916 rule-based
paraphrases, for 1,948 pending drafts altogether. The projected training dataset
has 716 reviews in each quality class after every draft passes human review.
By default, pending/rejected rows do not enter training or change experiment
hashes. An explicit `--allow-unreviewed-augmentations` option includes pending
rows for experimental runs while keeping their status; rejected rows stay excluded.
The source CSV, validation/test memberships, and real OOF manifest remain intact.

The augmentation file stores only the fields you need to author/review drafts:

| Field | What the groupmate does |
| --- | --- |
| `review_id` | Give each synthetic draft its own unique ID. |
| `source_review_id` | Identify an original training review being paraphrased. |
| `review_text` | Preserve the source experience, opinions, uncertainty and language style. |
| `text_label` | Preserve the source text-quality target independently of its final label. |
| `ground_truth` | Preserve the source final quality label; disputed labels need separate review. |
| `aspect_annotations` | JSON annotations with exact evidence from the new text. Authentic paraphrases must retain all source category/sentiment pairs, including mixed sentiments. |
| `image_mode` | `source` reuses original photos, `none` uses no photos, `donor` uses another original review's photos for a reviewed controlled mismatch. |
| `image_source_review_id` | Leave blank for source/none; give the original donor's review ID for donor mode. |
| `augmentation_method` | Normally `paraphrase`; donor mode requires `controlled_image_mismatch`. |
| `review_status` | Start with `pending`; use `approved` after human review, or `rejected` for unusable drafts. |
| `reviewed_by` | Name/initials of the actual human who approved the row. |
| `review_notes` | Record label justification, evidence checks, image inspection and any concerns. |

The loader inherits product ID, title, description, platform/source metadata, star
rating and OOF fold. These fields are not generated or overridden by the LLM.
Images are resolved from real CSV rows; arbitrary or invented URL fields are
rejected. Non-Authentic aspect targets may be `[]`: both ABSA trainers use
ground-truth Authentic rows, while own quality training uses all four classes.

Review the source and draft together using the written annotation guidelines.
Text is primary for relevance/information; images and ratings support quality
interpretation. A rating/photo mismatch alone does not establish deception or
irrelevance. Confirm that the quality label still holds with the draft's selected
images, including when `image_mode=none` removes supporting visual evidence.
Inspect actual photos before approval; URL equality checks do not verify their
content or accessibility. Reviewers must resolve questionable source labels or
aspect targets rather than multiply them. The pilot notes flag these concerns.

Images from validation/test or another OOF fold cannot be donated. Reused URLs
must occur only in the source's training fold, even for source-image mode. The
current original dataset has two URL values shared across partitions and six
shared across partitions/folds altogether; these are blocked for new augmentation.
The loader does not alter those original rows or claim the original image overlap
is resolved. Different URLs containing identical photos need separate checking.
Use a different image donor or `none` when a URL is blocked. Donor mode is limited
to already deceptive/irrelevant sources; it never automatically changes a label.

From the project root, check all pilot drafts without approving them:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_augmentation.py --check-drafts
```

Add `--audit-images` to print shared URLs. For more generation, export training
sources and use [scripts/prompts/review_augmentation.txt](../scripts/prompts/review_augmentation.txt)
with [scripts/prompts/absa_annotation.txt](../scripts/prompts/absa_annotation.txt):

```powershell
.\.venv\Scripts\python.exe scripts/prepare_augmentation.py --export-sources reports/generated/augmentation_sources_01.json --limit-per-class 5
```

Source export rotates through OOF folds and product groups. It does not approve
source labels. Generate a training balance report for quality labels, distinct
review/category/polarity counts, source-product coverage and human-review notes:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_augmentation.py --check-drafts --balance-report reports/generated/augmentation_balance_01.json
```

Use a new report filename for each batch. These reports distinguish original
examples from approved additions and pending proposals. Balance batch 01 spans
all five source folds and limits total drafts to three per original source. Its
Authentic additions target negative aesthetics, neutral packaging/seller/design,
negative beauty texture and mixed performance. Some source labels/targets are
flagged in the row notes: confirm them before approval rather than generating
more copies. In particular, do not synthesize neutral description accuracy from
size surprise or concern about other buyers' reviews without a reviewer-stated
comparison of advertised and received characteristics. More paraphrases do not
create new independent products or real experiences.

Full generation is available through
[scripts/balance_augmentation.py](../scripts/balance_augmentation.py). It is an
offline rule-based generator using conservative phrase replacements and ordering
of independent tagged fields, not a pretrained LLM. It preserves source labels,
all Authentic aspect/polarity pairs, exact transformed evidence, listing context,
ratings, product IDs and folds. Punctuation/case-only variants are rejected.
Source annotations that conflict with known taxonomy constraints are excluded
from automatic expansion, and the report records targets it could not fill.

```powershell
.\.venv\Scripts\python.exe scripts/balance_augmentation.py --aspect-minimum 30 --report reports/generated/augmentation_full_balance_next.json
```

The generator first increases sparse Authentic aspect/polarity support, then
fills every quality class to the resulting largest class. It saves an exact
backup of the previous augmentation CSV beside the report before replacing it.
It leaves the real CSV and split manifest intact. A rerun on an already balanced
file adds nothing. Use a fresh report filename for a different experiment.

The current experiment uses `data/splits_dataset.json` and focuses on quality
classification. It rebuilds drafts from original training sources after the
dataset/splits changed. Run this only when intentionally regenerating the file;
choose an unused report name:

```powershell
.\.venv\Scripts\python.exe scripts/balance_augmentation.py --splits data/splits_dataset.json --fresh --quality-only --balance-text-labels --report reports/generated/quality_balance_dataset_next.json
```

`--fresh` ignores old drafts and saves an exact backup before replacement.
`--quality-only` adds Deceptive, LIV and Irrelevant paraphrases, with no Authentic
or ABSA augmentation. `--balance-text-labels` considers both label distributions
when selecting sources. Each column uses its own largest original class as its
target floor; label disagreement can make the final counts differ. It never
copies `ground_truth` into `text_label` to force agreement. Source `text_label`
trains RoBERTa/OOF; source `ground_truth` trains XGBoost and determines ABSA eligibility.

The current `data/augmented_reviews.csv` contains 1,652 pending synthetic rows:
668 Deceptive, 335 LIV and 649 Irrelevant by `ground_truth`. Combined with the
1,100 original training reviews, the projected training distributions are:

| Class | `text_label` | `ground_truth` |
| --- | ---: | ---: |
| Authentic | 682 | 688 |
| Deceptive | 707 | 688 |
| LIV | 674 | 688 |
| Irrelevant | 689 | 688 |

There are 2,752 training rows. Validation remains 226 original reviews and test
remains 230 original reviews. ABSA still trains on the 688 original ground-truth
Authentic training reviews. Of the drafts, 1,078 reference source images and 574
have no selected images because URLs were absent or crossed partition/fold scopes.
Image URL lineage checks do not establish image availability or visual correctness.

The report is `reports/generated/quality_balance_dataset_20261015.json` and the
previous file is backed up beside it as `.previous.csv`. The generator uses
rule-based phrase substitutions, not an LLM. All drafts remain pending; there
are no recorded human approvals. The largest source contributes 188 variants.
There are only 20 original ground-truth Deceptive training reviews; paraphrases
add wording variation, not independent reviews or evidence of deceptive intent.

Generated outputs remain pending. By default, training includes only approved
rows. For an immediate experimental run, explicitly include pending drafts:

```powershell
# Project root: run all stages, preserving previous model outputs.
python pipeline.py --splits data/splits_dataset.json --augmentations data/augmented_reviews.csv --allow-unreviewed-augmentations --work-dir data/quality_balanced_v5/features --output models/quality_balanced_v5 --epochs 3
```

This uses all 1,652 pending rows without changing their status or inventing a
human reviewer. Structural checks still run: source labels, exact aspect evidence,
image URL scope, product separation and source OOF fold inheritance. Rejected
rows are always excluded. Metadata records `unreviewed_rows`; this is an
experimental model trained on unreviewed synthetic labels. Real validation/test
rows are unchanged. Keep `--allow-unreviewed-augmentations` on direct stage and
evaluation commands for the same experiment, or checkpoint provenance will differ.
Do not edit the augmentation file during training or before evaluation.

For the baseline, run from `baselines/clip-ca-cg`:

```powershell
.\.venv\Scripts\python.exe cache_clip_features.py --augmentations ../../data/augmented_reviews.csv --allow-unreviewed-augmentations
.\.venv\Scripts\python.exe main.py --augmentations ../../data/augmented_reviews.csv --allow-unreviewed-augmentations --epochs 3
```

The baseline still trains ABSA only on Authentic rows; deceptive additions train
AuthentiCheck's quality classifier. Its outputs follow the baseline YAML paths.
When evaluating this experiment, use:

```powershell
# Project root, after training finishes:
python stage2/evaluate_models.py --data data/test_reviews.csv --splits data/splits_dataset.json --augmentations data/augmented_reviews.csv --allow-unreviewed-augmentations --bundle models/quality_balanced_v5 --output results/quality_balanced_v5_test.json
# Baseline folder, after baseline training finishes:
.\.venv\Scripts\python.exe evaluate.py --augmentations ../../data/augmented_reviews.csv --allow-unreviewed-augmentations --output outputs/augmented_experiment_test.json
```

For a reviewed augmentation run, leave off the experimental flag. After human
approval, run from the root:

```powershell
.\.venv\Scripts\python.exe pipeline.py --augmentations data/augmented_reviews.csv --validate-only
.\.venv\Scripts\python.exe pipeline.py --augmentations data/augmented_reviews.csv --work-dir data/augmented_run/features --output models/augmented_run --epochs 3
```

From `baselines/clip-ca-cg`, use the same file for validation, cache preparation
and training:

```powershell
.\.venv\Scripts\python.exe main.py --augmentations ../../data/augmented_reviews.csv --validate-only
.\.venv\Scripts\python.exe cache_clip_features.py --augmentations ../../data/augmented_reviews.csv
.\.venv\Scripts\python.exe main.py --augmentations ../../data/augmented_reviews.csv
```

Alternatively set baseline `data.augmentations` in its YAML config to that path.
Keep output folders separate when retaining earlier baseline runs. Selected
Authentic additions affect baseline/own ABSA; deceptive/irrelevant additions help
own quality training. Balance text_label and ground_truth counts separately, as
well as aspect/sentiment pairs.

Pass the **same frozen augmentation CSV** to final evaluation so checkpoint
provenance includes exactly the accepted training rows:

```powershell
# Project root:
.\.venv\Scripts\python.exe stage2/evaluate_models.py --data data/test_reviews.csv --augmentations data/augmented_reviews.csv --bundle models/augmented_run --output results/augmented_run_test.json
# Baseline folder:
.\.venv\Scripts\python.exe evaluate.py --augmentations ../../data/augmented_reviews.csv --output outputs/augmented_run_test.json
```

These evaluate original real test rows only. Changing accepted draft text,
annotations, lineage, image selection or reviewer notes changes the experiment
fingerprint and requires rebuilding training outputs. Without `--augmentations`
(or the baseline config setting), the previous original-only workflow still works.
New-data prediction and both APIs do not need the augmentation CSV; they use the
trained checkpoint as before.

OOF means **out-of-fold**: each training review receives quality probabilities
from a RoBERTa model fitted on the other four folds. Those probabilities train
XGBoost. Final RoBERTa is a separate fit on the full training partition and is
used for validation/new reviews. The fold models are not passed into final
RoBERTa, and normal prediction does not load them.

### Train AuthentiCheck

Run from the project root:

```powershell
.\.venv\Scripts\python.exe pipeline.py --data data/test_reviews.csv --splits data/splits_real_v2.json --epochs 3 --batch-size 2
```

The pipeline generates OOF probabilities, trains final RoBERTa, extracts six
features, fits XGBoost, then trains independent ABSA. Test products are excluded.
All quality-text fitting and prediction uses title + description + buyer review.
Fold training uses its fixed epoch setting; held-fold reviews do not select its
checkpoint. Final RoBERTa is selected using validation macro-F1 against text_label;
XGBoost is selected against ground_truth;
XGBoost class weights come from training data. ABSA uses human-labeled Authentic
reviews and selects its checkpoint using detection and sentiment validation
macro-F1. Its validation-selected detection threshold controls which aspects
appear, not which sentiment wins.

For a separate experiment, choose separate destinations:

```powershell
.\.venv\Scripts\python.exe pipeline.py --data data/test_reviews.csv --splits data/experiment_01/splits.json --work-dir data/experiment_01/features --output models/experiment_01 --epochs 3 --batch-size 2
```

**After the review-budget and separate-target corrections, regenerate quality models and features.**
The old review-only, v3 combined-input and v4 ground-truth-trained RoBERTa/XGBoost weights (including
`models/augmented_experiment/`) are rejected by the current API. Use a fresh
destination while retaining the shared splits and augmentation CSV:

```powershell
.\.venv\Scripts\python.exe pipeline.py --data data/test_reviews.csv --splits data/splits_real_v2.json --augmentations data/augmented_reviews.csv --allow-unreviewed-augmentations --work-dir data/text_target_v5/features --output models/text_target_v5 --epochs 3 --batch-size 2
```

This runs new OOF fits, final quality RoBERTa, six-feature extraction, XGBoost,
and ABSA. The independent ABSA input remains review text; a compatible saved
ABSA model may also be retained when rebuilding the quality components manually.
The CSV and split fingerprint policy are unchanged by this code update, so an
unchanged CSV can keep its existing shared product assignments. Current quality
artifacts use `product-context-six-features-v5-text-target`.

### Continue interrupted RoBERTa training on the GPU

AuthentiCheck and the baseline have separate environments. A CUDA-enabled
baseline environment does not enable CUDA in the root `.venv`. Check the
Python environment that will actually run AuthentiCheck:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For this RTX 4060 setup, the verified CUDA build is `torch==2.11.0+cu128`.
Stop Python training before replacing PyTorch. To install that build in the
root environment, use the official CUDA package index:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

The interrupted quality-balanced run saved the final RoBERTa at epoch 2,
step 2,752. Its five OOF models and OOF probability table are already complete.
After CUDA verifies as available, continue that same ten-epoch run:

```powershell
.\.venv\Scripts\python.exe pipeline.py --splits data/splits_dataset.json --augmentations data/augmented_reviews.csv --allow-unreviewed-augmentations --work-dir data/quality_balanced_v5/features --output models/quality_balanced_v5 --epochs 10 --batch-size 2 --start-at roberta --resume-roberta models/quality_balanced_v5/dost_roberta/checkpoints/checkpoint-2752 --require-cuda
```

Ten epochs means ten total, including the two completed epochs. The command
restores weights, optimizer, scheduler and training progress, skips OOF, then
continues feature extraction, XGBoost and ABSA. For a later interruption,
replace the checkpoint path with that run's latest complete saved checkpoint.
Do not use a deleted or partial checkpoint. The resume validator rejects
changed data, labels, batch size, epoch target, token limit or optimizer mode.
In particular, do not add `--low-memory` to a run originally using AdamW;
that switches to Adafactor. `--require-cuda` prevents silent CPU training.
The run records its resume checkpoint, prior metadata and new device/version.
A CPU checkpoint has no CUDA random-number-generator state. On the first
CPU-to-GPU resume, Transformers may log that it could not restore CUDA RNG
state. Weights, optimizer, scheduler and step progress still restore; the GPU
random sequence differs from uninterrupted CPU training. Record the device and
package change when describing the experiment.

### Train CLIP-CA-CG

For a groupmate training the current ten-aspect multi-label model, start at the
repository root, pull the shared CSV/splits/code, then enter the baseline folder:

```powershell
git pull origin main
cd baselines\clip-ca-cg
```

Set up the baseline's own environment once. On an NVIDIA machine:

```powershell
.\setup_environment.ps1
```

Use `.\setup_environment.ps1 -CpuOnly` on a CPU-only machine. The script prints
whether CUDA is available. Then validate, cache train/validation CLIP features,
and train for the current three-epoch experiment:

```powershell
.\.venv\Scripts\python.exe main.py --augmentations ../../data/augmented_reviews.csv --allow-unreviewed-augmentations --validate-only
.\.venv\Scripts\python.exe cache_clip_features.py --augmentations ../../data/augmented_reviews.csv --allow-unreviewed-augmentations
.\.venv\Scripts\python.exe main.py --augmentations ../../data/augmented_reviews.csv --allow-unreviewed-augmentations --epochs 3
```

These explicit Python paths do not require environment activation. Run the
commands in order and stop if a command fails. The first cache run may download
pretrained assets and buyer images. It prepares training/validation only, not
the final test set. Passing the augmentation file preserves experiment
provenance; the baseline still fits only ground-truth Authentic reviews.
The current quality-only drafts add no ABSA training reviews, so baseline
training uses 688 original Authentic reviews and validates on 126.
The selected model is saved to
`outputs/ten_aspect_multilabel_v2/checkpoints/best.pt`, with tokenizer, metadata
and history beside it. These generated files are ignored by Git and are created
locally; groupmates do not need to receive your `outputs` folder. Final testing
is a separate step after training settings are frozen.

[configs/config.yaml](../baselines/clip-ca-cg/configs/config.yaml) points to
`../../data/test_reviews.csv` and `../../data/splits_real_v2.json`. It defaults to five
epochs, batch size 2, maximum text length 128, up to five photos, and automatic
device selection. Caching prepares training/validation CLIP features and photos.
Training/validation use ground-truth Authentic reviews and their six-category
annotations. A fresh run replaces its configured checkpoint files; use separate
output directories in a copied config to retain experiments.

### Where to change epochs

| Training route | Setting |
| --- | --- |
| Full AuthentiCheck pipeline | Pass `--epochs 3`, or change `EPOCHS` in [model_contract.py](../model_contract.py). The value applies to each OOF fold, final quality RoBERTa, and independent ABSA. |
| Standalone Stage 1 training scripts | Pass `--epochs`, or change the same shared `EPOCHS` default. The full pipeline supplies its selected value. |
| Standalone ABSA training script | Pass `--epochs`, or change the same shared `EPOCHS` default. |
| Baseline training | Change `training.epochs` in [configs/config.yaml](../baselines/clip-ca-cg/configs/config.yaml), or run baseline `main.py --epochs 5`. |

AuthentiCheck paths and training defaults are together near the top of
[model_contract.py](../model_contract.py). The former separate Stage 1 settings
file has been removed. The shared training settings are:

```python
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
```

`--epochs`, `--batch-size`, and `--max-length` override these defaults for one
training run. The learning-rate default is shared by quality RoBERTa and ABSA;
CLIP remains frozen and XGBoost has its own tree settings. Baseline settings stay
in its separate YAML file. Restart a training command after editing defaults.

XGBoost trains decision trees and has no epoch setting. Changing a default does
not change existing weights; run a new training experiment and select its saved
bundle/checkpoint for the API. Both APIs keep loaded models in memory, so restart
the service when replacing its selected model. The AuthentiCheck practice bundle
was trained for one epoch under the older review-only input; the baseline's
current training history contains five. Retrain the quality components before
using that older AuthentiCheck bundle with the current API.

### When the dataset changes

Manifests contain a dataset fingerprint. Editing CSV content invalidates the old
manifest. Create a new version from the project root:

```powershell
.\.venv\Scripts\python.exe stage1/prepare_splits.py --data data/test_reviews.csv --splits data/splits_real_v3.json
```

Use `--splits data/splits_real_v3.json` for AuthentiCheck. Set baseline `data.splits`
to `../../data/splits_real_v3.json` and train against the same snapshot. Preserve the
old CSV/splits/models if retaining that experiment. Development splits can stay
ignored while annotation changes; archive the frozen experiment for final work.

### Evaluate selected models

For the default AuthentiCheck bundle, run from the project root:

```powershell
.\.venv\Scripts\python.exe stage2/evaluate_models.py --data data/test_reviews.csv --splits data/splits_real_v2.json --bundle models/own_model_v2 --output results/own_model_test.json
```

For a current input-compatible experiment, pass its matching `--bundle`,
`--splits`, and a new report filename. Evaluation checks training/validation
membership and available dataset fingerprints. It reports quality metrics,
aspect detection, sentiment given human target categories, and product sentiment
errors with/without filtering. Product sentiment averages category outputs
using positive `+1`, neutral `0`, negative `-1`; it is separate from average
stars. Undefined product aggregates are excluded from paired error analysis.

For the baseline, run from `baselines/clip-ca-cg`:

```powershell
.\.venv\Scripts\python.exe cache_clip_features.py --partition test
.\.venv\Scripts\python.exe evaluate.py
```

This saves `outputs/test_metrics.json` for ground-truth Authentic test reviews.
Existing final reports are not overwritten; use a new `--output` for a separate
experiment. Freeze training choices before final testing. A new filename does
not make a test set used for tuning an unbiased final evaluation set.

For ABSA comparison, score the same human-labeled Authentic test IDs and all ten
categories. Both reports contain `absa_evaluation`, with ground-truth pairs,
predicted pairs, sentiment given human target categories, and per-product
results. Detection and end-to-end aspect/sentiment pairs are measured separately.
Multi-label sentiment reports give TP/FP/FN/TN and a 2-by-2 confusion matrix per
polarity; a single 3-by-3 exclusive-class matrix is not appropriate. Accuracy
means exact agreement of the predicted and gold sentiment sets.

After training, use matching frozen data/splits/augmentations for both evaluation
commands. For the quality-balanced experiment, run from the project root:

```powershell
.\.venv\Scripts\python.exe stage2/evaluate_models.py --data data/test_reviews.csv --splits data/splits_dataset.json --augmentations data/augmented_reviews.csv --allow-unreviewed-augmentations --bundle models/quality_balanced_v5 --output results/quality_balanced_v5_test.json
```

Then, from `baselines/clip-ca-cg`:

```powershell
.\.venv\Scripts\python.exe cache_clip_features.py --partition test --augmentations ../../data/augmented_reviews.csv --allow-unreviewed-augmentations
.\.venv\Scripts\python.exe evaluate.py --augmentations ../../data/augmented_reviews.csv --allow-unreviewed-augmentations --output outputs/ten_aspect_multilabel_v2/test_metrics.json
```

Finally, from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/compare_absa.py --own results/quality_balanced_v5_test.json --baseline baselines/clip-ca-cg/outputs/ten_aspect_multilabel_v2/test_metrics.json --output results/absa_ten_aspect_comparison.json
```

The comparison rejects incompatible taxonomies, policies, test review IDs,
review text, image URLs, gold labels, and ABSA training/validation cohorts or
contents. It pairs products and retains the raw records and confusion counts.
It does not perform a significance test; the statistical protocol still needs
finalization. The SOP 2 quality comparison and SOP 3 filtering ablation remain
in the AuthentiCheck report.

## 5. Saved files and new-review prediction

| Location, relative to the project root | Contents |
| --- | --- |
| `data/splits_real_v2.json` | Active shared review/product partitions and OOF folds |
| `data/training_v2/oof_probabilities.csv` | OOF probabilities for training reviews |
| `data/training_v2/6d_features.csv` | Training/validation features for XGBoost |
| `models/own_model_v2/dost_roberta/` | Final quality encoder, tokenizer, configuration, and metadata |
| `models/own_model_v2/xgboost_meta_classifier.json` | Quality filter, with adjacent `.metadata.json` |
| `models/own_model_v2/absa_model/` | `model.pt`, `absa_config.json`, `encoder/config.json`, tokenizer, and metadata |
| `models/own_model_v2/oof/fold_0/` through `fold_4/` | Training-feature generation models |
| `models/own_model_v2/manifest.json` | Bundle versions, environment, taxonomy, and experiment settings |
| `models/practice_run/` | Historical review-only practice bundle; its quality weights are incompatible with current combined-input inference |
| `models/context_run/` | Suggested destination for new combined-input training; created when that training command is run |
| `results/*.json` | AuthentiCheck prediction/evaluation reports |
| `reports/generated/*.json` | Draft annotation/synthetic generation records |
| `baselines/clip-ca-cg/outputs/ten_aspect_multilabel_v2/checkpoints/` | New ten-aspect multi-label `best.pt`, tokenizer, training metadata, and epoch history |
| `baselines/clip-ca-cg/outputs/clip_cache/` | Content-keyed features and `encoder.json` with pinned CLIP revision |
| `baselines/clip-ca-cg/outputs/image_cache/` | Cached buyer photos |
| `baselines/clip-ca-cg/outputs/model_cache/` | Downloaded pretrained model assets |
| `baselines/clip-ca-cg/outputs/*.json` | Baseline predictions and evaluation reports |

Final RoBERTa saves epoch checkpoints under its `checkpoints/` directory, limits
retention, and exports the selected model in `dost_roberta/`. ABSA saves the best
validation model to `absa_model/model.pt`; its history is in training metadata.
The baseline saves its new best model to `outputs/ten_aspect_multilabel_v2/checkpoints/best.pt` and records
epoch results in `training_history.json`.

The existing `models/practice_run/` quality weights are historical review-only
weights. Keep them with their original metadata; the current API rejects their
input contract. Retraining to `models/context_run/` creates a compatible bundle.

If restoring an existing baseline, keep its configured checkpoint directory,
`outputs/clip_cache/`, and `outputs/image_cache/` together. The model cache can
also be copied; otherwise uncached reviews may download CLIP assets. The saved
CLIP revision must match training. Old six-aspect checkpoints stay in
`outputs/checkpoints/` and are rejected by the new model contract. An empty `outputs` directory is created as
needed during training; cloning source alone does not provide trained files.

The tracked `models/manifest.json` and [scripts/setup_models.py](../scripts/setup_models.py)
describe an older download bundle. That installer does not create the current
ten-category model. Compatibility is determined from saved versions/configuration,
not the directory name.

### Predict a new CSV with AuthentiCheck

New data needs review text and integer stars. Include IDs for matching/grouping
and the listing title/description when available; photos are optional. Human
labels and annotations are not needed. After training the combined-input bundle,
run from the root:

```powershell
.\.venv\Scripts\python.exe stage2/online_inference.py --data data/new_reviews.csv --roberta-model models/context_run/dost_roberta --xgb-path models/context_run/xgboost_meta_classifier.json --absa-dir models/context_run/absa_model --output results/new_review_predictions.json
```

The API bundle environment variable does not set these CLI arguments. To use
the default trained bundle, omit the three model-path options. ABSA-only diagnosis
is available through `stage2/predict_absa.py`; it bypasses the full quality filter.

### Predict with the baseline CLI

Run from `baselines/clip-ca-cg`:

```powershell
.\.venv\Scripts\python.exe predict.py --text "Sira agad ang material" --prepare-cache --output outputs/single_review.json
.\.venv\Scripts\python.exe predict.py --csv new_reviews.csv --partition all --prepare-cache --output outputs/new_predictions.json
```

A new baseline CSV needs `review_id`, `review_text`, and `review_image_urls`.
Photo cells can be blank. With `--partition all`, it needs no split file. For
single-review photos, add `--image-url "https://..."`, repeating it as needed.
Single-review output is an aspect list; CSV output wraps lists under
`reviews[].aspects` with IDs. `--prepare-cache` computes missing features before
prediction. No detected categories gives `aspects: []`; aspect text is full
review context, not an extracted evidence snippet.

## 6. Run the APIs and connect the frontend

### Start both services

**Terminal A: project root, after training the combined-input bundle.**

```powershell
$env:AUTHENTICHECK_MODEL_BUNDLE = "models/context_run"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

`AUTHENTICHECK_MODEL_BUNDLE` selects all three AuthentiCheck models for the API.
Relative paths resolve from the project root. Without it, the default is
`models/own_model_v2`. Restart the API after changing the setting. It affects
API inference; pipeline training still uses its default or explicit `--output`.

The example requires the new bundle to exist. Selecting the older review-only
practice bundle produces a readiness/input-contract error until its quality
models and features are rebuilt.

**Terminal B: open at the project root, then enter the baseline folder.**

```powershell
cd baselines\clip-ca-cg
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

The baseline loads `best.pt` from its configured `output.checkpoint_dir` once and serializes predictions.
`CLIP_CA_CG_CHECKPOINT` can select another checkpoint before startup; relative
paths resolve from the baseline folder. The tokenizer path is recorded in the
checkpoint configuration. Keep matching CLIP metadata available at startup.

| Service | Health | Readiness | Interactive API docs | Prediction |
| --- | --- | --- | --- | --- |
| AuthentiCheck | `http://127.0.0.1:8000/health` | `http://127.0.0.1:8000/ready` | `http://127.0.0.1:8000/docs` | `POST /analyze` |
| CLIP-CA-CG | `http://127.0.0.1:8001/health` | `http://127.0.0.1:8001/ready` | `http://127.0.0.1:8001/docs` | `POST /predict-aspects` |

`/health` confirms the server is running; `/ready` must report `ready: true`
before model use. Missing/incompatible artifacts produce readiness errors and
HTTP 503 for predictions. API requests accept 1 to 20 reviews, text up to 2,000
characters, and up to five HTTP/HTTPS photo URLs per review. Use unique IDs.
Requests with invalid fields return 422; unexpected inference failures return
500. New uncached photos/features may take longer. Both APIs permit localhost
web frontends and Chrome/Edge extension origins.

### Request and response fields

AuthentiCheck needs platform/product context and review stars. A valid example:

```json
{
  "schemaVersion": "1.0",
  "platform": "shopee",
  "url": "https://shopee.ph/product/demo",
  "productTitle": "Product name",
  "productDescription": "Visible description",
  "reviews": [
    {"id": "review_001", "text": "Matibay ang material.", "rating": 5, "hasImage": false, "imageUrls": []}
  ]
}
```

Only send supported review fields to AuthentiCheck; keep extraction eligibility
and missing-field diagnostics in the frontend. Its quality model requires an
integer rating from 1 to 5. The baseline needs only this subset:

```json
{
  "schemaVersion": "1.0",
  "reviews": [
    {"id": "review_001", "text": "Matibay ang material.", "imageUrls": []}
  ]
}
```

The baseline ignores additional ratings/platform/product fields, so the existing
AuthentiCheck payload can also be sent to it. It uses no ground-truth labels.
Baseline response structure, with illustrative predictions:

```json
{
  "schemaVersion": "1.0",
  "modelVersion": "adapted-clip-ca-cg-aspects-v1",
  "reviews": [
    {
      "id": "review_001",
      "aspects": [
        {"category": "product_quality", "text": "Matibay ang material.", "sentiment": "positive"}
      ]
    }
  ]
}
```

| Meaning | AuthentiCheck | Baseline |
| --- | --- | --- |
| Review identifier | `reviews[].id` | `reviews[].id` |
| Review text | `reviews[].text` | Request text; also complete normalized text in each aspect |
| Review quality | `reviews[].label`, probabilities, confidence | Not provided |
| Aspect predictions | `reviews[].aspectSentiment` | `reviews[].aspects` |
| Category | Aspect item's `category` | Aspect item's `category` |
| Sentiment | `Positive`, `Neutral`, `Negative` | `positive`, `neutral`, `negative` |
| Quality/UI summaries | `reviewCount`, `counts`, `authenticShare`, `verifiedRating`, `sentimentCounts`, `aspects` | Not provided |

AuthentiCheck's `authenticShare` is the percentage classified Authentic, not
statistical model confidence. `verifiedRating` averages stars of reviews
classified Authentic. Predicted sentiments and product aggregates are separate
from these stars. The baseline preserves every ID even if its aspect list is empty.

### Frontend developer handoff

The baseline HTTP service is implemented. The remaining frontend work is:

1. Send the same IDs, text, and buyer-photo URLs to both endpoints.
2. Join responses by ID and show results on the same review row.
3. Compare the ten shared categories and normalize sentiment names to lowercase.
4. For reviews AuthentiCheck rejects, show **ABSA skipped by quality filter**.
   Distinguish this from an analyzed review with no detected aspects.
5. Keep AuthentiCheck's quality summaries with AuthentiCheck. Show loading,
   empty, unavailable-input, and API-error states; render review text with `textContent`.

| File | Connection work |
| --- | --- |
| [extension/background.js](../extension/background.js) | Add a separate baseline message/HTTP call; current AuthentiCheck call uses port 8000. |
| [extension/content.js](../extension/content.js) | Send matched reviews, join IDs, and render the comparison. |
| [extension/popup.js](../extension/popup.js) | Extend settings if exposing a baseline endpoint or comparison mode. |
| [prototype/app.js](../prototype/app.js) | Connect real requests/rendering if using this standalone demo as the frontend. |

The existing extension's `normalizeApiResult()` requires AuthentiCheck quality
counts, labels, and summaries. Baseline responses need their own comparison
handler. The extension manifest already allows localhost requests. For a web
frontend, a baseline request can use:

```javascript
async function getBaselinePredictions(reviews) {
  const response = await fetch("http://127.0.0.1:8001/predict-aspects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      schemaVersion: "1.0",
      reviews: reviews.map(({ id, text, imageUrls }) => ({ id, text, imageUrls: imageUrls ?? [] }))
    })
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `Baseline failed (${response.status})`);
  if (result.schemaVersion !== "1.0" || !Array.isArray(result.reviews)) throw new Error("Invalid baseline response");
  return new Map(result.reviews.map(review => [review.id, review.aspects]));
}
```

This is a handoff example; the frontend files do not call it yet. Validate that
returned IDs/categories match the request and handle timeouts/errors in the UI.
The existing AuthentiCheck extension call has a timeout of at least 60 seconds.

## 7. Browser extraction and interface behavior

Load the `extension/` folder through Chrome/Edge's **Load unpacked** option in
developer mode. Open a Shopee Philippines or Lazada Philippines product page,
scroll until buyer reviews are visible, and use the floating badge. Reload the
extension after source changes. To use trained results, enable **Use model API**
in its popup and set `http://127.0.0.1:8000/analyze`.

The default local estimate uses Tagalog/Taglish heuristics for interface testing.
It is not the trained model. API failures are shown explicitly; the user can
deliberately select the labeled local preview. The standalone prototype also
uses illustrative data until real calls are connected. Serve it from the root:

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory prototype
```

Open `http://127.0.0.1:5500` so localhost CORS rules apply.

### Buyer-review boundary

Adapters extract one record per buyer review already loaded in the page:

```json
{
  "id": "shopee-1",
  "text": "Buyer-authored review text only",
  "rating": 5,
  "imageUrls": ["https://example.com/buyer-photo.jpg"],
  "hasImage": true,
  "analysisEligible": true,
  "missingFields": []
}
```

Text, stars, and up to five photos must come from the same review-card root.
Missing text is `""`; an undetected rating is `null`; `hasImage` follows the URL
list. `analysisEligible` currently indicates available written text. Reviews
without text remain visible as **Not analyzed** and are excluded from model
requests. Missing ratings still need to be handled for AuthentiCheck's quality
model; do not invent them.

Remove seller/shop replies before reading buyer text/media. Exclude reviewer
usernames/avatars, seller names/logos, timestamps, variation labels, buttons,
like counts, and report controls. Never read account, cart, payment, delivery
address, or checkout data. Deduplicate nested card matches and duplicate
nonempty text; rating-only/photo-only cards remain separate evidence records.
Process the first 20 loaded reviews without automated scrolling, pagination,
or private marketplace API calls.

Diagnostics include candidate/accepted cards, analyzable reviews, duplicates,
reviews without text, detected stars, buyer photos, and removed seller responses.
Use them to inspect extraction before interpreting predictions. Marketplace DOM
changes belong in the appropriate [extractor](../extension/extractors/);
shared buyer-only rules live in `buyer-reviews.js`.

Keep loading, empty, model-error, recovery, and result states clear. Evidence
search, quality filters, input coverage, and JSON export refer to the actual
analyzed reviews. The interface supports keyboard use and responsive layouts.
Buyer photo bytes are processed in memory by AuthentiCheck; the baseline
caches photos/features locally. Review content is sent to the local service
when the user enables API mode.

## 8. File map, tests, and troubleshooting

Read this map as a directory guide, rather than a list of scripts to run.
**Entry points** are commands you launch; **supporting files** are called by those
commands automatically. The training commands in section 4 already coordinate
the required supporting files.

### Folder overview

All paths in this table start at `Authenticheck-ThesisProject/`.

| Folder | What it contains / when the team needs it |
| --- | --- |
| [data/](../data/) | The shared input CSV, saved product splits, and generated training features. Open this for dataset work. |
| [data/samples/](../data/samples/) | Small examples of messages exchanged by the extension and API. These are interface examples. |
| [stage1/](../stage1/) | Quality-text training, product split preparation, and six-feature preparation. |
| [stage2/](../stage2/) | XGBoost training, aspect/sentiment training, complete prediction, and final evaluation. The folder name is historical: XGBoost still belongs to the quality stage. |
| [models/](../models/) | AuthentiCheck's saved learned parameters, tokenizers, settings, and training records. See section 5 for the bundle layout. |
| [baselines/](../baselines/) | Alternative models used for comparison; currently contains the adapted CLIP-CA-CG project. |
| [baselines/clip-ca-cg/configs/](../baselines/clip-ca-cg/configs/) | Baseline training settings and data/output paths. |
| [baselines/clip-ca-cg/datasets/](../baselines/clip-ca-cg/datasets/) | Baseline code that reads the shared CSV and prepares text, photos, and labels. |
| [baselines/clip-ca-cg/models/](../baselines/clip-ca-cg/models/) | Python code defining the baseline architecture. `encoders/` reads text/photos; `fusion/` combines their information. |
| [baselines/clip-ca-cg/training/](../baselines/clip-ca-cg/training/) | Baseline learning, validation, and metric calculations. |
| `baselines/clip-ca-cg/outputs/` | Generated baseline checkpoints, cached photos/features, predictions, and evaluation reports. Created by the relevant commands. |
| [extension/](../extension/) | The actual Chrome/Edge tool: review extraction, settings, API connection, and result panel. |
| [extension/extractors/](../extension/extractors/) | Rules for finding buyer reviews on Shopee and Lazada pages. |
| [prototype/](../prototype/) | A standalone interface demonstration with illustrative data. |
| [tests/](../tests/) | Automated AuthentiCheck checks and small browser test pages. |
| [baselines/clip-ca-cg/tests/](../baselines/clip-ca-cg/tests/) | Automated baseline architecture/data/API checks. |
| [scripts/](../scripts/) | Optional compact development training/serving/diagnostics and an older model-bundle installer. Pretrained thesis training uses `pipeline.py`. |
| [docs/](./) | This project guide. Keep team explanations here instead of creating separate guides. |
| `results/` | AuthentiCheck prediction and evaluation reports, created by the relevant commands. |
| `reports/generated/` | Records of draft annotation filling and synthetic practice reviews. These describe dataset edits. |

**Two folders named `models` have different jobs:** root `models/` stores
AuthentiCheck's trained files; baseline `models/` defines how its neural network
works. The baseline's new trained files go to `outputs/ten_aspect_multilabel_v2/checkpoints/`.

### Files at the project root

| File | Plain-language purpose |
| --- | --- |
| [README.md](../README.md) | The short project overview, extension installation steps, and quick start. |
| [pipeline.py](../pipeline.py) | AuthentiCheck training entry point. Coordinates validation, splits, OOF fitting, final quality training, XGBoost, and ABSA. |
| [main.py](../main.py) | AuthentiCheck API entry point. Loads the selected trained bundle and answers extension requests; it does not train on those requests. |
| [model_contract.py](../model_contract.py) | Shared settings and rulebook: dataset/model paths, epochs, batch size, learning rate, quality labels, ten aspects, sentiments, six-feature order, combined quality-text builder, input version, rating/photo formats, and API bundle selection. |
| [model_data.py](../model_data.py) | Reads and checks the CSV, identifies dataset changes, and assigns products to saved partitions/folds. |
| [training_augmentation.py](../training_augmentation.py) | Shared by both models. Adds approved synthetic training reviews, inherits original context/product/fold, validates exact evidence and image lineage, and blocks held-out sources/images. |
| [model_metrics.py](../model_metrics.py) | Calculates quality scores and shared multi-label ABSA scores, TP/FP/FN/TN counts, cohort signatures and per-product results for both models. |
| [scripts/compare_absa.py](../scripts/compare_absa.py) | Checks that both held-out ABSA reports use matching data and labels, then produces paired product metrics and retains the ground-truth/prediction records. |
| [requirements.txt](../requirements.txt) | Python libraries needed by AuthentiCheck training and its API. |
| [.gitignore](../.gitignore) | Tells Git to leave out environments, large weights, caches, generated reports, and other local files. It does not delete them. |

`model_data.py` contains the split logic; `stage1/prepare_splits.py` is the
small command that calls it. They share one implementation. Similarly, the
metrics helper calculates scores while evaluation scripts choose which model
and reviews to score.

### AuthentiCheck Stage 1 files

| File | What it does | Used during |
| --- | --- | --- |
| [prepare_splits.py](../stage1/prepare_splits.py) | Checks the CSV and creates/reuses the shared train/validation/test and five-fold assignment. | Preparation command |
| [train_roberta.py](../stage1/train_roberta.py) | Fine-tunes DOST RoBERTa on review + bounded title/description context to predict the four text_label classes; OOF fitting uses the same code. | Training |
| [generate_oof_features.py](../stage1/generate_oof_features.py) | Trains five fold models and scores each held-out training fold. Those predictions become XGBoost training inputs. | Training-feature preparation |
| [features.py](../stage1/features.py) | Produces the same six numbers during training and prediction: four combined-text probabilities, combined-text/photo similarity, and normalized stars. No usable photo gives similarity `0`. | Supporting code for training and prediction |
| [extract_6d_features.py](../stage1/extract_6d_features.py) | Writes the training/validation feature CSV. Uses OOF probabilities for training and final RoBERTa probabilities for validation. | Training-feature preparation |

The `oof/` models help build training features. The final `dost_roberta/` model
handles new reviews. They are separate fits with separate purposes.

### AuthentiCheck Stage 2 files

| File | What it does | Used during |
| --- | --- | --- |
| [train_xgboost.py](../stage2/train_xgboost.py) | Learns the final quality decision from the six features and selects settings using validation reviews. | Training |
| [absa_model.py](../stage2/absa_model.py) | Defines the ten-aspect detector and sentiment classifier, including how their trained weights are saved and loaded. | Supporting model code |
| [fine_tune_absa.py](../stage2/fine_tune_absa.py) | Trains independent DOST ABSA on human-labeled Authentic training reviews; chooses the checkpoint, aspect threshold and multi-label sentiment threshold using validation. | Training |
| [online_inference.py](../stage2/online_inference.py) | Uses combined listing/review text for quality and CLIP, sends the buyer review alone to ABSA if predicted Authentic, and builds summaries. The API uses this prediction code. | New-review prediction / API |
| [predict_absa.py](../stage2/predict_absa.py) | Runs ABSA separately for diagnosis. Can filter by an existing predicted quality label; it does not run the full quality classifier itself. | Optional prediction command |
| [evaluate_models.py](../stage2/evaluate_models.py) | Checks model/data provenance and scores selected models on reserved test products, including ABSA with and without the predicted quality filter. | Final evaluation command |

During live use, saved weights are loaded and reused. Training scripts and
final evaluation scripts are not called every time someone analyzes a product.

### Baseline entry points and settings

The following paths start inside `baselines/clip-ca-cg/`. Run its commands with
its own environment; it reads the same root dataset and split file.

| File | Plain-language purpose |
| --- | --- |
| [main.py](../baselines/clip-ca-cg/main.py) | Baseline training entry point, or CSV/split checking with `--validate-only`. Unlike root `main.py`, this starts training. |
| [api.py](../baselines/clip-ca-cg/api.py) | Baseline API entry point. Loads the selected checkpoint once and serves `/predict-aspects`, `/health`, and `/ready`. |
| [predict.py](../baselines/clip-ca-cg/predict.py) | Command for predicting aspects from a new review or CSV and saving JSON. |
| [evaluate.py](../baselines/clip-ca-cg/evaluate.py) | Final evaluation command for the selected checkpoint on shared test products. |
| [cache_clip_features.py](../baselines/clip-ca-cg/cache_clip_features.py) | Downloads usable buyer photos and computes reusable frozen CLIP features before training/evaluation. |
| [inference.py](../baselines/clip-ca-cg/inference.py) | Supporting predictor used by the API and command scripts. Loads trained weights and maps model outputs into aspect JSON. |
| [runtime.py](../baselines/clip-ca-cg/runtime.py) | Reads settings, resolves file paths, chooses CPU/GPU, checks checkpoint compatibility, and saves JSON reports. |
| [configs/config.yaml](../baselines/clip-ca-cg/configs/config.yaml) | Dataset/split paths, epochs, learning rate, image/text limits, device settings, and output destinations. |
| [requirements.txt](../baselines/clip-ca-cg/requirements.txt) | Libraries for the baseline's separate Python environment. |
| [setup_environment.ps1](../baselines/clip-ca-cg/setup_environment.ps1) | Installs compatible baseline dependencies, including the CPU or CUDA/Torchvision setup. |
| [benchmark_model.py](../baselines/clip-ca-cg/benchmark_model.py) | Optional GPU-memory check with random weights/inputs. Saves a resource report; it does not produce a trained model or an accuracy result. |
| [.gitignore](../baselines/clip-ca-cg/.gitignore) | Additional rules keeping baseline environments, checkpoints, and generated caches/reports out of Git. |

### Baseline data, model, and training helpers

These are supporting files called automatically by the baseline entry points.

| File inside the baseline | Plain-language purpose |
| --- | --- |
| [datasets/authenticheck_data.py](../baselines/clip-ca-cg/datasets/authenticheck_data.py) | Maps the existing CSV and annotations to six baseline aspects, checks the shared split manifest, and builds training targets. |
| [datasets/multimodal_dataset.py](../baselines/clip-ca-cg/datasets/multimodal_dataset.py) | Packages text tokens, real photos, cached CLIP vectors, and aspect labels into batches. Marks absent/padded photos so the model can ignore them. |
| [datasets/image_store.py](../baselines/clip-ca-cg/datasets/image_store.py) | Downloads and reuses public buyer photos locally. Unavailable photos stay missing. |
| [datasets/clip_cache.py](../baselines/clip-ca-cg/datasets/clip_cache.py) | Stores frozen text/image features keyed by review content and checks which CLIP version created them. |
| [models/model.py](../baselines/clip-ca-cg/models/model.py) | Assembles the baseline network with ten aspect outputs and three independent sentiment outputs per aspect. |
| [models/encoders/text_encoder.py](../baselines/clip-ca-cg/models/encoders/text_encoder.py) | Uses pretrained RoBERTa and a bidirectional GRU to represent review words with surrounding context. |
| [models/encoders/image_encoder.py](../baselines/clip-ca-cg/models/encoders/image_encoder.py) | Uses ResNet50 to represent regions of real buyer photos. Frozen by default. |
| [models/encoders/clip_encoder.py](../baselines/clip-ca-cg/models/encoders/clip_encoder.py) | Computes frozen pretrained CLIP text/photo vectors for the cache. Supplies a zero image vector and score for missing photos. |
| [models/fusion/cross_attention.py](../baselines/clip-ca-cg/models/fusion/cross_attention.py) | Lets text focus on relevant photo regions and photos focus on relevant words, while ignoring padding and missing photos. |
| [models/fusion/gating.py](../baselines/clip-ca-cg/models/fusion/gating.py) | Learns how to combine text, image, and joint information for the final aspect predictions. |
| [models/fusion/projection.py](../baselines/clip-ca-cg/models/fusion/projection.py) | Older vector-size conversion helper. The current network uses its own linear layers and does not import this file. |
| [training/train.py](../baselines/clip-ca-cg/training/train.py) | Runs all epochs, checks validation performance, saves the best checkpoint/tokenizer, and records training history and dataset membership. |
| [training/engine.py](../baselines/clip-ca-cg/training/engine.py) | Compares predictions with annotations and updates learned parameters in small batches. Learns aspect presence and sentiment for annotated aspects, with options that reduce GPU-memory use. |
| [training/eval.py](../baselines/clip-ca-cg/training/eval.py) | Collects predictions, calculates aspect/sentiment metrics, and chooses the detection threshold using validation predictions. |
| [datasets/__init__.py](../baselines/clip-ca-cg/datasets/__init__.py), [models/__init__.py](../baselines/clip-ca-cg/models/__init__.py) | Empty package markers that let Python import those folders. They contain no model logic. |

Baseline flow: CSV/splits -> cached CLIP/photos -> batched inputs -> text and
image encoders -> cross-attention and gating -> aspect/sentiment heads -> JSON.
Training adds human targets and learning updates; prediction loads the selected
weights and returns outputs. Full-review text is returned as context for each
aspect; the baseline does not learn to extract opinion snippets.

### Extension and prototype files

The extension is the tool installed in Chrome/Edge. Its HTML defines visible
controls, CSS defines appearance, and JavaScript supplies behavior.

| File | Plain-language purpose |
| --- | --- |
| [extension/manifest.json](../extension/manifest.json) | Tells Chrome/Edge the extension's name, permissions, supported sites, and which scripts to load. |
| [extension/background.js](../extension/background.js) | Handles extension messages and the AuthentiCheck API connection, including timeouts and connection errors. |
| [extension/content.js](../extension/content.js) | Runs on the product page: collects reviews through the extractors, builds the panel, requests analysis, validates responses, and displays results. Also contains the optional local heuristic estimate. |
| [extension/content.css](../extension/content.css) | Styles the floating badge and analysis panel. |
| [extension/popup.html](../extension/popup.html) | The small settings window opened from the browser toolbar. |
| [extension/popup.js](../extension/popup.js) | Loads/saves settings such as API mode and endpoint, shows page status, and requests a rescan. |
| [extension/popup.css](../extension/popup.css) | Styles that settings window. |
| [extension/extractors/buyer-reviews.js](../extension/extractors/buyer-reviews.js) | Shared buyer-review extraction rules, including keeping text/stars/photos together, excluding seller responses, and removing duplicate matches. |
| [extension/extractors/shopee.js](../extension/extractors/shopee.js) | Shopee-specific product/review detection and selectors. |
| [extension/extractors/lazada.js](../extension/extractors/lazada.js) | Lazada-specific product/review detection and selectors. |
| [extension/extractors/common.js](../extension/extractors/common.js) | Older shared extraction helpers. The current extension manifest loads `buyer-reviews.js` instead, which supplies the shared helpers used by the adapters. |
| [prototype/index.html](../prototype/index.html) | The standalone demo product page and interface layout. |
| [prototype/styles.css](../prototype/styles.css) | The prototype's visual styling. |
| [prototype/app.js](../prototype/app.js) | Demo interactions, staged analysis, and illustrative results. Real API calls still need to be connected here if this is used as the model frontend. |

### Shared data and older setup files

| File | Plain-language purpose |
| --- | --- |
| [data/test_reviews.csv](../data/test_reviews.csv) | The working review dataset for both models. The split manifest decides which rows are training, validation, or test. |
| [data/augmented_reviews.csv](../data/augmented_reviews.csv) | Separate synthetic review drafts with source IDs, image selection and review status. Approved rows enter training by default; the explicit experimental flag also includes pending drafts without changing their status. |
| [data/samples/extension_api_payload.json](../data/samples/extension_api_payload.json) | Example extension request, useful for understanding/testing the message format. |
| [data/samples/extension_api_response.json](../data/samples/extension_api_response.json) | Example AuthentiCheck response for frontend development; its values are samples. |
| [data/samples/capability_challenge.json](../data/samples/capability_challenge.json) | Small diagnostic review cases with provisional expected answers requiring human review; separate from final accuracy evaluation. |
| [models/manifest.json](../models/manifest.json) | Manifest for the older download-bundle layout. Each newly trained bundle has its own manifest. |
| [scripts/setup_models.py](../scripts/setup_models.py) | Installs/verifies an older model archive according to that manifest. Current training uses `pipeline.py` instead. |
| [scripts/train_development_model.py](../scripts/train_development_model.py) | Optional offline training exercise using a compact random encoder, one epoch, and no photos. Its vocabulary includes the combined quality text. Saves to `models/development_v3/` and `data/development_v3/`; it does not train the pretrained DOST thesis model. |
| [scripts/prompts/absa_annotation.txt](../scripts/prompts/absa_annotation.txt) | Reusable LLM prompt for the fixed ten aspects and exact evidence phrases. Preserves distinct sentiments for one category; outputs require human review. It annotates existing text and does not generate training reviews. |
| [scripts/prompts/review_augmentation.txt](../scripts/prompts/review_augmentation.txt) | Prompt for meaning-preserving training review drafts. Keeps source labels, exact aspect evidence and pending human review; does not invent image URLs. |
| [scripts/prepare_augmentation.py](../scripts/prepare_augmentation.py) | Checks draft structure, writes training balance/source-review reports, reports image reuse, and exports training sources across products/folds for generation. Does not approve drafts or train a model. |
| [scripts/balance_augmentation.py](../scripts/balance_augmentation.py) | Generates the complete quality-class shortfall and sparse ABSA targets from training reviews using reproducible rule-based paraphrases. Preserves provenance, rejects known annotation conflicts, backs up earlier drafts and records projected counts; outputs remain pending. |
| [scripts/serve_development_model.py](../scripts/serve_development_model.py) | Serves that compact bundle through the AuthentiCheck API with images omitted. Use `--port 8002` when the baseline occupies port 8001. |
| [scripts/check_model_capabilities.py](../scripts/check_model_capabilities.py) | Sends provisional diagnostic cases to a selected endpoint and checks compact ABSA separately. For port 8002 pass `--endpoint http://127.0.0.1:8002/analyze`; reports go to `reports/generated/`. |
| [docs/PROJECT_GUIDE.md](PROJECT_GUIDE.md) | This shared explanation of the models, dataset, outputs, APIs, interface, and defense terminology. |
| [docs/DEVELOPMENT_MODEL_TESTING.md](DEVELOPMENT_MODEL_TESTING.md) | Existing supplementary instructions for the optional compact workflow. The pretrained-model setup is in this project guide. |
| [docs/MANUSCRIPT_COMPLETION_AUDIT.md](MANUSCRIPT_COMPLETION_AUDIT.md) | Historical assessment using a proposed implementation rubric. Its dated score is separate from model accuracy or current validation results. |

Generated split/feature files, per-epoch checkpoints, tokenizers, model
configurations, and reports are explained in [section 5](#5-saved-files-and-new-review-prediction).
Paths such as `data/practice_run/` or `models/practice_run/` are experiment
destinations, rather than extra implementations of the model.

### What the test files do

**Software checks and final dataset evaluation answer different questions.**
Software checks ask whether the code follows its rules, for example whether a
review without a photo gets CLIP score `0`. Final evaluation asks how accurately
the trained model predicts human-labeled reviews from unseen products.

The two maintained test folders have separate jobs:

| File in root `tests/` | What it checks |
| --- | --- |
| [test_own_model.py](../tests/test_own_model.py) | CSV/label rules, product separation, OOF behavior, no-image score `0`, mixed sentiments, ABSA filtering, and summary calculations. |
| [test_own_model_integration.py](../tests/test_own_model_integration.py) | A complete miniature train/save/load/predict/evaluate round trip using a temporary tiny encoder and mocked photo features. Reuses fixtures from `test_own_model.py`. |
| [test_api_contract.py](../tests/test_api_contract.py) | AuthentiCheck request/response format and review-count limits, using a fake predictor rather than saved thesis weights. |
| [check_extension_contract.mjs](../tests/check_extension_contract.mjs) | Extension script configuration and sample API message formats. Runs with Node. |
| [test_extension_runtime.mjs](../tests/test_extension_runtime.mjs) | Extension response validation, message handling, and API errors/timeouts in a simulated environment. Runs with Node. |
| [extractor-fixture.html](../tests/extractor-fixture.html) | Small browser page for checking buyer text/photo extraction and seller-response exclusion. |
| [shopee-fallback-fixture.html](../tests/shopee-fallback-fixture.html) | Checks extraction from alternative Shopee markup. |
| [shopee-rating-only-fixture.html](../tests/shopee-rating-only-fixture.html) | Checks a review containing stars but no written text. |
| [panel-fixture.html](../tests/panel-fixture.html) | Checks the panel, analysis payload, and displayed results using controlled reviews and a simulated API. |
| [panel-lazy-text-fixture.html](../tests/panel-lazy-text-fixture.html) | Checks handling of review text that appears after the page first loads. |
| [panel-invalidated-context-fixture.html](../tests/panel-invalidated-context-fixture.html) | Checks recovery when reloading/disabling an extension invalidates its page connection. |
| [assets/review-photo.svg](../tests/assets/review-photo.svg) | Local example buyer picture for the browser fixtures. It is outside the thesis dataset. |

| File in baseline `tests/` | What it checks |
| --- | --- |
| [test_adaptation.py](../baselines/clip-ca-cg/tests/test_adaptation.py) | Six-category targets, mixed sentiments, product separation, feature-cache reuse, missing/real photos, attention behavior, learning updates, and saved checkpoint/JSON behavior. Uses miniature offline models. |
| [test_api.py](../baselines/clip-ca-cg/tests/test_api.py) | Baseline HTTP input/output, retained review IDs, predictor reuse, missing-model/cache errors, and allowed frontend origins. Uses a fake predictor. |

**Recommendation: keep both test folders in Git and leave them collapsed during
normal dataset/model work.** Normal training and APIs do not import them, so
removing them would not remove a trained model; it would remove these checks.
The tiny models exist only to run checks quickly, and temporary learned files
are cleaned up. Test logs can show one epoch or poor miniature-model scores;
those are software exercises and are not the real training results.

The redundant baseline launcher and older installer test have already been
removed from the current repository. The retained files cover current behavior.
Keep `test_own_model.py` if keeping its integration test, because that integration
test imports its sample-data helper.

For defense preparation, groupmates should explain the model flow and outputs;
the person maintaining the code can run these checks after relevant changes.
They do not need to run before every new review prediction.

From the root, run all Python checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py" -v
```

From `baselines/clip-ca-cg`, run baseline/API checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Extension checks, when Node is installed, run from the root:

```powershell
node tests/check_extension_contract.mjs
node tests/test_extension_runtime.mjs
```

The HTML fixtures are separate browser checks. Serve the project root, then open
the desired page under `/tests/`; its page title/result indicates PASS or FAIL:

```powershell
.\.venv\Scripts\python.exe -m http.server 5501 --bind 127.0.0.1
```

For example, open `http://127.0.0.1:5501/tests/extractor-fixture.html`.
These controlled pages do not establish that extraction works on every current
marketplace page; check the actual page diagnostics too.

### Local tool folders and navigation

| Item | Meaning |
| --- | --- |
| `.venv/` in each model project | Installed Python and libraries. Use its interpreter, but keep the folder collapsed. |
| `__pycache__/` | Automatically generated Python bytecode. It can be regenerated and is ignored by Git. |
| `.git/` | Git's repository history and bookkeeping. Leave it managed by Git. |
| `.vscode/` | Local editor settings. It is outside the model's learned behavior. |
| `debug.log`, other logs | Local diagnostics, not saved model weights or final accuracy reports. |

Open `Authenticheck-ThesisProject/` as the VS Code project when working on this
repository. Keep `.venv`, caches, saved weights, and tests collapsed, and open the
folder relevant to the current task. Baseline source is in `baselines/clip-ca-cg/`.
The generated output folders are not additional source folders to maintain.

### Troubleshooting

| Problem | Check |
| --- | --- |
| `/health` works but `/ready` is false | Read its `error`; confirm matching weights/tokenizers/metadata and the selected folder, then restart. |
| Dataset fingerprint mismatch | Generate a new split version and use it for both newly trained models. |
| Five product folds cannot be formed | Check product/class coverage; a row-level fallback is not used. |
| CLIP revision mismatch or cached image bytes missing | Restore the matching cache; rebuild features with `--force` if photos/features are inconsistent. |
| HTTP 422 | Check schema version, supported fields, nonblank text, review count, and photo URLs; baseline IDs must be unique. |
| Inference fails with missing/invalid stars | AuthentiCheck requires integer ratings 1 to 5; inspect extraction instead of filling invented ratings. |
| Stage 1 input-contract mismatch | Regenerate OOF probabilities/features and retrain quality RoBERTa/XGBoost for the combined input. Select the rebuilt bundle and restart the API. |
| No baseline comparison appears | The service exists, but the frontend comparison message/call/rendering still needs implementation. |
| No models/outputs after cloning | Train locally or restore a compatible bundle; ignored files are not supplied by Git. |

## 9. Thesis scope and baseline sources

Current code supports the fixed taxonomy, shared product splits, OOF features,
separate final testing, missing-image handling, both prediction APIs, and local
practice runs. Final real-dataset training/evaluation, annotation agreement,
formal statistical tests, empirical Tagalog/Taglish CLIP alignment, and frontend
comparison integration remain work to complete. Use human-labeled unseen products
for final accuracy claims, and record dataset versions, seeds, environments,
thresholds, training settings, and model versions for reproducibility.

The baseline is an adaptation for this project's categories, frozen encoders,
missing photos, and aspect-level outputs. Its reference does not establish an
exact reproduction of the paper's reported experiments.

Starting repository: [charlesczar/CLIP-CA-CG-model](https://github.com/charlesczar/CLIP-CA-CG-model),
commit `fa5cf5dc36cf278c57c2eb2a25bb217c5ca92e56`.
Architecture reference: [Lu et al. (2024)](https://thesai.org/Downloads/Volume15No2/Paper_90-Cross_Modal_Sentiment_Analysis_Based_on_CLIP_Image.pdf).


<!-- published-augmented-experiment-bundle -->

## Download the completed experimental AuthentiCheck bundle

The final inference bundle is tracked under `models/augmented_experiment/` using Git LFS.
It includes RoBERTa, XGBoost, ABSA, their tokenizers/configurations and provenance.
OOF fold weights and intermediate checkpoints are not needed for inference.
The model was trained with unreviewed synthetic drafts; `publication.json` records
verification and dataset provenance. Publishing is not a final test evaluation.

Run from the repository root after pulling:

```powershell
git lfs install
git lfs pull --include="models/augmented_experiment/**"
$env:AUTHENTICHECK_MODEL_BUNDLE = "models/augmented_experiment"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The API loads the selected bundle when it starts. Frozen multilingual CLIP models
are downloaded separately on first use; no CLIP fine-tuning weights were produced.

### Diagnose Stage 1 before another full training run

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/audit_stage1.py --allow-unreviewed-augmentations
```

This checks the `quality_balanced_v5` bundle and saved feature provenance, then
fits 24 small diagnostic XGBoost candidates using training rows and scores them
on validation rows. It compares probabilities alone, probabilities with rating,
probabilities with image similarity, and all six features, with three tree
settings and two training populations. The original-only population removes
synthetic rows from XGBoost fitting; its OOF encoders still used eligible
synthetic rows, so this is not a complete no-augmentation experiment.

Read `reports/generated/stage1_validation_audit.json`: `validation_baselines`
contains the current models' validation scores, `candidate_trials` ranks the
diagnostic fits by validation macro-F1, and `validation_errors` lists incorrect
current XGBoost predictions with their review text and features. No test
predictions are computed and no model weights are replaced. Choose a new
`--output` path to preserve a previous audit. A better validation score is a
candidate for a future experiment, not a new test result.

### SOP 3: compare product sentiment separately for each aspect

New runs of `stage2/evaluate_models.py` include `aspect_filtering_ablation` in
addition to the existing pooled `filtering_ablation`. To add this section to
an existing report without retraining, image downloads, or model inference:

```powershell
.\.venv\Scripts\python.exe stage2/evaluate_models.py --from-report results/quality_balanced_v5_test.json --output results/quality_balanced_v5_test_per_aspect.json
```

The source report is preserved. The new report contains all its original sections
plus the new per-aspect evaluation. The saved human gold pairs must cover all
ground-truth Authentic test reviews. Both prediction approaches must cover the
same reviews and use the same ABSA predictions, with only the quality gate changed.

For every review and aspect, distinct Negative, Neutral, and Positive labels map
to -1, 0, and +1. Average those values within the review first, then average the
review scores within each product/aspect. For example, a review with both Positive
and Negative for performance contributes 0 once; it does not count as two reviews.
Repeated evidence for the same polarity does not add weight. The reference uses
human annotations only from ground-truth Authentic reviews. The filtered score
uses reviews predicted Authentic; the unfiltered score uses every review. Both
approaches use our same trained ABSA model, not the separate CLIP-CA-CG model.

Inside `aspect_filtering_ablation`:

| Field | What to read |
| --- | --- |
| `per_product_aspect` | All product/aspect reference and predicted scores, review counts, absolute errors, paired eligibility, and exclusion reasons. |
| `by_aspect` | Paired filtered and unfiltered MAEs for each of the ten aspects, with reference, eligible, excluded, and missing-prediction counts. |
| `overall.macro_mae_across_aspects` | Average of aspect MAEs, giving each aspect with an eligible pair equal weight. |
| `overall.pooled_product_aspect_mae` | Average across all eligible product/aspect errors; aspects with more eligible products have more weight. |

Lower MAE means closer to the human reference. Positive `improvement` means
unfiltered error minus filtered error is positive, so filtering helped.
Both approaches' MAEs always use the same eligible product/aspect pairs.
Undefined scores are JSON `null`, never zero. Missing-prediction counts are
reported among products with a reference; predictions for aspects with no
reference are recorded separately as `*_without_ground_truth_count`. Exclusions
must be reported alongside MAE, because missed aspects would otherwise disappear
from the error comparison. An aspect without evaluable pairs has null MAEs.

These scores use a different aggregation unit from the old pooled product score,
so their numerical values need not match `filtering_ablation`. Statistical tests
remain pending; multiple aspects of the same product must not automatically be
treated as independent product samples.
