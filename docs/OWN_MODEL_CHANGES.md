# Technical notes for our model

For the everyday file map, commands, and output directories, read
[ModelNavigation.md](../ModelNavigation.md) first. This page records details
needed when changing the code or planning an experiment.

## Training and validation rules

- One saved manifest assigns approximately 70/15/15 training/validation/test
  partitions. Each product stays in one partition.
- The training partition has five product-disjoint OOF folds. Each fold model
  trains on four folds and predicts the remaining fold. The held fold is not
  used to select its checkpoint.
- Final RoBERTa trains on the full training partition. Validation macro-F1
  selects its checkpoint. Test reviews are not used in training.
- XGBoost training uses OOF probabilities; validation uses final RoBERTa
  probabilities. CLIP similarity and normalized stars complete both feature sets.
- XGBoost candidates use class weights calculated from training data.
  Validation macro-F1 selects their settings.
- ABSA starts independently from the pretrained DOST encoder. Only
  human-labeled Authentic training reviews supervise it. Authentic validation
  reviews select the checkpoint and detection threshold.

The splitter requires all four quality classes in each partition, at least five
training products per quality class, and all four fitting classes in every OOF
run. It fails if it cannot construct suitable product groups; it does not switch
to a split that mixes reviews from the same product across partitions.

ABSA training requires all three sentiments in the Authentic training subset.

## Features and model outputs

RoBERTa and CLIP receive review text; product titles, descriptions, and source
labels are not learned inputs. Offline and online extraction use the same code.

The six values are:

```text
P(Authentic), P(Deceptive), P(LIV), P(Irrelevant),
maximum review-image cosine similarity over usable matched images,
(stars - 1) / 4
```

Stars must be integers from 1 to 5. At least one matched buyer image must be
accessible and decodable during feature generation. Missing inputs cause an
error rather than an invented rating or similarity.

ABSA has ten independent category probabilities. Its sentiment head reads
the target category definition together with the review. The saved validation
threshold determines which categories are detected during inference.

Each review/category has one adjudicated sentiment. Repeated same-sentiment
evidence is merged; conflicting sentiments require annotation resolution.
The model predicts categories and sentiments, not evidence spans.

Sensory eligibility uses product category metadata. That mapping is deferred
during current annotation work, but the existing code still enforces it during
training/inference. Definitions remain in [ASPECT_TAXONOMY.md](ASPECT_TAXONOMY.md).

## Experiment records and final testing

The split manifest contains a dataset fingerprint. Changing dataset content
invalidates reuse of that manifest; create a new experiment manifest.

A separate preparation command is available:

```powershell
python pipeline.py --data data/reviews.csv --splits data/splits.json --prepare-only
```

For a separate experiment, keep its models, intermediates, and splits together:

```powershell
python pipeline.py --data data/reviews.csv --splits data/experiment_01/splits.json --work-dir data/experiment_01/features --output models/experiment_01 --epochs 3 --batch-size 2
```

Its held-out evaluation command is:

```powershell
python stage2/evaluate_models.py --data data/reviews.csv --splits data/experiment_01/splits.json --bundle models/experiment_01 --output results/experiment_01_test.json
```

Evaluation checks saved training/validation membership against the manifest.
Quality metrics use all four fixed labels; sentiment metrics use all three.
Standalone sentiment evaluation supplies the human target category, separately
from predicted-category detection evaluation.

Product sentiment averages valid review/category predictions using +1/0/-1.
It is separate from average stars. Products with no valid predictions have an
undefined aggregate. The filtering comparison reports absolute errors and
excludes undefined pairs from paired analysis.

Existing final-report files are not overwritten. Freeze model choices before
final testing. A new output filename alone does not make a previously inspected
test set suitable for tuning and then reporting unbiased final performance.

## Work still deferred

- Training and measuring performance on the finalized real dataset.
- Product category mapping before training with sensory annotations.
- Lu et al. comparison adapters/routes and their older category projection.
- Formal statistical tests and annotation agreement measurements.
- Reviewed synthetic augmentation; the pipeline does not generate it.
- Empirical validation of Tagalog/Taglish CLIP alignment.

The extension's heuristic preview is separate from the trained model.
Software-test scores are also separate from thesis performance evidence.
