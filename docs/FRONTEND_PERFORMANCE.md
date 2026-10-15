# Model performance in the extension

The **Model performance** tab sits beside **Reviews** and shows task comparisons,
evaluation provenance, per-class metrics and confusion matrices. The separate
**Test results** tab focuses on the labeled CSV test split: accuracy, macro
precision, macro recall, macro F1, and review-ID examples of ground truth versus
prediction. Both tabs remain available during loading, empty and error states.
Missing scores show **Not evaluated**. JSON exports include published evaluation
records.

Scores come from `extension/performance-data.js`, independently of the live API.
They describe the named evaluated bundle, not necessarily the connected API model.
No held-out reports are included yet. Local estimates never receive model scores.

`model_metrics.py` computes scores from supplied target and prediction arrays;
it does not open the CSV or choose the test partition. `stage2/evaluate_models.py`
loads the CSV and split manifest, checks training provenance, selects only the
test partition, runs inference, and calls the metrics helper. Live marketplace
reviews have no ground-truth annotations available to this extension.

The final XGBoost quality predictions are evaluated against `ground_truth`.
The RoBERTa-only ablation also uses `ground_truth` for comparison. A separate
RoBERTa text-target evaluation uses `text_label`; the UI names these targets.

## Publish measured results

First run the existing held-out evaluations described in `PROJECT_GUIDE.md`, using
the dataset, augmentation settings and splits that match each trained checkpoint.
Do not use training or validation scores in this section. Do not repeatedly tune
against held-out test results.

For example, after training the compatible `text_target_v5` bundle:

```powershell
.\.venv-dev\Scripts\python.exe stage2/evaluate_models.py --data data/test_reviews.csv --splits data/splits_real_v2.json --bundle models/text_target_v5 --output results/text_target_v5_test.json
```

Use the exact CSV, splits, and augmentation settings used for that bundle. If
training included augmentations, pass the same `--augmentations` path and, only
if used during training, `--allow-unreviewed-augmentations`. Older incompatible
weights must be rebuilt before they can produce a current test report. The
evaluator refuses to overwrite an existing final report.

Then publish compact summaries from the resulting reports, for example:

```powershell
node scripts/publish_performance.mjs --own results/text_target_v5_test.json --bundle text_target_v5 --dataset test_reviews.csv --split splits_real_v2.json --evaluated-at 2026-10-16
```

The date must be the actual evaluation date. The bundle identifies the evaluated
model version; use an immutable version or artifact digest when available.
Dataset and split names must identify the actual evaluation data. The publisher
stores a SHA-256 of each source report and excludes review text and usernames.
Up to 20 review-ID/ground-truth/prediction examples are included in report order;
these are a bounded preview, while aggregate metrics use the full test split.
The UI identifies the number of recorded comparisons and displayed examples.

Optionally append `--baseline baselines/clip-ca-cg/outputs/test_metrics.json
--baseline-version <checkpoint-version>` when a real baseline test report exists.
The dataset/split options apply to both reports: verify both used that source.
Reload the unpacked extension and refresh the marketplace tab after publishing.

AuthentiCheck's text-only quality ablation and full quality model use the same
report and test reviews. The external baseline has no four-class quality output
and uses six ABSA aspects versus AuthentiCheck's ten. The UI therefore displays
their separate scopes without ranking them. A defensible direct ABSA comparison
requires a separate evaluation on shared aspects, labels and review IDs; this
publisher does not create that evaluation.

Accuracy is not reported for multilabel aspect detection in the existing reports.
Class precision, recall, F1 and support are calculated from each report's confusion
matrix, using zero for undefined class precision/recall. Missing metrics display
as unavailable, not zero.
