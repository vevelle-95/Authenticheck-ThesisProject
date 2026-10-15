# Model performance in the extension

The expandable **Model performance** section appears above loading, empty, error,
and product result states. Its summary always shows the published quality test
Macro-F1, or **Not evaluated**. Expanding it shows macro-precision, macro-recall,
accuracy, task comparison, evaluation provenance, per-class metrics and confusion
matrices when available. JSON exports include the published evaluation record.

Scores come from `extension/performance-data.js`, independently of the live API.
They describe the named evaluated bundle, not necessarily the connected API model.
No held-out reports are included yet. Local estimates never receive model scores.

## Publish measured results

First run the existing held-out evaluations described in `PROJECT_GUIDE.md`, using
the dataset, augmentation settings and splits that match each trained checkpoint.
Do not use training or validation scores in this section. Do not repeatedly tune
against held-out test results.

Then publish compact summaries from the resulting reports, for example:

```powershell
node scripts/publish_performance.mjs --own results/augmented_experiment_test.json --bundle augmented_experiment --dataset test_reviews.csv --split splits_real_v2.json --evaluated-at 2026-10-15
```

The date must be the actual evaluation date. The bundle identifies the evaluated
model version; use an immutable version or artifact digest when available.
Dataset and split names must identify the actual evaluation data. The publisher
stores a SHA-256 of each source report and excludes raw reviews and predictions.

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
