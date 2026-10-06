# Manuscript-based system completion audit

Assessment date: October 6, 2026. Source: the supplied AuthentiCheck manuscript,
particularly Chapter 1 Scope and Limitations and Chapter 3 System Architecture,
including Output and Visualization Layer.

Source: https://docs.google.com/document/d/1B0DKF8nduDa0JpJlAh68k_yOtmIMmM2HiZAHVppnjZQ/edit?tab=t.ftjyr72uu91

This is a proposed engineering rubric, not a department-approved percentage.
Ten core capabilities have equal weight. Implemented and verified on available
test inputs = 1; implemented but with missing artifacts or incomplete live
verification = 0.5; absent = 0. Passing a software test establishes behavior on
those inputs, not reliable model accuracy or readiness on all marketplace pages.

| Manuscript capability | Score | Evidence and remaining work |
|---|---:|---|
| Shopee/Lazada review text, buyer images, and stars as input | 0.5 | Adapters and API exist. Live extraction on both marketplaces still needs verification; reviewed production dataset is unfinished. |
| Fine-tuned DOST-RoBERTa four-class text probabilities | 0.5 | Training/inference code exists and miniature training round-trip passed. Current served bundle uses a compact random encoder, not a newly trained DOST v2 model. |
| Frozen M-CLIP image/text similarity | 0.5 | Feature extraction code exists. Development serving omits images and integration tests mock image embeddings; real image inference is not verified. |
| Integer star validation and min-max normalization | 1 | Shared normalization tested; API rejects missing/fractional stars and extension excludes ineligible reviews. |
| Six-dimensional feature concatenation | 1 | Shared feature order and six-feature checks verified in model tests and training integration. |
| XGBoost four-class review-quality classification | 0.5 | Trained compact-bundle classifier runs. Full DOST/M-CLIP feature bundle remains to be trained and verified. |
| Authentic-only gate before ABSA | 1 | Automated test confirms only predicted Authentic reviews reach ABSA. |
| Aspect detection and aspect-conditioned polarity | 0.5 | Separate ABSA training/reloading/inference tested. Current compact predictions are degenerate; usable pretrained practice bundle remains unfinished. |
| Product-level sentiment aggregation | 1 | Backend aggregation and filtered/unfiltered absolute-error computation implemented and tested on fixtures. Real-data validation remains pending. |
| Classification, per-review aspects, aggregates, charts in user interface | 0.5 | Review classes, aspect summaries, charts and export exist. Per-review aspect polarities and numeric product sentiment aggregates are not fully exposed by the panel. |

Total: 7/10 = **70% estimated core system implementation completion** under this
rubric. The four full-credit capabilities are software operations verified on
fixtures. This is not a prediction-accuracy score, validated deployment readiness,
or proof of meeting the department's 90% threshold.

## Path to 90% under this rubric

Bring four partial capabilities to full credit: deploy and verify the pretrained
DOST v2 text model; verify real M-CLIP image inference; train and verify XGBoost
using the intended six features; complete the per-review aspect and product-level
sentiment presentation. This adds 2 points (20 percentage points). Fix ABSA's
prediction behavior and verify live marketplace extraction too; those are core
defense risks even if a numeric checklist reaches 90%.

## Research completion is separate

The manuscript also requires reviewed ground truth, shared product-disjoint
partitions, training-only augmentation where used, macro-F1 comparisons with
text-only DOST and the aspect-conditioned CLIP-CA-CG adaptation, filtering
absolute-error comparisons, and hypothesis testing. Some evaluation and baseline
code exists, but completed real-data experiments are not established. The own-model
evaluation script explicitly defers statistical tests. These tasks are not counted
as completed by the core system score above.

The manuscript excludes formal speed, memory, time-complexity, and scalability
evaluation; these should not be invented as thesis completion requirements.
Video analysis is out of scope. English challenge examples were exploratory and
do not establish completion for the manuscript's Tagalog/Taglish target domain.
