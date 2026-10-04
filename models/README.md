# Saved models

Training with `pipeline.py` saves the updated models in
`models/own_model_v2/`. Read [ModelNavigation.md](../ModelNavigation.md) for
the training commands.

## Current output layout

```text
models/
  own_model_v2/
    dost_roberta/
      config.json
      model.safetensors (or pytorch_model.bin)
      tokenizer files
      training_metadata.json
      checkpoints/
    xgboost_meta_classifier.json
    xgboost_meta_classifier.metadata.json
    absa_model/
      model.pt
      absa_config.json
      encoder/config.json
      tokenizer/
      training_metadata.json
    oof/
      fold_0/ ... fold_4/
    manifest.json
```

| Artifact | Used for |
| --- | --- |
| `dost_roberta/` | Predicting four quality probabilities from review text. |
| `xgboost_meta_classifier.json` | Predicting the final quality label from six features. |
| `absa_model/` | Predicting ten-category aspects and their sentiments. |
| `oof/` | Generating training features; normal inference does not load these fold models. |
| Metadata and manifest files | Recording labels, settings, training membership, and validation results. |

The pipeline creates these directories during training on the finalized dataset.
The software integration test creates its own temporary models and removes them
after the check; it does not create your trained thesis bundle here.

CLIP uses pretrained encoders downloaded/cached by the model library. The
pipeline does not fine-tune CLIP or save it in this bundle.

## Earlier artifacts

`models/dost_roberta/`, `models/absa_model/`, and the XGBoost file directly
under `models/` are from the earlier architecture. The backend now defaults
to `own_model_v2/`.

The tracked `models/manifest.json` and `scripts/setup_models.py` describe and
install/verify the **older download bundle**. Running that installer does not
produce the updated ten-category model. The new pipeline writes a separate
`models/own_model_v2/manifest.json`.

Trained weights and generated bundles are excluded from Git by `.gitignore`.
