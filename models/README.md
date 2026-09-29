# AuthentiCheck model artifacts

This directory intentionally keeps trained weights out of normal Git history.
`manifest.json` is tracked because it defines the model version, class order,
feature order, and required artifact layout.

Expected local layout:

```text
models/
  manifest.json
  dost_roberta/
    config.json
    model.safetensors (or pytorch_model.bin)
    tokenizer files
  xgboost_meta_classifier.json
  absa_model/
    absa_config.json
    model.pt
```

The normal development path creates these files locally from the synthetic
integration dataset:

```powershell
pip install -r requirements.txt
python pipeline.py
python scripts/setup_models.py --verify-only
```

If the team later distributes a trained bundle, place its URL and SHA-256 digest
in `manifest.json`, or install a bundle already on the computer:

```powershell
python scripts/setup_models.py --archive C:\path\to\authenticheck-models.zip
```

The archive must contain the `models/` directory shown above. The setup script
rejects unsafe archive paths and verifies the manifest after extraction.
