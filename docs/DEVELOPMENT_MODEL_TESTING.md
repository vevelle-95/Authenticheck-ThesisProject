# Development model testing

The compact development bundle exercises the v2 software pipeline before final
annotations are available. It uses `data/test_reviews.csv`, including draft and
synthetic annotations, a randomly initialized compact RoBERTa encoder, and one
training epoch. It is not the pretrained DOST thesis model. Buyer images are
omitted during training and serving; their similarity feature is zero.

The training workflow retains product-disjoint partitions and five-fold
out-of-fold features. Validation selects checkpoints and the ABSA threshold.
The training script checks inference on four validation records and does not
evaluate the held-out test partition. Results cannot support thesis accuracy
claims.

From the repository root, run the offline integration test:

```powershell
.\.venv-dev\Scripts\python.exe -m unittest discover -s tests -p test_own_model_integration.py
```

Train the isolated bundle:

```powershell
.\.venv-dev\Scripts\python.exe scripts/train_development_model.py
```

Artifacts are saved in `models/development_v2/`; training intermediates are in
`data/development_v2/`. The manifest records the test limitations and smoke
response. Existing `models/own_model_v2/` and older weights are preserved.

Start the development API:

```powershell
.\.venv-dev\Scripts\python.exe scripts/serve_development_model.py
```

In the extension popup enable **Use model API**, set the endpoint to
`http://127.0.0.1:8001/analyze`, and save. Reload the extension and refresh the
marketplace tab after changing extension source. Scroll to reviews and rescan.
The panel labels this bundle **Development model**. Use `/ready` on port 8001
to check readiness. Keep the server terminal open; Ctrl+C stops it.

Written text and an integer star rating from 1 to 5 are required for analysis.
The extension retains unrated reviews as **Not analyzed** and omits them from
model requests. Direct requests with missing or fractional ratings return HTTP
422 rather than reaching inference. Development evidence names the compact
encoder and explicitly states that buyer images are omitted.

When the finalized dataset is ready, use `pipeline.py` with the pretrained DOST
encoder to train the thesis bundle, then switch back to the normal port 8000 API.
