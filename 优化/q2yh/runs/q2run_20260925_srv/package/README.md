# Q2 frozen inference package

Rebuild the frozen model and reproduce the 30 Attachment-3 predictions:

```bash
python infer.py --input data/attachment3_input.npz --out /tmp/q2_out --device cuda:0
```

Device note: the frozen predictions shipped in `submission/` were produced on CUDA
(`--device cuda:1`).  Re-running on CUDA reproduces `q2_predictions.csv` bit-for-bit
(clean-process check: logits/raw max abs diff 0.0).  Running on CPU reproduces the same
30 rows and the same polarities, but the float32 logits drift by up to ~5e-4, so the
CPU output is not byte-identical to the shipped CSV.

Environment note: if a broken user-site TensorFlow/protobuf shadows the interpreter,
run with `PYTHONNOUSERSITE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0`.

The shared text encoder is stored once under `resources/bert_shared`; the candidate
checkpoint holds fusion/head weights only.  See `model/config.json` and `package.json`.
