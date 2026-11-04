# Real-error QA configs

This directory is intentionally tracked so the default split path exists after checkout.

Generate the frozen Lung_L split with:

```bash
python scripts/real_error_qa/build_patient_splits.py \
  --data-root ../radformation-data/data \
  --roi Lung_L \
  --qa-test-fraction 0.20 \
  --n-folds 5 \
  --seed 88
```

By default this writes:

- `patient_splits_lung_l.json`
- `patient_splits_lung_l.csv`

The JSON is then consumed by `scripts/real_error_qa/export_nnunet_lung_l.py`.
