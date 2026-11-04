# Real-error QA: Lung_L nnU-Net v2 pilot

This pilot freezes patient identities before any model training and creates two disjoint cohorts:

- `qa_development`: used to train segmentation models and to generate out-of-fold (OOF) predictions for QA training.
- `qa_test`: never used to train segmentation or QA models; used only for final evaluation.

Within `qa_development`, every patient belongs to exactly one of five OOF validation folds. Reuse the same split JSON for every segmentation architecture.

## 1. Create the frozen split

```bash
python scripts/real_error_qa/build_patient_splits.py \
  --data-root ../radformation-data/data \
  --roi Lung_L \
  --qa-test-fraction 0.20 \
  --n-folds 5 \
  --seed 88 \
  --output-json configs/real_error_qa/patient_splits_lung_l.json
```

The script also writes `patient_splits_lung_l.csv` beside the JSON. Do not regenerate this split with a different seed after experiments begin.

## 2. Export nnU-Net v2 raw data

Set nnU-Net's three storage locations first. Example:

```bash
export nnUNet_raw=/scratch/shaohan/nnUNet_raw
export nnUNet_preprocessed=/scratch/shaohan/nnUNet_preprocessed
export nnUNet_results=/scratch/shaohan/nnUNet_results
```

Then export the data:

```bash
python scripts/real_error_qa/export_nnunet_lung_l.py \
  --data-root ../radformation-data/data \
  --split-json configs/real_error_qa/patient_splits_lung_l.json \
  --dataset-id 501 \
  --dataset-name RadformationLungLRealError
```

By default CT files are symlinked to avoid duplicating large volumes. Approved `Lung_L` masks are rewritten as binary masks in CT geometry using nearest-neighbor resampling if necessary.

The resulting raw dataset contains:

```text
Dataset501_RadformationLungLRealError/
├── dataset.json
├── imagesTr/       # QA-development CTs
├── labelsTr/       # QA-development approved Lung_L masks
├── imagesTs/       # fully held-out QA-test CTs
├── qa_test_gt/     # final-evaluation GT; never used for training
├── splits_final.json
├── real_error_case_manifest.csv
└── real_error_dataset_metadata.json
```

## 3. Plan and preprocess

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

nnU-Net reads custom folds from the *preprocessed* dataset directory, so install the frozen split after preprocessing:

```bash
python scripts/real_error_qa/export_nnunet_lung_l.py \
  --data-root ../radformation-data/data \
  --split-json configs/real_error_qa/patient_splits_lung_l.json \
  --dataset-id 501 \
  --dataset-name RadformationLungLRealError \
  --install-preprocessed-split
```

The exporter is deterministic and rewrites the same raw dataset. `--install-preprocessed-split` copies the same fold definition to:

```text
$nnUNet_preprocessed/Dataset501_RadformationLungLRealError/splits_final.json
```

## 4. Train the five OOF segmenters

For each fold `0..4`, train `3d_fullres`:

```bash
nnUNetv2_train 501 3d_fullres 0
nnUNetv2_train 501 3d_fullres 1
nnUNetv2_train 501 3d_fullres 2
nnUNetv2_train 501 3d_fullres 3
nnUNetv2_train 501 3d_fullres 4
```

For the real-error dataset, each QA-development patient must be predicted by the fold model for which that patient was in `val`, never by a model trained on that patient. The later OOF-inference script should enforce this from `patient_splits_lung_l.json` rather than infer it from filenames.

## Leakage rules

1. `qa_test` patients never enter segmentation training, QA training, threshold selection, or model selection.
2. Split by patient, never by contour/prediction row.
3. OOF QA-development predictions must come from the fold that excluded that patient from training.
4. Keep `qa_test_gt/` separate from `labelsTr/`.
5. Reuse this exact split for nnU-Net, SegResNet, SwinUNETR, and downstream QA comparisons.

## Train the five 3d_fullres OOF folds

After preprocessing and installing the frozen `splits_final.json`, submit one
Slurm array job:

```bash
sbatch --array=0-4 scripts/real_error_qa/train_nnunet_lung_l.slurm
```

The script uses the Hugo partition/account and the existing project email
settings. By default it activates the `nnunet` conda environment and trains
`Dataset501` with `3d_fullres`. It passes `--npz` so validation probabilities
are retained for later QA uncertainty features.

Useful submission-time overrides are environment variables, for example:

```bash
sbatch --array=0-4 \
  --export=ALL,NNUNET_ENV=nnunet,NNUNET_DATASET_ID=501 \
  scripts/real_error_qa/train_nnunet_lung_l.slurm
```

Do not train a fold on a different split file. The study depends on each of the
102 development cases being predicted by exactly the model whose training set
excluded that case.

## Collect OOF validation predictions

`nnUNetv2_train` runs validation when a fold finishes, so no second inference
pass is needed for the 102 development cases. Once all five array jobs finish:

```bash
conda activate nnunet
export nnUNet_raw=/scratch/shaohan/nnUNet_raw
export nnUNet_results=/scratch/shaohan/nnUNet_results

python scripts/real_error_qa/collect_nnunet_oof.py \
  --split-json configs/real_error_qa/patient_splits_lung_l.json \
  --dataset-id 501 \
  --dataset-name RadformationLungLRealError \
  --configuration 3d_fullres \
  --require-probabilities
```

The default canonical output is:

```text
/scratch/shaohan/contour_QA/real_error_segqa/
  predictions/oof/nnunet/Lung_L/3d_fullres/
    segmentations/
    probabilities/
    oof_predictions.csv
    collection_summary.json
```

The collector uses the frozen split as the source of truth and fails if an
expected validation prediction is missing or duplicated. A successful run for
the current pilot should report exactly 102 OOF predictions.

The next workflow stage is to compare these predictions against the approved
Lung_L contours in native CT geometry and build the real-error metric manifest
(Dice, 1-Dice, HD95, ASSD, and volume error).
