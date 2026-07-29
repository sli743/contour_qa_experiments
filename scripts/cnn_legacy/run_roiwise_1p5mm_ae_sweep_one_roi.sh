#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Unified ROI-wise 1.5 mm vanilla AE + better probe sweep
# ============================================================
# Run from /home/shaohan/VariantVAE
#
# Example:
#   bash run_roiwise_1p5mm_ae_sweep_one_roi.sh Lung_L
#   bash run_roiwise_1p5mm_ae_sweep_one_roi.sh Heart
#
# Env overrides:
#   RUN_TRAIN=0 RUN_PROBE_SWEEP=1 bash run_roiwise_1p5mm_ae_sweep_one_roi.sh Heart
#   CROP_XYZ=224,224,224 bash run_roiwise_1p5mm_ae_sweep_one_roi.sh Lung_L

ROI="${1:-}"
if [[ -z "${ROI}" ]]; then
  echo "Usage: bash $0 <ROI_NAME>"
  exit 1
fi

RAW_DATA_DIR="${RAW_DATA_DIR:-../radformation-data/data}"
PREP_ROOT_BASE="${PREP_ROOT_BASE:-../radformation-data/preprocessed_roiwise_sweep_1p5mm}"
CONFIG_DIR="${CONFIG_DIR:-configs/roiwise_1p5mm_sweep}"
SAVE_DIR="${SAVE_DIR:-/scratch/shaohan/experiments/VariantVAE}"

RESAMPLE_MM="${RESAMPLE_MM:-1.5}"
TRAIN_RATIO="${TRAIN_RATIO:-0.70}"
VALID_RATIO="${VALID_RATIO:-0.15}"
TEST_RATIO="${TEST_RATIO:-0.15}"
SEED="${SEED:-88}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_PROBE_SWEEP="${RUN_PROBE_SWEEP:-1}"
SKIP_PREPROCESS_IF_EXISTS="${SKIP_PREPROCESS_IF_EXISTS:-1}"

EVAL_SCRIPT="${EVAL_SCRIPT:-eval_post_dae_probe_sitk_morph_sweep_v2.py}"

CUT_LEVELS="${CUT_LEVELS:-0,0.1,0.2,0.3,0.4,0.5}"
LOCAL_FRAC_LEVELS="${LOCAL_FRAC_LEVELS:-0,0.1,0.2,0.3,0.4,0.5}"
VOXEL_LEVELS="${VOXEL_LEVELS:-0,1,2,3,4,5}"
DROP_INF_IN_PLOTS="${DROP_INF_IN_PLOTS:-1}"
SAVE_VIS="${SAVE_VIS:-0}"

BATCH_SIZE="${BATCH_SIZE:-4}"
ENC_CHS="${ENC_CHS:-[16, 32, 64, 128]}"
DEC_CHS="${DEC_CHS:-[128, 64, 32, 16]}"
Z_DIM="${Z_DIM:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-400}"
ALPHA="${ALPHA:-0.8}"
KLD_WEIGHT="${KLD_WEIGHT:-1.0e-3}"

mkdir -p "${CONFIG_DIR}"

case "${ROI}" in
  A_Aorta)        LABEL_ID=1;  GROUP="heart_core";        DEFAULT_CROP_XYZ="160,160,160" ;;
  A_Pulmonary)    LABEL_ID=2;  GROUP="heart_core";        DEFAULT_CROP_XYZ="160,160,160" ;;
  BrachPlx_L)     LABEL_ID=3;  GROUP="brachial";          DEFAULT_CROP_XYZ="160,128,96" ;;
  BrachPlx_R)     LABEL_ID=4;  GROUP="brachial";          DEFAULT_CROP_XYZ="160,128,96" ;;
  Bronchus)       LABEL_ID=5;  GROUP="airway";            DEFAULT_CROP_XYZ="128,128,128" ;;
  Carina)         LABEL_ID=6;  GROUP="airway";            DEFAULT_CROP_XYZ="128,128,128" ;;
  Esophagus)      LABEL_ID=7;  GROUP="mediastinum_tubes"; DEFAULT_CROP_XYZ="128,128,256" ;;
  Heart)          LABEL_ID=8;  GROUP="heart_core";        DEFAULT_CROP_XYZ="160,160,160" ;;
  Liver)          LABEL_ID=9;  GROUP="upper_abdomen";     DEFAULT_CROP_XYZ="192,192,128" ;;
  Lung_L)         LABEL_ID=10; GROUP="lungs_chest";       DEFAULT_CROP_XYZ="192,192,160" ;;
  Lung_R)         LABEL_ID=11; GROUP="lungs_chest";       DEFAULT_CROP_XYZ="192,192,160" ;;
  Ribs)           LABEL_ID=12; GROUP="lungs_chest";       DEFAULT_CROP_XYZ="192,192,160" ;;
  SpinalCord)     LABEL_ID=13; GROUP="spine";             DEFAULT_CROP_XYZ="96,96,192" ;;
  Stomach)        LABEL_ID=14; GROUP="upper_abdomen";     DEFAULT_CROP_XYZ="192,192,128" ;;
  Trachea)        LABEL_ID=15; GROUP="airway";            DEFAULT_CROP_XYZ="128,128,128" ;;
  V_VenaCava_S)   LABEL_ID=16; GROUP="mediastinum_tubes"; DEFAULT_CROP_XYZ="128,128,256" ;;
  NS_Chestwall)   LABEL_ID=17; GROUP="lungs_chest";       DEFAULT_CROP_XYZ="192,192,160" ;;
  *) echo "ERROR: Unknown ROI: ${ROI}"; exit 1 ;;
esac

CROP_XYZ="${CROP_XYZ:-${DEFAULT_CROP_XYZ}}"
PROJ_SIZE=$(python - <<PY
x, y, z = [int(v) for v in "${CROP_XYZ}".split(",")]
print(f"[{x}, {y}, {z}]")
PY
)

ROI_SAFE="${ROI}"
PREP_ROOT="${PREP_ROOT_BASE}/${GROUP}/${ROI_SAFE}"
CONFIG_PATH="${CONFIG_DIR}/label_ae_${ROI_SAFE}_roiwise_1p5mm.yaml"
EXP_NAME="label_ae_${ROI_SAFE}_roiwise_1p5mm"

SPLIT_DIR="${PREP_ROOT}/patient_splits"
TRAIN_IDS="${SPLIT_DIR}/training1_patients.txt"
VALID_IDS="${SPLIT_DIR}/validation1_patients.txt"
TEST_IDS="${SPLIT_DIR}/testing1_patients.txt"

TRAIN_OUT="${PREP_ROOT}/training1"
VALID_OUT="${PREP_ROOT}/validation1"
TEST_OUT="${PREP_ROOT}/testing1"

TRAIN_LABEL_DIR="${TRAIN_OUT}/${GROUP}/${ROI_SAFE}/labels"
VALID_LABEL_DIR="${VALID_OUT}/${GROUP}/${ROI_SAFE}/labels"
TEST_LABEL_DIR="${TEST_OUT}/${GROUP}/${ROI_SAFE}/labels"

RUN_DIR="${SAVE_DIR}/${EXP_NAME}_alpha${ALPHA}_kld0.001_wmuniform"
PROBE_SWEEP_DIR="${RUN_DIR}/probe_eval_sweep"

mkdir -p "${SPLIT_DIR}" "${CONFIG_DIR}"

echo "============================================================"
echo "ROI-wise 1.5 mm AE + better probe sweep"
echo "ROI:          ${ROI}"
echo "Label ID:     ${LABEL_ID}"
echo "Group:        ${GROUP}"
echo "Crop XYZ:     ${CROP_XYZ}"
echo "proj_size:    ${PROJ_SIZE}"
echo "Preprocess:   ${PREP_ROOT}"
echo "Config:       ${CONFIG_PATH}"
echo "Experiment:   ${EXP_NAME}"
echo "============================================================"

if [[ ! -f "${TRAIN_IDS}" || ! -f "${VALID_IDS}" || ! -f "${TEST_IDS}" ]]; then
  echo "[1/5] Creating deterministic split for ${ROI}..."
  python - <<PY
from pathlib import Path
import re, random

raw = Path("${RAW_DATA_DIR}")
image_dir = raw / "images"
label_dir = raw / "labels-approved"
split_dir = Path("${SPLIT_DIR}")
split_dir.mkdir(parents=True, exist_ok=True)

roi = "${ROI}"
ids = []
for p in sorted(image_dir.glob("patient*_image.nii.gz")):
    m = re.search(r"patient(\d+)_image\.nii\.gz$", p.name)
    if not m:
        continue
    pid = int(m.group(1))
    lab = label_dir / f"patient{pid}_{roi}_approved.nii.gz"
    if lab.exists():
        ids.append(pid)

if not ids:
    raise RuntimeError(f"No eligible patients found for ROI={roi}")

rng = random.Random(${SEED})
rng.shuffle(ids)

n = len(ids)
n_train = max(1, int(round(n * ${TRAIN_RATIO})))
n_valid = max(1, int(round(n * ${VALID_RATIO})))
if n_train + n_valid >= n:
    n_train = max(1, n - 2)
    n_valid = 1

train = sorted(ids[:n_train])
valid = sorted(ids[n_train:n_train+n_valid])
test = sorted(ids[n_train+n_valid:])
if not test and valid:
    test = [valid.pop()]
if not valid and len(train) > 1:
    valid = [train.pop()]

(split_dir / "training1_patients.txt").write_text("\n".join(map(str, train)) + "\n")
(split_dir / "validation1_patients.txt").write_text("\n".join(map(str, valid)) + "\n")
(split_dir / "testing1_patients.txt").write_text("\n".join(map(str, test)) + "\n")

print(f"Eligible: {n}")
print(f"Train/valid/test: {len(train)}/{len(valid)}/{len(test)}")
PY
else
  echo "[1/5] Reusing existing split files: ${SPLIT_DIR}"
fi

readarray -t TRAIN_PATIENTS < "${TRAIN_IDS}"
readarray -t VALID_PATIENTS < "${VALID_IDS}"
readarray -t TEST_PATIENTS < "${TEST_IDS}"

run_preprocess_split () {
  local split_name="$1"
  local out_dir="$2"
  shift 2
  local patients=("$@")

  local done_marker="${out_dir}/.preprocess_done"
  if [[ "${SKIP_PREPROCESS_IF_EXISTS}" == "1" && -f "${done_marker}" && -d "${out_dir}/${GROUP}/${ROI_SAFE}/labels" ]]; then
    echo "[2/5] ${split_name}: preprocessing already exists; skipping."
    return
  fi

  if [[ "${RUN_PREPROCESS}" != "1" ]]; then
    echo "[2/5] ${split_name}: RUN_PREPROCESS=0; skipping."
    return
  fi

  echo "[2/5] ${split_name}: preprocessing ${#patients[@]} patients for group=${GROUP}, ROI=${ROI}..."
  rm -rf "${out_dir}"

  python radformation_preprocess_roiwise.py     -d "${RAW_DATA_DIR}"     -o "${out_dir}"     -l approved     -r "${RESAMPLE_MM}"     --groups "${GROUP}"     --patients "${patients[@]}"     --crop "${GROUP}=${CROP_XYZ}"     -m     --debug-json     -v

  touch "${done_marker}"

  if [[ ! -d "${out_dir}/${GROUP}/${ROI_SAFE}/labels" ]]; then
    echo "ERROR: expected ROI labels not found:"
    echo "  ${out_dir}/${GROUP}/${ROI_SAFE}/labels"
    echo "Check ${out_dir}/debug/preprocess_skipped.csv and preprocess_roi_debug.csv"
    exit 1
  fi
}

run_preprocess_split "training1" "${TRAIN_OUT}" "${TRAIN_PATIENTS[@]}"
run_preprocess_split "validation1" "${VALID_OUT}" "${VALID_PATIENTS[@]}"
run_preprocess_split "testing1" "${TEST_OUT}" "${TEST_PATIENTS[@]}"

echo "[3/5] Writing config: ${CONFIG_PATH}"

cat > "${CONFIG_PATH}" <<YAML
save_dir: ${SAVE_DIR}
exp_name: ${EXP_NAME}
train: True
valid: True
test: True

train_dir: ${TRAIN_LABEL_DIR}
valid_dir: ${VALID_LABEL_DIR}
test_dir: ${TEST_LABEL_DIR}
train_sample_ratio: 1.0

proj_size: ${PROJ_SIZE}
proj_spacing: [${RESAMPLE_MM}, ${RESAMPLE_MM}, ${RESAMPLE_MM}]
proj_direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
save_volume: True

manual_seed: 88
resume: False
resume_path: ${SAVE_DIR}/${EXP_NAME}/checkpoints/
use_tf32: False
device: cuda:0

init_lr: 1.0e-4
batch_size: ${BATCH_SIZE}
num_epochs: ${NUM_EPOCHS}
weight_decay: 1.0e-4
recon_loss: dice_bce
kld_weight: [${KLD_WEIGHT}]
alpha: [${ALPHA}]
weight_mode: uniform
gamma: 2.0
beta_cyc_annealing: False

model: ae
dim: 3
residual_block: True
blocks_per_level: 1
in_ch: 1
out_ch: 1
train_label: True

enc_chs: ${ENC_CHS}
dec_chs: ${DEC_CHS}
bottle_ch: null
z_dim: ${Z_DIM}

normalization: Instance
activation: LeakyRelu
final_activation: Identity
downsample: Stride2
upsample: Stride2

ckpt_enabled: True
save_interval: 5
save_best: True
ckpt_file_load: final_weights.pth

pre_processing:
  normalize: False
  MinMaxNormalize: False
  clip: False
  clip_percent: 0.5
  train_augment: False
  valid_augment: False
  post_dae_train: False

lr_scheduler:
  type: no_decay
  factor: 0.9
  spe_arg: 10

early_stopping:
  type: loss_improve
  patience: 30
  min_epoch: 50
  min_improve: 0.0001

label_names:
  0: Background
  1: A_Aorta
  2: A_Pulmonary
  3: BrachPlx_L
  4: BrachPlx_R
  5: Bronchus
  6: Carina
  7: Esophagus
  8: Heart
  9: Liver
  10: Lung_L
  11: Lung_R
  12: Ribs
  13: SpinalCord
  14: Stomach
  15: Trachea
  16: V_VenaCava_S
  17: NS_Chestwall

label_mask: [0, ${LABEL_ID}]
YAML

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[4/5] Training/testing model..."
  python main.py -c "${CONFIG_PATH}"
else
  echo "[4/5] RUN_TRAIN=0; skipping training."
fi

if [[ "${RUN_PROBE_SWEEP}" == "1" ]]; then
  echo "[5/5] Running better probe severity sweep..."
  if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "ERROR: missing eval script: ${EVAL_SCRIPT}"
    exit 1
  fi

  CMD=(
    python "${EVAL_SCRIPT}"
    -c "${CONFIG_PATH}"
    --cut_levels "${CUT_LEVELS}"
    --local_frac_levels "${LOCAL_FRAC_LEVELS}"
    --voxel_levels "${VOXEL_LEVELS}"
  )

  if [[ "${DROP_INF_IN_PLOTS}" == "1" ]]; then
    CMD+=(--drop_inf_in_plots)
  fi
  if [[ "${SAVE_VIS}" == "1" ]]; then
    CMD+=(--save_vis)
  fi

  echo "${CMD[@]}"
  "${CMD[@]}"
else
  echo "[5/5] RUN_PROBE_SWEEP=0; skipping better probe sweep."
fi

echo
echo "Done for ROI=${ROI}."
echo "Config:     ${CONFIG_PATH}"
echo "Run dir:    ${RUN_DIR}"
echo "Sweep dir:  ${PROBE_SWEEP_DIR}"
