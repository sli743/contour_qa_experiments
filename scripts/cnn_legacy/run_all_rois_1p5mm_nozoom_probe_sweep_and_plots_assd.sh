#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Unified ROI-wise 1.5 mm AE training + better probe sweep + plots
# WITHOUT zoom augmentation, with patient-variability-aware plots.
# ============================================================
# Run from /home/shaohan/VariantVAE
#
# Required scripts in current directory:
#   radformation_preprocess_roiwise.py
#   main.py
#   eval_post_dae_probe_sitk_morph_sweep_assd.py
#   plot_probe_sweep_unified_patient_variability_assd.py
#
# Defaults:
#   - all 17 ROIs
#   - 1.5 mm ROI-wise preprocessing
#   - no training/validation augmentation
#   - better probe sweep with Dice, HD95, and ASSD metrics
#   - old requested-level plots + patient-variability-aware volume-change plots
#
# Examples:
#   ROIS="Bronchus Lung_L" bash run_all_rois_1p5mm_nozoom_probe_sweep_and_plots.sh
#   RUN_TRAIN=0 RUN_PROBE_SWEEP=1 RUN_PLOTS=1 bash run_all_rois_1p5mm_nozoom_probe_sweep_and_plots.sh
#   BATCH_SIZE=2 ROIS="Lung_L Lung_R" bash run_all_rois_1p5mm_nozoom_probe_sweep_and_plots.sh

RAW_DATA_DIR="${RAW_DATA_DIR:-../radformation-data/data}"
PREP_ROOT_BASE="${PREP_ROOT_BASE:-../radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom}"
CONFIG_DIR="${CONFIG_DIR:-configs/roiwise_1p5mm_nozoom_assd_allrois}"
SAVE_DIR="${SAVE_DIR:-/scratch/shaohan/experiments/VariantVAE}"

RESAMPLE_MM="${RESAMPLE_MM:-1.5}"
TRAIN_RATIO="${TRAIN_RATIO:-0.70}"
VALID_RATIO="${VALID_RATIO:-0.15}"
SEED="${SEED:-88}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_PROBE_SWEEP="${RUN_PROBE_SWEEP:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
SKIP_PREPROCESS_IF_EXISTS="${SKIP_PREPROCESS_IF_EXISTS:-1}"

EVAL_SCRIPT="${EVAL_SCRIPT:-eval_post_dae_probe_sitk_morph_sweep_assd_vis.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-plot_probe_sweep_unified_patient_variability_assd.py}"

CUT_LEVELS="${CUT_LEVELS:-0,0.1,0.2,0.3,0.4,0.5}"
LOCAL_FRAC_LEVELS="${LOCAL_FRAC_LEVELS:-0,0.1,0.2,0.3,0.4,0.5}"
VOXEL_LEVELS="${VOXEL_LEVELS:-0,1,2,3,4,5}"
DROP_INF_IN_PLOTS="${DROP_INF_IN_PLOTS:-1}"
SAVE_VIS="${SAVE_VIS:-1}"

VOLUME_BIN_WIDTH="${VOLUME_BIN_WIDTH:-2.5}"
ERROR_BAND="${ERROR_BAND:-sd}"

BATCH_SIZE="${BATCH_SIZE:-4}"
ENC_CHS="${ENC_CHS:-[16, 32, 64, 128]}"
DEC_CHS="${DEC_CHS:-[128, 64, 32, 16]}"
Z_DIM="${Z_DIM:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-400}"
ALPHA="${ALPHA:-0.8}"
KLD_WEIGHT="${KLD_WEIGHT:-1.0e-3}"

AUGMENT_PROB="${AUGMENT_PROB:-0.0}"
ZOOM_PROB="${ZOOM_PROB:-0.0}"
ZOOM_MIN="${ZOOM_MIN:-1.00}"
ZOOM_MAX="${ZOOM_MAX:-1.00}"

ROIS="${ROIS:-A_Aorta A_Pulmonary BrachPlx_L BrachPlx_R Bronchus Carina Esophagus Heart Liver Lung_L Lung_R Ribs SpinalCord Stomach Trachea V_VenaCava_S NS_Chestwall}"

mkdir -p "${CONFIG_DIR}"

roi_metadata () {
  local roi="$1"
  case "${roi}" in
    A_Aorta)        echo "1 heart_core 160,160,160" ;;
    A_Pulmonary)    echo "2 heart_core 160,160,160" ;;
    BrachPlx_L)     echo "3 brachial 160,128,96" ;;
    BrachPlx_R)     echo "4 brachial 160,128,96" ;;
    Bronchus)       echo "5 airway 128,128,128" ;;
    Carina)         echo "6 airway 128,128,128" ;;
    Esophagus)      echo "7 mediastinum_tubes 128,128,256" ;;
    Heart)          echo "8 heart_core 160,160,160" ;;
    Liver)          echo "9 upper_abdomen 192,192,128" ;;
    Lung_L)         echo "10 lungs_chest 192,192,160" ;;
    Lung_R)         echo "11 lungs_chest 192,192,160" ;;
    Ribs)           echo "12 lungs_chest 192,192,160" ;;
    SpinalCord)     echo "13 spine 96,96,192" ;;
    Stomach)        echo "14 upper_abdomen 192,192,128" ;;
    Trachea)        echo "15 airway 128,128,128" ;;
    V_VenaCava_S)   echo "16 mediastinum_tubes 128,128,256" ;;
    NS_Chestwall)   echo "17 lungs_chest 192,192,160" ;;
    *) return 1 ;;
  esac
}

make_split () {
  local roi="$1"
  local split_dir="$2"
  local train_ids="$3"
  local valid_ids="$4"
  local test_ids="$5"

  if [[ -f "${train_ids}" && -f "${valid_ids}" && -f "${test_ids}" ]]; then
    echo "  Reusing split files: ${split_dir}"
    return
  fi

  python - <<PY
from pathlib import Path
import re, random

raw = Path("${RAW_DATA_DIR}")
image_dir = raw / "images"
label_dir = raw / "labels-approved"
split_dir = Path("${split_dir}")
split_dir.mkdir(parents=True, exist_ok=True)
roi = "${roi}"

ids = []
for p in sorted(image_dir.glob("patient*_image.nii.gz")):
    m = re.search(r"patient(\d+)_image\.nii\.gz$", p.name)
    if not m:
        continue
    pid = int(m.group(1))
    if (label_dir / f"patient{pid}_{roi}_approved.nii.gz").exists():
        ids.append(pid)

if not ids:
    raise RuntimeError(f"No eligible patients for ROI={roi}")

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

print(f"  Eligible patients: {n}")
print(f"  Train/valid/test: {len(train)}/{len(valid)}/{len(test)}")
PY
}

preprocess_split () {
  local split_name="$1"
  local out_dir="$2"
  local group="$3"
  local roi="$4"
  local crop_xyz="$5"
  shift 5
  local patients=("$@")

  local done_marker="${out_dir}/.preprocess_done"
  if [[ "${SKIP_PREPROCESS_IF_EXISTS}" == "1" && -f "${done_marker}" && -d "${out_dir}/${group}/${roi}/labels" ]]; then
    echo "  ${split_name}: preprocessing exists; skip."
    return
  fi

  if [[ "${RUN_PREPROCESS}" != "1" ]]; then
    echo "  ${split_name}: RUN_PREPROCESS=0; skip."
    return
  fi

  echo "  ${split_name}: preprocessing ${#patients[@]} patients..."
  rm -rf "${out_dir}"

  python radformation_preprocess_roiwise.py \
    -d "${RAW_DATA_DIR}" \
    -o "${out_dir}" \
    -l approved \
    -r "${RESAMPLE_MM}" \
    --groups "${group}" \
    --patients "${patients[@]}" \
    --crop "${group}=${crop_xyz}" \
    -m \
    --debug-json \
    -v

  touch "${done_marker}"

  if [[ ! -d "${out_dir}/${group}/${roi}/labels" ]]; then
    echo "ERROR: expected labels not found: ${out_dir}/${group}/${roi}/labels"
    exit 1
  fi
}

write_config () {
  local config_path="$1"
  local exp_name="$2"
  local train_label_dir="$3"
  local valid_label_dir="$4"
  local test_label_dir="$5"
  local proj_size="$6"
  local label_id="$7"

  cat > "${config_path}" <<YAML
save_dir: ${SAVE_DIR}
exp_name: ${exp_name}
train: True
valid: True
test: True

train_dir: ${train_label_dir}
valid_dir: ${valid_label_dir}
test_dir: ${test_label_dir}
train_sample_ratio: 1.0

proj_size: ${proj_size}
proj_spacing: [${RESAMPLE_MM}, ${RESAMPLE_MM}, ${RESAMPLE_MM}]
proj_direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
save_volume: True

manual_seed: 88
resume: False
resume_path: ${SAVE_DIR}/${exp_name}/checkpoints/
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
  augment_prob: ${AUGMENT_PROB}
  zoom_prob: ${ZOOM_PROB}
  zoom_min: ${ZOOM_MIN}
  zoom_max: ${ZOOM_MAX}
  rotate_prob: 0.0
  translate_prob: 0.0
  require_foreground: True

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

label_mask: [0, ${label_id}]
YAML
}

for ROI in ${ROIS}; do
  meta="$(roi_metadata "${ROI}")" || { echo "Unknown ROI: ${ROI}"; exit 1; }
  read -r LABEL_ID GROUP DEFAULT_CROP_XYZ <<< "${meta}"
  CROP_XYZ_ROI="${CROP_XYZ:-${DEFAULT_CROP_XYZ}}"
  PROJ_SIZE="$(python - <<PY
x,y,z=[int(v) for v in "${CROP_XYZ_ROI}".split(",")]
print(f"[{x}, {y}, {z}]")
PY
)"

  PREP_ROOT="${PREP_ROOT_BASE}/${GROUP}/${ROI}"
  SPLIT_DIR="${PREP_ROOT}/patient_splits"
  TRAIN_IDS="${SPLIT_DIR}/training1_patients.txt"
  VALID_IDS="${SPLIT_DIR}/validation1_patients.txt"
  TEST_IDS="${SPLIT_DIR}/testing1_patients.txt"

  TRAIN_OUT="${PREP_ROOT}/training1"
  VALID_OUT="${PREP_ROOT}/validation1"
  TEST_OUT="${PREP_ROOT}/testing1"

  TRAIN_LABEL_DIR="${TRAIN_OUT}/${GROUP}/${ROI}/labels"
  VALID_LABEL_DIR="${VALID_OUT}/${GROUP}/${ROI}/labels"
  TEST_LABEL_DIR="${TEST_OUT}/${GROUP}/${ROI}/labels"

  EXP_NAME="label_ae_${ROI}_roiwise_1p5mm_nozoom"
  CONFIG_PATH="${CONFIG_DIR}/${EXP_NAME}.yaml"
  RUN_DIR="${SAVE_DIR}/${EXP_NAME}_alpha${ALPHA}_kld0.001_wmuniform"
  SWEEP_CSV="${RUN_DIR}/probe_eval_sweep/probe_eval_sweep_detailed.csv"
  PLOT_OUT="${RUN_DIR}/probe_eval_sweep/unified_patient_variability_plots_assd"

  mkdir -p "${SPLIT_DIR}"

  echo "============================================================"
  echo "ROI=${ROI} label=${LABEL_ID} group=${GROUP} crop=${CROP_XYZ_ROI}"
  echo "============================================================"

  make_split "${ROI}" "${SPLIT_DIR}" "${TRAIN_IDS}" "${VALID_IDS}" "${TEST_IDS}"
  readarray -t TRAIN_PATIENTS < "${TRAIN_IDS}"
  readarray -t VALID_PATIENTS < "${VALID_IDS}"
  readarray -t TEST_PATIENTS < "${TEST_IDS}"

  preprocess_split training1 "${TRAIN_OUT}" "${GROUP}" "${ROI}" "${CROP_XYZ_ROI}" "${TRAIN_PATIENTS[@]}"
  preprocess_split validation1 "${VALID_OUT}" "${GROUP}" "${ROI}" "${CROP_XYZ_ROI}" "${VALID_PATIENTS[@]}"
  preprocess_split testing1 "${TEST_OUT}" "${GROUP}" "${ROI}" "${CROP_XYZ_ROI}" "${TEST_PATIENTS[@]}"

  echo "  Writing config: ${CONFIG_PATH}"
  write_config "${CONFIG_PATH}" "${EXP_NAME}" "${TRAIN_LABEL_DIR}" "${VALID_LABEL_DIR}" "${TEST_LABEL_DIR}" "${PROJ_SIZE}" "${LABEL_ID}"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "  Training/testing..."
    python main.py -c "${CONFIG_PATH}"
  else
    echo "  RUN_TRAIN=0; skipping training."
  fi

  if [[ "${RUN_PROBE_SWEEP}" == "1" ]]; then
    if [[ ! -f "${EVAL_SCRIPT}" ]]; then
      echo "ERROR: missing ${EVAL_SCRIPT}"
      exit 1
    fi
    echo "  Running better probe sweep..."
    CMD=(
      python "${EVAL_SCRIPT}"
      -c "${CONFIG_PATH}"
      --cut_levels "${CUT_LEVELS}"
      --local_frac_levels "${LOCAL_FRAC_LEVELS}"
      --voxel_levels "${VOXEL_LEVELS}"
    )
    [[ "${DROP_INF_IN_PLOTS}" == "1" ]] && CMD+=(--drop_inf_in_plots)
    [[ "${SAVE_VIS}" == "1" ]] && CMD+=(--save_vis)
    "${CMD[@]}"
  else
    echo "  RUN_PROBE_SWEEP=0; skipping probe sweep."
  fi

  if [[ "${RUN_PLOTS}" == "1" ]]; then
    if [[ ! -f "${PLOT_SCRIPT}" ]]; then
      echo "ERROR: missing ${PLOT_SCRIPT}"
      exit 1
    fi
    if [[ ! -f "${SWEEP_CSV}" ]]; then
      echo "WARNING: missing sweep CSV; skipping plots: ${SWEEP_CSV}"
    else
      echo "  Plotting old + patient-variability-aware plots..."
      python "${PLOT_SCRIPT}" \
        --csv "${SWEEP_CSV}" \
        --out_dir "${PLOT_OUT}" \
        --x_col abs_input_volume_change_pct \
        --volume_bin_width "${VOLUME_BIN_WIDTH}" \
        --error_band "${ERROR_BAND}" \
        --drop_inf
    fi
  else
    echo "  RUN_PLOTS=0; skipping plots."
  fi

  echo "Completed ROI=${ROI}"
  echo "  Config: ${CONFIG_PATH}"
  echo "  Run:    ${RUN_DIR}"
  echo "  Plots:  ${PLOT_OUT}"
done

echo "All ROIs completed."
