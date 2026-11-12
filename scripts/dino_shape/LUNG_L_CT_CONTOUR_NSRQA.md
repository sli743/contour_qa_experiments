# CT + contour NSR-QA (Lung_L, Month1+2 -> Month3)

Branch `feature/ct-contour-3dino`. Adds the CT crop as a second input to NSR-QA.

## Model (`models/dino_shape/dino_image_contour_autoencoder.py`)

- The same frozen 3DINO encoder is run on the contour and on the CT crop.
- **Token fusion** (`image_conditioning.fusion`):
  - `gated_add` (default): contour tokens + gate * proj(CT tokens). The gate is per-channel and starts at zero.
  - `concat`: a 1x1 mix initialised to pass the contour tokens through unchanged.
- **Full-resolution CT stem**: 2 convs feeding the last `image_skip_last_n_stages` decoder stages through zero-initialised 1x1 convs.
- **Starting point**: at initialisation the output equals a contour-only decoder, so CT is used only as far as training finds it useful.
- **`image_present`** switches every CT pathway off. Training uses it for CT dropout (`train_image_dropout`, default 0.3) so the decoder cannot turn into a pure CT segmenter. Evaluation uses it for the CT-reliance check (`--image-mode absent`).
- **`contour_input: empty`** gives the CT-only segmenter arm with the identical network.
- **CT intensities**:
  - The roiwise training crops are uint8 (−1024…3324 HU). The Month3 deployment crops are raw HU.
  - Both are converted to HU and quantised to the same uint8 grid, then mapped from `hu_window` [−1000, 2000] to [−1, 1].
- **Token cache**: CT tokens are cached per case during training, since the encoder is frozen and the CT crop never changes.
- **Logging**: `logs/image_guidance.csv` records the gate and injection norms per epoch, showing whether the CT pathway opens.

## Arms

| task | arm | change vs v3 contour-only config |
|---|---|---|
| 0 | `ct_contour_corruption` | + CT |
| 1 | `ct_contour_no_corruption` | + CT, clean_probability 1.0 |
| 2 | `ct_only_segmenter` | + CT, contour input empty, clean_probability 1.0, no CT dropout |
| – | `contour_only_v3` | linked from the v3 family-validated study |
| – | `contour_only_no_corruption` | linked from the month-temporal ablation study |

In addition, arms 0 and 1 are re-evaluated with the CT off, giving `<arm>_ct_absent`.

## Run

Smoke test first (2 epochs, arm 0, with eval):

```bash
SMOKE=1 bash scripts/dino_shape/submit_lung_l_ct_contour_nsrqa.sh
```

Full study:

```bash
bash scripts/dino_shape/submit_lung_l_ct_contour_nsrqa.sh
```

## Outputs

Scores land in `<study>/evaluations_floor_corrected/seed91/<arm>_floor_corrected_scores.csv`, the same layout used by `analysis/plots/plot_arm_ranking_comparison.py` and `plot_ood_vs_qa_scores.py`.
