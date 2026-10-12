# Released results

Images are identified by neutral IDs: `img_000001`, ... (real images) and
`vf_000001`, ... (synthetic virtual-flaw images).

## `splits/assignments.csv`

One row per image used in any partition. `class` (normal/defect); `heldout` (1 =
fixed held-out test set); `tune` (train/val role in the tuning split);
`p100`, `p101` (`val<k>` = validation image of fold k of that partition,
`train` = training image in every fold); `synthetic` (1 = virtual-flaw image,
used for training in the VF condition only).

## `scores/<method>/<condition>/p<P>_f<F>_s<S>_<part>.npz`

Per-image defect scores of one trained model. `P` = partition seed (100, 101),
`F` = fold (0-4), `S` = training seed, `part` = `heldout` (fixed held-out set,
5,000 normal + 150 defective images) or `val` (the fold's own validation set,
used only to calibrate the FPR <= 5% threshold).

Arrays: `path` (image ID), `label` (1 = defective), `score` (defect score;
higher = more likely defective). All held-out files contain the same 5,150 images; the order is the same within a method and differs between methods (match rows by `path`).

| method | condition | runs | Paper |
|---|---|---|---|
| defectcon | main | 10 partitions x seeds 42, 202, 303 | Tables 3-6, Fig. 3, App. E |
| defectcon | vf / vf_tuned / vf_widened | 10 x seed 42 | Tables 9, 10, Section 5.5 |
| bgad | main | 10 x seeds 42, 202, 303 | Tables 3-6 |
| bgad | vf / vf_tuned | 10 x seed 42 | Tables 9, 10 |
| supcon, uninet | main / vf | 10 x seed 42 | Tables 3, 9 |

`defectcon/vf_tuned` has held-out scores only; `bgad/vf_tuned` has validation
scores for partition seed 101 only. DefectCon scores use bilinear resizing;
BGAD and UniNet use the Lanczos resizing of their implementations.

## `metrics/`

| file | content |
|---|---|
| `ablation_cv_heldout.csv` | ablation cells D1, P1, proj (= J1), L1: held-out metrics per partition (`partition` = s<P>f<F>); A0 is `scores/defectcon/main/*_s42_*` |
| `ablation_tune_heldout.csv` | ablation cells trained on the tuning split (A0 is the sensitivity centre; L1 is the contrastive-weight-0 point) |
| `sensitivity_heldout.csv` | sensitivity points (Table C); `source=reused-A0` marks the centre |
| `vf_tuned_operating_point.csv` | Table 10 DefectCon recall/precision/F1 at FPR <= 5% (mean, SD over 10 runs) |
| `silhouette.csv` | cosine silhouette of the held-out embeddings, ten seed-42 models (App. D) |

Metric columns: `auroc`, `ap` (average precision), `pauc5_std` / `pauc10_std`
(McClish-standardised partial AUC up to 5% / 10% FPR), `fpr5_P`/`fpr5_R`/`fpr5_F1`
(precision/recall/F1 at the threshold calibrated to 5% FPR on validation).
Other columns are auxiliary.

## `curves/defectcon/main/p<P>_f<F>_s42.csv`

Per-epoch training curves of the seed-42 runs (Fig. 4): validation AP and AUROC,
trailing-5 smoothed validation AP, training losses (`train_loss_con`,
`train_loss_ce`), seconds per epoch.

## `tuning/`

Optuna journals of the three DefectCon searches (`final_novf` = baseline,
`final_vf` = VF, `final_vf3` = widened VF search), trial tables, and the
training-length calibration runs (`budget_calibration_*.json`).

## `figure_inputs/gradcam_ids.csv`

The held-out images shown in Fig. 5.
