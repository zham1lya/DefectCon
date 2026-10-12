# DefectCon

Code and released results for the paper *Choosing Defect Detectors by
False-Alarm Budget: A Controlled Study of Supervised Contrastive Learning for
Radiographic Weld Inspection*.

DefectCon trains a dilated ResNet-18 end to end with a supervised contrastive
loss on two augmented views and a focal classification loss, using
fixed-composition batches of normal and defective radiographs. On a fixed
held-out set of industrial weld radiographs it is compared with BGAD, UniNet and
SupCon over two repeated 5-fold partitions and three training seeds, with
paired partition-level statistics, an operating-region analysis, an ablation,
a sensitivity sweep and a study of virtual-flaw (VF) augmentation.

```bibtex
@misc{defectcon,
  title  = {Choosing Defect Detectors by False-Alarm Budget: A Controlled Study of Supervised Contrastive Learning for Radiographic Weld Inspection},
  author = {Zhamilya Saparova and Sudong Lee},
  year   = {2026},
  note   = {Under review},
  url    = {https://github.com/zham1lya/DefectCon}
}
```

## Installation

Python 3.9; experiments used PyTorch 2.0.1 (CUDA 11.8) on NVIDIA RTX 3090 GPUs.

```bash
pip install -r requirements.txt
pip install -e .
```

Training needs a CUDA GPU with 24 GB (dilated layer3/layer4 at 320x320, batch 128,
AMP). Everything under "Reproducing the paper" that reads `results/` runs on CPU.

## Data

The radiographs are confidential and are not distributed. Images are identified
by neutral IDs (`img_000001`, ... for real images, `vf_000001`, ... for synthetic
virtual-flaw images). `results/` contains the split definitions and the
per-image held-out and validation scores of every model of the main
comparisons, so every reported metric and statistic can be recomputed without
the images. For the ablation cells and the sensitivity sweep, per-fold metric
files are provided (`results/metrics/`), not per-image scores. See
[results/README.md](results/README.md).

## Reproducing the paper

All numbers used in the paper text and the baseline recall/precision/F1 table cells
(`paper/paper_numbers.tex`, extracted from the manuscript source) are checked:

```bash
python -m pytest tests/test_paper_numbers.py
```

Tables 3-6 and Fig. 3 (main comparison) in one command, from the released
scores on CPU (about 20 s; output identical to the shipped `results/metrics/`
files and Fig. 3):

```bash
bash scripts/reproduce_main.sh            # or: bash scripts/reproduce_main.sh <output dir>
```

The individual commands:

| Paper item | Command | Output |
|---|---|---|
| Table 3 (absolute results) | `python scripts/aggregate.py --condition main` | `results/metrics/main_summary.csv` |
| Table 4 (paired comparison) | `python scripts/analyze.py paired` | `results/metrics/paired.json` |
| Table 5 (seed vs partition variance) | `python scripts/analyze.py variance` | `results/metrics/variance.json` |
| Section 5.2.2 (per training seed) | `python scripts/analyze.py per-seed` | `results/metrics/per_seed.json` |
| Section 5.2.3 (ensembles) | `python scripts/analyze.py ensembles` | `results/metrics/ensembles.json` |
| Table 6, Section 5.3 (TPR at FPR, ROC crossing, pAUC by bound) | `python scripts/analyze.py operating-region` | `results/metrics/operating_region.json` |
| Section 5.3 (natural defect rate) | `python scripts/analyze.py natural-rate` | `results/metrics/natural_rate.json` |
| Figure 3 | `python scripts/figures.py operating-region` | `figures/fig3_operating_region.{png,pdf}` |
| Figure 4 | `python scripts/figures.py convergence` | `figures/fig4_convergence.png` |
| Tables 7-8 (ablation) | `python scripts/aggregate.py --condition ablation` | `results/metrics/ablation_summary.csv` |
| Table 9 (VF, baseline hyperparameters) | `python scripts/aggregate.py --condition vf` | `results/metrics/vf_summary.csv` |
| Table 10 (VF, VF-tuned hyperparameters) | `python scripts/aggregate.py --condition vf_tuned` | `results/metrics/vf_tuned_summary.csv`; Rec/Prec/F1 in `results/metrics/vf_tuned_operating_point.csv` |
| Section 5.5 (widened VF search) | `python scripts/aggregate.py --condition vf_widened`; `python scripts/analyze.py paired-vf` | `results/metrics/vf_widened_summary.csv`, `paired_vf.json` |
| Table 11 (computational cost) | measured once, see below | — |
| Table B (search spaces, selected values) | `configs/search/*.yaml`, `configs/hparams/*.yaml`; `python scripts/tune.py --search-space defectcon_baseline --export-trials --journal results/tuning/final_novf.journal` | `results/tuning/final_novf_trials.csv` |
| Table C (sensitivity) | `python scripts/aggregate.py --condition sensitivity` | `results/metrics/sensitivity_summary.csv` |
| Appendix D (silhouette) | `results/metrics/silhouette.csv`; recompute with `scripts/embeddings.py silhouette` (GPU, images) | |
| Figure 5 (Grad-CAM) | `python scripts/gradcam.py --checkpoint <p101_f0_s42>/model.pth` (GPU, images, private ID mapping) | `figures/fig5_gradcam.png` |
| Figure 7 (UMAP) | `python scripts/embeddings.py umap --checkpoint <p101_f0_s42>/model.pth` (GPU, images) | `figures/fig7_umap.png` |
| Appendix E (per-partition AP) | `python scripts/aggregate.py --condition per_partition` | `results/metrics/main_per_partition_ap.csv` |

The natural-rate estimates of Section 5.3 use the dataset-level defect rate of
1.41% printed in the paper (the held-out rate is 150/5,150 = 2.91%).

Table 11 was measured on one NVIDIA RTX 3090 with AMP: 11.57 M parameters
(11.40 M at inference), 18.68 GMAC per 320x320 image, 44.2 MB FP32 checkpoint,
7.52 ± 0.30 h per fold, 11.97 GB peak allocated memory, 2.41 ms single-image
latency and 438 images/s end to end with eight loader workers (200 timed runs
after warm-up).

## Training from scratch

Set three paths (the defaults are `data/images`, `data/splits` and `runs/` in the
repository):

```bash
export DEFECTCON_DATA=/path/to/images     # noVF/{train,test}/{normal,defect}/*.png, VF/train/defect/*.png
export DEFECTCON_SPLITS=/path/to/splits   # split lists, see below
export DEFECTCON_OUT=/path/to/runs
```

The released split definitions (`results/splits/assignments.csv`) were built
with the procedure of Appendix A (position-level grouping, class-blind held-out
set, multilabel-stratified tuning split and 5-fold partitions with seeds 100 and
101); the construction script is not released because it depends on the
confidential file-naming scheme. For your own data, provide one directory per
split (`$DEFECTCON_SPLITS/tune`, `cv_seed<P>/fold<F>`, and `_vf` copies for VF
training) holding `{train,val,test}_{normal,defect}.txt` with one image path per
line, plus `heldout_normal.txt` / `heldout_defective.txt` with image file names.

```bash
# main comparison: 2 partition seeds x 5 folds x 3 training seeds
python scripts/train.py --split cv --partition-seed 100 101 --fold 0 1 2 3 4 --seeds 42 202 303
# VF under baseline / VF-tuned / widened hyperparameters
python scripts/train.py --split cv --condition vf --hparams baseline
python scripts/train.py --split cv --condition vf --hparams vf_tuned
python scripts/train.py --split cv --condition vf --hparams vf_widened
# ablation cells and sensitivity points
python scripts/train.py --split cv --ablation L1
python scripts/train.py --split tune --ablation A0
python scripts/train.py --split tune --sensitivity temperature=0.05
# re-score a saved checkpoint into any folder
python scripts/score.py --checkpoints runs/cv_novf_baseline/p100_f0_s42/model.pth --split cv:100:0 --out scores/
# hyperparameter search (two workers per study)
python scripts/tune.py --search-space defectcon_baseline --enqueue-only
CUDA_VISIBLE_DEVICES=0 python scripts/tune.py --search-space defectcon_baseline --worker-index 0
```

Each run writes `model.pth`, `curve.csv`, `meta.json`, `scores_val.npz` and
`scores_heldout.npz`. `--dry-run` prints the schedule of every run; the step
budget is fixed (21,780 optimizer steps without VF, 24,750 with VF) and the epoch
count follows from the training-set size and batch composition.

**Main comparison from scratch (no VF).** Train the 30 runs, score them into
a results tree, and run the analyses on it:

```bash
python scripts/train.py --split cv --partition-seed 100 101 --fold 0 1 2 3 4 --seeds 42 202 303
export DEFECTCON_RESULTS=$PWD/my_results   # keep the released results/ untouched
python scripts/score.py --runs $DEFECTCON_OUT/cv_novf_baseline/p*_f*_s*
cp -r results/scores/bgad my_results/scores/  # baseline scores (or your own BGAD runs)
bash scripts/reproduce_main.sh $PWD/my_results/metrics
```

`score.py --runs` reads each run's `meta.json` and writes
`<results>/scores/defectcon/<main|vf|vf_tuned|vf_widened>/p<P>_f<F>_s<S>_{val,heldout}.npz`;
it does not overwrite existing files unless `--overwrite` is given. Each run
takes about 7.5 h on one RTX 3090 (Table 11), so the 30 runs take about 226
GPU-hours (about 2.5 days on four GPUs, one run per GPU).

**Smoke test.** `python -m pytest tests/test_smoke.py` checks the whole chain on
tiny synthetic images in a temporary folder, on CPU in about 40 s: fake splits ->
`train.py --smoke-test` (6 steps, batch size 4) -> `score.py --runs` ->
`reproduce_main.sh`. The other 59 runs the analyses need are filled with noisy
copies of the trained run's scores, so the numbers are meaningless. It does not
read real data or `private/` and does not write to `results/` or `figures/`.

Commands that need the real radiographs (not distributed): `train.py`, `tune.py`,
`score.py`, `embeddings.py` (silhouette, Fig. 7) and `gradcam.py` (Fig. 5; it
also needs the private ID mapping). Everything else runs from `results/`.

**Determinism.** All seeds are set in `defectcon.engine.seed_everything` and
cuDNN runs in deterministic mode for training and scoring. Different GPUs,
drivers or library versions can still change results slightly (for BGAD, a
rebuilt environment changed seed-42 results by 0.0002 AUROC and 0.0017 AP).

## Baselines

The baselines were run with their official implementations; their code is not
included. Per-image scores of all three are in `results/scores/`.

| Method | Upstream repository | Settings | Preprocessing |
|---|---|---|---|
| BGAD (Yao et al., 2023) | https://github.com/xcyao00/BGAD | supervised setting; hyperparameters searched with the protocol of Appendix B (Table B); 84 epochs without VF, 119 with VF | resize with Lanczos (the implementation's filter) at training and test time |
| UniNet (Wei et al., 2025) | https://github.com/pangdatangtt/UniNet | supervised setting; default hyperparameters of the official implementation; 320x320 input | resize with Lanczos at training and test time |
| SupCon (Khosla et al., 2020) | https://github.com/HobbitLong/SupContrast | supervised contrastive pre-training of a 1-channel ResNet-18 followed by a classifier; default hyperparameters; 320x320 input | as in the official implementation |

Recall, precision and F1 at FPR <= 5% are computed in the same way for every
method: the score threshold is the 95th percentile of the run's own validation
normals and is applied unchanged to the held-out set; tables report mean and
sample SD (n - 1) over runs (`defectcon.metrics.operating_point`,
`tests/test_paper_numbers.py::test_baseline_operating_point`).

The exact upstream commits were not recorded. All baselines used the same
partitions, the same held-out set and training seed 42 (BGAD also 202 and 303).

## Repository layout

```
configs/    architecture/protocol, hyperparameters, search spaces, ablation cells, sensitivity points
defectcon/  package: data, model, losses, sampler, transforms, training, scoring, metrics, statistics
scripts/    train, score, tune, aggregate, analyze, figures, reproduce_main.sh, gradcam, embeddings, anonymize
results/    split definitions, per-image scores, metric files, curves, tuning journals
tests/      paper-number test, pipeline smoke test
paper/      numbers reported in the paper (macros and tested table cells)
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The
radiographs are not included and cannot be released.
