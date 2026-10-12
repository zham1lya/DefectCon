"""Pipeline smoke test on tiny synthetic data (CPU, about a minute).

Builds fake radiographs and split lists in a temporary folder, then runs the
released commands end to end:

    train.py --smoke-test  ->  score.py --runs  ->  reproduce_main.sh
    (Table 3 aggregation, paired statistics, variance components, per-seed and
    ensemble analyses, operating region, Fig. 3)

One run (partition 100, fold 0, seed 42) is trained and scored. The analyses
need all 30 runs of DefectCon and of BGAD, so the other score files are filled
with noisy copies of the trained run's scores. The test checks that every step
finishes and writes its outputs; the numbers themselves are meaningless. It
reads no real images, nothing in private/, and writes nothing to results/ or
figures/.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SIZE = 64
COUNTS = {"train": (12, 6), "val": (6, 3), "heldout": (16, 6)}   # (normal, defect)


def fake_image(rng: np.random.Generator, defect: bool) -> Image.Image:
    img = rng.normal(120, 20, (SIZE, SIZE))
    if defect:
        y, x = rng.integers(8, SIZE - 16, 2)
        img[y:y + 8, x:x + 8] += 80
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def build_fake_data(root: Path) -> dict:
    """Images under root/images and split lists under root/splits; returns the environment."""
    rng = np.random.default_rng(0)
    data, splits = root / "images", root / "splits"
    fold = splits / "cv_seed100" / "fold0"
    fold.mkdir(parents=True)
    lists = {}
    for part, counts in COUNTS.items():
        subset = "test" if part == "heldout" else "train"
        for kind, n in zip(("normal", "defect"), counts):
            d = data / "noVF" / subset / kind
            d.mkdir(parents=True, exist_ok=True)
            files = []
            for i in range(n):
                f = d / f"{part}_{kind}_{i:03d}.png"
                fake_image(rng, kind == "defect").save(f)
                files.append(str(f))
            lists[(part, kind)] = files
    for kind in ("normal", "defect"):
        for part in ("train", "val"):
            (fold / f"{part}_{kind}.txt").write_text("\n".join(lists[(part, kind)]) + "\n")
        (fold / f"test_{kind}.txt").write_text("\n".join(lists[("heldout", kind)]) + "\n")
    for kind, name in (("normal", "heldout_normal.txt"), ("defect", "heldout_defective.txt")):
        (splits / name).write_text("\n".join(Path(p).name for p in lists[("heldout", kind)]) + "\n")
    env = dict(os.environ, DEFECTCON_DATA=str(data), DEFECTCON_SPLITS=str(splits),
               DEFECTCON_OUT=str(root / "runs"), DEFECTCON_RESULTS=str(root / "results"),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MPLBACKEND="Agg")
    return env


def fill_other_runs(results: Path) -> None:
    """Noisy copies of the trained run's scores for every other run the analyses read."""
    rng = np.random.default_rng(1)
    src = results / "scores" / "defectcon" / "main"
    trained = {part: np.load(src / f"p100_f0_s42_{part}.npz") for part in ("val", "heldout")}
    wanted = [("defectcon", "main", (42, 202, 303)), ("bgad", "main", (42, 202, 303)),
              ("defectcon", "vf", (42,)), ("defectcon", "vf_tuned", (42,))]
    for method, condition, seeds in wanted:
        for ps in (100, 101):
            for k in range(5):
                for s in seeds:
                    for part, z in trained.items():
                        f = results / "scores" / method / condition / f"p{ps}_f{k}_s{s}_{part}.npz"
                        if f.exists():
                            continue
                        f.parent.mkdir(parents=True, exist_ok=True)
                        score = np.clip(z["score"] + rng.normal(0, 0.05, len(z["score"])), 0, 1)
                        np.savez_compressed(f, path=z["path"], label=z["label"], score=score)


def snapshot(*dirs: Path) -> dict:
    return {str(f): (f.stat().st_size, f.stat().st_mtime_ns)
            for d in dirs if d.exists() for f in d.rglob("*") if f.is_file()}


def run(cmd: list, env: dict) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode == 0, f"{' '.join(map(str, cmd))} failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return p


def test_pipeline(tmp_path):
    released = snapshot(REPO / "results", REPO / "figures")
    env = build_fake_data(tmp_path)
    py = sys.executable

    run([py, "scripts/train.py", "--split", "cv", "--partition-seed", "100", "--fold", "0",
         "--seeds", "42", "--smoke-test", "--device", "cpu", "--num-workers", "0"], env)
    run_dir = tmp_path / "runs" / "cv_novf_baseline" / "p100_f0_s42"
    for name in ("model.pth", "curve.csv", "meta.json", "scores_val.npz", "scores_heldout.npz"):
        assert (run_dir / name).exists(), name

    run([py, "scripts/score.py", "--runs", str(run_dir), "--device", "cpu"], env)
    scores = tmp_path / "results" / "scores" / "defectcon" / "main"
    held = np.load(scores / "p100_f0_s42_heldout.npz")
    assert len(held["score"]) == sum(COUNTS["heldout"])
    assert np.array_equal(held["score"], np.load(run_dir / "scores_heldout.npz")["score"])
    again = subprocess.run([py, "scripts/score.py", "--runs", str(run_dir), "--device", "cpu"],
                           cwd=REPO, env=env, capture_output=True, text=True)
    assert again.returncode != 0 and "--overwrite" in again.stderr

    fill_other_runs(tmp_path / "results")
    out = tmp_path / "out"
    run(["bash", "scripts/reproduce_main.sh", str(out)], env)
    for name in ("main_summary.csv", "paired.json", "variance.json", "per_seed.json",
                 "ensembles.json", "operating_region.json", "fig3_operating_region.png"):
        assert (out / name).exists(), name

    assert snapshot(REPO / "results", REPO / "figures") == released, "released files were modified"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
