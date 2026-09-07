#!/usr/bin/env python3
"""Format-level sanity check that does not need GPU / DINOv2 / the real dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import discover_samples, group_folder
from src.postprocess import maps_to_uint8, sample_score_from_views, smooth_map
from src.submission import SubmissionBuilder, zip_submission


def make_dummy(root: Path, n_cls=2, n_samples=3):
    cats = ["battery", "toy_tire"][:n_cls]
    for c in cats:
        for i in range(1, n_samples + 1):
            d = root / c / f"S{i:04d}"
            d.mkdir(parents=True, exist_ok=True)
            for v in range(5):
                arr = np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{v}.png")
    return cats


def main():
    tmp = ROOT / "outputs" / "_dummy"
    data = tmp / "Train"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    make_dummy(data)
    samples = discover_samples(data)
    assert len(samples) == 6, samples
    print(f"discovered {len(samples)} samples:", [group_folder(c, s) for c, s, _ in samples])

    builder = SubmissionBuilder()
    for c, s, _ in samples:
        maps = np.random.rand(5, 448, 448).astype(np.float32) * 0.2
        maps = smooth_map(maps, sigma=4.0)
        score = sample_score_from_views(maps, max_ratio=0.01, reduce="max")
        builder.add_sample(group_folder(c, s), score, maps_to_uint8(maps, scale=2.0))
    out = tmp / "submit"
    zpath = tmp / "my_submission.zip"
    builder.save(str(out), zip_path=str(zpath))
    zip_submission(str(out), str(tmp / "my_submission2.zip"))

    import zipfile
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert "submission.csv" in names
    assert any(n.startswith("predicted_masks/") and n.endswith("0_mask.png") for n in names)
    print(f"zip ok: {zpath}  ({len(names)} entries)")

    from src.refine import NormalStats, border_weight, refine_map

    m = np.zeros((32, 32), dtype=np.float32)
    m[10:14, 10:14] = 0.4
    out = refine_map(m, mean=np.full((8, 8), 0.05, np.float32), std=np.full((8, 8), 0.02, np.float32),
                     gamma=1.4, border=4, use_fg=False)
    assert out.shape == (32, 32) and out.min() >= 0
    assert out[12, 12] > out[0, 0]
    w = border_weight(32, 32, 4)
    assert w[16, 16] > w[0, 0]
    st = NormalStats(hw=8)
    st.update("battery", 0, m)
    st.update("battery", 0, m + 0.01)
    mu, sd = st.mean_std("battery", 0)
    assert mu is not None and sd is not None
    tmp_npz = tmp / "stats.npz"
    st.save(str(tmp_npz))
    st2 = NormalStats.load(str(tmp_npz))
    assert st2.mean_std("battery", 0)[0].shape == (8, 8)
    assert st2.view_scores, "view_scores lost on save/load"
    st.update("battery", 1, m + 0.02)
    g_lo = st.view_gate("battery", 0, 0.01, k=1.0, temp=0.3, floor=0.0)
    g_hi = st.view_gate("battery", 0, 1.0, k=1.0, temp=0.3, floor=0.0)
    assert 0.0 <= g_lo < g_hi <= 1.0
    print("refine ok")
    print("sanity check passed")


if __name__ == "__main__":
    main()
