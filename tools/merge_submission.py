#!/usr/bin/env python3
"""Merge two submissions: image scores from A, masks from B.

Use this when one run has better I-AUROC and another has better P-AP.

    python tools/merge_submission.py \
        --scores-csv outputs/run_cls/submission.csv \
        --mask-dir outputs/run_seg/predicted_masks \
        --out-dir outputs/merged \
        --zip outputs/merged.zip
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.submission import zip_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-csv", required=True, help="submission.csv with the better I-AUROC")
    p.add_argument("--mask-dir", required=True, help="predicted_masks/ with the better P-AP")
    p.add_argument("--out-dir", default="outputs/merged")
    p.add_argument("--zip", default="outputs/merged.zip")
    args = p.parse_args()

    src_csv = Path(args.scores_csv)
    src_masks = Path(args.mask_dir)
    if src_masks.name != "predicted_masks" and (src_masks / "predicted_masks").is_dir():
        src_masks = src_masks / "predicted_masks"
    if not src_csv.is_file():
        raise SystemExit(f"missing {src_csv}")
    if not src_masks.is_dir():
        raise SystemExit(f"missing {src_masks}")

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    dest_masks = out / "predicted_masks"
    dest_masks.mkdir(parents=True)

    rows = []
    with open(src_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gf = row["group_folder"].strip()
            rows.append((gf, row["anomaly_score"]))
            src = src_masks / gf
            dst = dest_masks / gf
            if not src.is_dir():
                raise SystemExit(f"no masks for {gf} under {src_masks}")
            dst.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                png = src / f"{i}_mask.png"
                if not png.is_file():
                    raise SystemExit(f"missing {png}")
                shutil.copy2(png, dst / f"{i}_mask.png")

    with open(out / "submission.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_folder", "anomaly_score"])
        w.writerows(rows)

    z = zip_submission(str(out), args.zip)
    print(f"merged {len(rows)} samples → {z}")
    print("scores from", src_csv)
    print("masks  from", src_masks)


if __name__ == "__main__":
    main()

