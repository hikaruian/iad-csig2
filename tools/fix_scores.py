#!/usr/bin/env python3
"""Rewrite submission.csv so anomaly_score is in [0, 1] without re-running the model.

    python tools/fix_scores.py outputs/submission/submission.csv
    python tools/fix_scores.py outputs/my_submission.zip
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.postprocess import squash_score  # noqa: E402


def fix_csv(path: Path) -> tuple[float, float, int]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["group_folder", "anomaly_score"]
        for row in reader:
            row = dict(row)
            row["anomaly_score"] = f"{squash_score(row['anomaly_score']):.8f}"
            rows.append(row)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    vals = [float(r["anomaly_score"]) for r in rows]
    return (min(vals) if vals else 0.0, max(vals) if vals else 0.0, len(vals))


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip())
        sys.exit(2)
    src = Path(sys.argv[1])
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    if src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="fix_scores_"))
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(tmp)
        csvs = list(tmp.rglob("submission.csv"))
        if not csvs:
            raise SystemExit("zip has no submission.csv")
        mn, mx, n = fix_csv(csvs[0])
        bak = src.with_suffix(src.suffix + ".bak")
        shutil.copy2(src, bak)
        with zipfile.ZipFile(src, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            root = csvs[0].parent
            zf.write(csvs[0], arcname="submission.csv")
            pred = root / "predicted_masks"
            if pred.is_dir():
                for png in sorted(pred.rglob("*.png")):
                    zf.write(png, arcname=str(Path("predicted_masks") / png.relative_to(pred)))
        print(f"updated {src} (backup {bak})  n={n} min={mn:.6f} max={mx:.6f}")
        return
    mn, mx, n = fix_csv(src)
    print(f"updated {src}  n={n} min={mn:.6f} max={mx:.6f}")


if __name__ == "__main__":
    main()

