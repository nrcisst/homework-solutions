"""Reproducible detector flags, not human-confirmed language labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import pandas as pd
from langdetect import DetectorFactory, detect, LangDetectException
from download_data import DOMAINS, ROLES, ROOT, download


def load_rows(data_dir):
    pieces = []
    for domain in DOMAINS:
        for role in ROLES:
            part = pd.read_json(Path(data_dir) / domain / f"{role}.jsonl", lines=True)
            assert set(part.columns) == {"text", "label"}
            part["row_id"] = [f"difraud:{domain}:{role}:{i}" for i in range(len(part))]
            part["domain"] = domain
            part["original_split"] = role
            pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def detect_one(text):
    DetectorFactory.seed = 42
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def language_audit(frame, output_dir=ROOT, workers=4):
    path = Path(output_dir) / "language_predictions.csv"
    if path.exists():
        result = pd.read_csv(path)
        if result["row_id"].tolist() == frame["row_id"].tolist():
            return result
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        langs = list(pool.map(detect_one, frame["text"], chunksize=200))
    result = frame[["row_id", "domain", "label"]].copy()
    result["detected_language"] = langs
    result.to_csv(path, index=False)
    summary = result.groupby(["domain", "label"]).agg(
        rows=("row_id", "size"),
        non_english_flags=("detected_language", lambda s: ((s != "en") & (s != "unknown")).sum()),
        unknown=("detected_language", lambda s: (s == "unknown").sum()),
    ).reset_index()
    summary["flagged_percent"] = 100 * summary["non_english_flags"] / summary["rows"]
    summary.to_csv(Path(output_dir) / "language_counts.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Language audit seconds: {time.perf_counter() - start:.3f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    language_audit(load_rows(download(args.data_dir)), workers=args.workers)
