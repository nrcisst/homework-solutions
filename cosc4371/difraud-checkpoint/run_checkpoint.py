"""Audit, train once, select a validation threshold, score, and run the supplied checker."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from download_data import DOMAINS, REVISION, ROOT, download
from language_audit import load_rows, language_audit

SEED = 42
TEST_CAP = 500


def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def stratifier(frame):
    return frame["domain"] + ":" + frame["label"].astype(str)


def duplicate_stats(frame, field):
    groups = frame.groupby(field, sort=False)
    sizes = groups.size()
    return {
        "groups": int((sizes > 1).sum()),
        "extra_rows": int((sizes - 1).clip(lower=0).sum()),
        "rows_in_groups": int(sizes[sizes > 1].sum()),
        "original_cross_split_groups": int((groups["original_split"].nunique() > 1).sum()),
        "conflicting_label_groups": int((groups["label"].nunique() > 1).sum()),
    }


def run(data_dir, workers):
    total_start = time.perf_counter()
    raw = load_rows(download(data_dir))
    assert raw["row_id"].is_unique
    missing_text = raw["text"].isna() | raw["text"].astype(str).str.strip().eq("")
    invalid_label = ~raw["label"].isin([0, 1])
    audit = {
        "revision": REVISION, "raw_rows": len(raw), "raw_columns": ["text", "label"],
        "class_counts": {str(k): int(v) for k, v in raw["label"].value_counts().items()},
        "missing_target": int(raw["label"].isna().sum()),
        "invalid_target": int(invalid_label.sum()),
        "missing_or_empty_text": int(missing_text.sum()),
        "entity_ids": "unavailable: source files contain only text and label",
        "time_range": "unavailable: no row timestamps",
    }
    if missing_text.any() or invalid_label.any():
        raise ValueError("Unexpected missing/invalid data; inspect before changing the analysis.")
    raw["normalized"] = raw["text"].map(normalize)
    audit["exact_text"] = duplicate_stats(raw, "text")
    audit["normalized_text"] = duplicate_stats(raw, "normalized")
    audit["normalization_rule"] = "Unicode lower(); remove characters outside \\w and whitespace; collapse whitespace; strip"
    domain_audit = raw.groupby("domain").agg(rows=("label", "size"), positive=("label", "sum"))
    domain_audit["negative"] = domain_audit["rows"] - domain_audit["positive"]
    domain_audit["positive_percent"] = 100 * domain_audit["positive"] / domain_audit["rows"]
    domain_audit.to_csv(ROOT / "domain_audit.csv")
    print("Audit:", audit, flush=True)

    # Remove ambiguous labels and collapse text-equivalent groups before splitting.
    conflicts = raw.groupby("normalized")["label"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    conflict_mask = raw["normalized"].isin(conflict_keys)
    unambiguous = raw.loc[~conflict_mask].copy()
    repeated_mask = unambiguous.duplicated("normalized", keep="first")
    kept = unambiguous.loc[~repeated_mask].copy()
    audit["conflict_rows_removed"] = int(conflict_mask.sum())
    audit["additional_duplicate_rows_removed"] = int(repeated_mask.sum())
    audit["clean_rows"] = len(kept)

    # Fixed 80/10/10 split, stratified jointly by domain and class.
    train, holdout = train_test_split(kept, test_size=0.2, random_state=SEED,
                                     stratify=stratifier(kept))
    validation, candidates = train_test_split(holdout, test_size=0.5, random_state=SEED,
                                              stratify=stratifier(holdout))
    langs = language_audit(raw, workers=workers).set_index("row_id")["detected_language"]
    raw["detected_language"] = raw["row_id"].map(langs)
    candidates = candidates.assign(detected_language=candidates["row_id"].map(langs))
    english = candidates.loc[candidates["detected_language"] == "en"].copy()
    parts = []
    for domain in DOMAINS:
        part = english.loc[english["domain"] == domain]
        if len(part) > TEST_CAP:
            part, _ = train_test_split(part, train_size=TEST_CAP, random_state=SEED,
                                       stratify=part["label"])
        parts.append(part)
    test = pd.concat(parts)
    audit["test_candidates"] = len(candidates)
    audit["test_non_english_or_unknown_excluded"] = len(candidates) - len(english)
    audit["test_english_excluded_by_cap"] = len(english) - len(test)
    audit["test_cap_per_domain"] = TEST_CAP
    audit["retained_rows"] = len(train) + len(validation) + len(test)
    audit["original_roles_changed_before_test_sampling"] = int(
        train["original_split"].ne("train").sum()
        + validation["original_split"].ne("validation").sum()
        + candidates["original_split"].ne("test").sum())
    audit["retained_non_english_flags"] = {
        name: int(((frame["row_id"].map(langs) != "en") & (frame["row_id"].map(langs) != "unknown")).sum())
        for name, frame in [("train", train), ("validation", validation), ("test", test)]}
    leakage = {}
    for field in ["text", "normalized"]:
        for name, other in [("test", test), ("validation", validation)]:
            leakage[f"train_rows_matching_{name}_{field}"] = int(train[field].isin(set(other[field])).sum())
        leakage[f"validation_rows_matching_test_{field}"] = int(validation[field].isin(set(test[field])).sum())
    assert all(value == 0 for value in leakage.values()), leakage
    audit["leakage"] = leakage
    dump("audit.json", audit)
    split = pd.concat([frame.assign(split=name) for name, frame in
                       [("train", train), ("validation", validation), ("test", test)]])
    split["dataset"] = "DIFrauD"
    split[["row_id", "dataset", "split", "label"]].to_csv(ROOT / "split.csv", index=False)
    split[["row_id", "domain", "original_split", "split"]].to_csv(ROOT / "row_metadata.csv", index=False)
    test[["row_id", "domain", "label", "detected_language"]].to_csv(ROOT / "paired_test_ids.csv", index=False)
    print("Splits:", split.groupby("split")["label"].agg(["size", "sum"]).to_dict(), flush=True)

    model = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                                    min_df=2, max_features=50000, sublinear_tf=True, dtype=np.float32)),
            ("char", TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="char_wb",
                                    ngram_range=(3, 5), min_df=2, max_features=30000,
                                    sublinear_tf=True, dtype=np.float32)),
        ])),
        ("classifier", LogisticRegression(C=1.0, solver="liblinear", max_iter=2000,
                                         random_state=SEED, class_weight=None)),
    ])
    print("Fitting TF-IDF and logistic regression...", flush=True)
    fit_start = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(train["text"], train["label"])
    fit_seconds = time.perf_counter() - fit_start
    print(f"Fit seconds: {fit_seconds:.3f}", flush=True)
    validation_start = time.perf_counter()
    p_validation = model.predict_proba(validation["text"])[:, 1]
    fpr, tpr, thresholds = roc_curve(validation["label"], p_validation, drop_intermediate=False)
    eligible = np.isfinite(thresholds) & (fpr <= 0.05)
    threshold = float(thresholds[eligible].min())
    val_pred = p_validation >= threshold
    y_val = validation["label"].to_numpy()
    val_fpr = float(val_pred[y_val == 0].mean())
    val_recall = float(val_pred[y_val == 1].mean())
    assert val_fpr <= 0.05
    validation_seconds = time.perf_counter() - validation_start
    (ROOT / "threshold.txt").write_text(repr(threshold) + "\n")

    # Score each test row exactly once; reuse those scores for aggregate reporting.
    test_start = time.perf_counter()
    test = test.assign(score=model.predict_proba(test["text"])[:, 1])
    test_seconds = time.perf_counter() - test_start
    rows = []
    domain_metrics = []
    for domain in DOMAINS:
        part = test.loc[test["domain"] == domain].copy()
        part["test_set"] = "English_" + domain
        rows.append(part[["row_id", "test_set", "label", "score"]])
        domain_metrics.append({"domain": domain, "n": len(part),
                               "AUROC": roc_auc_score(part["label"], part["score"]),
                               "AP": average_precision_score(part["label"], part["score"])})
    pooled = test.assign(test_set="English_pooled")
    rows.append(pooled[["row_id", "test_set", "label", "score"]])
    pd.concat(rows).to_csv(ROOT / "scores.csv", index=False)
    metrics_frame = pd.DataFrame(domain_metrics)
    metrics_frame.to_csv(ROOT / "domain_point_metrics.csv", index=False)
    aggregate = {
        metric: {"mean": float(metrics_frame[metric].mean()),
                 "weighted_mean": float(np.average(metrics_frame[metric], weights=metrics_frame["n"]))}
        for metric in ["AUROC", "AP"]}
    dump("aggregate_metrics.json", aggregate)
    cpu = "unknown CPU"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    versions = {package: importlib.metadata.version(package) for package in
                ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "langdetect", "threadpoolctl"]}
    run_info = {
        "seed": SEED, "threshold": threshold, "validation_fpr": val_fpr,
        "validation_recall": val_recall, "fit_seconds": fit_seconds,
        "validation_seconds": validation_seconds, "test_score_seconds": test_seconds,
        "feature_count": int(sum(len(transform.vocabulary_) for _, transform in
                                 model.named_steps["features"].transformer_list)),
        "lr_iterations": model.named_steps["classifier"].n_iter_.tolist(),
        "python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
        "visible_cpus": os.cpu_count(), "fit_threads": 1, "versions": versions,
        "total_seconds_before_checker": time.perf_counter() - total_start,
    }
    dump("run_info.json", run_info)

    cm = confusion_matrix(test["label"], test["score"] >= threshold, labels=[0, 1])
    dump("confusion_matrix.json", {"labels": [0, 1], "counts": cm.tolist()})
    fig, ax = plt.subplots(figsize=(4.6, 3.35))
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=15,
                    color="white" if cm[i,j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-deceptive", "Deceptive"],
           yticklabels=["Non-deceptive", "Deceptive"], xlabel="Predicted label", ylabel="Actual label",
           title="English pooled test set")
    fig.tight_layout()
    fig.savefig(ROOT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    command = [sys.executable, str(ROOT / "check_checkpoint.py"), "--split", str(ROOT / "split.csv"),
               "--scores", str(ROOT / "scores.csv"), "--threshold", repr(threshold)]
    print("Running the instructor's checker (1,000 bootstrap samples per test set)...", flush=True)
    checked = subprocess.run(command, capture_output=True, text=True, check=True)
    (ROOT / "checker_output.txt").write_text(checked.stdout)
    if checked.stderr:
        (ROOT / "checker_stderr.txt").write_text(checked.stderr)
    print(checked.stdout, flush=True)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ["split.csv", "scores.csv", "paired_test_ids.csv", "threshold.txt"]}
    dump("output_hashes.json", hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.data_dir, args.workers)
