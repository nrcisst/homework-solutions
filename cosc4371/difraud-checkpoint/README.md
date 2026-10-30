# COSC 4371: DIFrauD non-English samples checkpoint

Biniam Habte. Instructor-assigned Project 1. The distributed project list calls
this topic Project 2; the assigned number is retained.

## What is finished

This checkpoint audits all 21 pinned DIFrauD JSONL files, removes normalized
duplicate text, creates a new stratified split, and runs one word/character
TF-IDF logistic-regression baseline. It freezes 500 detector-English test rows
per domain. No Spanish/French results or trained transformer are claimed.

`split.csv` and `scores.csv` are actual generated results, not examples. Both
belong in the submitted repository. `checker_output.txt` is the instructor's
unmodified checker output. The same prediction appears once in its domain test
set and once in `English_pooled`; those are overlapping summaries, not two
independent experiments. Each test text is scored once per reproduction run.

## Run from a fresh copy

Use Python 3.12.14 (tested) with internet access. From this folder:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_checkpoint.py
```

Windows activation: `.venv\Scripts\activate`.

The last command downloads the pinned data, verifies SHA-256 checksums, runs the
audit and detector, fits the model, picks a threshold using validation only,
writes the two CSVs and other evidence, creates the figure, and runs the supplied
checker with that exact threshold. No token or dataset-loading script is used.
If using an existing local copy, use
`python run_checkpoint.py --data-dir /path/to/difraud` with the seven domain
subfolders. All 21 files must match `data_manifest.json`.

The detector uses four worker processes. On a smaller machine use
`python run_checkpoint.py --workers 2`. The model fit uses one numerical thread.
The feature vocabulary is learned from training data only. Reported runtime is
in `run_info.json`; other machines may take longer.

Language predictions are cached locally after their first run. To redo that
audit, delete `language_predictions.csv` before running. The cache is excluded
from the delivered ZIP; a fresh copy recomputes it. `langdetect` uses seed 42
for each text. It produces estimates, not human-confirmed language labels.

To run just the instructor's checker, copy the exact value from `threshold.txt`:

```sh
python check_checkpoint.py --split split.csv --scores scores.csv --threshold VALUE
```

`VALUE` is a placeholder only in this README; the pipeline itself passes the
actual validation-selected value automatically. The report prints that value.

## Split and duplicate rules

1. Load files in the fixed domain order in `download_data.py`, then train,
   validation, test order, preserving each file's row order.
2. Normalize text with Python Unicode lowercase; remove characters outside
   `\w` and whitespace; collapse whitespace; strip. This defines a limited
   near-duplicate check, not a semantic similarity check.
3. Remove every row in a normalized-text group with conflicting labels. Keep
   the first row of each remaining normalized-text group, across all domains.
4. Split 80/20 using `train_test_split(random_state=42)`, stratified jointly by
   domain and label. Split the 20% holdout in half with the same rule.
5. Keep only detector-English test candidates. Within each domain, select up to
   500 candidates using a stratified sample, seed 42. Test exclusions are never
   moved into training or validation. No validation or test performance was
   used to choose this cap, model, or preprocessing.
6. Check exact and normalized-text overlap for all three split pairs; require
   zero. The files have no person, sender, campaign, or time identifiers, so
   this does not establish entity-independent or future-time generalization.

Training and validation retain their original language mix. Language filtering
applies only to the paired evaluation sample. The proposed follow-up translates
these exact IDs into Spanish and French and scores them with the frozen model.

## Main outputs

- `split.csv`: every retained training, validation and test row, with globally
  unique source-file-and-line IDs.
- `scores.csv`: continuous class-1 probabilities for all domain test sets and
  the pooled set.
- `paired_test_ids.csv`: the fixed source items for future translations.
- `audit.json`, `domain_audit.csv`, `language_counts.csv`: measured audit counts.
- `run_info.json`, `threshold.txt`: runtime, versions, threshold and validation
  performance.
- `checker_output.txt`: all rates, 95% intervals and warnings.
- `aggregate_metrics.json`: mean and test-count-weighted domain AP/AUROC. Both
  means coincide because all seven domain test sets have 500 rows. These are
  not the same statistic as pooled AP/AUROC.
- `confusion_matrix.png`: the pooled baseline figure.
- `data_manifest.json`, `output_hashes.json`: source and output SHA-256 hashes.
- `reproduction_check.json`: independent clean-directory reproduction evidence.

## Data and citations

Dataset: https://huggingface.co/datasets/redasers/difraud

Pinned revision: `f8fa04aff083aa22c6878d005b4655e34a4cbd85`.
All seven domains' `train.jsonl`, `validation.jsonl`, and `test.jsonl` are used.
The dataset card lists MIT; the publisher's license is in `source_LICENSE.txt`.
Raw texts are downloaded locally and omitted from the package and Git history.

Boumber, D. A., Qachfar, F. Z., and Verma, R. (2024). Domain-Agnostic Adapter
Architecture for Deception Detection: Extensive Evaluations with the DIFrauD
Benchmark. LREC-COLING 2024. https://aclanthology.org/2024.lrec-main.468/

Language detector: https://github.com/Mimino666/langdetect

Planned models (not executed in this checkpoint):
https://huggingface.co/FacebookAI/xlm-roberta-base
https://huggingface.co/Helsinki-NLP/opus-mt-en-es
https://huggingface.co/Helsinki-NLP/opus-mt-en-fr

AI assistance: OpenAI ChatGPT/Codex generated code and report text and helped
check the executed results. Numerical results were generated by the included
code and instructor-supplied checker. No human language review is claimed.

## Submission

Public code and results:
https://github.com/nrcisst/homework-solutions/tree/main/cosc4371/difraud-checkpoint

Submit the checkpoint PDF containing that link. To regenerate it after running
the baseline, use:

```sh
python make_report.py --repo-url https://github.com/nrcisst/homework-solutions/tree/main/cosc4371/difraud-checkpoint
```

The plot and PDF are generated locally; the repository includes their code and
the numerical evidence. The assignment requests one PDF plus the accessible
code link. No private dataset, credentials, or raw message texts are uploaded.

Follow the dated plan in the report for translations, language review and the
transformer. The compute budget explicitly uses assumed multipliers; only the
baseline runtime has been measured.
