# COSC 4371 final checkpoint: DIFrauD language comparison

Biniam Habte, Project 1, Fall 2026.

This folder extends the adjacent `difraud-checkpoint` midsemester submission.
The variable is test-text language (English, Spanish, French), the primary metric
is AUROC, and the practical rule is **yes** for an absolute change >= 0.09;
**no** for a smaller change. Exactly 0.09 is yes. There was no prespecified
indecision band. This is a practical threshold, not a significance test.

## Completed checkpoint scope

The final artifacts report the frozen word/character TF-IDF logistic-regression
classifier on the same 3,500 source IDs in all three languages. Each language
retains all 500 examples per domain, both classes, and the original labels.
Seven domain summaries and one pooled summary are reported. Pooled predictions
are reused domain predictions, not independent observations.

XLM-R-base has a short training-time benchmark only. Its classification head is
not a completed fitted detector and no transformer performance is reported.
The remaining 24 XLM-R cells and other unfinished project elements are explicitly
dated in `completion_plan.json` and the PDF.

## Reproduce

Use Python 3.12.14, at least 10 GB of system memory, and sufficient disk space.
Keep model caches on disk rather than a RAM-backed temporary filesystem.
Install CPU PyTorch first if a GPU build is not needed:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python run_final.py
```

Run from this directory. Keep the adjacent `../difraud-checkpoint` directory;
it contains the immutable IDs, source manifest, baseline probabilities and data
downloader that this extension verifies. A fresh reproduction downloads public
source data and pinned model weights. No credentials are needed.

The steps can also be run separately:

```sh
python prepare_baseline.py
python translate_test.py --lang es
python translate_test.py --lang fr
python repair_capped_translations.py --lang es
python repair_capped_translations.py --lang fr
python apply_translation_corrections.py
python score_comparisons.py
python finalize_reviews.py
python check_final_checkpoint.py --split split.csv --scores scores.csv --errors errors.csv --metric auroc --reference English --direction either > checker_output.txt
python summarize_results.py
python validate_artifacts.py
python make_report.py
```

`prepare_baseline.py` uses the committed split order rather than resampling. It
refits the unchanged LR pipeline and requires its validation threshold and all
English test scores to reproduce the midsemester values within 1e-10. It checks
exact and normalized overlaps for all three split boundaries. The vocabulary
and IDF are fitted on training only. Only raw message text is used as a feature.

The alert threshold is the lowest observed validation probability satisfying
FPR <= 5%. Spanish and French reuse that exact threshold. AUROC is computed on
continuous class-1 probabilities, not predicted labels.

The CSV schemas and checker script follow the instructor's final-checkpoint
handout. The unmodified checker uses 1,000 row-bootstrap draws per cell with its
value/column-derived seed. `summarize_results.py` extracts its printed AUROC
intervals and computes the separate mean-domain intervals (within-domain row
bootstrap, 1,000 draws, seed 42). Equal domain sizes make the unweighted and
test-count-weighted domain means identical; neither is pooled AUROC.

## Translation details and limitations

Model names and exact revisions are in `translation_config_es.json` and
`translation_config_fr.json`. Both are the OPUS-MT models named at midsemester.
Inference uses CTranslate2 int8, greedy decoding (beam size 1), two CPU threads,
and token-budget batching. Sentences are split at punctuation followed by
whitespace, or newlines. Each sentence is tokenized without truncation; longer
ones are divided into at most 256 source tokens. Every segment is translated
and the results are joined in source order. The output cap is 512 tokens per
segment. The scoring script refuses results if any unresolved segment reaches
that cap. Before scoring, capped messages are retried by the repair script:
their segments are retransmitted, and capped segments use beam size 4 and a
no-repeat-3-gram rule while retaining their context. If a segment still caps,
it is recursively bisected. Every initial cap and repaired row is recorded.
This deterministic recovery rule was fixed before translated scores were seen.

This is a CPU implementation choice, fixed before scoring the translated
messages. Full input coverage does not guarantee faithful translation: a
translation model can omit or distort content. Item-level AI-assisted review
of 20 pairs per language is recorded in `translation_review.csv`; independent
bilingual verification remains a limitation and a planned follow-up.

Before classifier scoring, AI review found clear claim-direction, polarity or
active/passive reversals. Six phrase corrections in four translated messages
were applied using `translation_corrections.json` and
`apply_translation_corrections.py`. The audit is explicit: these are reviewed
machine translations with targeted post-edits, not entirely untouched OPUS-MT
output. Other wording errors and omissions remain in the review notes. The
training data, test IDs, inherited labels and validation threshold were not
changed. No classifier outcome was used to select a correction.

Translations are resumable local JSONL caches. Each row records its original
ID, inherited label, source SHA-256, input/output token counts and cap status.
These caches and model weights are intentionally excluded from Git. The code
regenerates them using the pinned public sources. The packaged run archive
includes the generated translations for reuse; raw source data and model
weights are not bundled.

## Error analysis

The sampled cell is Spanish / LR pooled, selected before individual errors were
inspected. `score_comparisons.py` sorts by row ID and samples up to ten false
positives and ten false negatives separately with `random_state=42`; if fewer
exist, it selects all of them. `error_categories.csv` stores the reviewed
assignment of exactly one category per sampled ID; `category_rules.json`
defines the content-based rules. The script verifies the sampled IDs before
writing `errors.csv`. Categories are descriptive, not claims about model
causality. No catch-all category is used. Review used AI assistance.

## Main files

- `split.csv`, `scores.csv`, `errors.csv`: required committed evidence.
- `checker_output.txt`: complete unedited instructor-checker stdout.
- `checker_auroc.csv`, `verdicts.csv`: values shown in the report.
- `aggregate_metrics.csv`: separate mean/weighted domain summary.
- `run_info.json`: measured LR times, threshold and overlap checks.
- `point_metrics.csv`: AUROC/AP/precision/recall/FPR and error counts.
- `translation_stats_*.json`, `translation_config_*.json`: measured translation
  times, coverage and pinned settings.
- `translation_review.csv`, `translation_review_summary.json`: qualitative audit.
- `transformer_benchmark.json`: actual CPU timing, not performance results.
- `benchmark_transformer.py`: repeat the training-only timing benchmark; the
  evaluation pipeline does not use the resulting temporary weights.
- `benchmark_language_audit.py`, `language_audit_benchmark.json`: measured
  detector throughput for the remaining full disagreement audit; not a claim
  that the second-detector review is finished.
- `completion_plan.json`: statuses and work targets for remaining project work.
- `provenance.json`, `output_hashes.json`: source and output hashes.
- `make_report.py`: produces `Habte_Final_Checkpoint.pdf` with a five-page body
  and full checker appendix.

The benchmark starts from the pinned XLM-R-base checkpoint, initializes its
classification head with seed 42, and measures three real AdamW updates on six
training rows (batch size 2, padded/truncated length 256, four CPU threads).
The first update is excluded from the steady-time estimate. The planned
two-epoch run extrapolates from the last two updates; download/tokenization,
manual review and reporting are not included. Contention and a short sample
limit the precision of this estimate. No test item enters this benchmark.

## Sources

- DIFrauD: https://huggingface.co/datasets/redasers/difraud
- Dataset revision: `f8fa04aff083aa22c6878d005b4655e34a4cbd85`
- Boumber, Qachfar and Verma (2024), *Domain-Agnostic Adapter Architecture for
  Deception Detection: Extensive Evaluations with the DIFrauD Benchmark*:
  https://aclanthology.org/2024.lrec-main.468/
- Spanish model: https://huggingface.co/Helsinki-NLP/opus-mt-en-es
- French model: https://huggingface.co/Helsinki-NLP/opus-mt-en-fr
- Transformer: https://huggingface.co/FacebookAI/xlm-roberta-base
- CTranslate2 model support: https://opennmt.net/CTranslate2/guides/transformers.html

DIFrauD is distributed under MIT; the original license is retained in the
midsemester folder. OPUS-MT model cards list Apache 2.0.

AI use: OpenAI ChatGPT/Codex helped write code/report text, review translation
and error samples, and verify executed results. All reported numerical outcomes
come from the included programs and the instructor's checker. AI review is not
independent bilingual or human verification.
