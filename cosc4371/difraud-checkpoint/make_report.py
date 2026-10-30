"""Build the checkpoint PDF directly from executed results and checker output."""
import argparse
from html import escape
import json
import math
from pathlib import Path
import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parent
NAMES = {"fake_news": "Fake news", "job_scams": "Jobs", "phishing": "Phishing",
         "political_statements": "Political", "product_reviews": "Reviews", "sms": "SMS",
         "twitter_rumours": "Twitter", "pooled": "Pooled"}


def load(name):
    return json.loads((ROOT / name).read_text())


def read_checker():
    text = (ROOT / "checker_output.txt").read_text()
    sets = {}
    for block in text.split("Test set: ")[1:]:
        name, rest = block.split("\n", 1)
        row = {"n": int(re.search(r"test rows\s+(\d+)", rest)[1]),
               "positive": int(re.search(r"positive rows\s+(\d+)", rest)[1]),
               "positive_percent": re.search(r"positive rows\s+\d+ \(([^)]+)\)", rest)[1],
               "random_AP": re.search(r"random-scorer AP\s+([0-9.]+)", rest)[1]}
        for metric in ["precision", "recall", "false-positive rate", "AP", "AUROC"]:
            found = re.search(r"^  " + re.escape(metric) + r"\s+([^\n]+)", rest, re.M)
            if found is None:
                raise ValueError(f"Missing {metric} for {name}")
            row[metric] = found[1].strip()
        sets[name] = row
    assert len(sets) == 8
    return sets


def main(repo_url):
    audit, info = load("audit.json"), load("run_info.json")
    aggregate, cm = load("aggregate_metrics.json"), load("confusion_matrix.json")["counts"]
    scores = read_checker()
    split = pd.read_csv(ROOT / "split.csv")
    languages = pd.read_csv(ROOT / "language_counts.csv")
    domain_audit = pd.read_csv(ROOT / "domain_audit.csv").set_index("domain")
    half_widths = {}
    for key, item in scores.items():
        bounds = re.search(r"\[([0-9.]+), ([0-9.]+)\]", item["AUROC"])
        half_widths[key] = (float(bounds[2]) - float(bounds[1])) / 2
    widest = max(half_widths, key=half_widths.get)
    half = half_widths[widest]
    delta = (math.floor(half * 100) + 1) / 100
    baseline_minutes = info["fit_seconds"] / 60
    score_minutes = info["test_score_seconds"] / 60
    budget = 103 * baseline_minutes + 2 * score_minutes
    flags = int(languages["non_english_flags"].sum())
    unknown = int(languages["unknown"].sum())

    # Use embedded fonts when available, with standard PDF fonts as a portable fallback.
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    if (font_dir / "DejaVuSerif.ttf").exists():
        pdfmetrics.registerFont(TTFont("Body", str(font_dir / "DejaVuSerif.ttf")))
        pdfmetrics.registerFont(TTFont("BodyBold", str(font_dir / "DejaVuSerif-Bold.ttf")))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
        body, bold = "Body", "BodyBold"
    else:
        body, bold = "Times-Roman", "Times-Bold"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyTextCustom", fontName=body, fontSize=9.3, leading=12.2, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallCustom", fontName=body, fontSize=8, leading=10.3, spaceAfter=4))
    styles.add(ParagraphStyle(name="TableCustom", fontName=body, fontSize=7.7, leading=10))
    styles.add(ParagraphStyle(name="TableCenter", parent=styles["TableCustom"], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="TableCompact", fontName=body, fontSize=7.5, leading=9))
    styles.add(ParagraphStyle(name="HeadingCustom", fontName=bold, fontSize=12.5, leading=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="TitleCustom", fontName=bold, fontSize=16, leading=20, spaceAfter=7))
    story = []

    def p(text, small=False):
        story.append(Paragraph(text, styles["SmallCustom" if small else "BodyTextCustom"]))

    def heading(text):
        story.append(Paragraph(text, styles["HeadingCustom"]))

    def table(rows, widths, font="TableCustom"):
        cells = [[Paragraph(str(cell), styles[font]) for cell in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5 if font == "TableCompact" else 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 if font == "TableCompact" else 4),
        ]))
        story.extend([t, Spacer(1, 7)])

    def metric(value):
        return value.replace(" [", "<br/>[")

    story.append(Paragraph("Midsemester Checkpoint", styles["TitleCustom"]))
    p("Biniam Habte | COSC 4371 | October 30, 2026<br/><b>Project 1: DIFrauD Non-English Samples</b>")
    p("Project number follows the instructor's assignment. The distributed project list numbers this topic 2.", True)
    if repo_url:
        p('Code: <link href="' + escape(repo_url, quote=True) + '">' + escape(repo_url) + '</link>', True)
    else:
        p("<b>DRAFT: add a public repository or accessible shared-folder URL before submitting.</b>", True)
    heading("1. Experimental design")
    p("<b>Question.</b> How much does deception detection change when the same DIFrauD test messages are translated into Spanish and French? I will compare logistic regression and a multilingual transformer across the seven domains. This could help decide whether translated messages need a different model or human review.")
    p("<b>Variable.</b> The test-message language changes: original English, Spanish, or French.")
    p("<b>Fixed.</b> Within each classifier's comparison, the training data, fitted model, validation threshold, source test IDs, and labels stay the same. Only the test text is translated.")
    p("<b>Changes since the proposal.</b> AUROC replaces F1 as the primary metric. I now remove duplicate text globally and drop conflicting-label groups before splitting. The detector and near-duplicate checks below are fresh runs with explicit rules, so their counts replace the initial audit estimates. The language comparison and two planned classifiers are unchanged.")
    p("<b>Split.</b> After cleaning, use an 80/20 stratified random split, then divide the holdout equally into validation and test. Both steps use seed 42 and joint domain/class strata. Select up to 500 detector-English test items per domain, stratified by class with seed 42. Time and entity splits are unavailable because the source has no timestamps or entity IDs.")
    p("<b>Primary metric.</b> AUROC measures ranking quality and is less sensitive to the different class shares across domains than precision or AP. Recall and false-positive rate at the validation-selected threshold remain useful for triage.")
    p("<b>Comparison table: logistic-regression AUROC.</b> A dash means the run is still planned. All baseline cells use the fixed source IDs. The transformer will have a separate table with the same layout.")
    ds = list(NAMES)[:-1]
    rows = [["Language"] + [NAMES[d] for d in ds] + ["Pooled"]]
    rows.append(["English"] + [scores["English_" + d]["AUROC"].split()[0] for d in ds] + [scores["English_pooled"]["AUROC"].split()[0]])
    rows.extend([[lang] + ["-"] * 8 for lang in ["Spanish", "French"]])
    table(rows, [62] + [56.25] * 8, "TableCenter")
    p(f"Mean domain AUROC: <b>{aggregate['AUROC']['mean']:.4f}</b>. Test-count-weighted mean: <b>{aggregate['AUROC']['weighted_mean']:.4f}</b>. They match because each domain has 500 test rows. Pooled AUROC is a separate calculation over all 3,500 rows.")
    p("The English baseline is complete. It does not yet show whether translation improves or hurts performance.")
    p(f"<b>Code and reproduction.</b> Use Python {info['python']} and the pinned requirements.txt. After installing dependencies, run <font face='Courier'>python run_checkpoint.py</font>. It downloads the pinned files, runs the audit, trains the baseline, writes split.csv and scores.csv, and runs the unchanged checker with the exact validation-selected threshold. Both CSVs belong in the linked repository. The README gives the full setup; reproduction_check.json records the clean-directory check.", True)

    story.append(PageBreak())
    heading("2. Data and split")
    p("<b>Dataset and fit.</b> DIFrauD contains labeled deceptive and non-deceptive texts from seven domains. It fits this question because translating the same texts allows a language comparison without changing their underlying examples or labels. This addresses the proposal feedback.")
    p('I loaded all 21 JSONL files from <link href="https://huggingface.co/datasets/redasers/difraud">redasers/difraud</link>, revision <font size="7.5">f8fa04aff083aa22c6878d005b4655e34a4cbd85</font>. Each domain has train, validation, and test files. SHA-256 checksums are in data_manifest.json. The dataset is described by Boumber, Qachfar, and Verma (2024). The card lists MIT; raw texts are downloaded rather than bundled.', True)
    p("One row is one text. Label 1 means deceptive; label 0 means non-deceptive. The authors harmonized source labels and describe Cleanlab processing and manual corrections. Labels are inherited, not independently verified here.", True)
    table([
        ["Audit on all loaded rows", "Measured result"],
        ["Rows / source columns", "95,854 / 2 (text, label)"],
        ["Classes", "37,282 deceptive (38.89%); 58,572 non-deceptive (61.11%)"],
        ["Missing target / missing or blank text", "0 / 0"],
        ["Exact duplicates: identical full text", "93 groups; 208 rows in groups; 115 extra rows; 34 groups cross original splits"],
        ["Near duplicates: normalized text", "670 groups; 1,472 rows in groups; 802 extra rows; 263 groups cross original splits"],
        ["Repeated entities / time range", "Unavailable: no sender, user, campaign, or timestamp fields"],
    ], [195, 317])
    p("Near-duplicate rule: lowercase, remove punctuation, collapse whitespace, and strip. This catches formatting variants, not paraphrases or every shared template. An extra duplicate is any row after the first in its group.", True)
    lang_rows = [["Domain", "Rows", "Deceptive<br/>n (%)", "Flagged non-English<br/>label 0: n (%)", "Flagged non-English<br/>label 1: n (%)"]]
    for d in ds:
        da = domain_audit.loc[d]
        l0 = languages[(languages.domain == d) & (languages.label == 0)].iloc[0]
        l1 = languages[(languages.domain == d) & (languages.label == 1)].iloc[0]
        lang_rows.append([NAMES[d], f"{int(da.rows):,}", f"{int(da.positive):,} ({da.positive_percent:.2f}%)",
                          f"{int(l0.non_english_flags)} ({l0.flagged_percent:.2f}%)",
                          f"{int(l1.non_english_flags)} ({l1.flagged_percent:.2f}%)"])
    table(lang_rows, [67, 51, 113, 141, 140])
    p(f"langdetect 1.0.9, seed 42: {flags} non-English flags ({100*flags/audit['raw_rows']:.3f}%) and {unknown} unknowns (all non-deceptive SMS). Percentages above use each domain/class count. These are detector estimates. Short SMS can be misidentified; human review and a second detector are still planned.", True)
    split_rows = [["DIFrauD split", "Rows", "Deceptive", "Non-deceptive", "Deceptive %"]]
    for role in ["train", "validation", "test"]:
        part = split[split.split == role]
        pos = int(part.label.sum())
        split_rows.append([role, f"{len(part):,}", f"{pos:,}", f"{len(part)-pos:,}", f"{100*pos/len(part):.2f}%"])
    table(split_rows, [108, 101, 101, 101, 101])
    p(f"<b>Removals and moves.</b> Remove {audit['conflict_rows_removed']} conflicting-label rows and {audit['additional_duplicate_rows_removed']} other duplicate rows, leaving {audit['clean_rows']:,}. Repartitioning changes the original role of {audit['original_roles_changed_before_test_sampling']:,} rows. Of {audit['test_candidates']:,} test candidates, exclude {audit['test_non_english_or_unknown_excluded']} language flags/unknowns and {audit['test_english_excluded_by_cap']:,} through the sample cap. Keep {audit['retained_rows']:,} rows in split.csv. Excluded test rows never return to training.", True)
    p("<b>Leakage check.</b> Training rows exactly matching test: <b>0</b>; matching validation: <b>0</b>. Normalized-text matches are also 0, as are validation/test matches. Global duplicate removal protects every split boundary. Unobserved shared entities remain a limitation. Training and validation still include 521 and 66 non-English flags, respectively; only the paired test sample is language-filtered.", True)

    story.append(PageBreak())
    heading("3. Baseline and metrics")
    p("<b>Pipeline.</b> Use raw text only. Both TF-IDF branches lowercase and strip accents. Word unigrams/bigrams use at most 50,000 features; character-within-word 3-5 grams use at most 30,000. Both use min_df=2, sublinear term frequency, and L2 normalization. Concatenate them, then fit logistic regression (C=1, L2 penalty, liblinear, max_iter=2000, no class weighting, seed 42). Vocabularies and IDF are fitted on training only.")
    p(f"<b>Measured run.</b> TF-IDF plus classifier fitting took {info['fit_seconds']:.2f} seconds ({baseline_minutes:.2f} minutes), using {info['feature_count']:,} features. Hardware: {escape(info['cpu'])}, Linux CPU, one numerical thread; no GPU. Validation/scoring took {info['validation_seconds']:.2f}/{info['test_score_seconds']:.2f} seconds. The model converged in {info['lr_iterations'][0]} iterations.", True)
    p(f"<b>Positive class and threshold.</b> Positive = deceptive (1). Predict positive when score &gt;= <b>{repr(info['threshold'])}</b>. Select the lowest observed validation-score threshold with validation FPR at most 5%. It gives validation FPR {info['validation_fpr']:.4f} and recall {info['validation_recall']:.4f}. No test results select the threshold.")
    p("<b>Checker results.</b> Every interval below is the instructor checker's 95% row-bootstrap interval (1,000 resamples, seed 0). Domain names refer to English_&lt;domain&gt;; Pooled refers to English_pooled. Each text is predicted once; pooled reporting reuses those scores.", True)
    rows = [["Test set", "n / positive n (%)", "Random AP", "AP [95% interval]", "AUROC [95% interval]"]]
    for d in [*ds, "pooled"]:
        m = scores["English_" + d]
        rows.append([NAMES[d], f"{m['n']:,} / {m['positive']:,}<br/>({m['positive_percent']})", m["random_AP"], metric(m["AP"]), metric(m["AUROC"])])
    table(rows, [61, 99, 60, 146, 146], "TableCompact")
    rows = [["Test set", "Precision [95% interval]", "Recall [95% interval]", "FPR [95% interval]"]]
    for d in [*ds, "pooled"]:
        m = scores["English_" + d]
        rows.append([NAMES[d], metric(m["precision"]), metric(m["recall"]), metric(m["false-positive rate"])])
    table(rows, [62, 150, 150, 150], "TableCompact")
    failed = [NAMES[d] for d in [*ds, "pooled"] if float(scores['English_'+d]['AP'].split()[0]) <= float(scores['English_'+d]['random_AP']) or float(scores['English_'+d]['AUROC'].split()[0]) <= .5]
    p("<b>Random reference.</b> " + ("AP exceeds the positive share and AUROC exceeds 0.5 in every test set." if not failed else "The following sets do not beat both random references: " + ", ".join(failed) + "."), True)
    high = [NAMES[d] for d in [*ds, "pooled"] if any(float(scores['English_'+d][m].split()[0]) >= .99 for m in ['precision','recall','AP','AUROC'])]
    p("<b>Too-good check.</b> " + ((", ".join(high) + " reach 0.99 on at least one checked metric. Exact and normalized train/test overlap are both 0. Only text enters the model; labels and domain names are not features. Source-specific wording or templates can still make this split easier than deployment.") if high else "None."), True)
    small_sets = [f"{NAMES[d]} ({scores['English_'+d]['positive']} positive)" for d in ds if scores['English_'+d]['positive'] < 30 or scores['English_'+d]['n'] - scores['English_'+d]['positive'] < 30]
    if small_sets:
        p("<b>Small-class warning.</b> " + ", ".join(small_sets) + " has fewer than 30 examples in one class. Its interval is less stable; the decision margin in Section 5 uses the widest baseline interval.", True)
    p("Jobs precision is 1.0000 from just one positive prediction; recall is only 0.0476. Zero observed false positives and the degenerate bootstrap interval do not establish zero future risk.", True)
    p("<b>Reproducibility.</b> See the code link and setup command in Section 1. Package versions, data checksums, seed 42, both CSVs, and clean-directory reproduction evidence are included.", True)

    story.append(PageBreak())
    heading("4. Figure")
    story.append(Image(str(ROOT / "confusion_matrix.png"), width=271, height=197, hAlign="CENTER"))
    pooled = scores["English_pooled"]
    p(f"<b>Figure 1.</b> Count confusion matrix for the DIFrauD English_pooled baseline: {pooled['n']:,} test rows, {pooled['positive_percent']} deceptive. Rows are actual labels; columns are predictions.", True)
    p(f"The model catches {cm[1][1]:,} of {pooled['positive']:,} deceptive messages (recall {pooled['recall'].split()[0]}) before translation. The translated versions will show how much of that detection carries over.", True)
    heading("5. Plan to finish")
    p("Dates below are work targets, not a claimed course deadline.", True)
    table([
        ["Required element", "Status", "Finish date"],
        ["Explain dataset fit; audit, clean, split, and run one baseline", "Done", "Oct 30, 2026"],
        ["Count non-English samples by domain/class; review flags with a second detector and inspect disagreements", "Started", "Nov 1, 2026"],
        ["Translate the fixed 3,500 English test items into Spanish and French; inspect 20 pairs per language", "Not started", "Nov 2, 2026"],
        ["Score Spanish and French with the frozen logistic-regression model", "Not started", "Nov 2, 2026"],
        ["Train XLM-RoBERTa-base on the same training IDs; select its threshold on validation only", "Not started", "Nov 3, 2026"],
        ["Compare both classifiers on all three languages, per domain and aggregate mean/weighted performance", "Started: English LR only", "Nov 4, 2026"],
        ["Interpret language effects, check limitations, and rerun the final package", "Started", "Nov 5, 2026"],
    ], [335, 89, 88])
    p(f"<b>Main comparison.</b> Compare English, Spanish, and French on the same source IDs for each fixed classifier. Use the same training and validation IDs for both models, with a separate validation-selected threshold per model. Primary metric: AUROC. The widest baseline interval is {NAMES[widest.removeprefix('English_')]}: {scores[widest]['AUROC']}; half-width = {half:.5f}. Call an absolute language difference of at least <b>{delta:.2f} AUROC</b> a yes; a smaller difference is a no (not resolved beyond this margin). Report direction, every domain, and both aggregate means. This is a practical cutoff, not a multiple-comparison significance test.", True)
    p(f"<b>Remaining compute budget.</b> Let B = measured fit time = {baseline_minutes:.2f} minutes. Reserve 60B for one transformer fit, 20B per translation language, and 3B for transformer scoring, plus two LR scoring passes ({2*score_minutes:.2f} minutes). Total: 103B + {2*score_minutes:.2f} = <b>{budget:.1f} minutes</b>, excluding manual review and downloads. The multipliers are planning assumptions, not measured transformer/translation runtimes; benchmark them before the full runs and revise the budget.", True)
    p("<b>Sources.</b> Dataset and license: huggingface.co/datasets/redasers/difraud. Boumber, Qachfar, and Verma (2024): aclanthology.org/2024.lrec-main.468. Detector: github.com/Mimino666/langdetect. Planned models: FacebookAI/xlm-roberta-base and Helsinki-NLP/opus-mt-en-es, opus-mt-en-fr on Hugging Face. Full links are in README.md.", True)
    p("<b>AI use.</b> OpenAI ChatGPT/Codex helped generate code and check executed results. The included code and supplied checker produced the numerical results.", True)

    output = ROOT / "Habte_Midsemester_Checkpoint.pdf"
    def footer(canvas, doc):
        canvas.setFont(body, 8)
        canvas.drawString(50, 25, "Biniam Habte | COSC 4371 | Project 1")
        canvas.drawRightString(562, 25, str(doc.page))
    SimpleDocTemplate(str(output), pagesize=letter, leftMargin=50, rightMargin=50,
                      topMargin=38, bottomMargin=40, title="DIFrauD Midsemester Checkpoint",
                      author="Biniam Habte").build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "report_metadata.json").write_text(json.dumps({"repo_url": repo_url or None,
        "submission_ready": bool(repo_url), "decision_delta": delta, "widest_half_width": half,
        "remaining_compute_budget_minutes": budget}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default="")
    args = parser.parse_args()
    if args.repo_url and not args.repo_url.startswith("https://"):
        parser.error("--repo-url must be an HTTPS link to the accessible repository or folder")
    main(args.repo_url)
