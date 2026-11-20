"""Build the five-page checkpoint body and the complete, unedited checker appendix."""
from pathlib import Path
import json,html,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Preformatted
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
ROOT=Path(__file__).resolve().parent
URL='https://github.com/nrcisst/homework-solutions/tree/main/cosc4371/difraud-final'
LANGS=['English','Spanish','French']
DOMAINS=['fake_news','job_scams','phishing','political_statements','product_reviews','sms','twitter_rumours','pooled']
NAMES=['Fake news','Jobs','Phishing','Political','Reviews','SMS','Twitter','Pooled']
def main():
    info=json.loads((ROOT/'run_info.json').read_text())
    ci=pd.read_csv(ROOT/'checker_auroc.csv')
    point=pd.read_csv(ROOT/'point_metrics.csv')
    verdict=pd.read_csv(ROOT/'verdicts.csv')
    agg=pd.read_csv(ROOT/'aggregate_metrics.csv')
    benchmark=json.loads((ROOT/'transformer_benchmark.json').read_text())
    audit_bench=json.loads((ROOT/'language_audit_benchmark.json').read_text())
    categories=json.loads((ROOT/'category_rules.json').read_text())
    review=json.loads((ROOT/'translation_review_summary.json').read_text())
    errors=pd.read_csv(ROOT/'errors.csv')
    scores=pd.read_csv(ROOT/'scores.csv')
    e=errors.merge(scores,on=['row_id','value','column'],validate='one_to_one')
    nfp=int((e.label==0).sum());nfn=int((e.label==1).sum())
    stats={l:json.loads((ROOT/f'translation_stats_{c}.json').read_text()) for l,c in [('Spanish','es'),('French','fr')]}
    pooled=ci[ci.column=='LR / pooled'].set_index('value').loc[LANGS]
    fig,ax=plt.subplots(figsize=(7,3.35))
    vals=pooled.auroc.to_numpy()
    ax.errorbar(range(3),vals,yerr=np.array([vals-pooled.low.to_numpy(),pooled.high.to_numpy()-vals]),fmt='o',capsize=6,color='#245b83',markersize=7,linewidth=1.5)
    ax.axhline(.5,color='gray',ls='--',lw=.8,label='Random ranking')
    ax.set(xticks=range(3),xticklabels=LANGS,ylabel='AUROC',ylim=(.45,1.0),xlim=(-.45,2.45))
    for i,v in enumerate(vals):ax.annotate(f'{v:.4f}',(i,v),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left',frameon=False)
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=10,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=8.2,leading=10.5,spaceAfter=5))
    styles.add(ParagraphStyle(name='Cell2',fontName='Helvetica',fontSize=8,leading=10))
    styles.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=17,leading=21,spaceAfter=9))
    styles.add(ParagraphStyle(name='Head2',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=7,spaceAfter=7))
    styles.add(ParagraphStyle(name='Code2',fontName='Courier',fontSize=7.7,leading=10,spaceAfter=7))
    story=[]
    def p(s,style='Body2'):story.append(Paragraph(s,styles[style]))
    def h(s):p(s,'Head2')
    def table(rows,widths):
        data=[[Paragraph(str(c),styles['Cell2']) for c in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.4,colors.gray),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.append(t);story.append(Spacer(1,9))
    def page():story.append(PageBreak())
    p('Final Project Checkpoint','Title2')
    p('Biniam Habte | COSC 4371 | November 19, 2026<br/>Project 1: DIFrauD Non-English Samples')
    p('Code and results: <link href="'+URL+'">'+URL+'</link>','Small2')
    h('1. Main comparison')
    p('<b>Question.</b> How much does deception detection change when the same DIFrauD test messages are translated from English into Spanish and French? I compare the languages within each fixed classifier, across seven domains and in the pooled sample.')
    p('<b>(a) What I ran.</b> I completed English, Spanish and French for the word/character TF-IDF logistic-regression classifier (LR). All 3,500 test IDs from the midsemester checkpoint are retained: 500 per domain, including 1,267 deceptive and 2,233 non-deceptive items. Training, validation, labels, classifier and threshold are fixed across languages. The original split and stratified sample use seed 42. There is no new subsample.')
    p('Spanish and French were generated with the planned OPUS-MT models. Messages are split at sentence punctuation/newlines, with longer segments divided into at most 256 source tokens. Every segment is translated and joined in order. The CPU implementation uses CTranslate2 int8 and greedy decoding. Output-capped segments are retried with beam 4 and no-repeat-3-gram decoding, then divided further only if still capped. These rules were fixed before translated classifier scores were examined. No input text is truncated; initial cap counts and repairs are recorded.')
    p(f'<b>Changes since midsemester.</b> The baseline was refitted on the same ordered IDs; its maximum probability difference from saved English scores was {info["baseline_max_score_difference"]:.2g}. AI review identified clear meaning reversals, so six phrase corrections in four translated messages were logged and applied before scoring (translation_corrections.json). The results therefore use reviewed machine translations with these targeted post-edits. XLM-R has only a timing benchmark. The variable, baseline, primary metric and decision rule are unchanged.')
    p(f'<b>(b) Threshold.</b> LR predicts deceptive when score &gt;= <b>{info["threshold"]!r}</b>. This is the lowest observed validation-score threshold with validation FPR at most 5%; validation FPR is {info["validation_fpr"]:.4f} and recall is {info["validation_recall"]:.4f}. The same threshold is used for all languages. AUROC uses continuous scores, so this threshold affects the error sample and operating rates, not AUROC.')
    p('<b>(c) Leakage.</b> Same training set. Rerun exact-text and normalized-text overlap checks found 0 training/test matches, 0 training/validation matches and 0 validation/test matches. Normalization lowercases, removes punctuation and collapses whitespace. Only message text enters the model. These checks address the checker\'s 0.99 warnings, but do not rule out shared sources or templates; the data have no entity IDs or timestamps.')
    page()
    h('1. Main comparison (continued)')
    p('<b>(d) Results.</b> AUROC followed by the checker\'s 95% row-bootstrap interval (1,000 resamples per cell). Rows are the language values. Columns below are domain-specific views of the fixed LR classifier. English is the reference value. The pooled column reuses all domain predictions; it is not an independent test set.','Small2')
    for inds in [[0,1,2,3],[4,5,6,7]]:
        rows=[['Language']+[f'LR: {NAMES[i]}' for i in inds]]
        for lang in LANGS:
            cells=[]
            for i in inds:
                r=ci[(ci.value==lang)&(ci.column=='LR / '+DOMAINS[i])].iloc[0]
                cells.append(f'{r.auroc:.4f}<br/>[{r.low:.4f}, {r.high:.4f}]')
            rows.append([lang]+cells)
        table(rows,[58,119,119,119,119])
    bad=point[(point.ap<=point.positive/point.n)|(point.auroc<=.5)]
    if len(bad):
        p('Cells no better than a random scorer on AP or AUROC: '+ '; '.join(f'{r.value}, {r.column}' for r in bad.itertuples())+'. The English-trained text features may lose their association with deception after translation.','Small2')
    else:p('Cells with AP at or below the positive share, or AUROC at or below 0.5: none.','Small2')
    p('<b>Unfinished XLM-R cells:</b> English, Spanish and French in every domain and pooled are shown as -- individually in Appendix B. All 24 cells are dated in Section 6(a). Jobs has only 21 positive items in each language; its estimates are less stable.','Small2')
    p('<b>(e) Verdict.</b> Direction: <b>either</b>. The midsemester rule is yes for an absolute AUROC difference of at least 0.09 and no for a smaller difference. I retain that strict boundary for no, so exactly 0.09 is yes. These cutoffs concern practical size, not statistical significance; no does not establish equivalence or absence of an effect. No separate indecision band was prespecified. Changes below are the checker\'s absolute differences from English.','Small2')
    rows=[['LR column','Spanish change / verdict','French change / verdict']]
    for domain,name in zip(DOMAINS,NAMES):
        cells=[]
        for lang in ['Spanish','French']:
            r=verdict[(verdict.value==lang)&(verdict.column=='LR / '+domain)].iloc[0]
            direction='drop' if r.difference<0 else 'increase'
            cells.append(f'{r.change:.4f} ({direction}) / {r.verdict}')
        rows.append([name]+cells)
    table(rows,[120,207,207])
    page()
    h('2. Figure')
    story.append(Image(str(ROOT/'comparison.png'),width=520,height=249))
    p('<b>Figure 1.</b> LR / pooled; the same 3,500 test rows in English, Spanish and French; 36.20% deceptive. Points show AUROC and bars show the checker\'s 95% intervals.','Small2')
    es=verdict[(verdict.value=='Spanish')&(verdict.column=='LR / pooled')].iloc[0]
    fr=verdict[(verdict.value=='French')&(verdict.column=='LR / pooled')].iloc[0]
    p(f'The pooled AUROC changes by {es.difference:+.4f} in Spanish and {fr.difference:+.4f} in French relative to English, which gives {es.verdict} and {fr.verdict} under the 0.09 practical rule.')
    rows=[['Language','Mean domain AUROC [95% interval]','Test-count-weighted mean']]
    for r in agg.itertuples():rows.append([r.value,f'{r.mean_domain_auroc:.4f} [{r.low:.4f}, {r.high:.4f}]',f'{r.weighted_mean_domain_auroc:.4f}'])
    table(rows,[85,274,175])
    p('Both means coincide because every domain contains 500 items. These means differ from pooled AUROC. Mean intervals are computed by summarize_results.py with 1,000 within-domain row-bootstrap draws (seed 42), then averaging the seven AUROCs in each draw; they are separate from the supplied checker.','Small2')
    h('3. Claim and limitation')
    p(f'<b>(a) Claim.</b> For LR / pooled, changing the fixed English test texts to these OPUS-MT Spanish and French translations produces absolute AUROC changes of {es.change:.4f} and {fr.change:.4f}, yielding {es.verdict} and {fr.verdict} under the prespecified rule; this claim does not cover the unfinished XLM-R cells.')
    p('<b>(b) Limitation.</b> Translation can remove, simplify or alter details that affect deception, so a difference may reflect translation artifacts as well as language. The check that would settle this concern is independent bilingual review of label-relevant meaning, followed by rescoring corrected translations for the same IDs. The present results apply to these machine translations, not naturally authored Spanish or French.')
    p(f'An AI-assisted check inspected the same 20 randomly selected IDs per language (seed 42), giving 40 pairs. {html.escape(review["summary"])} This is not independent bilingual or human verification. The item-level observations are in translation_review.csv.','Small2')
    page()
    h('4. Error analysis')
    cell=point[(point.value=='Spanish')&(point.column=='LR / pooled')].iloc[0]
    p(f'<b>(a) Sample.</b> I selected Spanish / LR pooled before examining its individual errors. At the fixed validation threshold, this cell has {int(cell.fp)} false positives and {int(cell.fn)} false negatives. After sorting by row_id, I sampled {nfp} false positives and {nfn} false negatives without replacement, using pandas random_state=42 separately for each type. If fewer than 10 were available, all were taken.')
    p('<b>(b) Categories.</b> Each sampled row has exactly one category. The rules describe the text, not a proven causal explanation of the classifier\'s decision. Classification and category review used the original English text alongside its Spanish translation.','Small2')
    rows=[['Category and rule','FP','FN','Example row_id']]
    for c in categories:
        g=e[e.category==c['category']]
        if len(g):rows.append([f'<b>{html.escape(c["category"])}</b><br/>{html.escape(c["rule"])}',str(int((g.label==0).sum())),str(int((g.label==1).sum())),html.escape(g.row_id.iloc[0])])
    table(rows,[292,30,30,182])
    p('The errors.csv categories and counts match the supplied checker. No catch-all category is used. Categories were assigned with AI assistance; they are descriptive qualitative annotations.','Small2')
    h('5. Reproducibility')
    p('<b>(a) Files and command.</b> The linked repository contains split.csv, scores.csv, errors.csv, pinned translation configurations, baseline verification, the error/translation review records and scripts. The midsemester folder is retained. Each translated copy uses its original row_id and label.','Small2')
    p('From cosc4371/difraud-final, install requirements.txt and run <font name="Courier">python run_final.py</font>. The pipeline downloads checksum-verified DIFrauD files, refits and verifies the English model, translates both languages, writes scores.csv and errors.csv, runs the checker and builds this report. Downloads and model inference require internet access and adequate CPU memory.','Small2')
    story.append(Preformatted('python check_final_checkpoint.py --split split.csv --scores scores.csv\n  --errors errors.csv --metric auroc --reference English --direction either',styles['Code2']))
    p('Type the checker command on one line. Translation model revisions are pinned in translation_config_es.json and translation_config_fr.json. Scores are class-1 probabilities. The unchanged instructor checker is included; its SHA-256 is recorded in provenance.json.','Small2')
    p('<b>(b) Checker output.</b> Appendix A contains the complete output, including warnings. Results in Section 1(d) and changes in 1(e) are taken from that run. The high-score leakage checks and Jobs small-class warning are addressed above.','Small2')
    page()
    h('6. Plan to finish')
    p('<b>(a) Required elements.</b> This updates the seven elements in midsemester Section 5(a). Dates are work targets, not an asserted course deadline. The checkpoint\'s LR comparison is complete; the broader project is still in progress.','Small2')
    plan=json.loads((ROOT/'completion_plan.json').read_text())
    table([['Required element','Status','Finish target']]+[[r['element'],r['status'],r['date']] for r in plan],[348,85,101])
    p('Every unfinished XLM-R cell (English, Spanish and French across seven domains plus pooled, 24 cells) has the same finish target shown for the model comparison. No transformer point estimate, threshold or interval is presented as completed.','Small2')
    h('(b) Remaining compute')
    total=benchmark['training_minutes_estimate']+benchmark['validation_and_test_minutes_estimate']+audit_bench['estimated_minutes']
    p(f'The completed LR fit took {info["fit_seconds"]/60:.2f} minutes. Spanish and French translation took {stats["Spanish"]["translation_seconds"]/60:.2f} and {stats["French"]["translation_seconds"]/60:.2f} minutes in the completed runs. Those tasks have no remaining compute budget.')
    p(f'For XLM-R-base, I measured three actual AdamW training steps on training rows only, with batch size 2, maximum length 256, four CPU threads and learning rate 2e-5. Step times were '+', '.join(f'{t:.2f}' for t in benchmark['training_steps_seconds'])+f' seconds. Excluding the first warm-up step gives {benchmark["steady_seconds_per_step"]:.2f} seconds per batch. The benchmark weights are discarded and are not an evaluated model.')
    p(f'Two full epochs over 76,040 training rows require 76,040 batch updates. At the measured rate, training is approximately <b>{benchmark["training_minutes_estimate"]:,.0f} minutes</b>. Two validation passes and scoring all three test languages add about <b>{benchmark["validation_and_test_minutes_estimate"]:,.0f} minutes</b>. The remaining language-detector audit adds approximately <b>{audit_bench["estimated_minutes"]:.0f} minutes</b>, extrapolated from 200 measured rows. Total remaining compute is approximately <b>{total:,.0f} minutes ({total/60:.1f} hours)</b> on this CPU.')
    p('This is an extrapolation from a short benchmark, not a completed full training run or a guaranteed runtime. It excludes downloads, tokenization, checker/report regeneration and manual review. A faster accelerator would need its own benchmark. Model selection will use validation only, followed by its own threshold satisfying validation FPR at most 5%.')
    h('Sources and AI use')
    p('DIFrauD: redasers/difraud, revision f8fa04aff083aa22c6878d005b4655e34a4cbd85 (MIT). Boumber, Qachfar and Verma (2024), <i>Domain-Agnostic Adapter Architecture for Deception Detection</i>, LREC-COLING: aclanthology.org/2024.lrec-main.468. Translation: Helsinki-NLP/opus-mt-en-es and opus-mt-en-fr (Apache 2.0). Planned classifier: FacebookAI/xlm-roberta-base. Full links and pinned revisions are in README.md.','Small2')
    p('OpenAI ChatGPT/Codex helped write the code and report, review translation/error samples, and check executed results. Numerical results come from the included code and the instructor\'s checker.','Small2')
    page()
    h('Appendix A. Full checker output')
    raw=(ROOT/'checker_output.txt').read_text()
    wrapped=[]
    for line in raw.splitlines():wrapped.extend(textwrap.wrap(line,width=112,replace_whitespace=False,drop_whitespace=False) or [''])
    for i in range(0,len(wrapped),70):
        if i:page();h('Appendix A. Full checker output (continued)')
        story.append(Preformatted('\n'.join(wrapped[i:i+70]),ParagraphStyle(name='Appendix',fontName='Courier',fontSize=7.5,leading=9.3)))
    page()
    h('Appendix B. Unfinished classifier cells')
    p('Classifier: XLM-RoBERTa-base. These are the remaining primary-metric AUROC cells. Every -- means unfinished, not zero. All cells have a finish target of November 25, 2026, as listed in Section 6(a). No evaluated transformer model or threshold is claimed.')
    for inds in [[0,1,2,3],[4,5,6,7]]:
        table([['Language']+[f'XLM-R: {NAMES[i]}' for i in inds]]+[[lang]+['--']*len(inds) for lang in LANGS],[58,119,119,119,119])
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.drawString(39,24,'Biniam Habte | COSC 4371 | Project 1');canvas.drawRightString(573,24,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/'Habte_Final_Checkpoint.pdf'),pagesize=(612,792),rightMargin=39,leftMargin=39,topMargin=36,bottomMargin=40)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

if __name__=='__main__':main()
