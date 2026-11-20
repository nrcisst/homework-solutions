"""Validate the saved qualitative review against its prespecified random sample."""
import json,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    test=pd.read_json(ROOT/'test_texts.jsonl',lines=True).set_index('row_id')
    expected={(i,lang) for i in test.sort_index().sample(n=20,random_state=42).index for lang in ['es','fr']}
    notes=json.loads((ROOT/'translation_review_notes.json').read_text())
    assert len(notes)==40
    assert {(r['row_id'],r['language']) for r in notes}==expected
    assert len({(r['row_id'],r['language']) for r in notes})==40
    assert all(r['status'] in ['no_major_issue','wording_error','meaning_risk'] for r in notes)
    frame=pd.DataFrame(notes)
    corrected={(r['row_id'],r['language']) for r in json.loads((ROOT/'translation_corrections.json').read_text())}
    frame['targeted_postedit_applied']=[(r['row_id'],r['language']) in corrected for r in notes]
    translations={lang:pd.read_json(ROOT/f'translations_{lang}.jsonl',lines=True).set_index('row_id') for lang in ['es','fr']}
    frame['translation_sha256']=[hashlib.sha256(translations[r['language']].loc[r['row_id'],'translation'].encode()).hexdigest() for r in notes]
    if (ROOT/'translation_review.csv').exists():
        previous=pd.read_csv(ROOT/'translation_review.csv').set_index(['row_id','language'])
        for r in frame.to_dict('records'):
            assert previous.loc[(r['row_id'],r['language']),'translation_sha256']==r['translation_sha256'], 'Translation changed; review must be updated'
    frame['seed']=42
    frame['reviewer']='OpenAI ChatGPT/Codex (AI-assisted qualitative review)'
    frame['independent_bilingual_verification']=False
    frame.to_csv(ROOT/'translation_review.csv',index=False)
    counts=frame.status.value_counts().to_dict()
    summary=f"Initially, {counts.get('wording_error',0)} pairs had wording problems and {counts.get('meaning_risk',0)} had omissions or meaning risks; {counts.get('no_major_issue',0)} showed no obvious major issue. Four messages received logged meaning corrections before scoring. This does not establish label preservation for the full test set."
    (ROOT/'translation_review_summary.json').write_text(json.dumps({'sample_seed':42,'pairs':40,'distinct_source_ids':20,'status_counts':counts,'summary':summary},indent=2)+'\n')
    print(summary)

if __name__=='__main__':main()
