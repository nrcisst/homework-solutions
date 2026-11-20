"""Independent alignment and report/checker consistency checks on the saved artifacts."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    split=pd.read_csv(ROOT/'split.csv')
    scores=pd.read_csv(ROOT/'scores.csv')
    errors=pd.read_csv(ROOT/'errors.csv')
    assert (ROOT/'split.csv').read_bytes()==(ROOT.parent/'difraud-checkpoint'/'split.csv').read_bytes()
    assert len(split)==89045 and split.row_id.is_unique
    assert len(scores)==21000
    assert not scores.duplicated(['row_id','value','column']).any()
    test=split[split.split=='test'].set_index('row_id')
    assert len(test)==3500 and int(test.label.sum())==1267
    assert np.array_equal(scores.label,test.loc[scores.row_id,'label'])
    assert set(scores.value)=={'English','Spanish','French'}
    assert scores.threshold.nunique()==1
    assert np.isfinite(scores[['score','threshold']].to_numpy()).all()
    for column,g in scores.groupby('column'):
        sets=[set(v.row_id) for _,v in g.groupby('value')]
        assert len(sets)==3 and sets[0]==sets[1]==sets[2]
        for _,v in g.groupby('value'):
            assert len(v)==(3500 if column=='LR / pooled' else 500)
    for value in scores.value.unique():
        p=scores[(scores.value==value)&(scores.column=='LR / pooled')].set_index('row_id')
        d=scores[(scores.value==value)&(scores.column!='LR / pooled')].set_index('row_id')
        assert np.array_equal(p.score,d.loc[p.index,'score'])
    merged=errors.merge(scores,on=['row_id','value','column'],validate='one_to_one')
    assert len(merged)==len(errors) and ((merged.score>=merged.threshold)!=merged.label).all()
    assert merged.category.notna().all() and merged.row_id.is_unique
    actual=pd.read_csv(ROOT/'verdicts.csv')
    text=(ROOT/'checker_output.txt').read_text().split("Difference from the baseline value 'English'",1)[1]
    column=None;checked=0
    for line in text.splitlines():
        if line.startswith('Column: '):column=line[8:]
        match=re.match(r'\s+(Spanish|French)\s+difference ([+-][\d.]+)\s+change ([+-][\d.]+)',line)
        if match:
            row=actual[(actual.value==match[1])&(actual.column==column)].iloc[0]
            assert f'{row.difference:+.4f}'==match[2]
            assert f'{row.change:+.4f}'==match[3]
            checked+=1
    assert checked==16
    result={'split_byte_identical_to_midsemester':True,'test_ids':3500,'score_rows':21000,'matched_checker_changes':checked,'sampled_errors':len(errors),'same_ids_and_labels_across_languages':True,'pooled_predictions_equal_domain_predictions':True,'status':'passed'}
    (ROOT/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
