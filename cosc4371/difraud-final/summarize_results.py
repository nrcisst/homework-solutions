"""Extract exact checker intervals, then compute separately labeled domain-mean intervals."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent

def main():
    text=(ROOT/'checker_output.txt').read_text()
    rows=[]
    for block in text.split('\nValue: ')[1:]:
        header=block.splitlines()[0]
        if '   Column: ' not in header: continue
        value,column=header.split('   Column: ',1)
        m=re.search(r'\n  AUROC\s+([0-9.]+) \[([0-9.]+), ([0-9.]+)\]',block)
        if m: rows.append({'value':value.strip(),'column':column.strip(),'auroc':float(m[1]),'low':float(m[2]),'high':float(m[3])})
    results=pd.DataFrame(rows)
    assert len(results)==24, len(results)
    results.to_csv(ROOT/'checker_auroc.csv',index=False)
    scores=pd.read_csv(ROOT/'scores.csv')
    point=pd.read_csv(ROOT/'point_metrics.csv')
    verdicts=[]
    for r in point.to_dict('records'):
        if r['value']=='English':continue
        base=point[(point.value=='English')&(point.column==r['column'])].iloc[0]
        diff=r['auroc']-base.auroc
        verdicts.append({'value':r['value'],'column':r['column'],'difference':diff,'change':abs(diff),'verdict':'yes' if abs(diff)>=.09 else 'no'})
    pd.DataFrame(verdicts).to_csv(ROOT/'verdicts.csv',index=False)
    aggregate=[]
    for value in ['English','Spanish','French']:
        cells=scores[(scores.value==value)&(scores.column!='LR / pooled')]
        groups=[g.sort_values('row_id') for _,g in cells.groupby('column')]
        points=[roc_auc_score(g.label,g.score) for g in groups]
        draws=[];rng=np.random.default_rng(42)
        for _ in range(1000):
            values=[]
            for g in groups:
                y=g.label.to_numpy();s=g.score.to_numpy()
                idx=rng.integers(0,len(g),len(g))
                # A single-class draw has undefined AUROC and is excluded as a whole.
                if np.unique(y[idx]).size<2:break
                values.append(roc_auc_score(y[idx],s[idx]))
            if len(values)==len(groups):draws.append(np.mean(values))
        lo,hi=np.percentile(draws,[2.5,97.5])
        aggregate.append({'value':value,'mean_domain_auroc':float(np.mean(points)),'weighted_mean_domain_auroc':float(np.average(points,weights=[len(g) for g in groups])),'low':float(lo),'high':float(hi),'bootstrap_seed':42,'valid_draws':len(draws)})
    pd.DataFrame(aggregate).to_csv(ROOT/'aggregate_metrics.csv',index=False)
    print(results.to_string(index=False));print(pd.DataFrame(verdicts).to_string(index=False));print(pd.DataFrame(aggregate).to_string(index=False))

if __name__=='__main__':main()
