"""Write the final-checkpoint schema and a deterministic 10-FP/10-FN sample."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent
SEED=42

def main():
    info=json.loads((ROOT/'run_info.json').read_text())
    model=joblib.load(ROOT/'lr_model.joblib')
    assert model.classes_.tolist()==[0,1], 'Score column must be the class-1 probability'
    test=pd.read_json(ROOT/'test_texts.jsonl',lines=True).set_index('row_id')
    english=pd.read_csv(ROOT/'english_predictions.csv')
    frames=[];timings={}
    for value,lang in [('English',None),('Spanish','es'),('French','fr')]:
        frame=english.copy()
        if lang:
            translations=pd.read_json(ROOT/f'translations_{lang}.jsonl',lines=True).set_index('row_id')
            assert translations.index.is_unique and set(translations.index)==set(frame.row_id)
            assert np.array_equal(translations.loc[frame.row_id,'label'],frame.label)
            expected=test.loc[frame.row_id,'text'].map(lambda t:hashlib.sha256(t.encode()).hexdigest())
            assert np.array_equal(expected,translations.loc[frame.row_id,'source_sha256'])
            assert (translations.max_length_chunks==0).all(), 'Inspect output length caps before scoring'
            started=time.perf_counter()
            frame['score']=model.predict_proba(translations.loc[frame.row_id,'translation'])[:,1]
            timings[value]=time.perf_counter()-started
        frame['value']=value
        frame['threshold']=info['threshold']
        frame['column']='LR / '+frame.domain
        pooled=frame.copy();pooled['column']='LR / pooled'
        frames.extend([frame,pooled])
    scores=pd.concat(frames,ignore_index=True)[['row_id','value','column','label','score','threshold']]
    scores.to_csv(ROOT/'scores.csv',index=False)
    (ROOT/'scoring_times.json').write_text(json.dumps(timings,indent=2)+'\n')
    points=[]
    for (value,column),g in scores.groupby(['value','column'],sort=False):
        pred=g.score>=g.threshold
        points.append({'value':value,'column':column,'n':len(g),'positive':int(g.label.sum()),'auroc':roc_auc_score(g.label,g.score),'ap':average_precision_score(g.label,g.score),'recall':float(pred[g.label==1].mean()),'fpr':float(pred[g.label==0].mean()),'precision':float(g.label[pred].mean()) if pred.any() else None,'fp':int(((g.label==0)&pred).sum()),'fn':int(((g.label==1)&~pred).sum())})
    pd.DataFrame(points).to_csv(ROOT/'point_metrics.csv',index=False)
    print(pd.DataFrame(points).to_string(index=False),flush=True)
    # Select the cell a priori, then sample each error type without replacement.
    cell=scores[(scores.value=='Spanish')&(scores.column=='LR / pooled')].sort_values('row_id')
    sample=[]
    for label in [0,1]:
        part=cell[(cell.label==label)&((cell.score>=cell.threshold)!=cell.label)]
        selected=part.sample(n=min(10,len(part)),random_state=SEED)
        sample.append(selected)
    sampled=pd.concat(sample)
    if not (ROOT/'error_categories.csv').exists():
        sampled[['row_id','value','column']].assign(category='').to_csv(ROOT/'error_categories_template.csv',index=False)
    translated=pd.read_json(ROOT/'translations_es.jsonl',lines=True).set_index('row_id')
    examples=[]
    for row in sampled.to_dict('records'):
        row.update(domain=test.loc[row['row_id'],'domain'],english=test.loc[row['row_id'],'text'],spanish=translated.loc[row['row_id'],'translation'],error_type='FP' if row['label']==0 else 'FN')
        examples.append(row)
    (ROOT/'error_sample_texts.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    if (ROOT/'error_categories.csv').exists():
        categories=pd.read_csv(ROOT/'error_categories.csv')
        assert set(categories.row_id)==set(sampled.row_id)
        assert categories.row_id.is_unique and categories.category.notna().all()
        errors=sampled[['row_id','value','column']].merge(categories[['row_id','category']],on='row_id',validate='one_to_one')
        errors.to_csv(ROOT/'errors.csv',index=False)
    review=[]
    for lang in ['es','fr']:
        translated=pd.read_json(ROOT/f'translations_{lang}.jsonl',lines=True).set_index('row_id')
        for row_id in test.sort_index().sample(n=20,random_state=42).index:
            review.append({'row_id':row_id,'language':lang,'english':test.loc[row_id,'text'],'translation':translated.loc[row_id,'translation']})
    (ROOT/'translation_review_texts.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
