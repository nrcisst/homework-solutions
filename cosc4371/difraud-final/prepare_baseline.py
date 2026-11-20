"""Reuse the committed midsemester IDs; refit and verify the unchanged LR baseline."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time, json, hashlib, re, shutil, platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'difraud-checkpoint'
sys.path.insert(0, str(BASE))
from download_data import download, DOMAINS, ROLES

def load_data():
    data = download()
    parts = []
    for domain in DOMAINS:
        for role in ROLES:
            frame = pd.read_json(data/domain/f'{role}.jsonl', lines=True)
            frame['row_id'] = [f'difraud:{domain}:{role}:{i}' for i in range(len(frame))]
            frame['domain'] = domain
            parts.append(frame)
    return pd.concat(parts, ignore_index=True)

def main():
    start = time.perf_counter()
    raw = load_data().set_index('row_id')
    split = pd.read_csv(BASE/'split.csv')
    frame = split.join(raw[['text','domain']], on='row_id', validate='one_to_one')
    assert frame.text.notna().all()
    assert np.array_equal(raw.loc[frame.row_id, 'label'], frame.label)
    train = frame[frame.split == 'train']
    val = frame[frame.split == 'validation']
    test = frame[frame.split == 'test'].copy()
    assert test.row_id.tolist() == pd.read_csv(BASE/'paired_test_ids.csv').row_id.tolist()
    shutil.copyfile(BASE/'split.csv', ROOT/'split.csv')
    test.to_json(ROOT/'test_texts.jsonl', orient='records', lines=True, force_ascii=False)
    frame.to_pickle(ROOT/'retained_rows.pkl')
    norm = lambda t: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', t.lower())).strip()
    leakage = {}
    for kind, transform in [('exact', lambda x:x), ('normalized',norm)]:
        a,b,c = [part.text.map(transform) for part in [train,val,test]]
        leakage[f'train_test_{kind}'] = int(a.isin(set(c)).sum())
        leakage[f'train_validation_{kind}'] = int(a.isin(set(b)).sum())
        leakage[f'validation_test_{kind}'] = int(b.isin(set(c)).sum())
    assert not any(leakage.values())
    model = Pipeline([
        ('features', FeatureUnion([
            ('word', TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2), min_df=2, max_features=50000, sublinear_tf=True, dtype=np.float32)),
            ('char', TfidfVectorizer(lowercase=True, strip_accents='unicode', analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=30000, sublinear_tf=True, dtype=np.float32))])),
        ('classifier', LogisticRegression(C=1, solver='liblinear', max_iter=2000, random_state=42, class_weight=None))])
    print('Fitting unchanged LR on frozen IDs', flush=True)
    fit_start = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(train.text,train.label)
    fit_seconds = time.perf_counter()-fit_start
    vp = model.predict_proba(val.text)[:,1]
    fpr,tpr,thresholds = roc_curve(val.label,vp,drop_intermediate=False)
    threshold = float(thresholds[np.isfinite(thresholds)&(fpr<=.05)].min())
    old_threshold = float((BASE/'threshold.txt').read_text())
    assert abs(threshold-old_threshold)<1e-10, (threshold,old_threshold)
    score_start=time.perf_counter()
    test['score']=model.predict_proba(test.text)[:,1]
    score_seconds=time.perf_counter()-score_start
    old=pd.read_csv(BASE/'scores.csv').query("test_set == 'English_pooled'").set_index('row_id')
    difference=float(np.max(np.abs(test.score.to_numpy()-old.loc[test.row_id,'score'].to_numpy())))
    assert difference < 1e-10, difference
    test[['row_id','domain','label','score']].to_csv(ROOT/'english_predictions.csv', index=False)
    joblib.dump(model,ROOT/'lr_model.joblib')
    info={'seed':42,'threshold':threshold,'fit_seconds':fit_seconds,'score_seconds':score_seconds,'total_seconds':time.perf_counter()-start,'baseline_max_score_difference':difference,'leakage':leakage,'training_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'python':platform.python_version(),'visible_cpus':os.cpu_count(),'validation_fpr':float((vp[val.label==0]>=threshold).mean()),'validation_recall':float((vp[val.label==1]>=threshold).mean())}
    info['cpu']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown') if Path('/proc/cpuinfo').exists() else platform.processor()
    (ROOT/'run_info.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__': main()
