"""Time detector calls for the remaining disagreement audit; do not claim the audit is done."""
import json,time
from pathlib import Path
import pandas as pd
from lingua import LanguageDetectorBuilder
from langdetect import DetectorFactory,detect,LangDetectException
ROOT=Path(__file__).resolve().parent

def main():
    sample=pd.read_pickle(ROOT/'retained_rows.pkl').sample(n=200,random_state=42)
    start=time.perf_counter()
    detector=LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    detector.detect_language_of(sample.text.iloc[0])
    warmup=time.perf_counter()-start
    start=time.perf_counter()
    for text in sample.text:detector.detect_language_of(text)
    lingua_seconds=time.perf_counter()-start
    start=time.perf_counter()
    for text in sample.text:
        DetectorFactory.seed=42
        try:detect(text)
        except LangDetectException:pass
    langdetect_seconds=time.perf_counter()-start
    result={'sample_rows':200,'seed':42,'sample_source':'random retained rows; timing sample only','lingua_mode':'all languages, low accuracy','lingua_warmup_seconds':warmup,'lingua_seconds':lingua_seconds,'langdetect_seconds':langdetect_seconds,'raw_rows':95854,'estimated_minutes':(warmup+(lingua_seconds+langdetect_seconds)*95854/200)/60,'note':'Extrapolates single-process calls to all 95,854 raw rows. The disagreement review itself remains unfinished. Manual review is excluded.'}
    (ROOT/'language_audit_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
