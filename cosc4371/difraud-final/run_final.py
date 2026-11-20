"""Reproduce the completed LR comparisons and final-checkpoint report."""
import subprocess,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
if __name__=='__main__':
    run('prepare_baseline.py')
    for lang in ['es','fr']:run('translate_test.py','--lang',lang)
    for lang in ['es','fr']:run('repair_capped_translations.py','--lang',lang)
    run('apply_translation_corrections.py')
    run('score_comparisons.py')
    run('finalize_reviews.py')
    command=[sys.executable,'check_final_checkpoint.py','--split','split.csv','--scores','scores.csv','--errors','errors.csv','--metric','auroc','--reference','English','--direction','either']
    result=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True)
    (ROOT/'checker_output.txt').write_text(result.stdout)
    if result.stderr:print(result.stderr,file=sys.stderr)
    run('summarize_results.py')
    run('validate_artifacts.py')
    run('make_report.py')
    names=['split.csv','scores.csv','errors.csv','checker_output.txt','checker_auroc.csv','verdicts.csv','aggregate_metrics.csv']
    (ROOT/'output_hashes.json').write_text(json.dumps({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},indent=2)+'\n')
