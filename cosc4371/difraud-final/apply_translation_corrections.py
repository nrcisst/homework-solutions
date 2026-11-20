"""Apply the disclosed, review-based meaning corrections before scoring."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    corrections=json.loads((ROOT/'translation_corrections.json').read_text())
    audit=[]
    for lang in ['es','fr']:
        path=ROOT/f'translations_{lang}.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        lookup={r['row_id']:r for r in rows}
        for correction in [c for c in corrections if c['language']==lang]:
            row=lookup[correction['row_id']]
            before=row['translation']
            if correction['old'] in before:
                assert before.count(correction['old'])==1
                row['translation']=before.replace(correction['old'],correction['new'])
            else:
                assert correction['new'] in before, 'Neither original nor corrected phrase found'
            row['review_postedited']=True
            audit.append({'row_id':row['row_id'],'language':lang,'old':correction['old'],'new':correction['new'],'reason':correction['reason']})
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (ROOT/'translation_correction_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(f'Applied/verified {len(audit)} phrase corrections in {len(set((r["row_id"],r["language"]) for r in audit))} translated messages.')

if __name__=='__main__':main()
