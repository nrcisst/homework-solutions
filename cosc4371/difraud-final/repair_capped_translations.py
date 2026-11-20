"""Retry output-capped segments with smaller source chunks, before any scoring."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
import json,time,re
from pathlib import Path
import pandas as pd
import ctranslate2
from transformers import MarianTokenizer
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent

def run(lang):
    started=time.perf_counter()
    config=json.loads((ROOT/f'translation_config_{lang}.json').read_text())
    source=snapshot_download(config['model'],revision=config['revision'],local_files_only=True)
    tok=MarianTokenizer.from_pretrained(source)
    model=ctranslate2.Translator(str(ROOT/'models'/f'opus-en-{lang}-int8'),device='cpu',compute_type='int8',intra_threads=2)
    frame=pd.read_json(ROOT/'test_texts.jsonl',lines=True).set_index('row_id')
    path=ROOT/f'translations_{lang}.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if not any(r['max_length_chunks'] for r in rows) and (ROOT/f'translation_repairs_{lang}.json').exists():
        print(lang,'no unresolved caps; preserving repair audit',flush=True)
        return
    records=[]
    def translate(tokens):
        inp=tok.convert_ids_to_tokens(tokens+[tok.eos_token_id])
        hyp=model.translate_batch([inp],beam_size=4,no_repeat_ngram_size=3,max_input_length=0,max_decoding_length=512)[0].hypotheses[0]
        if len(hyp)>=512:
            if len(tokens)>32:
                middle=len(tokens)//2
                a,n1=translate(tokens[:middle]);b,n2=translate(tokens[middle:])
                return a+' '+b,n1+n2
        assert len(hyp)<512,'Unresolved output cap'
        return tok.decode(tok.convert_tokens_to_ids(hyp),skip_special_tokens=True),len(hyp)
    for row in rows:
        if not row['max_length_chunks']:continue
        original_caps=row['max_length_chunks']
        text=frame.loc[row['row_id'],'text']
        chunks=[]
        for sentence in re.split(r'(?<=[.!?])\s+|\n+',text):
            if sentence.strip():
                ids=tok.encode(sentence,add_special_tokens=False,truncation=False)
                chunks.extend(ids[i:i+256] for i in range(0,len(ids),256))
        inputs=[tok.convert_ids_to_tokens(chunk+[tok.eos_token_id]) for chunk in chunks]
        batch=model.translate_batch(inputs,max_batch_size=2048,batch_type='tokens',beam_size=1,max_input_length=0,max_decoding_length=512)
        outputs=[]
        for chunk,result in zip(chunks,batch):
            hypothesis=result.hypotheses[0]
            if len(hypothesis)>=512:
                outputs.append(translate(chunk))
            else:outputs.append((tok.decode(tok.convert_tokens_to_ids(hypothesis),skip_special_tokens=True),len(hypothesis)))
        row['translation']=' '.join(t for t,n in outputs)
        row['output_tokens']=sum(n for t,n in outputs)
        row['max_length_chunks']=0
        row['initial_cap_chunks']=original_caps
        row['cap_repair']='beam4/no-repeat-3 with full segment context; recursive bisection only if still capped'
        records.append({'row_id':row['row_id'],'initial_cap_chunks':original_caps,'remaining_cap_chunks':0})
        print(lang,row['row_id'],'repaired',flush=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    config['output_cap_recovery']='On capped messages, retranslate all segments; retry capped segments with beam4/no-repeat-3, retaining their context. Recursively bisect only if still capped. No unresolved caps permitted.'
    (ROOT/f'translation_config_{lang}.json').write_text(json.dumps(config,indent=2)+'\n')
    duration=time.perf_counter()-started
    (ROOT/f'translation_repairs_{lang}.json').write_text(json.dumps({'repaired_rows':records,'seconds':duration},indent=2)+'\n')
    stats=json.loads((ROOT/f'translation_stats_{lang}.json').read_text())
    stats['initial_cap_chunks']=sum(r.get('initial_cap_chunks',0) for r in rows)
    stats['repaired_rows']=sum(bool(r.get('initial_cap_chunks',0)) for r in rows)
    stats['max_length_chunks']=0
    stats['output_tokens']=sum(r['output_tokens'] for r in rows)
    stats['repair_seconds']=duration
    stats['translation_seconds']+=duration
    (ROOT/f'translation_stats_{lang}.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(lang,'repairs',len(records),'seconds',duration,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--lang',choices=['es','fr'],required=True)
    run(p.parse_args().lang)
