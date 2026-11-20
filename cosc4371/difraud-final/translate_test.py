"""Translate every frozen test item using pinned OPUS-MT weights, without truncation."""
import os
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse, json, time, hashlib, re
from pathlib import Path
import pandas as pd
import ctranslate2
from transformers import MarianTokenizer
from huggingface_hub import HfApi, snapshot_download

ROOT=Path(__file__).resolve().parent

def run(lang):
    started=time.perf_counter()
    model_name=f'Helsinki-NLP/opus-mt-en-{lang}'
    config_path=ROOT/f'translation_config_{lang}.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    revision=config.get('revision') or HfApi().model_info(model_name).sha
    config.update(model=model_name,revision=revision,engine='CTranslate2',compute_type='int8',beam_size=1,source_chunk_tokens=256,max_decoding_length=512,inter_threads=1,intra_threads=2,segmentation='sentence punctuation plus whitespace, or newline; then chunks of <=256 source tokens')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    print('Downloading',model_name,revision,flush=True)
    source=snapshot_download(model_name,revision=revision,allow_patterns=['config.json','generation_config.json','pytorch_model.bin','model.safetensors','source.spm','target.spm','vocab.json','tokenizer_config.json'],max_workers=4)
    tokenizer=MarianTokenizer.from_pretrained(source)
    converted=ROOT/'models'/f'opus-en-{lang}-int8'
    if not (converted/'model.bin').exists():
        ctranslate2.converters.TransformersConverter(source).convert(str(converted),quantization='int8',force=True)
    translator=ctranslate2.Translator(str(converted),device='cpu',compute_type='int8',inter_threads=1,intra_threads=2)
    frame=pd.read_json(ROOT/'test_texts.jsonl',lines=True)
    out=ROOT/f'translations_{lang}.jsonl'
    cached={}
    if out.exists():
        for line in out.read_text().splitlines():
            row=json.loads(line); cached[row['row_id']]=row
    if set(cached)==set(frame.row_id) and (ROOT/f'translation_stats_{lang}.json').exists():
        for row in frame.to_dict('records'):
            assert cached[row['row_id']]['source_sha256']==hashlib.sha256(row['text'].encode()).hexdigest()
        print('Reusing complete translation cache; preserving measured run statistics.',flush=True)
        return
    translation_start=time.perf_counter()
    progress_path=ROOT/f'translation_progress_{lang}.json'
    prior_seconds=json.loads(progress_path.read_text())['seconds'] if progress_path.exists() else 0.0
    completed=0
    with out.open('a') as f:
        # Work in 32-document batches, retaining every source token in ordered chunks.
        for offset in range(0,len(frame),32):
            batch=frame.iloc[offset:offset+32]
            docs=[]; sources=[]
            for row in batch.to_dict('records'):
                if row['row_id'] in cached: continue
                sentences=[s for s in re.split(r'(?<=[.!?])\s+|\n+',row['text']) if s.strip()]
                chunks=[];n_tokens=0
                for sentence in sentences:
                    ids=tokenizer.encode(sentence,add_special_tokens=False,truncation=False)
                    n_tokens+=len(ids)
                    chunks.extend(ids[j:j+256] for j in range(0,len(ids),256))
                chunks=chunks or [[]]
                begin=len(sources)
                for chunk in chunks:
                    sources.append(tokenizer.convert_ids_to_tokens(chunk+[tokenizer.eos_token_id]))
                docs.append((row,begin,len(sources),n_tokens))
            if not sources: continue
            results=translator.translate_batch(sources,max_batch_size=2048,batch_type='tokens',beam_size=1,max_input_length=0,max_decoding_length=512)
            for row,a,b,n_tokens in docs:
                hypotheses=[results[j].hypotheses[0] for j in range(a,b)]
                texts=[tokenizer.decode(tokenizer.convert_tokens_to_ids(h),skip_special_tokens=True) for h in hypotheses]
                result={'row_id':row['row_id'],'domain':row['domain'],'label':row['label'],'language':lang,'translation':' '.join(texts),'source_tokens':n_tokens,'chunks':b-a,'output_tokens':sum(map(len,hypotheses)),'max_length_chunks':sum(len(h)>=512 for h in hypotheses),'source_sha256':hashlib.sha256(row['text'].encode()).hexdigest()}
                assert result['translation'].strip()
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
                cached[row['row_id']]=result; completed+=1
            f.flush()
            elapsed=time.perf_counter()-translation_start
            progress_path.write_text(json.dumps({'rows':len(cached),'seconds':prior_seconds+elapsed})+'\n')
            print(f'{lang}: {len(cached)}/{len(frame)}; new={completed}; translate_seconds={elapsed:.1f}',flush=True)
    elapsed=time.perf_counter()-translation_start
    assert set(cached)==set(frame.row_id)
    stats={'rows':len(cached),'new_rows_this_run':completed,'translation_seconds_this_run':elapsed,'translation_seconds':prior_seconds+elapsed,'total_seconds_this_run':time.perf_counter()-started,'source_tokens':sum(r['source_tokens'] for r in cached.values()),'chunks':sum(r['chunks'] for r in cached.values()),'output_tokens':sum(r['output_tokens'] for r in cached.values()),'max_length_chunks':sum(r['max_length_chunks'] for r in cached.values())}
    (ROOT/f'translation_stats_{lang}.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(stats,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lang',choices=['es','fr'],required=True)
    run(p.parse_args().lang)
