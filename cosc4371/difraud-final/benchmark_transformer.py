"""Time real XLM-R training steps; these benchmark weights are NOT evaluation results."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
import time,json,math
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parent
REVISION='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    model_name='FacebookAI/xlm-roberta-base'
    tok=AutoTokenizer.from_pretrained(model_name,revision=REVISION,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_name,revision=REVISION,num_labels=2,local_files_only=True)
    model.train()
    frame=pd.read_pickle(ROOT/'retained_rows.pkl')
    train=frame[frame.split=='train'].sample(n=8,random_state=42)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,foreach=False)
    timings=[]
    for i in range(3):
        part=train.iloc[2*i:2*i+2]
        batch=tok(part.text.tolist(),padding='max_length',truncation=True,max_length=256,return_tensors='pt')
        batch['labels']=torch.tensor(part.label.tolist())
        start=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss=model(**batch).loss;loss.backward();opt.step()
        elapsed=time.perf_counter()-start;timings.append(elapsed)
        print('step',i,'seconds',elapsed,'loss',loss.item(),flush=True)
    model.eval()
    start=time.perf_counter()
    with torch.no_grad():model(**batch)
    inference_seconds=time.perf_counter()-start
    seconds_per_step=sum(timings[1:])/2
    training_steps=2*math.ceil(76040/2)
    result={'model':model_name,'revision':REVISION,'device':'cpu','threads':4,'batch_size':2,'max_length':256,'optimizer':'AdamW','learning_rate':2e-5,'planned_epochs':2,'training_steps_seconds':timings,'steady_seconds_per_step':seconds_per_step,'training_minutes_estimate':training_steps*seconds_per_step/60,'validation_and_test_minutes_estimate':math.ceil((9505*2+3500*3)/2)*inference_seconds/60,'inference_seconds_per_batch':inference_seconds,'training_rows':76040,'note':'Short benchmark only; no trained transformer results. Extrapolation excludes downloads/tokenization and assumes the measured CPU throughput. Validation threshold will use the midsemester FPR<=5% rule. Test results will not select a checkpoint.'}
    (ROOT/'transformer_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
