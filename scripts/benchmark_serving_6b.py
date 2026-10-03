"""Frozen generator-only and end-to-end benchmarks for Milestone 6B."""
from __future__ import annotations
import argparse,asyncio,json,math,os,subprocess,time
from pathlib import Path
import httpx,numpy as np,psutil
from transformers import AutoTokenizer
from retrieval.verification import build_messages,format_evidence

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/serving-6b-v1.json').read_text())
ART=Path(os.environ.get('RAG_ROOT', Path.home()/'rag')); RUN=ART/'runs/milestone6b-serving-v1'
model_path=Path(CFG['generator']['local_path']).expanduser()
if not model_path.is_absolute(): model_path=ART/model_path
TOKENIZER=AutoTokenizer.from_pretrained(model_path,local_files_only=True)


def dev_inputs():
    mapping=json.loads((ART/'runs/milestone5a-verification-v1/verification-mapping.json').read_text())['records']
    ranks=json.loads((ART/'runs/milestone5a-verification-v1/evidence-condition-rankings.json').read_text())['D1']
    corpus={str(x['doc_id']):x for x in map(json.loads,(ART/'data/scifact/original/corpus.jsonl').read_text().splitlines())}
    rows=[]
    for item in mapping:
        doc=corpus[str(ranks[item['claim_id']][0])]
        evidence,_=format_evidence([doc],TOKENIZER,CFG['generator']['max_evidence_tokens'])
        prompt=TOKENIZER.apply_chat_template(build_messages(item['claim'],evidence,'grounded-v2'),tokenize=False,add_generation_prompt=True)
        rows.append({'query_id':item['claim_id'],'claim':item['claim'],'prompt':prompt,'tokens':len(TOKENIZER.encode(prompt,add_special_tokens=False))})
    return sorted(rows,key=lambda x:int(x['query_id']))


async def gpu_sampler(stop,records):
    while not stop.is_set():
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits','--id=0'],text=True).strip()
            mem,util=map(float,raw.split(','))
            records.append({'t':time.time(),'memory_mib':mem,'utilization_percent':util})
        except Exception as e: records.append({'t':time.time(),'error':type(e).__name__})
        try: await asyncio.wait_for(stop.wait(),timeout=.2)
        except asyncio.TimeoutError: pass


async def streamed_completion(client,prompt):
    start=time.perf_counter(); first=None; text=''; usage={}
    payload={'model':CFG['generator']['served_model_name'],'prompt':prompt,'temperature':CFG['generator']['temperature'],
             'repetition_penalty':CFG['generator']['repetition_penalty'],'max_tokens':CFG['benchmark']['generator_max_tokens'],
             'seed':CFG['generator']['seed'],'stream':True,'stream_options':{'include_usage':True}}
    try:
        async with client.stream('POST','/v1/completions',json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data: '): continue
                data=line[6:]
                if data=='[DONE]': continue
                chunk=json.loads(data)
                usage=chunk.get('usage') or usage
                choices=chunk.get('choices') or []
                if choices:
                    piece=choices[0].get('text','')
                    if piece and first is None: first=time.perf_counter()
                    text+=piece
        end=time.perf_counter(); out=int(usage.get('completion_tokens') or len(TOKENIZER.encode(text,add_special_tokens=False)))
        ttft=(first-start if first else end-start)*1000
        tpot=((end-first)*1000/max(out-1,1)) if first else None
        return {'ok':True,'latency_ms':(end-start)*1000,'ttft_ms':ttft,'tpot_ms':tpot,'output_tokens':out,'input_tokens':int(usage.get('prompt_tokens') or len(TOKENIZER.encode(prompt,add_special_tokens=False)))}
    except Exception as e:
        return {'ok':False,'latency_ms':(time.perf_counter()-start)*1000,'error':f'{type(e).__name__}:{e}'}


async def verify_request(client,claim):
    start=time.perf_counter()
    try:
        r=await client.post('/verify',json={'claim':claim}); r.raise_for_status(); x=r.json()
        return {'ok':True,'latency_ms':(time.perf_counter()-start)*1000,'valid':x['valid'],'timing_ms':x['timing_ms']}
    except Exception as e: return {'ok':False,'latency_ms':(time.perf_counter()-start)*1000,'error':f'{type(e).__name__}:{e}'}


def summarize(records,wall_s,kind,meta,telemetry):
    good=[x for x in records if x['ok']]; lat=np.array([x['latency_ms'] for x in good])
    result={**meta,'kind':kind,'requests':len(records),'successful':len(good),'failure_rate':1-len(good)/len(records),
            'wall_seconds':wall_s,'request_throughput_per_s':len(good)/wall_s if wall_s else None,
            'latency_ms':{'mean':float(lat.mean()),'p50':float(np.percentile(lat,50)),'p95':float(np.percentile(lat,95))} if len(lat) else None,
            'gpu':{'peak_memory_mib':max((x['memory_mib'] for x in telemetry if 'memory_mib'in x),default=None),
                   'mean_utilization_percent':float(np.mean([x['utilization_percent'] for x in telemetry if 'utilization_percent'in x])) if telemetry else None},
            'cpu_system_memory_peak_percent':max((x.get('cpu_memory_percent',0) for x in records),default=0)}
    if kind=='generator':
        out=sum(x['output_tokens'] for x in good); inp=sum(x['input_tokens'] for x in good)
        result.update({'output_tokens':out,'input_tokens':inp,'generated_tokens_per_s':out/wall_s,
                       'total_tokens_per_s':(out+inp)/wall_s,
                       'ttft_ms':{'mean':float(np.mean([x['ttft_ms'] for x in good])),'p50':float(np.percentile([x['ttft_ms'] for x in good],50)),'p95':float(np.percentile([x['ttft_ms'] for x in good],95))},
                       'tpot_ms_mean':float(np.mean([x['tpot_ms'] for x in good if x['tpot_ms'] is not None]))})
    else:
        result['invalid_output_rate']=sum(not x.get('valid',False) for x in good)/len(good) if good else None
        for key in ['embedding','faiss','retrieval','prompt','generation','total']:
            vals=[x['timing_ms'][key] for x in good]
            result.setdefault('component_ms',{})[key]={'mean':float(np.mean(vals)),'p50':float(np.percentile(vals,50)),'p95':float(np.percentile(vals,95))}
    return result


async def measured(kind,items,concurrency,count,meta):
    telemetry=[]; stop=asyncio.Event(); sampler=asyncio.create_task(gpu_sampler(stop,telemetry))
    sem=asyncio.Semaphore(concurrency)
    base='http://127.0.0.1:8001' if kind=='generator' else 'http://127.0.0.1:8000'
    async with httpx.AsyncClient(base_url=base,timeout=240,limits=httpx.Limits(max_connections=concurrency)) as client:
        async def one(i):
            async with sem:
                result=await (streamed_completion(client,items[i%len(items)]['prompt']) if kind=='generator' else verify_request(client,items[i%len(items)]['claim']))
                result['cpu_memory_percent']=psutil.virtual_memory().percent
                return result
        started=time.perf_counter(); records=await asyncio.gather(*(one(i) for i in range(count))); wall=time.perf_counter()-started
    stop.set(); await sampler
    return summarize(records,wall,kind,meta,telemetry),records,telemetry


async def warmup(kind,items,count=20):
    summary,_,_=await measured(kind,items,min(4,count),count,{'warmup':True})
    if summary['failure_rate']>0: raise RuntimeError(f'{kind} warmup failures: {summary}')


async def main_async(phase):
    rows=dev_inputs(); output=RUN/f'{phase}-benchmark.json'
    if output.exists(): raise FileExistsError(output)
    all_results=[]; raw={}; telemetry={}
    if phase=='generator':
        chosen=[]
        for target in CFG['benchmark']['prompt_token_targets']:
            row=min(rows,key=lambda x:(abs(x['tokens']-target),int(x['query_id'])))
            chosen.append({**row,'target_tokens':target})
        await warmup('generator',chosen,CFG['benchmark']['warmup_requests'])
        for item in chosen:
            for c in CFG['benchmark']['concurrency']:
                key=f"tokens-{item['target_tokens']}-c{c}"
                summary,recs,tel=await measured('generator',[item],c,CFG['benchmark']['measured_requests'],{'target_input_tokens':item['target_tokens'],'actual_input_tokens':item['tokens'],'query_id':item['query_id'],'concurrency':c})
                all_results.append(summary); raw[key]=recs; telemetry[key]=tel; print(key,summary['request_throughput_per_s'],flush=True)
        payload={'representative_inputs':[{k:v for k,v in x.items() if k!='prompt' and k!='claim'} for x in chosen],'warmup_requests':20,'results':all_results,'raw':raw,'telemetry':telemetry}
    else:
        await warmup('end_to_end',rows,CFG['benchmark']['warmup_requests'])
        for c in CFG['benchmark']['concurrency']:
            key=f'c{c}'; summary,recs,tel=await measured('end_to_end',rows,c,CFG['benchmark']['measured_requests'],{'concurrency':c,'claim_count':len(rows)})
            all_results.append(summary); raw[key]=recs; telemetry[key]=tel; print(key,summary['request_throughput_per_s'],flush=True)
        payload={'representative_query_ids':[x['query_id'] for x in rows],'warmup_requests':20,'results':all_results,'raw':raw,'telemetry':telemetry}
    output.write_text(json.dumps(payload,indent=2)+'\n'); print(output)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['generator','end-to-end'],required=True); a=p.parse_args()
    asyncio.run(main_async(a.phase.replace('-','_')))
if __name__=='__main__': main()
