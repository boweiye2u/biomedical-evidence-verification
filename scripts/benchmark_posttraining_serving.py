from __future__ import annotations
import argparse,asyncio,json,subprocess,time
from pathlib import Path
import httpx,numpy as np,psutil
ROOT=Path(__file__).resolve().parents[1];CFG=json.loads((ROOT/'configs/posttraining/serving-tuned-3b-v1.json').read_text());RUN=Path(CFG['run_root'])
def inputs():
 mapping=json.loads(Path('/home/boweiye2/rag/runs/milestone5a-verification-v1/verification-mapping.json').read_text())['records'];return [{'claim_id':x['claim_id'],'claim':x['claim']} for x in sorted(mapping,key=lambda x:int(x['claim_id']))]
async def gpu_sampler(stop,values):
 while not stop.is_set():
  try:
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits','--id=0'],text=True).strip();mem,util=map(float,raw.split(','));values.append({'time':time.time(),'memory_mib':mem,'utilization_percent':util})
  except Exception as e:values.append({'time':time.time(),'error':type(e).__name__})
  try:await asyncio.wait_for(stop.wait(),.2)
  except asyncio.TimeoutError:pass
async def run_one(client,sem,item):
 async with sem:
  started=time.perf_counter()
  try:
   response=await client.post('/verify',json={'claim':item['claim']});response.raise_for_status();x=response.json();elapsed=(time.perf_counter()-started)*1000
   return {'ok':True,'claim_id':item['claim_id'],'latency_ms':elapsed,'valid':x['valid'],'decision':x['decision'],'rationale_sentences':x['rationale_sentences'],'retrieval_document_id':x['retrieval']['document_id'],'timing_ms':x['timing_ms'],'input_tokens':x['token_usage']['input_tokens'],'output_tokens':x['token_usage']['output_tokens'],'request_id':x['request_id'],'cpu_memory_percent':psutil.virtual_memory().percent}
  except Exception as e:return {'ok':False,'claim_id':item['claim_id'],'latency_ms':(time.perf_counter()-started)*1000,'error':f'{type(e).__name__}:{e}','cpu_memory_percent':psutil.virtual_memory().percent}
def pct(xs,p):return float(np.percentile(xs,p))
def summary(records,wall,concurrency,telemetry):
 good=[x for x in records if x['ok']];lat=[x['latency_ms'] for x in good];components={}
 for k in ('embedding','faiss','retrieval','prompt','generation','total'):
  vals=[x['timing_ms'][k] for x in good];components[k]={'mean':float(np.mean(vals)),'p50':pct(vals,50),'p95':pct(vals,95)}
 out=sum((x['output_tokens'] or 0) for x in good);inp=sum((x['input_tokens'] or 0) for x in good)
 return {'concurrency':concurrency,'requests':len(records),'successful':len(good),'failure_rate':1-len(good)/len(records),'invalid_output_rate':sum(not x['valid'] for x in good)/len(good) if good else None,'wall_seconds':wall,'requests_per_second':len(good)/wall,'latency_ms':{'mean':float(np.mean(lat)),'p50':pct(lat,50),'p95':pct(lat,95)},'component_ms':components,'tokens':{'input_total':inp,'output_total':out,'input_mean':inp/len(good),'output_mean':out/len(good),'output_tokens_per_second':out/wall,'total_tokens_per_second':(inp+out)/wall},'gpu':{'peak_memory_mib':max(x['memory_mib'] for x in telemetry if 'memory_mib'in x),'mean_utilization_percent':float(np.mean([x['utilization_percent'] for x in telemetry if 'utilization_percent'in x]))},'system_memory_peak_percent':max(x['cpu_memory_percent'] for x in records)}
async def measured(items,concurrency,count):
 sem=asyncio.Semaphore(concurrency);telemetry=[];stop=asyncio.Event();sampler=asyncio.create_task(gpu_sampler(stop,telemetry))
 async with httpx.AsyncClient(base_url='http://127.0.0.1:8000',timeout=240,limits=httpx.Limits(max_connections=concurrency)) as client:
  started=time.perf_counter();records=await asyncio.gather(*(run_one(client,sem,items[i%len(items)]) for i in range(count)));wall=time.perf_counter()-started
 stop.set();await sampler;return summary(records,wall,concurrency,telemetry),records,telemetry
async def main_async():
 out=RUN/'benchmark.json'
 if out.exists():raise FileExistsError(out)
 items=inputs();warm,_,_=await measured(items,4,CFG['benchmark']['warmup_requests'])
 if warm['failure_rate']:raise RuntimeError('warmup failed')
 results=[];raw={};tele={}
 for c in CFG['benchmark']['concurrency']:
  s,r,t=await measured(items,c,CFG['benchmark']['measured_requests']);results.append(s);raw[f'c{c}']=r;tele[f'c{c}']=t;print(json.dumps({'concurrency':c,'p50':s['latency_ms']['p50'],'p95':s['latency_ms']['p95'],'rps':s['requests_per_second']}),flush=True)
 payload={'status':'complete','warmup':warm,'workload_claim_ids':[x['claim_id'] for x in items],'quality_evaluation':False,'results':results,'raw':raw,'telemetry':tele};out.write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':asyncio.run(main_async())
