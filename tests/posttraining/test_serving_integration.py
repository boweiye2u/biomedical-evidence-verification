import json
from pathlib import Path
from fastapi.testclient import TestClient
from serving.posttraining_app import create_app,load_config
ROOT=Path(__file__).resolve().parents[2];CFG=json.loads((ROOT/'configs/posttraining/serving-tuned-3b-v1.json').read_text())
class Index:ntotal=5183
class Retriever:
 index=Index()
 def __init__(self):self.calls=[]
 def retrieve(self,claim,k=1):
  self.calls.append((claim,k));return [{'document_id':'42','score':.9,'title':'T','abstract':['first','second']}],{'embedding':1.,'faiss':.1,'retrieval':1.1}
class Verifier:
 def __init__(self,valid=True):self.valid=valid
 async def reachable(self):return True
 async def close(self):pass
 def build_prompt(self,claim,doc):return 'prompt',{'prompt_tokens':20,'sentence_count':2}
 async def generate(self,prompt,count):
  if self.valid:return {'raw_output':'{"decision":"SUPPORT","rationale_sentences":[1]}','parsed':{'decision':'SUPPORT','rationale_sentences':[1]},'decision':'SUPPORT','errors':[],'json_valid':True,'schema_valid':True,'decision_valid':True,'rationale_index_valid':True,'input_tokens':20,'output_tokens':14,'generation_ms':2.}
  return {'raw_output':'bad','parsed':None,'decision':None,'errors':['invalid_json'],'json_valid':False,'schema_valid':False,'decision_valid':False,'rationale_index_valid':False,'input_tokens':20,'output_tokens':1,'generation_ms':2.}
class Logger:
 def __init__(self):self.records=[]
 def write(self,x):self.records.append(x)
def client(valid=True):
 r,l=Retriever(),Logger();return TestClient(create_app(r,Verifier(valid),l)),r,l
def test_config_is_exact_m2_not_base_or_m3():
 c=load_config();assert c['deployment_candidate']=='M2';assert '/m2/' in c['generator']['local_path'];assert '/m3/' not in c['generator']['local_path'];assert c['generator']['weight_sha256']=={'model-00001-of-00002.safetensors':'c76fc29521b759e0b5f302116ec1c0a3d55d8b694846ee3b6e6dd0967a3cd2d2','model-00002-of-00002.safetensors':'66d2737fde38a876994f3f14ee27f7930fdcaf986a1839a00086d50d2298e828'}
def test_retrieval_and_schema_are_frozen():
 assert CFG['retriever']['revision']=='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a';assert CFG['retriever']['index']=='faiss.IndexFlatIP';assert CFG['retriever']['production_top_k']==1;assert CFG['parser']['schema_version']=='posttraining-v2'
def test_health_and_top_one_verify():
 api,r,l=client()
 with api:
  h=api.get('/health').json();x=api.post('/verify',json={'claim':'claim'}).json()
 assert h['status']=='ok' and h['deployment_candidate']=='M2'
 assert h['models']['served_model_name']=='posttraining-m2-3b';assert r.calls==[('claim',1)];assert x['decision']=='SUPPORT' and x['rationale_sentences']==[1] and x['valid'];assert x['token_usage']=={'input_tokens':20,'output_tokens':14};assert l.records[0]['rationale_index_valid']
def test_malformed_is_not_repaired():
 api,_,l=client(False)
 with api:x=api.post('/verify',json={'claim':'claim'}).json()
 assert not x['valid'] and x['decision'] is None and x['rationale_sentences']==[] and x['errors']==['invalid_json'];assert l.records[0]['raw_output']=='bad'
def test_benchmark_and_overwrite_guards_recorded():
 assert CFG['benchmark']['concurrency']==[1,2,4,8,16];assert CFG['completed_run_overwrite_allowed'] is False;assert CFG['generator']['gpu_memory_utilization']==.72

def test_benchmark_summary_records_latency_memory_and_components():
 from scripts.benchmark_posttraining_serving import summary
 timing={k:float(i+1) for i,k in enumerate(('embedding','faiss','retrieval','prompt','generation','total'))}
 records=[{'ok':True,'latency_ms':10.,'valid':True,'timing_ms':timing,'input_tokens':20,'output_tokens':4,'cpu_memory_percent':1.},{'ok':True,'latency_ms':20.,'valid':True,'timing_ms':timing,'input_tokens':22,'output_tokens':6,'cpu_memory_percent':2.}]
 x=summary(records,1.,2,[{'memory_mib':123.,'utilization_percent':50.}])
 assert x['latency_ms']['p50']==15. and x['latency_ms']['p95']==19.5
 assert x['gpu']['peak_memory_mib']==123. and x['tokens']['output_total']==10
 assert set(x['component_ms'])=={'embedding','faiss','retrieval','prompt','generation','total'}

def test_hashes_environment_and_completed_run_guard_are_recorded():
 assert CFG['generator']['finalist_config_sha256'] and CFG['retriever']['corpus_sha256'] and CFG['retriever']['index_sha256']
 assert CFG['environment']=='/home/boweiye2/rag/envs/serving' and CFG['git_commit']
 text=(ROOT/'scripts/benchmark_posttraining_serving.py').read_text();assert "if out.exists():raise FileExistsError(out)" in text
 assert CFG['scientific_quality_changes_allowed'] is False
