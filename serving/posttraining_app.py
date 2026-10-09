from __future__ import annotations
import asyncio,hashlib,json,os,time,uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any,Literal
from fastapi import FastAPI
from pydantic import BaseModel,Field,field_validator
from .logging_utils import JsonlRequestLogger
from .retriever import FrozenRetriever
from .posttraining_verifier import PosttrainingVerifier
ROOT=Path(__file__).resolve().parents[1]
CONFIG_PATH=ROOT/'configs/posttraining/serving-tuned-3b-v1.json'

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def load_config():
 path=Path(os.environ.get('POSTTRAINING_SERVING_CONFIG',CONFIG_PATH));c=json.loads(path.read_text());c['_config_path']=str(path);c['_config_sha256']=sha256(path)
 if c['deployment_candidate']!='M2' or 'm2/' not in c['generator']['local_path'].lower():raise ValueError('service must load frozen M2, not base 3B or M3')
 if c['generator']['served_model_name']!='posttraining-m2-3b':raise ValueError('unexpected served model identity')
 if c['retriever']['production_top_k']!=1 or c['retriever']['index']!='faiss.IndexFlatIP':raise ValueError('frozen retrieval changed')
 return c
class Claim(BaseModel):
 claim:str=Field(max_length=20000)
 @field_validator('claim')
 @classmethod
 def nonempty(cls,v):
  v=v.strip()
  if not v:raise ValueError('claim must not be empty')
  return v
class VerifyResponse(BaseModel):
 decision:Literal['SUPPORT','CONTRADICT','INSUFFICIENT']|None
 rationale_sentences:list[int]
 retrieval:dict[str,Any]
 timing_ms:dict[str,float]
 token_usage:dict[str,int|None]
 valid:bool
 errors:list[str]
 request_id:str

def create_app(retriever=None,verifier=None,request_logger=None):
 config=load_config();g={**config['generator'],'prompt_sha256':config['prompt']['sha256']}
 @asynccontextmanager
 async def lifespan(app):
  app.state.retriever=retriever or FrozenRetriever(config['retriever']);app.state.verifier=verifier or PosttrainingVerifier(g);app.state.request_logger=request_logger or JsonlRequestLogger(config['service']['request_log']);app.state.retrieval_lock=asyncio.Lock();yield
  close=getattr(app.state.verifier,'close',None)
  if close:await close()
 app=FastAPI(title='Frozen M2 Biomedical Verification Service',version='posttraining-v1',lifespan=lifespan)
 @app.get('/health')
 async def health():
  reachable=await app.state.verifier.reachable()
  return {'status':'ok' if reachable else 'degraded','retriever_loaded':app.state.retriever is not None,'faiss_loaded':getattr(app.state.retriever,'index',None) is not None,'generator_reachable':reachable,'deployment_candidate':'M2','schema_version':config['parser']['schema_version'],'models':{'retriever':config['retriever']['model_name'],'retriever_revision':config['retriever']['revision'],'generator':config['generator']['model_name'],'served_model_name':config['generator']['served_model_name'],'generator_revision':config['generator']['base_revision'],'checkpoint_path':config['generator']['local_path'],'checkpoint_weights':config['generator']['weight_sha256']},'config_sha256':config['_config_sha256']}
 @app.post('/retrieve')
 async def retrieve(request:Claim):
  started=time.perf_counter()
  async with app.state.retrieval_lock:docs,timing=await asyncio.to_thread(app.state.retriever.retrieve,request.claim,1)
  timing['total']=(time.perf_counter()-started)*1000
  return {'claim':request.claim,'documents':docs,'timing_ms':timing}
 @app.post('/verify',response_model=VerifyResponse)
 async def verify(request:Claim):
  rid=str(uuid.uuid4());started=time.perf_counter()
  async with app.state.retrieval_lock:docs,rt=await asyncio.to_thread(app.state.retriever.retrieve,request.claim,1)
  prompt_start=time.perf_counter();prompt,info=app.state.verifier.build_prompt(request.claim,docs[0]);prompt_ms=(time.perf_counter()-prompt_start)*1000
  generation=await app.state.verifier.generate(prompt,info['sentence_count']);total=(time.perf_counter()-started)*1000;parsed=generation['parsed'] or {}
  timing={**rt,'prompt':prompt_ms,'generation':generation['generation_ms'],'total':total};usage={'input_tokens':generation['input_tokens'] or info['prompt_tokens'],'output_tokens':generation['output_tokens']}
  result={'decision':parsed.get('decision'),'rationale_sentences':parsed.get('rationale_sentences',[]),'retrieval':{'document_id':docs[0]['document_id'],'score':docs[0]['score']},'timing_ms':timing,'token_usage':usage,'valid':generation['parsed'] is not None,'errors':generation['errors'],'request_id':rid}
  app.state.request_logger.write({'request_id':rid,'endpoint':'/verify','claim_char_count':len(request.claim),'input_token_count':usage['input_tokens'],'output_token_count':usage['output_tokens'],'retrieved_document_id':docs[0]['document_id'],'timing_ms':timing,'status':'ok' if result['valid'] else 'invalid_generation','parse_valid':result['valid'],'schema_valid':generation['schema_valid'],'decision_valid':generation['decision_valid'],'rationale_index_valid':generation['rationale_index_valid'],'parse_errors':result['errors'],'raw_output':generation['raw_output']})
  return result
 return app
app=create_app()
