import hashlib,json,os,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CFG=ROOT/'configs/posttraining/serving-tuned-3b-v1.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads(CFG.read_text());run=Path(c['run_root'])
 if run.exists():raise FileExistsError(f'refusing existing serving run: {run}')
 assert c['deployment_candidate']=='M2' and '/m2/' in c['generator']['local_path'] and '/m3/' not in c['generator']['local_path']
 assert {p.name:sha(p) for p in sorted(Path(c['generator']['local_path']).glob('*.safetensors'))}==c['generator']['weight_sha256']
 finalist=ROOT/c['generator']['finalist_config'];assert sha(finalist)==c['generator']['finalist_config_sha256']
 for key,file in [('corpus_sha256',c['retriever']['corpus']),('index_sha256',Path(c['retriever']['embedding_cache'])/'index.faiss'),('metadata_sha256',Path(c['retriever']['embedding_cache'])/'metadata.json')]:assert sha(file)==c['retriever'][key]
 ref=Path(c['regression']['reference_predictions']);assert sha(ref)==c['regression']['reference_predictions_sha256']
 run.mkdir(parents=True);Path(c['log_root']).mkdir(parents=True,exist_ok=False)
 env={'python':platform.python_version(),'pytorch':__import__('torch').__version__,'cuda_runtime':__import__('torch').version.cuda,'transformers':__import__('transformers').__version__,'vllm':subprocess.check_output([str(Path(c['environment'])/'bin/python'),'-c','import vllm;print(vllm.__version__)'],text=True).strip(),'fastapi':__import__('fastapi').__version__,'faiss':__import__('faiss').__version__,'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0],'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader','--id=0'],text=True).strip(),'git_commit':c['git_commit']}
 payload={'status':'passed_before_service_start','config_sha256':sha(CFG),'m2_checkpoint_path':c['generator']['local_path'],'m2_weight_sha256':c['generator']['weight_sha256'],'retrieval_hashes':{k:c['retriever'][k] for k in ('corpus_sha256','index_sha256','metadata_sha256')},'reference_predictions_sha256':c['regression']['reference_predictions_sha256'],'environment':env,'scientific_benchmark_rerun':False,'training_run':False}
 (run/'preflight.json').write_text(json.dumps(payload,indent=2)+'\n');(run/'environment.json').write_text(json.dumps(env,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
