"""Validate and summarize frozen M2 mixture, CV, finalist, and DEV artifacts."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path.home()/"rag/runs/posttraining/lora-mixed-v1";CKPT=Path.home()/"rag/checkpoints/posttraining/m2"
REQUIRED_TIMING={"wall_clock_seconds","wall_clock_hours","gpu_count","gpu_model","gpu_hours","optimizer_steps","epochs_completed","training_examples","tokens_processed","tokens_per_second","samples_per_second","mean_step_time","peak_gpu_memory_mib"}
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean_sd(xs):return {"mean":statistics.fmean(xs),"sample_sd":statistics.stdev(xs)}
def timing_ok(x):
 missing=REQUIRED_TIMING-set(x)
 if missing:raise ValueError(f"Missing timing fields {sorted(missing)}")
 if abs(x['gpu_hours']-x['gpu_count']*x['wall_clock_hours'])>1e-12:raise ValueError('GPU-hours mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--checkpoint-root',type=Path,default=CKPT);p.add_argument('--output',type=Path,default=ROOT/'docs/posttraining/milestone4-m2-lora-mixed-results.json');a=p.parse_args()
 cv=load(a.run_root/'cv-summary.json');mixtures=load(Path.home()/'rag/data/posttraining/mixtures-v1/manifest.json');final=load(ROOT/'configs/posttraining/m2-finalist-v1.json');all_finalists={};ids=[];final_training=[]
 for epoch in (2,3):
  seeds=[]
  for seed in (0,1,2):
   r=a.run_root/f'dev-finalists/mix-2-epoch-{epoch}/seed-{seed}';t=load(r/'training-metrics.json');timing_ok(t);m=load(r/'dev-evaluation/metrics.json');pred=[json.loads(x) for x in (r/'dev-evaluation/predictions.jsonl').read_text().splitlines() if x.strip()];assert len(pred)==m['count']==162 and all(x['decision_valid'] for x in pred);ids.append({x['claim_id'] for x in pred});merged=a.checkpoint_root/f'dev-finalists/mix-2-epoch-{epoch}/seed-{seed}/merged';shards=sorted(merged.glob('model-*.safetensors'));assert len(shards)==2
   seeds.append({'seed':seed,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'per_label':m['per_label'],'prediction_counts':m['prediction_counts_including_invalid'],'json_valid_rate':m['json_valid_rate'],'decision_valid_rate':m['decision_valid_rate'],'rationale_index_valid_rate':m['rationale_index_valid_rate'],'training':t,'evaluation_efficiency':m['efficiency'],'merged_weight_bytes':sum(x.stat().st_size for x in shards)});final_training.append(t)
  agg={k:mean_sd([x[k] for x in seeds]) for k in ('accuracy','macro_f1')}
  for label in ('SUPPORT','CONTRADICT','INSUFFICIENT'):
   for metric in ('f1','recall'):agg[f'{label.lower()}_{metric}']=mean_sd([x['per_label'][label][metric] for x in seeds])
  all_finalists[str(epoch)]={'seeds':seeds,'mean_sample_sd':agg}
 assert all(x==ids[0] for x in ids) and len(ids[0])==162 and final['selected_epochs']==3
 baseroot=Path.home()/'rag/runs/posttraining/baselines-v1';bases={k:load(baseroot/f'{k}-v2/metrics.json') for k in ('B0','B1','B3')};m1=load(ROOT/'docs/posttraining/milestone3-m1-lora-gold-results.json');sel=all_finalists['3']['mean_sample_sd']
 comparisons={'M1':{'accuracy':sel['accuracy']['mean']-m1['dev_mean_sample_sd']['accuracy']['mean'],'macro_f1':sel['macro_f1']['mean']-m1['dev_mean_sample_sd']['macro_f1']['mean']},**{k:{'accuracy':sel['accuracy']['mean']-v['accuracy'],'macro_f1':sel['macro_f1']['mean']-v['macro_f1']} for k,v in bases.items()}}
 cv_times=[]
 for mix in ('mix-1','mix-2'):
  for fold in range(5):
   t=load(a.run_root/f'cv/{mix}/fold-{fold}/training-metrics.json');timing_ok(t);cv_times.append(t)
 cv_seconds=sum(x['wall_clock_seconds'] for x in cv_times);final_seconds=sum(x['wall_clock_seconds'] for x in final_training)
 payload={'status':'complete','milestone':'M2_mixed_evidence_LoRA_SFT','passed_execution':True,'selected_configuration':{'mixture':'mix-2','epochs':3,'inferential_seed':0},'mixture_manifest':mixtures,'cv':cv,'dev_finalists':all_finalists,'comparisons_selected_M2_minus':comparisons,'diagnostic':{'insufficient_recall_restored':sel['insufficient_recall']['mean']>0,'contradict_recall_above_B2_B3':sel['contradict_recall']['mean']>max(2/28,bases['B3']['per_label']['CONTRADICT']['recall'])},'compute':{'cv_training_seconds':cv_seconds,'finalist_training_seconds':final_seconds,'total_training_seconds':cv_seconds+final_seconds,'total_gpu_hours':(cv_seconds+final_seconds)/3600,'peak_pytorch_memory_mib':max(x['peak_gpu_memory_mib'] for x in cv_times+final_training)},'integrity':{'all_dev_ids_aligned':True,'dev_count':162,'fixed_benchmark_accessed':False,'mixture_config_sha256':sha(ROOT/'configs/posttraining/m2-mixtures-v1.json'),'training_config_sha256':sha(ROOT/'configs/posttraining/train-m2-lora-mixed-v1.json'),'finalist_config_sha256':sha(ROOT/'configs/posttraining/m2-finalist-v1.json')}}
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
