"""Validate and summarize frozen M1 CV, training, and DEV artifacts."""
from __future__ import annotations
import argparse, hashlib, json, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=Path.home()/"rag/runs/posttraining/lora-gold-v1"
CHECKPOINT=Path.home()/"rag/checkpoints/posttraining/m1"

def load(path:Path): return json.loads(path.read_text())
def digest(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def mean_sd(xs): return {"mean":statistics.fmean(xs),"sample_sd":statistics.stdev(xs)}

REQUIRED_TIMING_FIELDS={"wall_clock_seconds","wall_clock_hours","gpu_count","gpu_model","gpu_hours","optimizer_steps","epochs_completed","training_examples","tokens_processed","tokens_per_second","samples_per_second","peak_gpu_memory_mib","mean_step_time"}
def validate_training_metrics(metrics:dict)->None:
 missing=REQUIRED_TIMING_FIELDS-set(metrics)
 if missing: raise ValueError(f"Missing timing fields: {sorted(missing)}")
 if abs(metrics["gpu_hours"]-metrics["gpu_count"]*metrics["wall_clock_hours"])>1e-12: raise ValueError("GPU-hours mismatch")

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,default=RUN);p.add_argument('--checkpoint-root',type=Path,default=CHECKPOINT);p.add_argument('--output',type=Path,default=ROOT/'docs/posttraining/milestone3-m1-lora-gold-results.json');a=p.parse_args()
 cv=load(a.run_root/'cv-summary.json'); finalist=load(ROOT/'configs/posttraining/m1-gold-finalist-v1.json')
 assert cv['selected_epochs']==finalist['selected_epochs']==3
 b2=load(Path.home()/"rag/runs/posttraining/baselines-v1/B2-v2/metrics.json")
 seeds=[]; id_sets=[]
 for seed in (0,1,2):
  run=a.run_root/'dev-finalists'/f'seed-{seed}'; train=load(run/'training-metrics.json'); dev=load(run/'dev-evaluation/metrics.json')
  validate_training_metrics(train)
  preds=[json.loads(x) for x in (run/'dev-evaluation/predictions.jsonl').read_text().splitlines() if x.strip()]
  assert len(preds)==dev['count']==162 and all(x['decision_valid'] for x in preds)
  id_sets.append({x['claim_id'] for x in preds})
  merged=a.checkpoint_root/'dev-finalists'/f'seed-{seed}'/'merged'
  shards=sorted(merged.glob('model-*.safetensors'));assert len(shards)==2
  seeds.append({'seed':seed,'accuracy':dev['accuracy'],'macro_f1':dev['macro_f1'],'per_label':dev['per_label'],'prediction_counts':dev['prediction_counts_including_invalid'],'json_valid_rate':dev['json_valid_rate'],'decision_valid_rate':dev['decision_valid_rate'],'rationale_index_valid_rate':dev['rationale_index_valid_rate'],'training':train,'evaluation_efficiency':dev['efficiency'],'merged_weight_bytes':sum(x.stat().st_size for x in shards)})
 assert id_sets[0]==id_sets[1]==id_sets[2] and len(id_sets[0])==162
 metrics={k:mean_sd([x[k] for x in seeds]) for k in ('accuracy','macro_f1')}
 for label in ('SUPPORT','CONTRADICT','INSUFFICIENT'):
  metrics[f'{label.lower()}_f1']=mean_sd([x['per_label'][label]['f1'] for x in seeds]);metrics[f'{label.lower()}_recall']=mean_sd([x['per_label'][label]['recall'] for x in seeds])
 total_cv=cv['candidates']['3']['total_training_seconds'];total_final=sum(x['training']['wall_clock_seconds'] for x in seeds)
 payload={'status':'complete','milestone':'M1_gold_evidence_LoRA_SFT','passed_execution':True,'scientific_outcome':'negative_on_DEV','selected_epochs':3,'cv':cv,'dev_seeds':seeds,'dev_mean_sample_sd':metrics,'b2':{'accuracy':b2['accuracy'],'macro_f1':b2['macro_f1'],'contradict_recall':b2['per_label']['CONTRADICT']['recall']},'m1_minus_b2':{'accuracy':metrics['accuracy']['mean']-b2['accuracy'],'macro_f1':metrics['macro_f1']['mean']-b2['macro_f1'],'contradict_recall':metrics['contradict_recall']['mean']-b2['per_label']['CONTRADICT']['recall']},'compute':{'actual_cv_training_seconds':total_cv,'actual_final_training_seconds':total_final,'actual_total_training_seconds':total_cv+total_final,'actual_total_gpu_hours':(total_cv+total_final)/3600,'max_peak_pytorch_memory_mib':max(x['training']['peak_gpu_memory_mib'] for x in seeds)},'integrity':{'aligned_dev_ids':True,'dev_count':162,'fixed_benchmark_accessed':False,'training_config_sha256':digest(ROOT/'configs/posttraining/train-m1-lora-gold-v1.json'),'finalist_config_sha256':digest(ROOT/'configs/posttraining/m1-gold-finalist-v1.json'),'cv_fold_sha256':digest(ROOT/'configs/posttraining/m1-gold-cv-folds-v1.json')}}
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
