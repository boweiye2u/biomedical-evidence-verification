"""Construct frozen deterministic M2 mixtures without DEV or benchmark access."""
from __future__ import annotations
import argparse, hashlib, json, math, random
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REQUIRED_FIELDS={"claim_id","doc_id","condition","source_split","claim","evidence","claim_gold_label","pair_annotation","context_label","rationale_sentences","annotation_basis","retrieval_rank","retrieval_score","filter_version"}

def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def read_jsonl(path:Path)->list[dict]:return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def stable_pool_seed(seed:int,pool:str)->int:return seed+{"A":101,"B":211,"C":307,"D":401}[pool]

def validate_pool(pool:str,rows:list[dict],spec:dict,train_ids:set[str],dev_ids:set[str])->None:
 if len(rows)!=spec["rows"]:raise ValueError(f"{pool} row count changed")
 seen=set()
 for row in rows:
  if not REQUIRED_FIELDS<=set(row):raise ValueError(f"{pool} missing fields")
  if row["source_split"]!="TRAIN" or str(row["claim_id"]) not in train_ids or str(row["claim_id"]) in dev_ids:raise ValueError(f"{pool} split leakage")
  if row["condition"]!=spec["condition"]:raise ValueError(f"{pool} condition mismatch")
  pair=(str(row["claim_id"]),str(row["doc_id"]))
  if pair in seen:raise ValueError(f"{pool} duplicate pair")
  seen.add(pair)
  if pool=="A":
   if row["pair_annotation"] not in {"SUPPORT","CONTRADICT"} or row["context_label"]!=row["pair_annotation"] or not row["rationale_sentences"]:raise ValueError("Invalid A semantics")
  else:
   if row["pair_annotation"] is not None or row["context_label"]!="INSUFFICIENT" or row["rationale_sentences"]!=[]:raise ValueError(f"Invalid {pool} semantics")
  if pool in {"B","C"} and row["filter_version"]!=spec["required_filter_version"]:raise ValueError(f"{pool} filter mismatch")
  if pool=="B" and (row["retrieval_score"] is None or row["retrieval_score"]>spec["maximum_score_inclusive"]+1e-12 or row.get("filter_status")!="approved"):raise ValueError("Unapproved B row")
  if pool=="D" and str(spec["required_source_seed"]) not in row["annotation_basis"]:raise ValueError("D source seed mismatch")

def hamilton_counts(percentages:dict[str,float],available:dict[str,int])->tuple[int,dict[str,int]]:
 total=math.floor(min(available[k]/p for k,p in percentages.items() if p>0))
 raw={k:total*p for k,p in percentages.items()};counts={k:math.floor(v) for k,v in raw.items()}
 for k in sorted(percentages,key=lambda x:(-(raw[x]-counts[x]),x))[:total-sum(counts.values())]:counts[k]+=1
 if any(counts[k]>available[k] for k in counts):raise ValueError("Hamilton allocation exceeds pool")
 return total,counts

def sample_mixture(pools:dict[str,list[dict]],name:str,spec:dict)->list[dict]:
 total,counts=hamilton_counts(spec["percentages"],{k:len(v) for k,v in pools.items()})
 if total!=spec["expected_total"] or counts!=spec["expected_counts"]:raise ValueError("Frozen effective mixture changed")
 selected=[]
 for pool in sorted(counts):
  values=sorted(pools[pool],key=lambda r:(int(r["claim_id"]),int(r["doc_id"])))
  random.Random(stable_pool_seed(spec["sampling_seed"],pool)).shuffle(values)
  for source in values[:counts[pool]]:
   row=dict(source);row["mixture_name"]=name;row["sampling_seed"]=spec["sampling_seed"];selected.append(row)
 selected.sort(key=lambda r:(int(r["claim_id"]),r["condition"],int(r["doc_id"])))
 pairs=[(str(r["claim_id"]),str(r["doc_id"])) for r in selected]
 if len(pairs)!=len(set(pairs)):raise ValueError(f"{name} cross-pool duplicate pair")
 return selected

def build(config_path:Path,output_root:Path|None=None)->dict:
 config=json.loads(config_path.read_text());split=json.loads((ROOT/"configs/splits/scifact_train_dev_v1.json").read_text());train_ids=set(map(str,split["train_ids"]));dev_ids=set(map(str,split["dev_ids"]))
 pools={}
 for pool,spec in config["source_pools"].items():
  path=Path(spec["path"])
  if sha256(path)!=spec["sha256"]:raise ValueError(f"{pool} source hash changed")
  pools[pool]=read_jsonl(path);validate_pool(pool,pools[pool],spec,train_ids,dev_ids)
 out=output_root or Path(config["output_root"])
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);manifest={"version":config["version"],"config_sha256":sha256(config_path),"benchmark_loaded":False,"source_hashes":{k:v["sha256"] for k,v in config["source_pools"].items()},"mixtures":{}}
 for name,spec in config["candidates"].items():
  rows=sample_mixture(pools,name,spec);path=out/f"{name}.jsonl"
  with path.open('w') as h:
   for row in rows:h.write(json.dumps(row,sort_keys=True)+"\n")
  counts=Counter({p:0 for p in pools});counts.update({"A_gold":"A","B_retrieved_non_gold":"B","C_hard_negative":"C","D_random_negative":"D"}[r["condition"]] for r in rows)
  labels=Counter(r["context_label"] for r in rows)
  manifest["mixtures"][name]={"path":str(path),"sha256":sha256(path),"sampling_seed":spec["sampling_seed"],"rows":len(rows),"counts":dict(counts),"effective_percentages":{k:counts[k]/len(rows) for k in sorted(counts)},"label_counts":dict(labels),"unique_claims":len({r["claim_id"] for r in rows}),"duplicate_pairs":0}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/posttraining/m2-mixtures-v1.json');p.add_argument('--output-root',type=Path);a=p.parse_args();print(json.dumps(build(a.config,a.output_root),indent=2))
if __name__=='__main__':main()
