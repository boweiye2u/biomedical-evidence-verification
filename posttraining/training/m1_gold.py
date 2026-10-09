"""Frozen gold-only Qwen2.5-3B LoRA training for M1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from posttraining.evaluation.baselines import build_messages, parse_output
from retrieval.verification import classification_metrics

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_gold(rows: list[dict], config: dict, train_ids: set[str], dev_ids: set[str]) -> None:
    expected = config["training_data"]
    if len(rows) != expected["rows"] or Counter(row["context_label"] for row in rows) != Counter({"SUPPORT": expected["support"], "CONTRADICT": expected["contradict"]}):
        raise ValueError("Frozen Condition A counts changed")
    pairs = set()
    for row in rows:
        if row["condition"] != expected["condition"] or row["source_split"] != "TRAIN":
            raise ValueError("M1 accepts Condition A TRAIN rows only")
        if row["claim_id"] not in train_ids or row["claim_id"] in dev_ids:
            raise ValueError("Training split leakage")
        if row["pair_annotation"] != row["context_label"] or row["context_label"] not in {"SUPPORT", "CONTRADICT"}:
            raise ValueError("Pair/context label mismatch")
        pair = row["claim_id"], row["doc_id"]
        if pair in pairs: raise ValueError("Duplicate gold pair")
        pairs.add(pair)
        rationale = row["rationale_sentences"]
        if not rationale or any(not isinstance(i, int) or i < 0 or i >= len(row["evidence"]["abstract"]) for i in rationale):
            raise ValueError("Invalid gold rationale")


def make_grouped_folds(rows: list[dict], folds: int, seed: int) -> list[list[str]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows: groups[row["claim_id"]].append(row)
    by_label: dict[str, list[str]] = defaultdict(list)
    for claim_id, values in groups.items():
        labels = {row["context_label"] for row in values}
        if len(labels) != 1: raise ValueError("Claim group has inconsistent labels")
        by_label[next(iter(labels))].append(claim_id)
    rng = random.Random(seed); assignments=[[] for _ in range(folds)]; label_rows={label:[0]*folds for label in by_label}; totals=[0]*folds
    for label in sorted(by_label):
        ids=by_label[label][:];rng.shuffle(ids);ids.sort(key=lambda cid:len(groups[cid]),reverse=True)
        for claim_id in ids:
            fold=min(range(folds),key=lambda i:(label_rows[label][i],totals[i],i))
            assignments[fold].append(claim_id);label_rows[label][fold]+=len(groups[claim_id]);totals[fold]+=len(groups[claim_id])
    return [sorted(ids,key=int) for ids in assignments]


def document(row: dict) -> dict:
    return {"doc_id": row["doc_id"], **row["evidence"]}


def target_text(row: dict) -> str:
    return json.dumps({"decision":row["context_label"],"rationale_sentences":row["rationale_sentences"]},separators=(",",":"))


def encode_training_row(row: dict, tokenizer, max_length: int) -> dict:
    prompt_messages=build_messages(row["claim"],document(row),[])
    prompt=tokenizer.apply_chat_template(prompt_messages,tokenize=True,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(prompt_messages+[{"role":"assistant","content":target_text(row)}],tokenize=True,add_generation_prompt=False)
    if full[:len(prompt)] != prompt: raise ValueError("Chat template prompt is not a prefix of supervised sequence")
    if len(full)>max_length: raise ValueError(f"Sequence length {len(full)} exceeds {max_length}")
    labels=[-100]*len(prompt)+full[len(prompt):]
    if not any(x!=-100 for x in labels) or any(x!=-100 for x in labels[:len(prompt)]): raise ValueError("Output-only mask failure")
    return {"input_ids":full,"labels":labels,"length":len(full),"answer_tokens":sum(x!=-100 for x in labels)}


class EncodedDataset(Dataset):
    def __init__(self, values): self.values=values
    def __len__(self): return len(self.values)
    def __getitem__(self,index): return self.values[index]


def collate(batch: list[dict], pad_id: int) -> dict:
    width=max(len(row["input_ids"]) for row in batch);ids=[];labels=[];mask=[]
    for row in batch:
        pad=width-len(row["input_ids"]);ids.append(row["input_ids"]+[pad_id]*pad);labels.append(row["labels"]+[-100]*pad);mask.append([1]*len(row["input_ids"])+[0]*pad)
    return {"input_ids":torch.tensor(ids),"labels":torch.tensor(labels),"attention_mask":torch.tensor(mask)}


def model_and_tokenizer(config: dict):
    path=config["local_model_path"]
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16).cuda()
    model.config.use_cache=False
    if config["optimization"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False});model.enable_input_require_grads()
    lc=config["lora"]
    model=get_peft_model(model,LoraConfig(r=lc["rank"],lora_alpha=lc["alpha"],lora_dropout=lc["dropout"],bias=lc["bias"],target_modules=lc["target_modules"],task_type="CAUSAL_LM"))
    return model,tokenizer


def gpu_name() -> str: return torch.cuda.get_device_name(0)


@torch.inference_mode()
def evaluate(model,tokenizer,rows: list[dict],batch_size: int=8) -> tuple[dict,list[dict]]:
    was_gc=model.is_gradient_checkpointing;model.gradient_checkpointing_disable();model.config.use_cache=True;model.eval();tokenizer.padding_side="left"
    records=[];start=time.perf_counter()
    for offset in range(0,len(rows),batch_size):
        part=rows[offset:offset+batch_size]
        prompts=[tokenizer.apply_chat_template(build_messages(row["claim"],document(row),[]),tokenize=False,add_generation_prompt=True) for row in part]
        tokens=tokenizer(prompts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        output=model.generate(**tokens,max_new_tokens=192,do_sample=False,repetition_penalty=1.05,pad_token_id=tokenizer.pad_token_id)
        generated=output[:,tokens["input_ids"].shape[1]:]
        for row,ids in zip(part,generated):
            raw=tokenizer.decode(ids,skip_special_tokens=True).strip();parsed=parse_output(raw,len(row["evidence"]["abstract"]))
            records.append({"claim_id":row["claim_id"],"doc_id":row["doc_id"],"gold_label":row["context_label"],"predicted_label":parsed["decision"] or "INVALID","raw_output":raw,**{k:parsed[k] for k in ("json_valid","schema_valid","decision_valid","rationale_index_valid","errors")}})
    elapsed=time.perf_counter()-start
    metrics=classification_metrics([r["gold_label"] for r in records],[r["predicted_label"] for r in records])
    metrics.update({"json_valid_rate":sum(r["json_valid"] for r in records)/len(records),"decision_valid_rate":sum(r["decision_valid"] for r in records)/len(records),"rationale_index_valid_rate":sum(r["rationale_index_valid"] for r in records)/len(records),"evaluation_seconds":elapsed})
    model.config.use_cache=False
    if was_gc: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    tokenizer.padding_side="right";return metrics,records


def train_run(config: dict,rows: list[dict],validation: list[dict],seed: int,epochs: int,output: Path,checkpoint_output: Path,save_epochs: set[int],merge: bool=False) -> dict:
    if output.exists(): raise FileExistsError(output)
    if checkpoint_output.exists(): raise FileExistsError(checkpoint_output)
    output.mkdir(parents=True);checkpoint_output.mkdir(parents=True);set_seed(seed);torch.cuda.reset_peak_memory_stats();model,tokenizer=model_and_tokenizer(config)
    encoded=[encode_training_row(row,tokenizer,config["optimization"]["max_sequence_length"]) for row in rows]
    opt_cfg=config["optimization"];micro=opt_cfg["micro_batch_size"];accum=opt_cfg["gradient_accumulation"]
    updates_per_epoch=math.ceil(math.ceil(len(encoded)/micro)/accum);total_updates=updates_per_epoch*opt_cfg["scheduler_horizon_epochs"]
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=opt_cfg["learning_rate"],weight_decay=opt_cfg["weight_decay"],betas=tuple(opt_cfg["betas"]),eps=opt_cfg["epsilon"])
    warmup=math.ceil(total_updates*opt_cfg["warmup_ratio"]);scheduler=get_linear_schedule_with_warmup(optimizer,warmup,total_updates)
    step=0;tokens=answer_tokens=0;trace=[];epoch_results={};training_seconds=0.0;total_start=time.perf_counter();torch.cuda.synchronize()
    for epoch in range(1,epochs+1):
        model.train()
        generator=torch.Generator().manual_seed(seed+epoch)
        loader=DataLoader(EncodedDataset(encoded),batch_size=micro,shuffle=True,generator=generator,collate_fn=lambda b:collate(b,tokenizer.pad_token_id))
        optimizer.zero_grad(set_to_none=True);group=[];torch.cuda.synchronize();epoch_train_start=time.perf_counter();group_start=None
        for batch_index,batch in enumerate(loader):
            if not group: group_start=time.perf_counter()
            batch={k:v.cuda(non_blocking=True) for k,v in batch.items()};loss=model(**batch).loss;loss.backward();group.append(float(loss.detach()))
            tokens+=int(batch["attention_mask"].sum());answer_tokens+=int((batch["labels"]!=-100).sum())
            final=batch_index+1==len(loader)
            if len(group)==accum or final:
                scale=1/len(group)
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(scale)
                grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),opt_cfg["max_gradient_norm"]));optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1;torch.cuda.synchronize()
                trace.append({"step":step,"epoch":epoch,"mean_micro_loss":sum(group)/len(group),"gradient_norm":grad,"learning_rate":scheduler.get_last_lr()[0],"step_seconds":time.perf_counter()-group_start});group=[];group_start=None
        torch.cuda.synchronize();training_seconds+=time.perf_counter()-epoch_train_start
        if epoch in save_epochs:
            adapter=checkpoint_output/f"epoch-{epoch}"/"adapter";adapter.mkdir(parents=True);model.save_pretrained(adapter,safe_serialization=True);tokenizer.save_pretrained(adapter)
            epoch_dir=output/f"epoch-{epoch}";epoch_dir.mkdir(parents=True)
            checkpoint_metrics={"cumulative_training_seconds":training_seconds,"cumulative_optimizer_steps":step,"cumulative_tokens_processed":tokens,"adapter_path":str(adapter)}
            if validation:
                metrics,predictions=evaluate(model,tokenizer,validation);checkpoint_metrics.update(metrics)
                with (epoch_dir/"predictions.jsonl").open("w") as h:
                    for row in predictions:h.write(json.dumps(row)+"\n")
                epoch_results[str(epoch)]=metrics
            (epoch_dir/"metrics.json").write_text(json.dumps(checkpoint_metrics,indent=2)+"\n")
            checkpoint_manifest={"epoch":epoch,"optimizer_step":step,"seed":seed,"config_sha256":sha256(ROOT/"configs/posttraining/train-m1-lora-gold-v1.json"),"training_data_sha256":config["training_data"]["sha256"],"model_revision":config["model_revision"],"tokenizer_revision":config["tokenizer_revision"],"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
            (adapter.parent/"checkpoint-manifest.json").write_text(json.dumps(checkpoint_manifest,indent=2)+"\n")
    torch.cuda.synchronize();total_wall=time.perf_counter()-total_start
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);total=sum(p.numel() for p in model.parameters())
    timing={"wall_clock_seconds":training_seconds,"wall_clock_hours":training_seconds/3600,"gpu_hours":training_seconds/3600,"mean_step_time":sum(x["step_seconds"] for x in trace)/len(trace),"tokens_per_second":tokens/training_seconds,"samples_per_second":len(rows)*epochs/training_seconds,"training_wall_clock_seconds":training_seconds,"training_wall_clock_hours":training_seconds/3600,"total_run_wall_clock_seconds":total_wall,"gpu_count":1,"gpu_model":gpu_name(),"training_gpu_hours":training_seconds/3600,"optimizer_steps":step,"epochs_completed":epochs,"training_examples":len(rows),"tokens_processed":tokens,"answer_tokens_processed":answer_tokens,"tokens_per_training_second":tokens/training_seconds,"samples_per_training_second":len(rows)*epochs/training_seconds,"peak_gpu_memory_mib":torch.cuda.max_memory_allocated()/2**20,"mean_optimizer_step_time":sum(x["step_seconds"] for x in trace)/len(trace),"trainable_parameters":trainable,"total_parameters":total,"trainable_percentage":100*trainable/total,"warmup_steps":warmup,"scheduler_horizon_steps":total_updates}
    (output/"training-metrics.json").write_text(json.dumps(timing,indent=2)+"\n");(output/"step-trace.json").write_text(json.dumps(trace,indent=2)+"\n")
    manifest={"seed":seed,"epochs":epochs,"config_sha256":sha256(ROOT/"configs/posttraining/train-m1-lora-gold-v1.json"),"training_data_sha256":config["training_data"]["sha256"],"model_revision":config["model_revision"],"tokenizer_revision":config["tokenizer_revision"],"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"checkpoint_output":str(checkpoint_output),"epoch_results":epoch_results,"timing":timing}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if merge:
        merged=checkpoint_output/"merged";merged.mkdir();merged_model=model.merge_and_unload();merged_model.config.use_cache=True;merged_model.save_pretrained(merged,safe_serialization=True,max_shard_size="4GB");tokenizer.save_pretrained(merged)
    del model;torch.cuda.empty_cache();return manifest


def load_context(args):
    config=json.loads(args.config.read_text());rows=read_jsonl(Path(config["training_data"]["path"]));split=json.loads((ROOT/"configs/splits/scifact_train_dev_v1.json").read_text());validate_gold(rows,config,set(map(str,split["train_ids"])),set(map(str,split["dev_ids"])))
    if sha256(Path(config["training_data"]["path"]))!=config["training_data"]["sha256"]:raise ValueError("Training data hash mismatch")
    return config,rows,split


def preflight(args,config,rows):
    tokenizer=AutoTokenizer.from_pretrained(config["local_model_path"],local_files_only=True);encoded=[encode_training_row(row,tokenizer,config["optimization"]["max_sequence_length"]) for row in rows]
    folds=make_grouped_folds(rows,config["cross_validation"]["folds"],config["cross_validation"]["seed"]);payload={"version":"m1-gold-cv-folds-v1","seed":config["cross_validation"]["seed"],"group":"claim_id","folds":[{"fold":i,"validation_claim_ids":ids} for i,ids in enumerate(folds)]}
    fold_path=ROOT/config["cross_validation"]["fold_file"]
    if fold_path.exists() and json.loads(fold_path.read_text())!=payload:raise FileExistsError(fold_path)
    fold_path.write_text(json.dumps(payload,indent=2)+"\n")
    names=[name for name,_ in AutoModelForCausalLM.from_pretrained(config["local_model_path"],local_files_only=True,torch_dtype=torch.bfloat16).named_modules()]
    for target in config["lora"]["target_modules"]:
        if not any(name.endswith(target) for name in names):raise ValueError(f"Missing target module {target}")
    result={"rows":len(rows),"unique_claims":len({r['claim_id'] for r in rows}),"max_sequence_tokens":max(x["length"] for x in encoded),"max_answer_tokens":max(x["answer_tokens"] for x in encoded),"fold_sizes":[sum(r['claim_id'] in set(ids) for r in rows) for ids in folds],"fold_claim_counts":[len(x) for x in folds],"target_modules":config["lora"]["target_modules"],"benchmark_loaded":False}
    args.run_root.mkdir(parents=True,exist_ok=True);(args.run_root/"preflight.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


def run_cv(args,config,rows):
    cvroot=args.run_root/"cv"
    if cvroot.exists():raise FileExistsError(cvroot)
    fold_data=json.loads((ROOT/config["cross_validation"]["fold_file"]).read_text())
    for fold in fold_data["folds"]:
        valid_ids=set(fold["validation_claim_ids"]);train=[r for r in rows if r["claim_id"] not in valid_ids];valid=[r for r in rows if r["claim_id"] in valid_ids]
        train_run(config,train,valid,config["cross_validation"]["seed"],3,cvroot/f"fold-{fold['fold']}",args.checkpoint_root/"cv"/f"fold-{fold['fold']}",set(config["optimization"]["candidate_epochs"]))
    aggregate={}
    for epoch in config["optimization"]["candidate_epochs"]:
        values=[];times=[]
        for fold in range(5):
            values.append(json.loads((cvroot/f"fold-{fold}"/f"epoch-{epoch}"/"metrics.json").read_text()))
        aggregate[str(epoch)]={"fold_macro_f1":[x["macro_f1"] for x in values],"mean_macro_f1":float(np.mean([x["macro_f1"] for x in values])),"fold_accuracy":[x["accuracy"] for x in values],"mean_accuracy":float(np.mean([x["accuracy"] for x in values])),"total_training_seconds":sum(x["cumulative_training_seconds"] for x in values),"total_gpu_hours":sum(x["cumulative_training_seconds"] for x in values)/3600}
    best=max(config["optimization"]["candidate_epochs"],key=lambda e:aggregate[str(e)]["mean_macro_f1"])
    other=min(config["optimization"]["candidate_epochs"])
    if best!=other and aggregate[str(best)]["mean_macro_f1"]-aggregate[str(other)]["mean_macro_f1"]<=config["cross_validation"]["tie_margin"]:best=other
    summary={"candidates":aggregate,"selected_epochs":best,"selection_rule":config["cross_validation"]};(args.run_root/"cv-summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))


def run_final(args,config,rows):
    summary=json.loads((args.run_root/"cv-summary.json").read_text());epochs=summary["selected_epochs"]
    train_run(config,rows,[],args.seed,epochs,args.run_root/"dev-finalists"/f"seed-{args.seed}",args.checkpoint_root/"dev-finalists"/f"seed-{args.seed}",{epochs},merge=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("preflight","cv","final"));p.add_argument("--seed",type=int);p.add_argument("--config",type=Path,default=ROOT/"configs/posttraining/train-m1-lora-gold-v1.json");p.add_argument("--run-root",type=Path,default=Path.home()/"rag/runs/posttraining/lora-gold-v1");p.add_argument("--checkpoint-root",type=Path,default=Path.home()/"rag/checkpoints/posttraining/m1");args=p.parse_args();config,rows,_=load_context(args)
    if args.phase=="preflight":preflight(args,config,rows)
    elif args.phase=="cv":run_cv(args,config,rows)
    else:
        if args.seed not in config["final_training"]["seeds"]:raise ValueError("Seed is not frozen")
        run_final(args,config,rows)


if __name__=="__main__":main()
