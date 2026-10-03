"""Run frozen Qwen verification inference for Milestone 5A."""
from __future__ import annotations
import argparse,json,os,random,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from retrieval.verification import build_messages,format_evidence

ROOT=Path(__file__).resolve().parents[1]
ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag")))
RUN=ART/"runs/milestone5a-verification-v1"

def load(path): return json.loads(path.read_text())
def read_jsonl(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def model_and_tokenizer(config):
    path=ART/config["model"]["local_path"]
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    tokenizer.padding_side="left"; tokenizer.truncation_side="right"
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id=tokenizer.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(
        path,local_files_only=True,dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa",
    )
    model.generation_config.do_sample=False
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None
    model.eval()
    return model,tokenizer

def build_records(condition,top_k,prompt_name,tokenizer,limit=None):
    config=load(ROOT/"configs/verification-5a-v1.json")
    mapping=load(RUN/"verification-mapping.json")["records"]
    rankings=load(RUN/"evidence-condition-rankings.json")[condition]
    corpus={str(x["doc_id"]):x for x in map(json.loads,(ART/"data/scifact/original/corpus.jsonl").read_text().splitlines())}
    if limit:
        by_label={}
        for item in mapping: by_label.setdefault(item["label"],item)
        mapping=[by_label[label] for label in ["SUPPORT","CONTRADICT","INSUFFICIENT"]]
    records=[]
    for item in mapping:
        query_id=item["claim_id"]
        document_ids=rankings[query_id][:top_k]
        documents=[corpus[doc] for doc in document_ids]
        evidence_text,evidence_tokens=format_evidence(documents,tokenizer,config["generation"]["max_evidence_tokens"])
        messages=build_messages(item["claim"],evidence_text,prompt_name)
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        prompt_tokens=len(tokenizer.encode(rendered,add_special_tokens=False))
        if prompt_tokens>config["generation"]["max_input_tokens"]:
            raise ValueError(f"Prompt {query_id} has {prompt_tokens} tokens")
        annotated=set(item["annotated_evidence_document_ids"])
        records.append({
            "query_id":query_id,"gold_label":item["label"],"condition":condition,"top_k":top_k,
            "prompt_name":prompt_name,"document_ids":document_ids,"annotated_evidence_document_ids":sorted(annotated,key=int),
            "any_annotated_evidence_retrieved":bool(annotated.intersection(document_ids)) if annotated else None,
            "document_count":len(document_ids),"evidence_token_count_before":evidence_tokens["before"],
            "evidence_token_count_after":evidence_tokens["after"],"evidence_truncated":evidence_tokens["truncated"],
            "prompt_token_count":prompt_tokens,"rendered_prompt":rendered,
        })
    return records

def generate(records,output,model,tokenizer,config):
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True,exist_ok=True)
    batch_size=config["generation"]["batch_size"]
    started=time.perf_counter()
    with output.open("w") as handle:
        for start in range(0,len(records),batch_size):
            batch=records[start:start+batch_size]
            texts=[item.pop("rendered_prompt") for item in batch]
            tokens=tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                generated=model.generate(
                    **tokens,do_sample=False,max_new_tokens=config["generation"]["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
                )
            new=generated[:,tokens.input_ids.shape[1]:]
            outputs=tokenizer.batch_decode(new,skip_special_tokens=True)
            for item,raw in zip(batch,outputs):
                item["raw_output"]=raw.strip(); handle.write(json.dumps(item)+"\n")
            handle.flush()
            print(f"{output.parent.name}: {min(start+batch_size,len(records))}/{len(records)}",flush=True)
    return {"count":len(records),"seconds":time.perf_counter()-started,"output":str(output)}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=["feasibility","development","final"],required=True)
    args=parser.parse_args(); config=load(ROOT/"configs/verification-5a-v1.json")
    random.seed(config["generation"]["seed"]); torch.manual_seed(config["generation"]["seed"]); torch.cuda.manual_seed_all(config["generation"]["seed"])
    model,tokenizer=model_and_tokenizer(config); results=[]
    if args.phase=="feasibility":
        records=build_records("D1",3,"grounded-v2",tokenizer,limit=3)
        results.append(generate(records,RUN/"feasibility-v3/generations.jsonl",model,tokenizer,config))
    elif args.phase=="development":
        for prompt_name in sorted(config["development"]["prompt_candidates"]):
            for top_k in config["development"]["top_k_candidates"]:
                records=build_records("D1",top_k,prompt_name,tokenizer)
                output=RUN/"development"/prompt_name/f"top-{top_k}"/"generations.jsonl"
                results.append(generate(records,output,model,tokenizer,config))
    else:
        selected=config.get("final_selection")
        if not selected or config["status"]!="final_prompt_and_budget_frozen": raise ValueError("Final configuration has not been frozen")
        for condition in ["D1","D2","D3"]:
            records=build_records(condition,selected["top_k"],selected["prompt_name"],tokenizer)
            output=RUN/"final"/condition/"generations.jsonl"
            results.append(generate(records,output,model,tokenizer,config))
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
