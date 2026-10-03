"""Run frozen Milestone 5A Qwen verifier on the R2 reranked top-1 evidence."""
from __future__ import annotations
import json,os,random
from pathlib import Path
import torch
from retrieval.reranking import assert_frozen_generator
from retrieval.verification import build_messages,format_evidence
from scripts.run_verification_5a import generate,model_and_tokenizer
ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone5b-reranking-v1"
def load(p): return json.loads(p.read_text())
def main():
    config5=load(ROOT/"configs/verification-5a-v1.json"); config=load(ROOT/"configs/reranking-5b-v1.json"); assert_frozen_generator(config5,config["frozen_generator"])
    mapping=load(ART/"runs/milestone5a-verification-v1/verification-mapping.json")["records"]
    rankings=load(RUN/"rankings.json")["R2"]
    corpus={str(x["doc_id"]):x for x in map(json.loads,(ART/"data/scifact/original/corpus.jsonl").read_text().splitlines())}
    random.seed(config5["generation"]["seed"]); torch.manual_seed(config5["generation"]["seed"]); torch.cuda.manual_seed_all(config5["generation"]["seed"])
    model,tokenizer=model_and_tokenizer(config5); records=[]
    selected=config5["final_selection"]; assert selected["top_k"]==1
    for item in mapping:
        q=item["claim_id"]; document_ids=rankings[q][:1]; documents=[corpus[x] for x in document_ids]
        evidence,counts=format_evidence(documents,tokenizer,config5["generation"]["max_evidence_tokens"])
        rendered=tokenizer.apply_chat_template(build_messages(item["claim"],evidence,selected["prompt_name"]),tokenize=False,add_generation_prompt=True)
        prompt_tokens=len(tokenizer.encode(rendered,add_special_tokens=False)); assert prompt_tokens<=config5["generation"]["max_input_tokens"]
        annotated=set(item["annotated_evidence_document_ids"])
        records.append({"query_id":q,"gold_label":item["label"],"condition":"R2","top_k":1,"prompt_name":selected["prompt_name"],"document_ids":document_ids,"annotated_evidence_document_ids":sorted(annotated,key=int),"any_annotated_evidence_retrieved":bool(annotated.intersection(document_ids)) if annotated else None,"document_count":1,"evidence_token_count_before":counts["before"],"evidence_token_count_after":counts["after"],"evidence_truncated":counts["truncated"],"prompt_token_count":prompt_tokens,"rendered_prompt":rendered})
    result=generate(records,RUN/"verification/R2/generations.jsonl",model,tokenizer,config5); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
