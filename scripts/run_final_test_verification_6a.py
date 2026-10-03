"""Run frozen Qwen T1 and conditional T_gold on official SciFact TEST."""
from __future__ import annotations
import json,os,random
from pathlib import Path
import torch
from retrieval.reranking import assert_frozen_generator
from retrieval.verification import build_messages,format_evidence
from scripts.run_verification_5a import generate,model_and_tokenizer
ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone6a-test-v1"
def load(p): return json.loads(p.read_text())
def save_new(p,x):
 if p.exists(): raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2)+"\n")
def build(condition,rankings,mapping,corpus,tokenizer,config5):
 records=[]; prompt_meta=[]
 for item in mapping:
  q=item["claim_id"]; document_ids=rankings[q][:1]; documents=[corpus[x] for x in document_ids]
  evidence,counts=format_evidence(documents,tokenizer,config5["generation"]["max_evidence_tokens"]); rendered=tokenizer.apply_chat_template(build_messages(item["claim"],evidence,config5["final_selection"]["prompt_name"]),tokenize=False,add_generation_prompt=True); prompt_tokens=len(tokenizer.encode(rendered,add_special_tokens=False)); assert prompt_tokens<=config5["generation"]["max_input_tokens"]
  ann=item["annotated_evidence_document_ids"]; retrieved=None if ann is None else (bool(set(document_ids)&set(ann)) if ann else None)
  record={"query_id":q,"gold_label":item["label"],"verification_labels_available":item["label"] is not None,"condition":condition,"top_k":1,"prompt_name":config5["final_selection"]["prompt_name"],"document_ids":document_ids,"annotated_evidence_document_ids":ann,"any_annotated_evidence_retrieved":retrieved,"document_count":len(document_ids),"evidence_token_count_before":counts["before"],"evidence_token_count_after":counts["after"],"evidence_truncated":counts["truncated"],"prompt_token_count":prompt_tokens,"rendered_prompt":rendered}
  records.append(record); prompt_meta.append({k:v for k,v in record.items() if k!="rendered_prompt"})
 return records,prompt_meta

def main():
 config=load(ROOT/"configs/final-test-6a-v1.json"); config5=load(ROOT/"configs/verification-5a-v1.json"); frozen={"model_name":config["verifier"]["model_name"],"model_revision":config["verifier"]["model_revision"],"precision":config["verifier"]["precision"],"prompt_name":config["verifier"]["prompt_name"],"prompt_sha256":config["verifier"]["prompt_sha256"],"top_k":config["verifier"]["evidence_depth"],"do_sample":config["verifier"]["do_sample"],"max_new_tokens":config["verifier"]["max_new_tokens"],"seed":config["verifier"]["seed"]}; assert_frozen_generator(config5,frozen)
 mapping_payload=load(RUN/"test-claim-mapping.json"); mapping=mapping_payload["records"]; rankings=load(RUN/"retrieval-rankings.json"); corpus={str(x["doc_id"]):x for x in map(json.loads,(ART/"data/scifact/original/corpus.jsonl").read_text().splitlines())}
 random.seed(config5["generation"]["seed"]); torch.manual_seed(config5["generation"]["seed"]); torch.cuda.manual_seed_all(config5["generation"]["seed"]); model,tokenizer=model_and_tokenizer(config5)
 conditions=[("T1",rankings)]
 if mapping_payload["verification_labels_available"]:
  gold={x["claim_id"]:x["annotated_evidence_document_ids"] for x in mapping}; conditions.append(("T_gold",gold))
 results=[]
 for condition,order in conditions:
  records,meta=build(condition,order,mapping,corpus,tokenizer,config5); save_new(RUN/"verification"/condition/"prompt-context-metadata.json",meta); out=RUN/"verification"/condition/"generations.jsonl"; results.append(generate(records,out,model,tokenizer,config5)); rows=[json.loads(x) for x in out.read_text().splitlines()]; save_new(RUN/"verification"/condition/"output-token-counts.json",{x["query_id"]:len(tokenizer.encode(x["raw_output"],add_special_tokens=False)) for x in rows})
 print(json.dumps({"conditions":[x[0] for x in conditions],"verification_labels_available":mapping_payload["verification_labels_available"],"results":results},indent=2))
if __name__=="__main__": main()
