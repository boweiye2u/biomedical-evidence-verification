"""Run the frozen Qwen verifier for P2 and G_rationale."""
from __future__ import annotations
import json,os,random
from pathlib import Path
import torch
from retrieval.passages import format_passage_evidence
from retrieval.reranking import assert_frozen_generator
from retrieval.verification import build_messages
from scripts.run_verification_5a import generate,model_and_tokenizer
ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone5c-passage-v1"
def load(p): return json.loads(p.read_text())
def build(condition,selected_by_query,mapping,tokenizer,config5):
 records=[]
 for item in mapping:
  q=item["claim_id"]; passages=selected_by_query[q]; document_ids=list(dict.fromkeys(str(x["document_id"]) for x in passages))
  evidence,counts=format_passage_evidence(passages,tokenizer,config5["generation"]["max_evidence_tokens"])
  rendered=tokenizer.apply_chat_template(build_messages(item["claim"],evidence,config5["final_selection"]["prompt_name"]),tokenize=False,add_generation_prompt=True)
  prompt_tokens=len(tokenizer.encode(rendered,add_special_tokens=False)); assert prompt_tokens<=config5["generation"]["max_input_tokens"]
  rationale={(str(doc),int(i)) for doc,sets in item["annotated_evidence"].items() for s in sets for i in s["sentence_ids"]}
  selected_pairs={(str(x["document_id"]),int(x["sentence_index"])) for x in passages}; annotated_docs=set(item["annotated_evidence_document_ids"])
  records.append({"query_id":q,"gold_label":item["label"],"condition":condition,"top_k":1,"prompt_name":config5["final_selection"]["prompt_name"],"document_ids":document_ids,"selected_passages":passages,"annotated_evidence_document_ids":sorted(annotated_docs,key=int),"annotated_rationale_pairs":[{"document_id":d,"sentence_index":i} for d,i in sorted(rationale,key=lambda x:(int(x[0]),x[1]))],"annotated_rationale_localized":bool(selected_pairs&rationale) if rationale else None,"any_annotated_evidence_retrieved":bool(set(document_ids)&annotated_docs) if annotated_docs else None,"document_count":len(document_ids),"evidence_token_count_before":counts["before"],"evidence_token_count_after":counts["after"],"evidence_truncated":counts["truncated"],"prompt_token_count":prompt_tokens,"rendered_prompt":rendered})
 return records

def main():
 config5=load(ROOT/"configs/verification-5a-v1.json"); config=load(ROOT/"configs/passage-localization-5c-v1.json"); assert_frozen_generator(config5,config["frozen_generator"])
 mapping=load(ART/"runs/milestone5a-verification-v1/verification-mapping.json")["records"]
 passages={x["query_id"]:[x["selected_passage"]] for x in load(RUN/"passage-rankings.json")}
 rationale={x["query_id"]:x["selected_gold_rationale"] for x in load(RUN/"rationale-mapping.json")["records"]}
 random.seed(config5["generation"]["seed"]); torch.manual_seed(config5["generation"]["seed"]); torch.cuda.manual_seed_all(config5["generation"]["seed"])
 model,tokenizer=model_and_tokenizer(config5); results=[]
 for condition,selected in [("P2",passages),("G_rationale",rationale)]:
  records=build(condition,selected,mapping,tokenizer,config5)
  results.append(generate(records,RUN/"verification"/condition/"generations.jsonl",model,tokenizer,config5))
 print(json.dumps(results,indent=2))
if __name__=="__main__": main()
