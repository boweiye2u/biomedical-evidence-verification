"""Build and score frozen sentence passages for Milestone 5C on DEV only."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from retrieval.passages import rank_passages,rationale_pairs,select_gold_rationale,sentence_passages,validate_sentence_preservation
from retrieval.verification import validate_dev_only,validate_exact_ids
ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone5c-passage-v1"
def load(p): return json.loads(p.read_text())
def save(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2)+"\n")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def score(model,tokenizer,pairs,batch_size):
 out=[]
 for start in range(0,len(pairs),batch_size):
  tokens=tokenizer(pairs[start:start+batch_size],truncation=True,padding=True,max_length=512,return_tensors="pt").to(model.device)
  with torch.inference_mode(): values=model(**tokens).logits.squeeze(1)
  out.extend(values.float().cpu().tolist())
 return out

def main():
 RUN.mkdir(parents=True,exist_ok=False)
 config=load(ROOT/"configs/passage-localization-5c-v1.json"); split=load(ROOT/"configs/splits/scifact_train_dev_v1.json")
 dev_ids=sorted(map(str,split["dev_ids"]),key=int); validate_dev_only(dev_ids,dev_ids); assert len(dev_ids)==162
 data=ART/"data/scifact/original"
 corpus={str(x["doc_id"]):x for x in map(json.loads,(data/"corpus.jsonl").read_text().splitlines())}
 mapping_records=load(ART/"runs/milestone5a-verification-v1/verification-mapping.json")["records"]
 mapping={x["claim_id"]:x for x in mapping_records}; validate_exact_ids(dev_ids,mapping,"verification mapping")
 raw=load(ART/config["document_retriever"]["rankings"]); validate_exact_ids(dev_ids,raw,"BGE DEV rankings")
 bge={q:[str(doc) for doc,_ in sorted(raw[q].items(),key=lambda x:-x[1])] for q in dev_ids}
 depth=config["document_retriever"]["candidate_depth"]
 used_docs=sorted({doc for q in dev_ids for doc in bge[q][:depth]},key=int)
 segmentation={}
 for doc_id in used_docs:
  passages=sentence_passages(corpus[doc_id]); validate_sentence_preservation(corpus[doc_id],passages)
  segmentation[doc_id]={"title":corpus[doc_id]["title"],"sentences":[{"sentence_index":x["sentence_index"],"text":x["text"]} for x in passages]}
 # Validate every original rationale index and frozen sentence text against the corpus.
 rationale_records=[]
 for q in dev_ids:
  item=mapping[q]
  for doc_id,sets in item["annotated_evidence"].items():
   for evidence_set in sets:
    for index,text in zip(evidence_set["sentence_ids"],evidence_set.get("sentence_texts",[])):
     assert corpus[doc_id]["abstract"][index]==text
  selected=select_gold_rationale(item,corpus)
  rationale_records.append({"query_id":q,"all_annotated_rationale_pairs":[{"document_id":d,"sentence_index":i} for d,i in sorted(rationale_pairs(item),key=lambda x:(int(x[0]),x[1]))],"selected_gold_rationale":selected,"selection_rule":config["gold_rationale_selection"]["rule"]})
 model_path=ART/config["reranker"]["local_path"]
 tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
 model=AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True).to("cuda:0").eval()
 query_candidates={}; all_pairs=[]; keys=[]; lengths=[]
 for q in dev_ids:
  candidates=[]
  for document_rank,doc_id in enumerate(bge[q][:depth],1):
   for passage in sentence_passages(corpus[doc_id]):
    candidate={**passage,"bge_document_rank":document_rank}
    candidates.append(candidate)
    all_pairs.append([mapping[q]["claim"],(passage["title"]+" "+passage["text"]).strip()])
    keys.append((q,len(candidates)-1))
    lengths.append(len(tokenizer(all_pairs[-1][0],all_pairs[-1][1],truncation=False)["input_ids"]))
  query_candidates[q]=candidates
 # Deterministic label-free feasibility on the first five numeric DEV queries.
 feasibility_keys=[i for i,(q,_) in enumerate(keys) if q in set(dev_ids[:5])]
 feasibility_pairs=[all_pairs[i] for i in feasibility_keys]
 first=score(model,tokenizer,feasibility_pairs,config["reranker"]["batch_size"])
 second=score(model,tokenizer,feasibility_pairs,config["reranker"]["batch_size"])
 max_diff=max(abs(a-b) for a,b in zip(first,second)); assert max_diff<=config["feasibility"]["determinism_tolerance"]
 started=time.perf_counter(); scores=score(model,tokenizer,all_pairs,config["reranker"]["batch_size"]); elapsed=time.perf_counter()-started
 by_query={q:[] for q in dev_ids}
 for (q,index),value,length in zip(keys,scores,lengths):
  by_query[q].append({**query_candidates[q][index],"reranker_score":value,"pre_truncation_token_count":length,"truncated":length>512})
 records=[]
 eligible=0; top1_hits=0; top3_hits=0; selected_doc_hits=0
 for q in dev_ids:
  candidates=by_query[q]; ordered=rank_passages(candidates,[x["reranker_score"] for x in candidates])
  targets=rationale_pairs(mapping[q]); annotated_docs=set(mapping[q]["annotated_evidence_document_ids"])
  eligible+=bool(targets); top1_hits+=bool(targets and (ordered[0]["document_id"],ordered[0]["sentence_index"]) in targets)
  top3_hits+=bool(targets and any((x["document_id"],x["sentence_index"]) in targets for x in ordered[:3]))
  selected_doc_hits+=bool(targets and ordered[0]["document_id"] in annotated_docs)
  records.append({"query_id":q,"claim":mapping[q]["claim"],"bge_top3_document_ids":bge[q][:3],"annotated_evidence_document_ids":mapping[q]["annotated_evidence_document_ids"],"annotated_rationale_pairs":[{"document_id":d,"sentence_index":i} for d,i in sorted(targets,key=lambda x:(int(x[0]),x[1]))],"candidates":candidates,"ranked_passages":ordered,"selected_passage":ordered[0]})
 summary={"scope":"frozen DEV only","query_count":len(dev_ids),"unique_candidate_documents":len(used_docs),"candidate_passages":len(all_pairs),"scoring_seconds":elapsed,"feasibility":{"query_ids":dev_ids[:5],"pair_count":len(feasibility_pairs),"max_repeat_abs_difference":max_diff,"score_direction":"higher raw logit is more relevant","labels_consulted":False},"tokenization":{"max_length":512,"truncated_pair_count":sum(x>512 for x in lengths),"max_pre_truncation_tokens":max(lengths)},"rationale_diagnostics":{"eligible_claims":eligible,"top1_hit_count":top1_hits,"top1_hit_rate":top1_hits/eligible,"top3_hit_count":top3_hits,"top3_hit_rate":top3_hits/eligible,"selected_annotated_document_count":selected_doc_hits,"selected_annotated_document_rate":selected_doc_hits/eligible},"hashes":{"config":sha(ROOT/"configs/passage-localization-5c-v1.json"),"bge_rankings":sha(ART/config["document_retriever"]["rankings"]),"model_config":sha(model_path/"config.json")},"test_qrels_or_labels_loaded":False}
 save(RUN/"sentence-segmentation.json",{"method":config["sentence_segmentation"],"documents":segmentation})
 save(RUN/"rationale-mapping.json",{"records":rationale_records})
 save(RUN/"passage-rankings.json",records); save(RUN/"localization-summary.json",summary)
 print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
