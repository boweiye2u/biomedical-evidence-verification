"""One-time frozen zero-shot BGE retrieval on official SciFact TEST."""
from __future__ import annotations
import csv,hashlib,io,json,os,tarfile,time,zipfile
from collections import Counter
from pathlib import Path
import faiss,numpy as np,torch
from retrieval.bge import BGE,MODEL,PREFIX
from retrieval.final_test import annotation_availability,validate_test_partition
from retrieval.metrics import evaluate
from retrieval.verification import map_scifact_annotation,validate_exact_ids
ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone6a-test-v1"
def load(p): return json.loads(p.read_text())
def save_new(p,x):
 if p.exists(): raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2)+"\n")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 config=load(ROOT/"configs/final-test-6a-v1.json"); pre=load(RUN/"pre-run-manifest.json")
 assert pre["test_records_loaded"] is False
 current_config_sha=sha(ROOT/"configs/final-test-6a-v1.json")
 if pre["config_sha256"]!=current_config_sha:
  correction=load(RUN/"implementation-corrections.json")
  assert correction["original_config_sha256"]==pre["config_sha256"] and correction["corrected_config_sha256"]==current_config_sha and correction["modeling_changes"] is False
 beir=ART/config["sources"]["beir_archive"]; original=ART/config["sources"]["original_archive"]
 assert sha(beir)==config["sources"]["beir_archive_sha256"] and sha(original)==config["sources"]["original_archive_sha256"]
 with zipfile.ZipFile(beir) as z:
  qrel_text=io.TextIOWrapper(z.open(config["sources"]["beir_qrels_member"]),encoding="utf-8")
  qrels={}
  for row in csv.DictReader(qrel_text,delimiter="\t"):
   q=str(row["query-id"]); qrels.setdefault(q,{})[str(row["corpus-id"])]=int(row["score"])
  queries={str(x["_id"]):x["text"] for x in map(json.loads,io.TextIOWrapper(z.open(config["sources"]["beir_queries_member"]),encoding="utf-8")) if str(x["_id"]) in qrels}
 with tarfile.open(original,"r:gz") as t:
  handle=t.extractfile(config["sources"]["original_test_claims_member"]); assert handle is not None
  official=[json.loads(line) for line in io.TextIOWrapper(handle,encoding="utf-8") if line.strip()]
 claims={str(x["id"]):x for x in official}; test_ids=sorted(qrels,key=int)
 validate_exact_ids(test_ids,queries,"BEIR TEST queries"); validate_exact_ids(test_ids,claims,"original SciFact TEST claims")
 split=load(ROOT/"configs/splits/scifact_train_dev_v1.json"); validate_test_partition(test_ids,list(map(str,split["train_ids"]))+list(map(str,split["dev_ids"])))
 for q in test_ids:
  if queries[q].strip()!=claims[q]["claim"].strip(): raise ValueError(f"BEIR/original claim mismatch {q}")
 available,reason=annotation_availability(official)
 corpus_beir={str(x["_id"]):x for x in map(json.loads,(ART/"data/scifact/beir/corpus.jsonl").read_text().splitlines())}
 corpus_original={str(x["doc_id"]):x for x in map(json.loads,(ART/"data/scifact/original/corpus.jsonl").read_text().splitlines())}
 if set(corpus_beir)!=set(corpus_original) or len(corpus_beir)!=config["retriever"]["corpus_size"]: raise ValueError("Corpus alignment failure")
 doc_ids=sorted(corpus_beir,key=int); cache=ART/config["retriever"]["embedding_cache"]
 metadata=load(cache/"metadata.json"); assert metadata["model"]==MODEL and metadata["revision"]==config["retriever"]["revision"] and metadata["query_prefix"]==PREFIX and metadata["doc_ids"]==doc_ids
 embeddings=np.load(cache/"embeddings.npy"); assert embeddings.shape==(5183,768) and embeddings.dtype==np.float32
 np.testing.assert_allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-5)
 torch.set_num_threads(4); faiss.omp_set_num_threads(4); torch.manual_seed(config["verifier"]["seed"]); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 bge=BGE(config["retriever"]["revision"]); index=faiss.IndexFlatIP(768); index.add(embeddings)
 sample=bge.encode([queries[q] for q in test_ids[:5]],query=True); repeat=bge.encode([queries[q] for q in test_ids[:5]],query=True); np.testing.assert_allclose(sample,repeat,atol=1e-7,rtol=1e-6)
 started=time.perf_counter(); query_embeddings=bge.encode([queries[q] for q in test_ids],query=True); values,indices=index.search(query_embeddings,100); elapsed=time.perf_counter()-started
 rankings={}; cosine={}
 for q,vs,ids in zip(test_ids,values,indices):
  order=[doc_ids[i] for i in ids]; rankings[q]=order; cosine[q]={doc:float(v) for doc,v in zip(order,vs)}
 metrics,per_query=evaluate(qrels,rankings)
 positive_ranks={}; all_ranks=[]
 for q in test_ids:
  pos={d for d,v in qrels[q].items() if v>0}; ranks=[rankings[q].index(d)+1 if d in rankings[q] else ">100" for d in sorted(pos,key=int)]; positive_ranks[q]=ranks; all_ranks.extend(ranks)
 mapped=[]
 if available:
  for q in test_ids:
   item=map_scifact_annotation(claims[q])
   for doc,sets in item["annotated_evidence"].items():
    for evidence_set in sets: evidence_set["sentence_texts"]=[corpus_original[doc]["abstract"][i] for i in evidence_set["sentence_ids"]]
   mapped.append(item)
 else:
  mapped=[{"claim_id":q,"claim":claims[q]["claim"],"label":None,"annotated_evidence":None,"annotated_evidence_document_ids":None,"cited_document_ids":[str(x) for x in claims[q].get("cited_doc_ids",[])]} for q in test_ids]
 counts=[sum(v>0 for v in qrels[q].values()) for q in test_ids]
 summary={"scope":"official held-out SciFact TEST","query_count":len(test_ids),"corpus_size":len(doc_ids),"metrics":metrics,"top1_known_positive_rate":sum(any(d==rankings[q][0] and rel>0 for d,rel in qrels[q].items()) for q in test_ids)/len(test_ids),"known_relevant_documents_per_query":{"min":min(counts),"median":float(np.median(counts)),"max":max(counts),"mean":float(np.mean(counts)),"distribution":dict(Counter(counts))},"positive_rank_distribution":{"at_1":sum(r==1 for r in all_ranks),"at_2_to_10":sum(isinstance(r,int) and 2<=r<=10 for r in all_ranks),"at_11_to_100":sum(isinstance(r,int) and 11<=r<=100 for r in all_ranks),"beyond_100":sum(r==">100" for r in all_ranks),"total_known_positives":len(all_ranks)},"query_inference_seconds":elapsed,"repeat_max_abs_difference":float(np.max(np.abs(sample-repeat))),"verification_labels_available":available,"verification_availability_reason":reason,"test_access_performed":True,"post_test_tuning_performed":False,"original_annotation_member_used":config["sources"]["original_test_claims_member"],"implementation_path_correction":True}
 save_new(RUN/"test-query-ids.json",test_ids); save_new(RUN/"retrieval-rankings.json",rankings); save_new(RUN/"retrieval-cosine-scores.json",cosine); save_new(RUN/"retrieval-per-query.json",per_query); save_new(RUN/"positive-ranks.json",positive_ranks); save_new(RUN/"test-claim-mapping.json",{"verification_labels_available":available,"reason":reason,"records":mapped}); save_new(RUN/"retrieval-summary.json",summary)
 print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
