"""Score frozen BGE candidates with MedCPT Cross-Encoder on DEV only."""
from __future__ import annotations
import csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from retrieval.metrics import evaluate
from retrieval.reranking import build_pairs,reranked_order,validate_candidate_set
from retrieval.verification import validate_dev_only,validate_exact_ids

ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone5b-reranking-v1"
def load(p): return json.loads(p.read_text())
def save(p,x): p.write_text(json.dumps(x,indent=2)+"\n")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def score_pairs(model,tokenizer,pairs,batch_size):
    scores=[]
    for start in range(0,len(pairs),batch_size):
        encoded=tokenizer(pairs[start:start+batch_size],truncation=True,padding=True,max_length=512,return_tensors="pt").to(model.device)
        with torch.inference_mode(): values=model(**encoded).logits.squeeze(1)
        scores.extend(values.float().cpu().tolist())
    return scores

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    config=load(ROOT/"configs/reranking-5b-v1.json"); split=load(ROOT/"configs/splits/scifact_train_dev_v1.json")
    dev_ids=sorted(map(str,split["dev_ids"]),key=int); validate_dev_only(dev_ids,dev_ids); assert len(dev_ids)==162
    data=ART/"data/scifact"; claims={str(x["id"]):x for x in map(json.loads,(data/"original/claims_train.jsonl").read_text().splitlines()) if str(x["id"]) in set(dev_ids)}
    corpus={str(x["doc_id"]):x for x in map(json.loads,(data/"original/corpus.jsonl").read_text().splitlines())}
    validate_exact_ids(dev_ids,claims,"DEV claims")
    raw=load(ART/config["primary_retriever"]["rankings"]); validate_exact_ids(dev_ids,raw,"BGE rankings")
    bge_order={q:[doc for doc,_ in sorted(raw[q].items(),key=lambda item:-item[1])] for q in dev_ids}
    assert all(len(x)==100 and len(set(x))==100 for x in bge_order.values())
    mapping={x["claim_id"]:x for x in load(ART/"runs/milestone5a-verification-v1/verification-mapping.json")["records"]}
    model_path=ART/config["reranker"]["local_path"]
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True); model=AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True).to("cuda:0").eval()
    depth=config["candidate_depth"]["diagnostic"]
    pairs=[]; pair_keys=[]; pre_lengths=[]
    for q in dev_ids:
        docs=[corpus[d] for d in bge_order[q][:depth]]; current=build_pairs(claims[q]["claim"],docs)
        for rank,(pair,doc) in enumerate(zip(current,docs),1):
            pairs.append(pair); pair_keys.append((q,str(doc["doc_id"]),rank)); pre_lengths.append(len(tokenizer(pair[0],pair[1],truncation=False)["input_ids"]))
    # Feasibility uses the first five numeric DEV IDs, independent of labels.
    feasibility_n=5*depth; first=score_pairs(model,tokenizer,pairs[:feasibility_n],config["reranker"]["batch_size"]); second=score_pairs(model,tokenizer,pairs[:feasibility_n],config["reranker"]["batch_size"])
    max_diff=max(abs(a-b) for a,b in zip(first,second)); assert max_diff<=config["feasibility"]["determinism_tolerance"]
    feasibility=[]
    for offset,q in enumerate(dev_ids[:5]):
        ids=bge_order[q][:depth]; values=first[offset*depth:(offset+1)*depth]; order=reranked_order(ids,values); validate_candidate_set(ids,order)
        feasibility.append({"query_id":q,"claim":claims[q]["claim"],"candidate_ids":ids,"reranked_ids":order,"scores":dict(zip(ids,values))})
    save(RUN/"feasibility.json",{"query_ids":dev_ids[:5],"selection":"first five numeric DEV IDs; labels not consulted","pair_format":config["reranker"]["pair_format"],"score_direction":"higher is more relevant, per official model card","max_repeat_abs_difference":max_diff,"candidate_sets_preserved":True,"records":feasibility})
    start=time.perf_counter(); values=score_pairs(model,tokenizer,pairs,config["reranker"]["batch_size"]); elapsed=time.perf_counter()-start
    by_query={q:[] for q in dev_ids}
    for (q,doc,rank),score,length in zip(pair_keys,values,pre_lengths): by_query[q].append({"document_id":doc,"bge_rank":rank,"bge_saved_rank_score":float(raw[q][doc]),"reranker_score":score,"pre_truncation_token_count":length,"truncated":length>512})
    rankings={"R1":{},"R2":{},"R3":{}}; records=[]
    for q in dev_ids:
        candidates=by_query[q]; ids=[x["document_id"] for x in candidates]; scores=[x["reranker_score"] for x in candidates]
        order20=reranked_order(ids,scores); order10=reranked_order(ids[:10],scores[:10]); validate_candidate_set(ids[:10],order10); validate_candidate_set(ids,order20)
        r1=bge_order[q]; r2=order10+r1[10:]; r3=order20+r1[20:]
        assert len(r2)==len(set(r2))==100 and len(r3)==len(set(r3))==100
        rankings["R1"][q]=r1; rankings["R2"][q]=r2; rankings["R3"][q]=r3
        rank20={doc:i+1 for i,doc in enumerate(order20)}
        for x in candidates: x["reranked_rank_within_top20"]=rank20[x["document_id"]]
        records.append({"query_id":q,"claim":claims[q]["claim"],"beir_known_positive_ids":mapping[q]["cited_document_ids"],"annotated_evidence_document_ids":mapping[q]["annotated_evidence_document_ids"],"candidates":candidates,"R1_top1":r1[0],"R2_top1":r2[0],"R3_top1":r3[0],"R2_order_top10":r2[:10],"R3_order_top20":r3[:20]})
    qrels={q:{} for q in dev_ids}
    qrels_path=data/"beir/qrels/train.tsv"; assert qrels_path.name=="train.tsv"
    for row in csv.DictReader(qrels_path.open(),delimiter="\t"):
        if row["query-id"] in qrels: qrels[row["query-id"]][row["corpus-id"]]=int(row["score"])
    metrics={}; per_query={}
    for condition in rankings:
        metrics[condition],per_query[condition]=evaluate(qrels,rankings[condition])
    eligible=[q for q in dev_ids if mapping[q]["annotated_evidence_document_ids"]]
    def coverage(condition,k): return sum(bool(set(rankings[condition][q][:k])&set(mapping[q]["annotated_evidence_document_ids"])) for q in eligible)/len(eligible)
    def changes(condition):
        out={"top1_unchanged":0,"annotated_evidence_promoted_to_top1":0,"annotated_evidence_demoted_from_top1":0,"beir_positive_promoted_to_top1":0,"beir_positive_demoted_from_top1":0}
        for q in dev_ids:
            before,after=rankings["R1"][q][0],rankings[condition][q][0]
            if before==after: out["top1_unchanged"]+=1
            ann=set(mapping[q]["annotated_evidence_document_ids"]); pos=set(mapping[q]["cited_document_ids"])
            out["annotated_evidence_promoted_to_top1"]+=before not in ann and after in ann
            out["annotated_evidence_demoted_from_top1"]+=before in ann and after not in ann
            out["beir_positive_promoted_to_top1"]+=before not in pos and after in pos
            out["beir_positive_demoted_from_top1"]+=before in pos and after not in pos
        return out
    summary={"scope":"frozen DEV only","query_count":len(dev_ids),"candidate_pairs":len(pairs),"scoring_seconds":elapsed,"score_direction":"higher raw logit is more relevant","max_length":512,"truncated_pair_count":sum(x>512 for x in pre_lengths),"max_pre_truncation_tokens":max(pre_lengths),"retrieval_metrics":metrics,"annotated_evidence_coverage":{c:{"top1":coverage(c,1),"top3":coverage(c,3)} for c in rankings},"changes_vs_R1":{"R2":changes("R2"),"R3":changes("R3")},"bge_score_semantics":config["primary_retriever"]["saved_score_semantics"],"test_qrels_or_labels_loaded":False,"hashes":{"config":sha(ROOT/"configs/reranking-5b-v1.json"),"bge_rankings":sha(ART/config["primary_retriever"]["rankings"]),"model_config":sha(model_path/"config.json")}}
    save(RUN/"per-query-reranking.json",records); save(RUN/"rankings.json",rankings); save(RUN/"retrieval-per-query.json",per_query); save(RUN/"reranking-summary.json",summary)
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
