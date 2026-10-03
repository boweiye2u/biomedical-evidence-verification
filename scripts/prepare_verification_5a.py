"""Prepare frozen DEV mappings and evidence conditions for Milestone 5A."""
from __future__ import annotations
import hashlib,json,os
from collections import Counter
from pathlib import Path
from retrieval.verification import map_scifact_annotation,normalize_ranking,validate_dev_only,validate_exact_ids

ROOT=Path(__file__).resolve().parents[1]
ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag")))
RUN=ART/"runs/milestone5a-verification-v1"

def load(path): return json.loads(path.read_text())
def save(path,value): path.write_text(json.dumps(value,indent=2)+"\n")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    config_path=ROOT/"configs/verification-5a-v1.json"
    config=load(config_path)
    split_path=ROOT/"configs/splits/scifact_train_dev_v1.json"
    split=load(split_path); dev_ids=sorted(map(str,split["dev_ids"]),key=int)
    validate_dev_only(dev_ids,dev_ids); assert len(dev_ids)==162
    data=ART/"data/scifact"
    claims_path=data/"original/claims_train.jsonl"
    corpus_path=data/"original/corpus.jsonl"
    assert claims_path.name=="claims_train.jsonl" and corpus_path.name=="corpus.jsonl"
    claims={str(row["id"]):row for row in map(json.loads,claims_path.read_text().splitlines()) if str(row["id"]) in set(dev_ids)}
    validate_exact_ids(dev_ids,claims,"original SciFact DEV claims")
    corpus={str(row["doc_id"]):row for row in map(json.loads,corpus_path.read_text().splitlines())}
    mapped=[]
    for claim_id in dev_ids:
        record=map_scifact_annotation(claims[claim_id])
        for document_id,evidence_sets in record["annotated_evidence"].items():
            if document_id not in corpus: raise ValueError(f"Missing evidence document {document_id}")
            abstract=corpus[document_id]["abstract"]
            for evidence_set in evidence_sets:
                for sentence_id in evidence_set["sentence_ids"]:
                    if sentence_id<0 or sentence_id>=len(abstract): raise ValueError(f"Invalid sentence {sentence_id} in {document_id}")
                evidence_set["sentence_texts"]=[abstract[index] for index in evidence_set["sentence_ids"]]
        mapped.append(record)
    counts=Counter(item["label"] for item in mapped)
    if counts!={"SUPPORT":70,"CONTRADICT":28,"INSUFFICIENT":64}: raise ValueError(counts)

    d1_path=ART/config["conditions"]["D1"]["rankings"]
    d2_path=ART/config["conditions"]["D2"]["rankings"]
    rankings={}
    for name,path in [("D1",d1_path),("D2",d2_path)]:
        raw=load(path); validate_exact_ids(dev_ids,raw,f"{name} rankings")
        rankings[name]={query_id:normalize_ranking(raw[query_id])[:100] for query_id in dev_ids}
        if not all(len(value)==100 for value in rankings[name].values()): raise ValueError(f"{name} ranking shorter than 100")
    rankings["D3"]={item["claim_id"]:item["annotated_evidence_document_ids"] for item in mapped}

    save(RUN/"verification-mapping.json",{
      "scope":"frozen DEV only","claim_count":len(mapped),"label_counts":dict(counts),
      "claims_with_annotated_evidence":sum(bool(x["annotated_evidence"]) for x in mapped),
      "claims_without_annotated_evidence":sum(not x["annotated_evidence"] for x in mapped),
      "annotated_evidence_document_assignments":sum(len(x["annotated_evidence"]) for x in mapped),
      "annotated_evidence_sets":sum(sum(len(v) for v in x["annotated_evidence"].values()) for x in mapped),
      "records":mapped,"test_labels_loaded":False,
    })
    save(RUN/"evidence-condition-rankings.json",rankings)
    save(RUN/"preparation-manifest.json",{
      "config_sha256":sha(config_path),"split_sha256":sha(split_path),"claims_train_sha256":sha(claims_path),
      "original_corpus_sha256":sha(corpus_path),"d1_rankings_sha256":sha(d1_path),"d2_rankings_sha256":sha(d2_path),
      "model_revision":config["model"]["revision"],"test_qrels_or_labels_loaded":False,
    })
    print(json.dumps({"run":str(RUN),"label_counts":dict(counts),"mapping_unambiguous":True},indent=2))

if __name__=="__main__": main()
