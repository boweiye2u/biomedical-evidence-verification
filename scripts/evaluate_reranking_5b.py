"""Evaluate R2 and perform frozen cluster-aware paired DEV analysis."""
from __future__ import annotations
import json,os
from collections import Counter
from pathlib import Path
from retrieval.reranking import cluster_macro_f1_bootstrap,macro_f1
from retrieval.statistics import classify_differences,cluster_bootstrap_ci,cluster_sign_flip_test,validate_clusters
from retrieval.verification import LABELS
from scripts.evaluate_verification_5a import evaluate
ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone5b-reranking-v1"
def load(p): return json.loads(p.read_text())
def save_new(p,x):
    if p.exists(): raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2)+"\n")
def main():
    config=load(ROOT/"configs/reranking-5b-v1.json"); r2_metrics,r2_records=evaluate(RUN/"verification/R2/generations.jsonl")
    save_new(RUN/"verification/R2/evaluation.json",{"metrics":r2_metrics,"records":r2_records})
    r1_payload=load(ART/"runs/milestone5a-verification-v1/final/D1/evaluation.json"); gold_payload=load(ART/"runs/milestone5a-verification-v1/final/D3/evaluation.json")
    r1_records=r1_payload["records"]; assert len(r1_records)==len(r2_records)==162
    r1={x["query_id"]:x for x in r1_records}; r2={x["query_id"]:x for x in r2_records}; assert set(r1)==set(r2)
    split=load(ROOT/"configs/splits/scifact_train_dev_v1.json"); query_ids=sorted(r1,key=int); dev=set(query_ids); groups=[]
    for original in split["groups"]:
        group=sorted(dev.intersection(original),key=int)
        if group: groups.append(group)
    validate_clusters(query_ids,groups)
    differences=[float(r2[q]["correct"])-float(r1[q]["correct"]) for q in query_ids]
    primary={"metric":"per-query accuracy","mean_difference":sum(differences)/len(differences),"counts":classify_differences(differences),
      "cluster_bootstrap_95ci":cluster_bootstrap_ci(differences,query_ids,groups,replicates=config["paired_analysis"]["bootstrap_replicates"],seed=config["paired_analysis"]["bootstrap_seed"]),
      "cluster_sign_flip":cluster_sign_flip_test(differences,query_ids,groups,permutations=config["paired_analysis"]["sign_flip_permutations"],seed=config["paired_analysis"]["sign_flip_seed"])}
    gold=[r1[q]["gold_label"] for q in query_ids]; p1=[r1[q]["predicted_label"] for q in query_ids]; p2=[r2[q]["predicted_label"] for q in query_ids]
    macro1=macro_f1(gold,p1,LABELS); macro2=macro_f1(gold,p2,LABELS)
    macro={"R1":macro1,"R2":macro2,"difference":macro2-macro1,"cluster_bootstrap_95ci":cluster_macro_f1_bootstrap(gold,p1,p2,query_ids,groups,LABELS,replicates=config["paired_analysis"]["bootstrap_replicates"],seed=config["paired_analysis"]["macro_f1_bootstrap_seed"]),"role":"exploratory; macro F1 is non-decomposable and no sign-flip p-value is reported"}
    rerank={x["query_id"]:x for x in load(RUN/"per-query-reranking.json")}; categories=Counter(); cases=[]
    for q in query_ids:
        before=r1[q]; after=r2[q]; info=rerank[q]; same_doc=info["R1_top1"]==info["R2_top1"]
        b_cov=before["any_annotated_evidence_retrieved"]; a_cov=after["any_annotated_evidence_retrieved"]
        if same_doc: category="top1_unchanged"
        elif before["predicted_label"]==after["predicted_label"]: category="top1_changed_prediction_unchanged"
        elif b_cov is False and a_cov is True and not before["correct"] and after["correct"]: category="annotated_evidence_promoted_prediction_fixed"
        elif b_cov is False and a_cov is True and not after["correct"]: category="annotated_evidence_promoted_model_still_wrong"
        elif b_cov is True and a_cov is False and before["correct"] and not after["correct"]: category="annotated_evidence_demoted_new_error"
        elif not before["correct"] and after["correct"]: category="changed_context_prediction_fixed_other"
        elif before["correct"] and not after["correct"]: category="changed_context_new_error_other"
        else: category="changed_context_same_correctness_different_prediction"
        categories[category]+=1
        if not same_doc:
            top=sorted(info["candidates"][:10],key=lambda x:x["reranked_rank_within_top20"])
            cases.append({"query_id":q,"claim":info["claim"],"gold_label":before["gold_label"],"annotated_evidence_document_ids":info["annotated_evidence_document_ids"],"beir_known_positive_ids":info["beir_known_positive_ids"],"R1_document_id":info["R1_top1"],"R2_document_id":info["R2_top1"],"R1_prediction":before["predicted_label"],"R2_prediction":after["predicted_label"],"R1_correct":before["correct"],"R2_correct":after["correct"],"transition_category":category,"conservative_failure_category":"unclear" if category not in ["annotated_evidence_promoted_prediction_fixed","annotated_evidence_demoted_new_error"] else ("reranker_promotes_correct_evidence" if "promoted" in category else "reranker_demotes_correct_evidence"),"top10_candidates_in_reranked_order":top})
    failure={"scope":"all 44 claims whose selected top-1 document changed; transition categories are descriptive, not causal","category_counts":dict(categories),"records":cases}
    save_new(RUN/"failure-analysis.json",failure)
    summary={"scope":"post-development frozen DEV analysis; not unseen-test inference","conditions":{"R1":r1_payload["metrics"],"R2":r2_metrics,"R_gold":gold_payload["metrics"]},"primary_paired":primary,"macro_f1_exploratory":macro,"grouping":{"count":len(groups),"sizes":dict(Counter(map(len,groups)))},"failure_analysis_counts":dict(categories),"R1_reused_from":"milestone5a-verification-v1/final/D1","R_gold_reused_from":"milestone5a-verification-v1/final/D3","test_qrels_or_labels_loaded":False}
    save_new(RUN/"verification-summary.json",summary); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
