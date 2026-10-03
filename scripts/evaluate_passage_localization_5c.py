"""Evaluate passage verification and cluster-aware paired DEV diagnostics."""
from __future__ import annotations
import json,os
from collections import Counter
from pathlib import Path
from retrieval.reranking import cluster_macro_f1_bootstrap,macro_f1
from retrieval.statistics import classify_differences,cluster_bootstrap_ci,cluster_sign_flip_test,validate_clusters
from retrieval.verification import LABELS
from scripts.evaluate_verification_5a import evaluate
ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone5c-passage-v1"
def load(p): return json.loads(p.read_text())
def save_new(p,x):
 if p.exists(): raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2)+"\n")
def conditional(records):
 subsets={"rationale_localized":[],"annotated_document_selected_rationale_not_localized":[],"annotated_document_not_selected":[]}
 for x in records:
  if not x["annotated_rationale_pairs"]: continue
  if x["annotated_rationale_localized"]: key="rationale_localized"
  elif x["any_annotated_evidence_retrieved"]: key="annotated_document_selected_rationale_not_localized"
  else: key="annotated_document_not_selected"
  subsets[key].append(x)
 result={k:{"count":len(v),"accuracy":sum(x["correct"] for x in v)/len(v) if v else None} for k,v in subsets.items()}
 combined=subsets["annotated_document_selected_rationale_not_localized"]+subsets["annotated_document_not_selected"]
 result["no_annotated_rationale_localized"]={"count":len(combined),"accuracy":sum(x["correct"] for x in combined)/len(combined) if combined else None}
 return result
def main():
 config=load(ROOT/"configs/passage-localization-5c-v1.json")
 p2_metrics,p2_records=evaluate(RUN/"verification/P2/generations.jsonl"); g_metrics,g_records=evaluate(RUN/"verification/G_rationale/generations.jsonl")
 save_new(RUN/"verification/P2/evaluation.json",{"metrics":p2_metrics,"records":p2_records})
 save_new(RUN/"verification/G_rationale/evaluation.json",{"metrics":g_metrics,"records":g_records})
 p1_payload=load(ART/"runs/milestone5a-verification-v1/final/D1/evaluation.json"); gf_payload=load(ART/"runs/milestone5a-verification-v1/final/D3/evaluation.json")
 p1={x["query_id"]:x for x in p1_payload["records"]}; p2={x["query_id"]:x for x in p2_records}; gf={x["query_id"]:x for x in gf_payload["records"]}; gr={x["query_id"]:x for x in g_records}
 query_ids=sorted(p1,key=int); assert set(query_ids)==set(p2)==set(gf)==set(gr) and len(query_ids)==162
 split=load(ROOT/"configs/splits/scifact_train_dev_v1.json"); dev=set(query_ids); groups=[]
 for original in split["groups"]:
  group=sorted(dev.intersection(map(str,original)),key=int)
  if group: groups.append(group)
 validate_clusters(query_ids,groups)
 diffs=[float(p2[q]["correct"])-float(p1[q]["correct"]) for q in query_ids]
 primary={"metric":"per-query accuracy","mean_difference":sum(diffs)/len(diffs),"counts":classify_differences(diffs),"cluster_bootstrap_95ci":cluster_bootstrap_ci(diffs,query_ids,groups,replicates=config["paired_analysis"]["bootstrap_replicates"],seed=config["paired_analysis"]["bootstrap_seed"]),"cluster_sign_flip":cluster_sign_flip_test(diffs,query_ids,groups,permutations=config["paired_analysis"]["sign_flip_permutations"],seed=config["paired_analysis"]["sign_flip_seed"])}
 gold=[p1[q]["gold_label"] for q in query_ids]; pred1=[p1[q]["predicted_label"] for q in query_ids]; pred2=[p2[q]["predicted_label"] for q in query_ids]
 m1=macro_f1(gold,pred1,LABELS); m2=macro_f1(gold,pred2,LABELS)
 macro={"P1":m1,"P2":m2,"difference":m2-m1,"cluster_bootstrap_95ci":cluster_macro_f1_bootstrap(gold,pred1,pred2,query_ids,groups,LABELS,replicates=config["paired_analysis"]["bootstrap_replicates"],seed=config["paired_analysis"]["macro_f1_bootstrap_seed"]),"role":"exploratory; macro F1 is non-decomposable and no sign-flip p-value is reported"}
 gold_diffs=[float(gr[q]["correct"])-float(gf[q]["correct"]) for q in query_ids]
 mismatch=[q for q in query_ids if gr[q]["document_ids"] and gf[q]["document_ids"] and gr[q]["document_ids"][0]!=gf[q]["document_ids"][0]]
 matched=[q for q in query_ids if q not in mismatch]
 matched_diffs=[float(gr[q]["correct"])-float(gf[q]["correct"]) for q in matched]
 gold_compare={"accuracy_difference":sum(gold_diffs)/len(gold_diffs),"counts":classify_differences(gold_diffs),"macro_f1_difference":g_metrics["macro_f1"]-gf_payload["metrics"]["macro_f1"],"document_mismatch_query_ids":mismatch,"same_document_sensitivity":{"count":len(matched),"accuracy_difference":sum(matched_diffs)/len(matched_diffs),"counts":classify_differences(matched_diffs),"macro_f1_difference":macro_f1([gf[q]["gold_label"] for q in matched],[gr[q]["predicted_label"] for q in matched],LABELS)-macro_f1([gf[q]["gold_label"] for q in matched],[gf[q]["predicted_label"] for q in matched],LABELS)},"role":"diagnostic descriptive comparison"}
 localization={x["query_id"]:x for x in load(RUN/"passage-rankings.json")}; counts=Counter(); failures=[]
 for q in query_ids:
  before=p1[q]; after=p2[q]; info=localization[q]; localized=after["annotated_rationale_localized"] is True; doc_selected=after["any_annotated_evidence_retrieved"] is True
  if not before["correct"] and after["correct"]: category="passage_localization_fixed_error"
  elif before["correct"] and not after["correct"]: category="full_abstract_correct_passage_wrong"
  elif localized and not after["correct"]: category="correct_rationale_selected_qwen_wrong"
  elif doc_selected and not localized: category="correct_document_wrong_sentence_selected"
  elif before["predicted_label"]!=after["predicted_label"] and after["predicted_label"]=="INSUFFICIENT": category="passage_selection_causes_abstention"
  elif before["predicted_label"]==after["predicted_label"]: category="prediction_unchanged"
  else: category="unclear"
  counts[category]+=1
  if before["predicted_label"]!=after["predicted_label"] or before["correct"]!=after["correct"]:
   failures.append({"query_id":q,"claim":info["claim"],"bge_top3_document_ids":info["bge_top3_document_ids"],"selected_passage":info["selected_passage"],"annotated_evidence_document_ids":info["annotated_evidence_document_ids"],"annotated_rationale_pairs":info["annotated_rationale_pairs"],"P1_prediction":before["predicted_label"],"P2_prediction":after["predicted_label"],"P1_correct":before["correct"],"P2_correct":after["correct"],"gold_label":before["gold_label"],"category":category,"note":"Descriptive transition; no causal mechanism assigned."})
 transitions={"prediction_unchanged":sum(p1[q]["predicted_label"]==p2[q]["predicted_label"] for q in query_ids),"prediction_changed":sum(p1[q]["predicted_label"]!=p2[q]["predicted_label"] for q in query_ids),"fixed_error":sum(not p1[q]["correct"] and p2[q]["correct"] for q in query_ids),"new_error":sum(p1[q]["correct"] and not p2[q]["correct"] for q in query_ids),"changed_prediction_same_correctness":sum(p1[q]["predicted_label"]!=p2[q]["predicted_label"] and p1[q]["correct"]==p2[q]["correct"] for q in query_ids),"passage_caused_abstention":sum(p1[q]["predicted_label"]!="INSUFFICIENT" and p2[q]["predicted_label"]=="INSUFFICIENT" for q in query_ids),"exact_rationale_selected_qwen_wrong":sum(p2[q]["annotated_rationale_localized"] is True and not p2[q]["correct"] for q in query_ids)}
 failure={"scope":"all claims whose P1 and P2 predictions or correctness differ","category_counts":dict(counts),"transition_counts":transitions,"records":failures}
 save_new(RUN/"failure-analysis.json",failure)
 summary={"scope":"post-development frozen DEV analysis; not unseen-test inference","conditions":{"P1":p1_payload["metrics"],"P2":p2_metrics,"G_full":gf_payload["metrics"],"G_rationale":g_metrics},"P2_conditional_verification":conditional(p2_records),"primary_paired":primary,"macro_f1_exploratory":macro,"gold_rationale_vs_full":gold_compare,"grouping":{"count":len(groups),"sizes":dict(Counter(map(len,groups)))},"failure_analysis_counts":dict(counts),"P1_reused_from":"milestone5a-verification-v1/final/D1","G_full_reused_from":"milestone5a-verification-v1/final/D3","test_qrels_or_labels_loaded":False}
 save_new(RUN/"verification-summary.json",summary); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
