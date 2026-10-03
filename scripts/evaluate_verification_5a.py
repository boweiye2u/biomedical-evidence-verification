"""Evaluate and freeze Milestone 5A DEV verification results."""
from __future__ import annotations
import argparse,hashlib,json,os
from collections import Counter
from pathlib import Path
from retrieval.verification import LABELS,classification_metrics,parse_and_validate_output,prompt_hash

ROOT=Path(__file__).resolve().parents[1]
ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag")))
RUN=ART/"runs/milestone5a-verification-v1"

def load(path): return json.loads(path.read_text())
def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def save_new(path,value):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2)+"\n")

def citation_is_valid(raw,allowed):
    try: value=json.loads(raw.strip())
    except Exception: return False
    ids=value.get("evidence_ids") if isinstance(value,dict) else None
    return isinstance(ids,list) and all(isinstance(x,str) for x in ids) and len(ids)==len(set(ids)) and set(ids)<=set(allowed)

def evaluate(path):
    items=rows(path); evaluated=[]; gold=[]; predicted=[]
    for item in items:
        value,errors=parse_and_validate_output(item["raw_output"],item["document_ids"])
        prediction=value["label"] if value else "INVALID"
        gold.append(item["gold_label"]); predicted.append(prediction)
        correct=prediction==item["gold_label"]
        if correct: category=None
        elif errors: category="citation_or_output_failure"
        elif item["gold_label"]=="INSUFFICIENT": category="insufficient_or_ambiguous_evidence"
        elif item["any_annotated_evidence_retrieved"] is False: category="annotated_evidence_not_retrieved"
        elif item["any_annotated_evidence_retrieved"] is True: category="evidence_retrieved_but_model_misclassified"
        else: category="unclear"
        evaluated.append({**item,"parsed_output":value,"validation_errors":errors,"predicted_label":prediction,
                          "correct":correct,"citation_ids_valid":citation_is_valid(item["raw_output"],item["document_ids"]),
                          "error_category":category})
    metrics=classification_metrics(gold,predicted)
    count=len(items); with_ann=[x for x in evaluated if x["any_annotated_evidence_retrieved"] is True]
    without_ann=[x for x in evaluated if x["any_annotated_evidence_retrieved"] is False]
    metrics.update({
      "invalid_output_count":sum(x["predicted_label"]=="INVALID" for x in evaluated),
      "invalid_output_rate":sum(x["predicted_label"]=="INVALID" for x in evaluated)/count,
      "citation_valid_count":sum(x["citation_ids_valid"] for x in evaluated),
      "citation_valid_rate":sum(x["citation_ids_valid"] for x in evaluated)/count,
      "abstention_count":sum(x["predicted_label"]=="INSUFFICIENT" for x in evaluated),
      "abstention_rate":sum(x["predicted_label"]=="INSUFFICIENT" for x in evaluated)/count,
      "annotated_evidence_coverage_count":len(with_ann),
      "annotated_evidence_eligible_count":len(with_ann)+len(without_ann),
      "annotated_evidence_coverage":len(with_ann)/(len(with_ann)+len(without_ann)) if with_ann or without_ann else None,
      "accuracy_when_annotated_evidence_retrieved":sum(x["correct"] for x in with_ann)/len(with_ann) if with_ann else None,
      "accuracy_when_annotated_evidence_not_retrieved":sum(x["correct"] for x in without_ann)/len(without_ann) if without_ann else None,
      "error_categories":dict(Counter(x["error_category"] for x in evaluated if x["error_category"])),
      "evidence_tokens":{"mean_before":sum(x["evidence_token_count_before"] for x in items)/count,
                         "mean_after":sum(x["evidence_token_count_after"] for x in items)/count,
                         "truncated_count":sum(x["evidence_truncated"] for x in items)},
    })
    return metrics,evaluated

def development():
    config=load(ROOT/"configs/verification-5a-v1.json"); candidates=[]
    for prompt in sorted(config["development"]["prompt_candidates"]):
        for k in config["development"]["top_k_candidates"]:
            path=RUN/"development"/prompt/f"top-{k}"/"generations.jsonl"
            metrics,evaluated=evaluate(path)
            save_new(path.parent/"evaluation.json",{"metrics":metrics,"records":evaluated})
            candidates.append({"prompt_name":prompt,"prompt_sha256":prompt_hash(prompt),"top_k":k,"metrics":metrics})
    candidates.sort(key=lambda x:(-x["metrics"]["macro_f1"],-x["metrics"]["accuracy"],x["metrics"]["invalid_output_rate"],x["top_k"],x["prompt_name"]))
    selected=candidates[0]
    save_new(RUN/"development-summary.json",{"selection_rule":config["development"]["selection_order"],"selected":selected,"candidates":candidates})
    config["status"]="final_prompt_and_budget_frozen"
    config["final_selection"]={"prompt_name":selected["prompt_name"],"prompt_sha256":selected["prompt_sha256"],"top_k":selected["top_k"],"selection_source":"D1 DEV development search only","metrics_at_selection":selected["metrics"]}
    (ROOT/"configs/verification-5a-v1.json").write_text(json.dumps(config,indent=2)+"\n")
    print(json.dumps({"selected":selected,"candidate_count":len(candidates)},indent=2))

def final():
    config=load(ROOT/"configs/verification-5a-v1.json"); summaries={}
    for condition in ["D1","D2","D3"]:
        path=RUN/"final"/condition/"generations.jsonl"; metrics,evaluated=evaluate(path)
        save_new(path.parent/"evaluation.json",{"metrics":metrics,"records":evaluated}); summaries[condition]=metrics
    save_new(RUN/"final-summary.json",{"config":config,"conditions":summaries,"test_qrels_or_labels_loaded":False})
    print(json.dumps(summaries,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--phase",choices=["development","final"],required=True); a=p.parse_args()
    development() if a.phase=="development" else final()
if __name__=="__main__": main()
