"""Build final Milestone 5A DEV report and machine-readable summary."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag")))
RUN=ART/"runs/milestone5a-verification-v1"

def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_new(path,value):
    if path.exists(): raise FileExistsError(path)
    path.write_text((json.dumps(value,indent=2)+"\n") if not isinstance(value,str) else value)

def main():
    config=load(ROOT/"configs/verification-5a-v1.json")
    mapping=load(RUN/"verification-mapping.json")
    development=load(RUN/"development-summary.json")
    final=load(RUN/"final-summary.json")["conditions"]
    records={c:load(RUN/f"final/{c}/evaluation.json")["records"] for c in ["D1","D2","D3"]}
    error_definitions={
      "annotated_evidence_not_retrieved":"Gold SUPPORT/CONTRADICT claim whose annotated evidence document is absent from the supplied top-k context.",
      "evidence_retrieved_but_model_misclassified":"At least one annotated evidence document is supplied, but the predicted label differs from the benchmark label.",
      "insufficient_or_ambiguous_evidence":"The original claim has no annotated evidence and the model predicts SUPPORT or CONTRADICT; this can reflect model error, unjudged evidence, or annotation scope.",
      "citation_or_output_failure":"Output fails strict JSON/schema/citation validation.",
      "unclear":"Available artifacts do not support a more specific category."
    }
    error_analysis={"scope":"conservative artifact-supported categories; not causal diagnosis","category_definitions":error_definitions,
                    "conditions":{c:[{k:x[k] for k in ["query_id","gold_label","predicted_label","document_ids","annotated_evidence_document_ids","any_annotated_evidence_retrieved","parsed_output","error_category"]} for x in records[c] if not x["correct"]] for c in records}}
    write_new(RUN/"error-analysis.json",error_analysis)
    d1={x["query_id"]:x for x in records["D1"]}; d2={x["query_id"]:x for x in records["D2"]}
    comparison={
      "same_top_document":sum(d1[q]["document_ids"]==d2[q]["document_ids"] for q in d1),
      "same_prediction":sum(d1[q]["predicted_label"]==d2[q]["predicted_label"] for q in d1),
      "D1_correct_D2_wrong":sum(d1[q]["correct"] and not d2[q]["correct"] for q in d1),
      "D2_correct_D1_wrong":sum(d2[q]["correct"] and not d1[q]["correct"] for q in d1),
      "exact_D1_replay":all(a["raw_output"]==b["raw_output"] for a,b in zip(
         [json.loads(x) for x in (RUN/"development/grounded-v2/top-1/generations.jsonl").read_text().splitlines()],
         [json.loads(x) for x in (RUN/"final/D1/generations.jsonl").read_text().splitlines()])),
    }
    result={
      "scope":"DEV-only verification; prompt and evidence budget selected on this same DEV set; no unseen-test inference",
      "model":config["model"],"generation":config["generation"],"final_selection":config["final_selection"],
      "mapping_summary":{k:mapping[k] for k in ["claim_count","label_counts","claims_with_annotated_evidence","claims_without_annotated_evidence","annotated_evidence_document_assignments","annotated_evidence_sets","test_labels_loaded"]},
      "conditions":config["conditions"],"development_candidates":development["candidates"],"final_metrics":final,
      "D1_D2_comparison":comparison,"error_category_definitions":error_definitions,
      "validation":{"tests_passed":22,"all_final_condition_counts":{c:len(records[c]) for c in records},"invalid_outputs":{c:final[c]["invalid_output_count"] for c in final},"test_qrels_or_labels_loaded":False},
      "hashes":{"config_sha256":sha(ROOT/"configs/verification-5a-v1.json"),"prompt_sha256":config["final_selection"]["prompt_sha256"],
                "inference_script_sha256":sha(ROOT/"scripts/run_verification_5a.py"),"evaluation_script_sha256":sha(ROOT/"scripts/evaluate_verification_5a.py")},
      "artifacts":{"run":str(RUN),"mapping":str(RUN/"verification-mapping.json"),"development":str(RUN/"development-summary.json"),
                   "final_summary":str(RUN/"final-summary.json"),"error_analysis":str(RUN/"error-analysis.json"),"source_config_dependency_snapshot":str(RUN/"source/exact-source-config-dependencies.tar.gz")},
      "limitations":["Prompt and top-k were selected and evaluated on the same 162 DEV claims.","No annotated evidence does not establish that no evidence exists.","D3 supplies annotated documents but is not a strict upper bound.","Condition differences combine retrieval-context and model-interaction effects."]
    }
    jp=ROOT/"docs/milestone5a-verification-dev-results.json"; write_new(jp,result)
    def metric_row(c):
      m=final[c]; return f"| {c} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['per_label']['SUPPORT']['accuracy']:.4f} / {m['per_label']['SUPPORT']['f1']:.4f} | {m['per_label']['CONTRADICT']['accuracy']:.4f} / {m['per_label']['CONTRADICT']['f1']:.4f} | {m['per_label']['INSUFFICIENT']['accuracy']:.4f} / {m['per_label']['INSUFFICIENT']['f1']:.4f} | {m['invalid_output_rate']:.4f} | {m['citation_valid_rate']:.4f} |"
    lines=[
      "# Milestone 5A — evidence-grounded biomedical claim verification on DEV","",
      "> **Scope:** Prompt and evidence-budget selection and all reported metrics use the same frozen 162-claim DEV set. These results characterize the completed DEV pipeline and are not estimates of unseen TEST generalization.","",
      "No retriever was trained, no TEST qrels or verification labels were read, and no reranking, serving, or performance benchmark was run.","",
      "## Verification mapping","",
      "All 162 frozen claim IDs align exactly with original SciFact training annotations. Labels are derived from the original evidence sets: **70 SUPPORT, 28 CONTRADICT, and 64 INSUFFICIENT**. The 98 claims with annotated evidence contain 108 document assignments and 189 alternative evidence sets. Empty evidence maps to the benchmark INSUFFICIENT label; it does not prove that no relevant evidence exists.","",
      "## Frozen model and prompt","",
      f"The local model is **{config['model']['name']}** at revision `{config['model']['revision']}`, loaded in BF16 with SDPA on one L40S. It was selected as a single practical 7B instruction model because it supports structured JSON generation, fits comfortably on one GPU, and is compatible with later vLLM work.","",
      f"Greedy decoding uses `max_new_tokens={config['generation']['max_new_tokens']}`, seed `{config['generation']['seed']}`, a 16,384-token input limit, and a 12,000-token evidence limit. The frozen prompt is **{config['final_selection']['prompt_name']}** with SHA-256 `{config['final_selection']['prompt_sha256']}`. It requires evidence-only decisions, exact supplied citation IDs, strict JSON, and abstention when the evidence is inadequate.","",
      "A three-example feasibility check exposed Qwen's tendency to convert `[DOC 123]` to `DOC_123`. The single permitted evidence-format variant changed identifiers to `[EVIDENCE_ID: 123]`; the final feasibility run produced exact IDs on SUPPORT and CONTRADICT and an empty list on INSUFFICIENT.","",
      "## Bounded DEV development","",
      "The complete search was two prompts × top-1/top-3/top-5 using D1 only. Selection was fixed as macro F1, accuracy, invalid-output rate, smaller top-k, then prompt name.","",
      "| Prompt | k | Accuracy | Macro F1 | Invalid | Citation validity | Annotated-evidence coverage |","|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in development["candidates"]:
      m=c["metrics"]; lines.append(f"| {c['prompt_name']} | {c['top_k']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['invalid_output_rate']:.4f} | {m['citation_valid_rate']:.4f} | {m['annotated_evidence_coverage']:.4f} |")
    lines += ["",f"The selected configuration is **{config['final_selection']['prompt_name']}, top-1**. More documents increased annotated-evidence coverage but reduced label metrics and strict-output validity for this model/prompt setup. No contexts reached the 12,000-token evidence cap.","",
      "## Final DEV results","",
      "D1 uses frozen zero-shot BGE. D2 uses the already-selected random-negative seed 20261003 epoch 3 checkpoint because it had the highest frozen retrieval NDCG among the selected random checkpoints; verification results played no role in that choice. D3 supplies the first annotated evidence document under the same top-1 budget and full-abstract format.","",
      "| Condition | Accuracy | Macro F1 | SUPPORT acc/F1 | CONTRADICT acc/F1 | INSUFFICIENT acc/F1 | Invalid rate | Citation validity |","|---|---:|---:|---:|---:|---:|---:|---:|",
      metric_row("D1"),metric_row("D2"),metric_row("D3"),"",
      "D1 and D2 both reach **0.7654 accuracy**; macro F1 is 0.75309 for D1 and 0.75308 for D2. They share the same top document on 129/162 claims and the same label on 156/162. D1 is uniquely correct on two claims and D2 on two, so the aggregate tie does not establish retrieval equivalence or a causal explanation.","",
      "D3 reaches **0.8765 accuracy / 0.8636 macro F1**. It is a diagnostic reference, not a strict upper bound: it uses only one annotated document, some claims have alternative evidence sets or documents, and full abstracts can obscure short rationales.","",
      "## Evidence diagnostics","",
      "| Condition | Annotated evidence coverage | Accuracy when retrieved | Accuracy when not retrieved | Abstention rate | Mean evidence tokens | Truncated |","|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ["D1","D2","D3"]:
      m=final[c]; missing="—" if m["accuracy_when_annotated_evidence_not_retrieved"] is None else f"{m['accuracy_when_annotated_evidence_not_retrieved']:.4f}"
      lines.append(f"| {c} | {m['annotated_evidence_coverage']:.4f} | {m['accuracy_when_annotated_evidence_retrieved']:.4f} | {missing} | {m['abstention_rate']:.4f} | {m['evidence_tokens']['mean_after']:.1f} | {m['evidence_tokens']['truncated_count']} |")
    lines += ["","Coverage is computed only for the 98 SUPPORT/CONTRADICT claims with annotated evidence. Absence means no annotated gold document was supplied; it does not prove that the retrieved document is useless.","",
      "## Conservative error analysis","",
      "| Condition | Annotated evidence not retrieved | Evidence retrieved, model misclassified | Insufficient/ambiguous annotation case | Citation/output failure |","|---|---:|---:|---:|---:|",
      f"| D1 | 7 | 14 | 17 | 0 |",f"| D2 | 6 | 15 | 17 | 0 |",f"| D3 | 0 | 20 | 0 | 0 |","",
      "Observed examples include abstention despite an annotated full abstract, difficulty reasoning about intervention direction (for example, knockout causing BBB disruption), and cases with empty original annotations where a retrieved abstract appears related enough for the model to predict SUPPORT or CONTRADICT. These are descriptive patterns. The artifacts do not establish annotation error, false-negative evidence, or a single model failure mechanism.","",
      "## Reproducibility and limitations","",
      "Final D1 reproduced every selected-development raw output exactly. All three conditions contain 162 aligned records, strict schema and citation validation is automatic, and all 22 tests pass. Model revision, prompt hash, frozen config, generations, token counts, evaluation records, and dependency versions are saved.","",
      "The main limitation is deliberate DEV reuse: the prompt and top-k were selected on the same claims reported here. TEST remains untouched and is required for confirmatory evaluation. D1/D2 differences cannot be attributed solely to retriever quality because changed contexts interact with the LLM.","",
      "## Commands","","```bash","source scripts/env.sh",'CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_verification_5a --phase feasibility','CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_verification_5a --phase development','python -m scripts.evaluate_verification_5a --phase development','CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_verification_5a --phase final','python -m scripts.evaluate_verification_5a --phase final','python -m pytest -q',"```","",
      "## Artifacts","",f"- Versioned external run: `{RUN}`",f"- Mapping: `{RUN/'verification-mapping.json'}`",f"- Development summary: `{RUN/'development-summary.json'}`",f"- Final summary and per-condition evaluations: `{RUN/'final-summary.json'}` and `{RUN/'final'}`",f"- Conservative error records: `{RUN/'error-analysis.json'}`",f"- Exact source/config/dependency snapshot: `{RUN/'source/exact-source-config-dependencies.tar.gz'}`","- Frozen config: `configs/verification-5a-v1.json`","","Milestone 5A stops here."]
    mp=ROOT/"docs/milestone5a-verification-dev-report.md"; write_new(mp,"\n".join(lines)+"\n")
    print(json.dumps({"markdown":str(mp),"json":str(jp),"D1_D2":comparison},indent=2))
if __name__=="__main__": main()
