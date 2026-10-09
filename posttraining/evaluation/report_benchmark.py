"""Render the completed frozen benchmark report from saved machine-readable artifacts."""
from __future__ import annotations
import hashlib,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('/home/boweiye2/rag/runs/posttraining/benchmark-eval-v1')
CONFIG=ROOT/'configs/posttraining/final-eval-v1.json'
OUT_JSON=ROOT/'docs/posttraining/milestone6-main-benchmark-results.json'
OUT_MD=ROOT/'docs/posttraining/milestone6-main-benchmark.md'

def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def f(x): return f'{x:.4f}'
def pvalue(x): return f'{x:.3e}' if x<.001 else f'{x:.4f}'

result=load(RUN/'final-results.json'); config=load(CONFIG)
dev={}
base_root=Path('/home/boweiye2/rag/runs/posttraining/baselines-v1')
for sid in ('B0','B1','B2','B3'):
    m=load(base_root/f'{sid}-v2'/'metrics.json'); dev[sid]={'seed0':{'accuracy':m['accuracy'],'macro_f1':m['macro_f1']}}
for sid,base in [('M1',Path('/home/boweiye2/rag/runs/posttraining/lora-gold-v1/dev-finalists')),('M2',Path('/home/boweiye2/rag/runs/posttraining/lora-mixed-v1/dev-finalists/mix-2-epoch-3'))]:
    values=[]
    for seed in range(3):
        m=load(base/f'seed-{seed}'/'dev-evaluation/metrics.json')
        values.append({'seed':seed,'accuracy':m['accuracy'],'macro_f1':m['macro_f1']})
    dev[sid]={'seeds':values,'mean_accuracy':statistics.fmean(x['accuracy'] for x in values),'mean_macro_f1':statistics.fmean(x['macro_f1'] for x in values)}
comparisons=[('M1','B2'),('M2','B3'),('M2','M1'),('M2','B0'),('M2','B1')]
stability=[]
for left,right in comparisons:
    left_source = dev[left]['seeds'] if 'seeds' in dev[left] else [dev[left]['seed0']] * 3
    left_values=[x['macro_f1'] for x in left_source]
    right_source = dev[right]['seeds'] if 'seeds' in dev[right] else [dev[right]['seed0']] * 3
    right_values=[x['macro_f1'] for x in right_source]
    differences=[a-b for a,b in zip(left_values,right_values)]
    mean_diff=statistics.fmean(left_values)-statistics.fmean(right_values)
    sign=lambda value: 1 if value>0 else -1 if value<0 else 0
    target=sign(differences[0])
    satisfied=target!=0 and sum(sign(x)==target for x in differences)>=2 and sign(mean_diff)==target
    stability.append({'left':left,'right':right,'dev_seed_macro_f1_differences':differences,'dev_mean_difference':mean_diff,'seed0_sign':target,'rule_satisfied':satisfied})
result['dev_context']=dev
result['seed_consistency']=stability
result['technical_corrections']=load(RUN/'technical-corrections.json')
result['environments']={'posttraining':load(RUN/'environment.json'),'serving':load(RUN/'serving-environment.json')}
result['snapshots']={'pre_benchmark_path':str(RUN/'source/pre-benchmark-source-config-dependencies.tar.gz'),'pre_benchmark_sha256':(RUN/'source/pre-benchmark-source-config-dependencies.sha256').read_text().split()[0]}
result['integrity']['final_eval_config_sha256']=sha(CONFIG)
OUT_JSON.write_text(json.dumps(result,indent=2)+'\n')

lines=[]
lines += ['# Milestone 6 — frozen main benchmark evaluation','',
'> **Disclosure:** This is a fixed 300-claim SciFact benchmark comparison. The split was previously evaluated in v1, and v1 error analysis helped motivate v2. It is not a pristine unseen test set and these results are not an unbiased estimate of generalization. The split was not used for v2 training or model selection; B0 compatibility reproduction was the only permitted pre-freeze access.','',
'No model was trained, no checkpoint was selected from benchmark performance, and no prompt, retrieval, parser, scoring, or decoding setting was changed. M3, FSDP, and serving integration were not run.','',
'## Frozen endpoint and identities','',
'E1—zero-shot BGE exact retrieval over 5,183 SciFact documents followed by the top-1 full abstract—is the sole primary endpoint. All six systems use identical ascending claim order, evidence, prompt/schema, greedy decoding, and claim-level gold scoring. E2–E4 were optional and were not run.','',
'| System | Checkpoint | Few-shot | Frozen config SHA-256 |', '|---|---|---:|---|']
for sid,s in config['systems'].items():
    lines.append(f"| {sid} | `{s['model_path']}` | {'yes' if s['fewshot'] else 'no'} | `{s['frozen_config_sha256']}` |")
lines += ['',f"Final evaluation config SHA-256: `{sha(CONFIG)}`. Exact shard hashes are preserved in `preflight.json` and `configs/posttraining/final-eval-v1.json`.",'','## Main E1 results','',
'| System | Accuracy | Macro F1 | SUPPORT F1/recall | CONTRADICT F1/recall | INSUFFICIENT F1/recall | Predicted S/C/I/invalid |',
'|---|---:|---:|---:|---:|---:|---:|']
for sid,m in result['systems'].items():
    pl=m['per_label']; counts=m['prediction_counts_including_invalid']
    lines.append(f"| {sid} | {f(m['accuracy'])} | {f(m['macro_f1'])} | {f(pl['SUPPORT']['f1'])} / {f(pl['SUPPORT']['recall'])} | {f(pl['CONTRADICT']['f1'])} / {f(pl['CONTRADICT']['recall'])} | {f(pl['INSUFFICIENT']['f1'])} / {f(pl['INSUFFICIENT']['recall'])} | {counts.get('SUPPORT',0)}/{counts.get('CONTRADICT',0)}/{counts.get('INSUFFICIENT',0)}/{counts.get('INVALID',0)} |")
lines += ['',
'M2 reaches **0.6733 accuracy / 0.6602 macro F1**. Relative to M1 it restores INSUFFICIENT recall from 0.0000 to 0.6786 while retaining much stronger CONTRADICT recall than B2/B3 (0.6562 versus 0.0312). Its 111/91/98 SUPPORT/CONTRADICT/INSUFFICIENT predictions are substantially more balanced than M1 or the untuned 3B baselines.','',
'## Structural validity','',
'| System | JSON | Schema | Decision | Rationale-index | Invalid output |','|---|---:|---:|---:|---:|---:|']
for sid,m in result['systems'].items():
    lines.append(f"| {sid} | {f(m['json_valid_rate'])} | {f(m['schema_valid_rate'])} | {f(m['decision_valid_rate'])} | {f(m['rationale_index_valid_rate'])} | {f(m['invalid_output_rate'])} |")
lines += ['','B3 produced three invalid outputs. B1 had one structurally valid decision with an invalid rationale index. Structural decision validity and rationale-index validity remain separate.','',
'## Restricted-choice calibration','',
'These are probabilities normalized only across SUPPORT, CONTRADICT, and INSUFFICIENT after scoring `<prompt> + {"decision": " + label`. They are conditional restricted-choice probabilities, not unrestricted generation probabilities. Structured generation remains the official task prediction.','',
'| System | ECE | Brier | NLL | Restricted argmax accuracy | Agreement with generation |','|---|---:|---:|---:|---:|---:|']
for sid,m in result['systems'].items():
    c=m['calibration']; lines.append(f"| {sid} | {f(c['ece'])} | {f(c['brier_score'])} | {f(c['nll'])} | {f(c['restricted_argmax_accuracy'])} | {f(c['restricted_generation_agreement_rate'])} |")
lines += ['','M1 (0.7300) and M2 (0.8167) fall materially below the predeclared approximately 95% agreement diagnostic. Their calibration path and free structured-generation path therefore differ; calibration values should not be interpreted as confidence in the generated decision.','',
'## Rationale quality','',
'Rationales are scored only where E1 retrieved an annotated document for a true SUPPORT/CONTRADICT claim. For multiple valid sets, the maximum rationale F1 is used.','',
'| System | Eligible / excluded | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
for sid,m in result['systems'].items():
    r=m['rationale']; lines.append(f"| {sid} | {r['eligible_count']} / {r['excluded_count']} | {f(r['precision'])} | {f(r['recall'])} | {f(r['f1'])} |")
lines += ['','Each system has 149 eligible records. The 151 exclusions are 112 gold INSUFFICIENT claims and 39 SUPPORT/CONTRADICT claims whose retrieved top-1 document lacks a valid gold rationale set.','',
'## Pre-registered paired comparisons','',
'Accuracy uses exact two-sided McNemar tests with Holm correction across tests 1–5 only. Macro-F1 intervals are unadjusted paired claim bootstrap percentile intervals with 10,000 resamples. Positive differences favor the left system.','',
'| Test | Comparison | Accuracy Δ | Discordance left-only/right-only | Raw p | Holm p | Macro-F1 Δ | 95% bootstrap CI |','|---:|---|---:|---:|---:|---:|---:|---:|']
for c in result['comparisons']:
    lines.append(f"| {c['id']} | {c['left']} − {c['right']} | {f(c['accuracy_difference'])} | {c['left_correct_right_wrong']}/{c['right_correct_left_wrong']} | {pvalue(c['mcnemar_exact_raw_p'])} | {pvalue(c['mcnemar_holm_adjusted_p'])} | {f(c['macro_f1_difference'])} | [{f(c['macro_f1_bootstrap_95_ci'][0])}, {f(c['macro_f1_bootstrap_95_ci'][1])}] |")
lines += ['','M1 is significantly worse than B2 on accuracy after Holm correction, and its macro-F1 interval is wholly negative. M2 has a clear macro-F1 advantage over B3, although its accuracy difference is not significant after correction. M2 strongly outperforms M1 on both analyses. Against B0 and B1, both M2 confidence intervals include zero and McNemar tests are not significant; this supports statistical comparability on this fixed benchmark rather than superiority or formal equivalence.','',
'## DEV corroboration and seed stability','',
'| System | Frozen DEV accuracy | Frozen DEV macro F1 | Benchmark accuracy | Benchmark macro F1 |','|---|---:|---:|---:|---:|']
for sid in ('B0','B1','B2','B3','M1','M2'):
    d=dev[sid]; da=d.get('mean_accuracy',d.get('seed0',{}).get('accuracy')); dm=d.get('mean_macro_f1',d.get('seed0',{}).get('macro_f1')); b=result['systems'][sid]
    lines.append(f"| {sid} | {f(da)} | {f(dm)} | {f(b['accuracy'])} | {f(b['macro_f1'])} |")
lines += ['','M1 and M2 DEV values are three-seed means; baseline values are their single frozen runs. For every pre-registered comparison, at least two of three tuned-model DEV seed differences—including seed 0—share the mean difference sign. The frozen seed-consistency rule is satisfied for all five comparisons. DEV and benchmark agree that M1 overpredicts CONTRADICT and never abstains, while M2 restores INSUFFICIENT behavior and retains a large CONTRADICT improvement over B2/B3.','',
'## Technical correction','',
'All six generation passes and B0 calibration completed before the only technical failure. B1 restricted-choice calibration exhausted GPU memory when batch size 6 materialized full-vocabulary logits for long few-shot prompts. No partial B1 calibration file was written. The failed state and traceback context were preserved in `technical-corrections.json`; calibration resumed at B1 with batch size 1. This changes throughput only: prompts, weights, continuations, total sequence likelihoods, normalization, generations, and metrics are unchanged. Generations and B0 calibration were not rerun.','',
'The merged-tokenizer legacy-regex warning seen in prior milestones recurred. Frozen tokenizer behavior was preserved, and zero-shot prompt lengths remained identical across B0, B2, M1, and M2.','',
'## Environment and commands','',
'- Generation: Python 3.11.17, PyTorch 2.7.0+cu126, Transformers 4.53.2, vLLM 0.9.2, CUDA 12.6.','- Calibration/statistics: Python 3.11.16, PyTorch 2.9.1+cu126, Transformers 4.57.6, CUDA 12.6.','- Hardware: one NVIDIA L40S.','- Repository commit recorded before the run: `db145b5e6526afefd7ad9fead083e4100fe53de2`.','',
'```bash','/home/boweiye2/rag/envs/posttraining/bin/python -m pytest -q tests/posttraining tests/test_verification.py tests/test_scifact_preparation.py','/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark preflight --config configs/posttraining/final-eval-v1.json','scripts/run_main_benchmark.sh','# After the documented B1 calibration OOM, resume only unfinished calibration with --batch-size 1','/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark calibrate --batch-size 1 --config configs/posttraining/final-eval-v1.json --system B1','/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark calibrate --batch-size 1 --config configs/posttraining/final-eval-v1.json --system B2 --system B3','/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark calibrate --batch-size 1 --config configs/posttraining/final-eval-v1.json --system M1','/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark calibrate --batch-size 1 --config configs/posttraining/final-eval-v1.json --system M2','/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark finalize --config configs/posttraining/final-eval-v1.json','```','',
'## Artifacts','',f"- External run: `{RUN}`",f"- Per-system predictions, raw generations, calibration scores, metrics, and validity: `{RUN}/systems/{{B0,B1,B2,B3,M1,M2}}/`",f"- Paired statistics and bootstrap distributions: `{RUN}/statistics/`",f"- Technical correction: `{RUN}/technical-corrections.json`",f"- Pre-benchmark snapshot: `{RUN}/source/pre-benchmark-source-config-dependencies.tar.gz`",f"- Pre-benchmark snapshot SHA-256: `{result['snapshots']['pre_benchmark_sha256']}`",'- Repository report results: `docs/posttraining/milestone6-main-benchmark-results.json`','',
'## Interpretation and stop point','',
'The fixed benchmark answers the five questions as follows: M1 does not beat B2; M2 improves substantially over B3 in macro F1; M2 clearly beats M1; and M2 performs near B0/B1 without a statistically reliable difference in either direction under the pre-registered analyses. This evidence is descriptive of a reused benchmark and does not establish unbiased generalization or formal model equivalence.','',
'The main benchmark passed. No implementation blocker remains before a separately authorized M3/FSDP stage. Milestone 6 stops here; no M3, retraining, post-benchmark tuning, or serving integration was performed.']
OUT_MD.write_text('\n'.join(lines)+'\n')
print(json.dumps({'report':str(OUT_MD),'results':str(OUT_JSON),'systems':{k:{'accuracy':v['accuracy'],'macro_f1':v['macro_f1']} for k,v in result['systems'].items()}},indent=2))
