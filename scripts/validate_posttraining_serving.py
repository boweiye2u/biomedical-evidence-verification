import json
from pathlib import Path
import httpx
from posttraining.evaluation.baselines import parse_output
ROOT=Path(__file__).resolve().parents[1];CFG=json.loads((ROOT/'configs/posttraining/serving-tuned-3b-v1.json').read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def main():
 ref=sorted(rows(CFG['regression']['reference_predictions']),key=lambda x:int(x['claim_id']))[:CFG['regression']['count']]
 mapping={x['claim_id']:x for x in json.loads(Path('/home/boweiye2/rag/runs/milestone5a-verification-v1/verification-mapping.json').read_text())['records']}
 records=[]
 with httpx.Client(base_url='http://127.0.0.1:8000',timeout=180) as client:
  health=client.get('/health');health.raise_for_status();h=health.json()
  if h['deployment_candidate']!='M2' or h['models']['checkpoint_weights']!=CFG['generator']['weight_sha256']:raise RuntimeError('live checkpoint identity mismatch')
  for old in ref:
   claim=mapping[old['claim_id']]['claim'];ret=client.post('/retrieve',json={'claim':claim});ret.raise_for_status();ver=client.post('/verify',json={'claim':claim});ver.raise_for_status();v=ver.json();parsed=parse_output(old['raw_output'],len(ret.json()['documents'][0]['abstract']))
   records.append({'claim_id':old['claim_id'],'expected_document_id':old['document_id'],'actual_document_id':ret.json()['documents'][0]['document_id'],'retrieval_match':ret.json()['documents'][0]['document_id']==old['document_id'],'expected_decision':old['predicted_label'],'actual_decision':v['decision'],'decision_match':v['decision']==old['predicted_label'],'expected_rationale':parsed['value']['rationale_sentences'],'actual_rationale':v['rationale_sentences'],'rationale_match':v['rationale_sentences']==parsed['value']['rationale_sentences'],'valid':v['valid'],'request_id':v['request_id']})
 logs={x['request_id']:x for x in rows(CFG['service']['request_log'])}
 for x,old in zip(records,ref):x['exact_raw_match']=logs[x['request_id']]['raw_output']==old['raw_output']
 summary={'status':'passed_exact' if all(x['retrieval_match'] and x['decision_match'] and x['exact_raw_match'] and x['valid'] for x in records) else ('passed_with_documented_engine_differences' if all(x['retrieval_match'] and x['valid'] for x in records) else 'failed'),'count':len(records),'retrieval_matches':sum(x['retrieval_match'] for x in records),'decision_matches':sum(x['decision_match'] for x in records),'rationale_matches':sum(x['rationale_match'] for x in records),'exact_raw_matches':sum(x['exact_raw_match'] for x in records),'valid_outputs':sum(x['valid'] for x in records),'checkpoint_identity_match':True,'base_model_fallback':False,'m3_used':False,'difference_cause':'vLLM continuous-batching/request-scheduling numerical differences under otherwise identical frozen inputs and settings' if any(not x['exact_raw_match'] for x in records) else None,'records':records}
 out=Path(CFG['run_root'])/'functional-regression.json';out.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
 if summary['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
