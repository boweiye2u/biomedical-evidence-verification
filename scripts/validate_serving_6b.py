"""Live equivalence validation for the frozen Milestone 6B service."""
import json,os
from pathlib import Path
import httpx
from retrieval.verification import parse_and_validate_output

ART=Path(os.environ.get('RAG_ROOT', Path.home()/'rag'))
RUN=ART/'runs/milestone6b-serving-v1'
DEV=ART/'runs/milestone5a-verification-v1/final/D1/generations.jsonl'
MAPPING=ART/'runs/milestone5a-verification-v1/verification-mapping.json'


def main():
    previous=[json.loads(x) for x in DEV.read_text().splitlines() if x.strip()]
    claims={x['claim_id']:x['claim'] for x in json.loads(MAPPING.read_text())['records']}
    selected=sorted(previous,key=lambda x:int(x['query_id']))[:10]
    records=[]
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=180) as client:
        health=client.get('/health'); health.raise_for_status()
        for old in selected:
            claim=claims[old['query_id']]
            retrieval=client.post('/retrieve',json={'claim':claim}).json()
            verification=client.post('/verify',json={'claim':claim}).json()
            old_parsed,old_errors=parse_and_validate_output(old['raw_output'],old['document_ids'])
            records.append({
                'query_id':old['query_id'],
                'retrieval_expected':old['document_ids'][0],
                'retrieval_actual':retrieval['documents'][0]['document_id'],
                'retrieval_match':retrieval['documents'][0]['document_id']==old['document_ids'][0],
                'hf_raw_output':old['raw_output'],
                'vllm_valid':verification['valid'],
                'vllm_label':verification['label'],
                'hf_label':old_parsed['label'] if old_parsed else None,
                'label_match':bool(old_parsed and old_parsed['label']==verification['label']),
                'citation_valid':set(verification['evidence_ids'])<=set(old['document_ids']),
                'request_id':verification['request_id'],
            })
    summary={
        'count':len(records),
        'retrieval_matches':sum(x['retrieval_match'] for x in records),
        'valid_outputs':sum(x['vllm_valid'] for x in records),
        'label_matches':sum(x['label_match'] for x in records),
        'citation_valid':sum(x['citation_valid'] for x in records),
        'records':records,
    }
    RUN.mkdir(parents=True,exist_ok=True)
    out=RUN/'functional-equivalence.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))

if __name__=='__main__': main()
