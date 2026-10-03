"""Frozen SciFact DEV only. Run from repo root: python -m scripts.run_dev_baselines."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import faiss
import pytrec_eval
from beir.retrieval.evaluation import EvaluateRetrieval
from retrieval.metrics import evaluate
from retrieval.bm25 import BM25
from retrieval.bge import BGE, MODEL, PREFIX
from retrieval.medcpt import MedCPT

ROOT=Path(__file__).resolve().parents[1]
ART=Path(os.environ.get('RAG_ROOT',str(Path.home()/'rag')))
DATA=ART/'data/scifact'
OUT=ART/'runs/scifact_dev_v1'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj): p.write_text(json.dumps(obj,indent=2)+'\n')
def synchronize(): torch.cuda.synchronize()
def stats(values): return {'mean_ms':float(np.mean(values)), 'p50_ms':float(np.percentile(values,50)), 'p95_ms':float(np.percentile(values,95)), 'count':len(values)}
def load_rows(path,key): return {str(r[key]):r for r in map(json.loads,path.read_text().splitlines())}

def main():
    torch.set_num_threads(4); faiss.omp_set_num_threads(4)
    torch.manual_seed(20261001)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    split_path=ROOT/'configs/splits/scifact_train_dev_v1.json'
    split=json.loads(split_path.read_text())
    for name, expected in split['input_sha256'].items(): assert sha(DATA/name)==expected
    corpus=load_rows(DATA/'beir/corpus.jsonl','_id')
    doc_ids=sorted(corpus,key=int); docs=[corpus[d] for d in doc_ids]
    claims=load_rows(DATA/'original/claims_train.jsonl','id')
    dev_ids=split['dev_ids']; qrels={q:{} for q in dev_ids}
    for row in csv.DictReader((DATA/'beir/qrels/train.tsv').open(),delimiter='\t'):
        if row['query-id'] in qrels: qrels[row['query-id']][row['corpus-id']]=int(row['score'])
    revisions=json.loads((ROOT/'configs/model-revisions.json').read_text())
    summary={'split_id':split['version'],'seed':split['seed'],'dev_count':len(dev_ids),'corpus_size':len(docs),'split_sha256':sha(split_path),'revisions':revisions,'gpu':torch.cuda.get_device_name(0),'cuda_runtime':torch.version.cuda,'cpu_threads':4,'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','sentence-transformers','beir','pytrec-eval-terrier','rank-bm25','faiss-cpu','numpy']},'systems':{}}
    def score_run(name,runs,scores,latencies,metadata):
        mean,per=evaluate(qrels,runs)
        trusted=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut_10','recall_10','recall_100'}).evaluate(scores)
        for q in dev_ids:
            for a,b in [('NDCG@10','ndcg_cut_10'),('Recall@10','recall_10'),('Recall@100','recall_100')]:
                assert abs(per[q][a]-trusted[q][b])<1e-9,(name,q,a)
        mrr=EvaluateRetrieval.evaluate_custom(qrels,scores,[10],metric='mrr')['MRR@10']
        assert abs(mean['MRR@10']-mrr)<1e-5
        write(OUT/f'{name}-rankings.json',scores)
        write(OUT/f'{name}-per-query.json',per)
        write(OUT/f'{name}-latencies.json',dict(zip(dev_ids,latencies)))
        summary['systems'][name]={'metrics':mean,'latency':stats(latencies),**metadata}
        write(OUT/'summary.json',summary)
        print(name,summary['systems'][name],flush=True)
    start=time.perf_counter(); bm=BM25(docs); build=time.perf_counter()-start
    for q in dev_ids[:5]: bm.search(claims[q]['claim'])
    runs={}; scores={}; lat=[]
    for q in dev_ids:
        start=time.perf_counter(); idx,values=bm.search(claims[q]['claim']); lat.append((time.perf_counter()-start)*1000)
        runs[q]=[doc_ids[i] for i in idx]
        # Stable tie policy encoded as strictly descending rank scores for metric parity.
        scores[q]={d:float(100-j) for j,d in enumerate(runs[q])}
    score_run('bm25',runs,scores,lat,{'index_build_seconds':build,'k1':1.5,'b':0.75,'epsilon':0.25,'tie_policy':'stable ascending numeric document ID','score_file_semantics':'rank-derived scores, not raw BM25 scores'})
    print('Loading BGE',flush=True)
    bge=BGE(revisions[MODEL]); torch.cuda.reset_peak_memory_stats()
    sample=[claims[q]['claim'] for q in split['train_ids'][:3]]
    a=bge.encode(sample,query=True); b=bge.encode(sample,query=True)
    assert a.shape==(3,768) and np.isfinite(a).all()
    np.testing.assert_allclose(np.linalg.norm(a,axis=1),1,atol=1e-6)
    np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-6)
    cache_meta={'model':MODEL,'revision':revisions[MODEL],'corpus_sha256':sha(DATA/'beir/corpus.jsonl'),'doc_ids':doc_ids,'document_format':'title + single space + abstract','query_prefix':PREFIX,'max_length':512,'pooling':'CLS','normalization':'L2','dtype':'float32','batch_size':32,'transformers':summary['versions']['transformers'],'torch':summary['versions']['torch']}
    key=hashlib.sha256(json.dumps(cache_meta,sort_keys=True).encode()).hexdigest()[:16]
    cache=ART/'embeddings'/f'scifact-bge-{key}';cache.mkdir(parents=True,exist_ok=True)
    encode_seconds=None
    if (cache/'embeddings.npy').exists():
        assert json.loads((cache/'metadata.json').read_text())==cache_meta
        embeddings=np.load(cache/'embeddings.npy'); cached=True
    else:
        synchronize(); start=time.perf_counter()
        embeddings=bge.encode([d['title']+' '+d['text'] for d in docs])
        synchronize(); encode_seconds=time.perf_counter()-start
        np.save(cache/'embeddings.npy',embeddings);write(cache/'metadata.json',cache_meta);cached=False
    assert embeddings.shape==(len(docs),768) and np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-5)
    start=time.perf_counter(); index=faiss.IndexFlatIP(768);index.add(embeddings);index_seconds=time.perf_counter()-start
    faiss.write_index(index,str(cache/'index.faiss'))
    for q in dev_ids[:5]: index.search(bge.encode([claims[q]['claim']],query=True),100)
    runs={};scores={};lat=[];encode_lat=[];search_lat=[]
    for q in dev_ids:
        synchronize(); start=time.perf_counter(); embedding=bge.encode([claims[q]['claim']],query=True);synchronize();mid=time.perf_counter()
        values,idx=index.search(embedding,100);end=time.perf_counter()
        lat.append((end-start)*1000);encode_lat.append((mid-start)*1000);search_lat.append((end-mid)*1000)
        runs[q]=[doc_ids[i] for i in idx[0]]
        scores[q]={d:float(100-j) for j,d in enumerate(runs[q])}
    score_run('bge',runs,scores,lat,{'corpus_encoding_seconds':encode_seconds,'embeddings_reused':cached,'index_build_seconds':index_seconds,'query_encoding_latency':stats(encode_lat),'search_latency':stats(search_lat),'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'embedding_bytes':embeddings.nbytes,'cache_key':key,'sanity_shape':list(a.shape),'repeat_max_abs_diff':float(np.max(np.abs(a-b))),'score_file_semantics':'rank-derived scores, not cosine scores'})
    # Record a few training-only nearest neighbors for inspection, without changing settings.
    values,idx=index.search(a,3)
    write(OUT/'bge-training-sanity.json',[{'query_id':q,'claim':claims[q]['claim'],'top3':[{'doc_id':doc_ids[i],'title':docs[i]['title'],'cosine':float(v)} for i,v in zip(indices,vs)]} for q,indices,vs in zip(split['train_ids'][:3],idx,values)])
    del bge;torch.cuda.empty_cache()
    print('Loading MedCPT encoders (sanity only)',flush=True)
    med=MedCPT(revisions)
    sample_docs=[corpus[str(claims[q]['cited_doc_ids'][0])] for q in split['train_ids'][:3]]
    mq=med.encode_queries(sample);ma=med.encode_articles(sample_docs)
    assert mq.shape==ma.shape==(3,768)
    np.testing.assert_allclose(mq,med.encode_queries(sample),atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(ma,med.encode_articles(sample_docs),atol=1e-7,rtol=1e-6)
    similarities=mq@ma.T;assert np.isfinite(similarities).all()
    pair_tokens=med.tokenizers['Article']([[d['title'],d['text']] for d in sample_docs],truncation=True,max_length=512)
    assert all(1 in t for t in pair_tokens['token_type_ids'])
    summary['medcpt_sanity']={'status':'passed','training_query_ids':split['train_ids'][:3],'query_shape':list(mq.shape),'article_shape':list(ma.shape),'score_matrix':similarities.tolist(),'pooling':'CLS','normalization':'none','score':'inner product','query_max_length':64,'article_max_length':512,'article_format':'tokenizer title/abstract pair; segment IDs checked','full_benchmark':False}
    write(OUT/'summary.json',summary)
    write(ROOT/'docs/scifact-dev-baselines.json',summary)
    print('COMPLETE',flush=True)
if __name__=='__main__': main()
