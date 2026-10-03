"""Milestone 4A: MedCPT DEV evaluation and TRAIN-only negative preparation. No training."""
import csv, hashlib, json, os, random, re, time
from pathlib import Path
import numpy as np
import torch
import faiss
import pytrec_eval
from beir.retrieval.evaluation import EvaluateRetrieval
from retrieval.medcpt import MedCPT
from retrieval.bge import BGE, MODEL
from retrieval.metrics import evaluate

ROOT=Path(__file__).resolve().parents[1]
ART=Path(os.environ.get('RAG_ROOT',str(Path.home()/'rag')))
DATA=ART/'data/scifact'
OUT=ART/'runs/milestone4a'
SEED=20261002

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n')
def load(path,key): return {str(r[key]):r for r in map(json.loads,path.read_text().splitlines())}
def fingerprint(doc):
    return hashlib.sha256(' '.join((doc['title']+' '+doc['text']).lower().split()).encode()).hexdigest()
def select_negatives(candidates, positives, fingerprints, count=5):
    excluded={fingerprints[d] for d in positives}; seen=set(); chosen=[]
    for d in candidates:
        f=fingerprints[d]
        if d in positives or f in excluded or f in seen: continue
        chosen.append(d);seen.add(f)
        if len(chosen)==count: return chosen
    raise ValueError('Insufficient unique negative candidates')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1
    split_path=ROOT/'configs/splits/scifact_train_dev_v1.json'
    split=json.loads(split_path.read_text())
    for f,h in split['input_sha256'].items(): assert sha(DATA/f)==h
    revisions=json.loads((ROOT/'configs/model-revisions.json').read_text())
    corpus=load(DATA/'beir/corpus.jsonl','_id');ids=sorted(corpus,key=int);docs=[corpus[d] for d in ids]
    claims=load(DATA/'original/claims_train.jsonl','id')
    qrels={}
    for r in csv.DictReader((DATA/'beir/qrels/train.tsv').open(),delimiter='\t'):
        qrels.setdefault(r['query-id'],{})[r['corpus-id']]=int(r['score'])
    med=MedCPT(revisions);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    cache_meta={'revisions':{k:v for k,v in revisions.items() if 'MedCPT' in k},'corpus_sha256':sha(DATA/'beir/corpus.jsonl'),'doc_ids':ids,'format':'title/abstract pair','pooling':'CLS','normalize':False,'article_max_length':512,'dtype':'float32','batch_size':32,'torch':torch.__version__}
    cache=ART/'embeddings'/('scifact-medcpt-'+hashlib.sha256(json.dumps(cache_meta,sort_keys=True).encode()).hexdigest()[:16]);cache.mkdir(parents=True,exist_ok=True)
    reused=(cache/'embeddings.npy').exists()
    if reused:
        assert json.loads((cache/'metadata.json').read_text())==cache_meta
        embeddings=np.load(cache/'embeddings.npy')
    else:
        embeddings=np.concatenate([med.encode_articles(docs[i:i+32]) for i in range(0,len(docs),32)])
        np.save(cache/'embeddings.npy',embeddings);save(cache/'metadata.json',cache_meta)
    torch.cuda.synchronize();enc_time=time.perf_counter()-start
    assert embeddings.shape==(5183,768) and np.isfinite(embeddings).all()
    index=faiss.IndexFlatIP(768);index.add(embeddings);faiss.write_index(index,str(cache/'index.faiss'))
    dev=split['dev_ids']
    for q in dev[:5]:index.search(med.encode_queries([claims[q]['claim']]),100)
    runs={};scores={};raw={};lat=[]
    for q in dev:
        torch.cuda.synchronize();start=time.perf_counter()
        emb=med.encode_queries([claims[q]['claim']]);torch.cuda.synchronize()
        values,indices=index.search(emb,100);lat.append(1000*(time.perf_counter()-start))
        runs[q]=[ids[i] for i in indices[0]]
        scores[q]={d:float(100-j) for j,d in enumerate(runs[q])}
        raw[q]={d:float(v) for d,v in zip(runs[q],values[0])}
    labels={q:qrels[q] for q in dev};means,per=evaluate(labels,runs)
    trusted=pytrec_eval.RelevanceEvaluator(labels,{'ndcg_cut_10','recall_10','recall_100'}).evaluate(scores)
    for q in dev:
        for a,b in [('NDCG@10','ndcg_cut_10'),('Recall@10','recall_10'),('Recall@100','recall_100')]: assert abs(per[q][a]-trusted[q][b])<1e-9
    assert abs(means['MRR@10']-EvaluateRetrieval.evaluate_custom(labels,scores,[10],metric='mrr')['MRR@10'])<1e-5
    med_summary={'metrics':means,'dev_queries':len(dev),'corpus_documents':len(ids),'corpus_encoding_seconds':None if reused else enc_time,'reused':reused,'mean_ms':float(np.mean(lat)),'p50_ms':float(np.percentile(lat,50)),'p95_ms':float(np.percentile(lat,95)),'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'revisions':cache_meta['revisions'],'cache':cache.name,'query_max_length':64,'article_max_length':512,'warmup':5,'query_batch_size':1,'cpu_threads':4,'device':torch.cuda.get_device_name(0)}
    save(OUT/'medcpt-summary.json',med_summary);save(OUT/'medcpt-rankings.json',scores);save(OUT/'medcpt-raw-scores.json',raw);save(OUT/'medcpt-per-query.json',per);save(OUT/'medcpt-latencies.json',dict(zip(dev,lat)))
    print('MedCPT',med_summary,flush=True)
    del med;torch.cuda.empty_cache()
    bge=BGE(revisions[MODEL])
    base=json.loads((ROOT/'docs/scifact-dev-baselines.json').read_text())
    bcache=ART/'embeddings'/('scifact-bge-'+base['systems']['bge']['cache_key'])
    meta=json.loads((bcache/'metadata.json').read_text())
    assert meta['revision']==revisions[MODEL] and meta['doc_ids']==ids and meta['corpus_sha256']==sha(DATA/'beir/corpus.jsonl')
    index=faiss.IndexFlatIP(768);index.add(np.load(bcache/'embeddings.npy'))
    train=split['train_ids'];query_emb=bge.encode([claims[q]['claim'] for q in train],query=True)
    values,indices=index.search(query_emb,50)
    fps={d:fingerprint(corpus[d]) for d in ids}
    rows_by={'random':[],'hard':[]}; pools={}; candidates={}
    for q,idx,vs in zip(train,indices,values):
        positives=set(qrels[q]); hard_candidates=[ids[i] for i in idx]
        shuffled=ids.copy();random.Random(SEED+int(q)).shuffle(shuffled)
        rnd=select_negatives(shuffled,positives,fps); hard=select_negatives(hard_candidates,positives,fps)
        pools[q]={'random':rnd,'hard':hard}
        candidates[q]=[{'doc_id':d,'score':float(v)} for d,v in zip(hard_candidates,vs)]
        for p in sorted(positives,key=int):
            for kind,neg in [('random',rnd),('hard',hard)]:
                rows_by[kind].append({'query_id':q,'positive_id':p,'negative_ids':neg})
    output=ART/'data/scifact/training_v1';output.mkdir(parents=True,exist_ok=True)
    for kind,records in rows_by.items():
        text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records)
        path=output/f'{kind}.jsonl'
        if path.exists(): assert path.read_text()==text,'Frozen data changed'
        else:path.write_text(text)
    save(OUT/'hard-candidates.json',candidates)
    paired=list(zip(rows_by['random'],rows_by['hard']))
    assert all((a['query_id'],a['positive_id'])==(b['query_id'],b['positive_id']) for a,b in paired)
    assert set(pools)==set(train) and not set(pools)&set(dev)
    for kind,rs in rows_by.items():
        for r in rs:
            assert len(r['negative_ids'])==5 and len({fps[d] for d in r['negative_ids']})==5
            assert not {fps[d] for d in r['negative_ids']} & {fps[d] for d in qrels[r['query_id']]}
    train_positive_docs={d for q in train for d in qrels[q]}
    audit={'seed':SEED,'train_queries':len(train),'positive_pairs_per_arm':len(paired),'negatives_per_pair':5,'groups_with_multiple_positives':sum(len(qrels[q])>1 for q in train),'corpus_duplicate_content_count':len(ids)-len(set(fps.values())),'random_hard_overlap_slots':sum(len(set(p['random'])&set(p['hard'])) for p in pools.values()),'cross_query_positive_negative_slots':{k:sum(d in train_positive_docs-set(qrels[q]) for q,p in pools.items() for d in p[k]) for k in rows_by},'known_positive_collisions':0,'negative_content_duplicates_within_query':0,'dev_queries_used_for_mining':0,'files':{k:{'path':f'data/scifact/training_v1/{k}.jsonl','sha256':sha(output/f'{k}.jsonl')} for k in rows_by},'split_sha256':sha(split_path),'corpus_sha256':sha(DATA/'beir/corpus.jsonl'),'bge_revision':revisions[MODEL]}
    sample=random.Random(SEED).sample(train,8)
    examples=[]
    for q in sample:
        examples.append({'query_id':q,'claim':claims[q]['claim'],'positive_titles':[corpus[d]['title'] for d in qrels[q]],'negatives':[{'kind':k,'doc_id':d,'title':corpus[d]['title'],'abstract':corpus[d]['text']} for k in ['hard','random'] for d in pools[q][k][:2]]})
    save(OUT/'audit-examples-full.json',examples)
    save(ROOT/'docs/milestone4a-results.json',{'medcpt':med_summary,'training_audit':audit})
    save(output/'manifest.json',audit)
    print('TRAINING AUDIT',audit,flush=True)
if __name__=='__main__':main()
