"""Exactly two one-seed runs; uses only training annotations and frozen DEV IDs."""
import csv,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import faiss
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from retrieval.training import embed,explicit_loss
from retrieval.metrics import evaluate

ROOT=Path(__file__).resolve().parents[1]
ART=Path(os.environ.get('RAG_ROOT',str(Path.home()/'rag')))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def rows(p,key):return {str(x[key]):x for x in map(json.loads,p.read_text().splitlines())}
def state_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():h.update(name.encode());h.update(p.cpu().numpy().tobytes())
    return h.hexdigest()
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

@torch.inference_mode()
def encode(model,tokenizer,texts,batch=32):
    return np.concatenate([embed(model,tokenizer(texts[i:i+batch],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')).cpu().numpy() for i in range(0,len(texts),batch)])

def main():
    cp=ROOT/'configs/training-4b-pilot-v1.json';c=json.loads(cp.read_text());t=c['training'];seed=t['development_seed']
    assert seed==20261002 and c['arms']==['random','hard'] and t['epochs']==3 and not c['loss']['in_batch_negatives']
    assert t['gradient_accumulation']==1 and t['precision']=='float32'
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4);faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    out=ART/'runs/milestone4b-pilot-v1';out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',c)
    sp=ROOT/c['split'];assert sha(sp)==c['split_sha256'];split=json.loads(sp.read_text())
    data=ART/'data/scifact'
    for f,h in split['input_sha256'].items():assert sha(data/f)==h
    corpus=rows(data/'beir/corpus.jsonl','_id');claims=rows(data/'original/claims_train.jsonl','id')
    ids=sorted(corpus,key=int);texts=[corpus[d]['title']+' '+corpus[d]['text'] for d in ids]
    textmap=dict(zip(ids,texts));dev=split['dev_ids'];prefix=c['model']['query_prefix']
    qrels={q:{} for q in dev}
    for r in csv.DictReader((data/'beir/qrels/train.tsv').open(),delimiter='\t'):
        if r['query-id'] in qrels:qrels[r['query-id']][r['corpus-id']]=int(r['score'])
    arms={}
    for arm in c['arms']:
        p=ART/c['data'][arm]['path'];assert sha(p)==c['data'][arm]['sha256']
        arms[arm]=list(map(json.loads,p.read_text().splitlines()))
        assert {r['query_id'] for r in arms[arm]}==set(split['train_ids'])
    assert [(r['query_id'],r['positive_id']) for r in arms['random']]==[(r['query_id'],r['positive_id']) for r in arms['hard']]
    summary={'seed':seed,'config_sha256':sha(cp),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'arms':{}}
    initial_hash=None
    for arm in c['arms']:
        seed_all(seed)
        model=AutoModel.from_pretrained(c['base_model'],revision=c['revision'],local_files_only=True).cuda()
        tokenizer=AutoTokenizer.from_pretrained(c['base_model'],revision=c['revision'],local_files_only=True)
        h=state_hash(model)
        if initial_hash is None:initial_hash=h
        assert h==initial_hash
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        no_decay=c['training']['no_decay_parameters']
        groups=[{'params':[p for n,p in model.named_parameters() if not any(s in n for s in no_decay)],'weight_decay':t['weight_decay']},{'params':[p for n,p in model.named_parameters() if any(s in n for s in no_decay)],'weight_decay':0.0}]
        opt=torch.optim.AdamW(groups,lr=t['learning_rate'],betas=tuple(t['betas']),eps=t['epsilon'])
        scheduler=get_linear_schedule_with_warmup(opt,t['warmup_steps'],t['total_optimizer_steps'])
        seed_all(seed)
        armout=out/arm;armout.mkdir();results=[];step=0
        probe=model.embeddings.word_embeddings.weight.detach().clone()
        for epoch in range(1,4):
            model.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            order=list(range(len(arms[arm])));random.Random(seed+epoch).shuffle(order)
            losses=[];norms=[];lrs=[];step_records=[]
            for offset in range(0,len(order),t['batch_size']):
                batch=[arms[arm][i] for i in order[offset:offset+t['batch_size']]]
                queries=[prefix+claims[r['query_id']]['claim'] for r in batch]
                documents=[textmap[d] for r in batch for d in [r['positive_id']]+r['negative_ids']]
                qt=tokenizer(queries,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                dt=tokenizer(documents,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                opt.zero_grad(set_to_none=True)
                loss,qe,de=explicit_loss(model,qt,dt,c['loss']['temperature'])
                if step==0:qe.retain_grad();de.retain_grad()
                assert torch.isfinite(loss)
                loss.backward()
                if step==0:
                    assert qe.grad.abs().sum()>0 and de.grad.abs().sum()>0
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),t['max_gradient_norm'],error_if_nonfinite=True)
                lr=opt.param_groups[0]['lr'];opt.step();scheduler.step();step+=1
                if step==2:assert not torch.equal(probe,model.embeddings.word_embeddings.weight)
                losses.append((float(loss.detach()),len(batch)));norms.append(float(norm));lrs.append(lr)
                step_records.append({'step':step,'loss':losses[-1][0],'batch_size':len(batch),'gradient_norm_before_clip':norms[-1],'learning_rate':lr})
                if step%10==0:print(arm,'epoch',epoch,'step',step,'loss',losses[-1][0],flush=True)
            torch.cuda.synchronize();train_seconds=time.perf_counter()-start
            assert step==epoch*t['steps_per_epoch']
            peak=torch.cuda.max_memory_allocated()/2**20
            checkpoint=ART/'checkpoints/milestone4b-pilot-v1'/arm/f'epoch-{epoch}';checkpoint.mkdir(parents=True,exist_ok=False)
            model.save_pretrained(checkpoint);tokenizer.save_pretrained(checkpoint)
            torch.save({'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'step':step,'seed':seed,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},checkpoint/'training_state.pt')
            model.eval();evalstart=time.perf_counter()
            embeddings=encode(model,tokenizer,texts)
            assert embeddings.shape==(5183,768) and np.isfinite(embeddings).all()
            index=faiss.IndexFlatIP(768);index.add(embeddings)
            query=encode(model,tokenizer,[prefix+claims[q]['claim'] for q in dev])
            _,idx=index.search(query,100);runs={q:[ids[i] for i in ix] for q,ix in zip(dev,idx)}
            metrics,per=evaluate(qrels,runs)
            result={'epoch':epoch,'step':step,'metrics':metrics,'train_loss':sum(v*n for v,n in losses)/sum(n for _,n in losses),'gradient_norm_mean':float(np.mean(norms)),'learning_rate_first':lrs[0],'learning_rate_last':lrs[-1],'train_seconds':train_seconds,'eval_seconds':time.perf_counter()-evalstart,'peak_allocated_mib':peak,'checkpoint':str(checkpoint.relative_to(ART)),'row_order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),'model_sha256':sha(checkpoint/'model.safetensors')}
            save(armout/f'epoch-{epoch}.json',result);save(armout/f'epoch-{epoch}-steps.json',step_records);save(armout/f'epoch-{epoch}-per-query.json',per);save(armout/f'epoch-{epoch}-rankings.json',runs)
            results.append(result);summary['arms'][arm]={'initial_state_sha256':h,'epochs':results}
            save(out/'summary.json',summary)
            print('EPOCH_RESULT',arm,json.dumps(result),flush=True)
        best=max(results,key=lambda r:r['metrics']['NDCG@10'])
        summary['arms'][arm]['selected_epoch']=best['epoch'];summary['arms'][arm]['selected_checkpoint']=best['checkpoint']
        save(out/'summary.json',summary)
        del model,opt,scheduler,groups,probe,qe,de,loss;torch.cuda.empty_cache()
    assert all(summary['arms']['random']['epochs'][i]['row_order_sha256']==summary['arms']['hard']['epochs'][i]['row_order_sha256'] for i in range(3))
    save(ROOT/'docs/milestone4b-pilot-results.json',summary)
    print('PILOT_COMPLETE',flush=True)
if __name__=='__main__':main()
