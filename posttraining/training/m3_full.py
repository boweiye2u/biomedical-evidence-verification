"""Frozen full-parameter Qwen2.5-3B training with PyTorch FSDP."""
from __future__ import annotations
import argparse,functools,hashlib,json,math,os,platform,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from transformers import AutoModelForCausalLM,AutoTokenizer,get_linear_schedule_with_warmup
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from posttraining.training.m1_gold import encode_training_row,collate,read_jsonl,sha256

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'configs/posttraining/train-m3-full-ft-v1.json'

class Encoded(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def setup():
    dist.init_process_group('nccl'); local=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local); return dist.get_rank(),local,dist.get_world_size()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def validate(config,rows):
    data=config['training_data']; assert len(rows)==data['rows']==573
    assert sha256(Path(data['path']))==data['sha256']=='7ca1013d3ff393eddb805f0e3ce5b5dc85b18055775c264dcdc073867514dfeb'
    condition_to_pool={'A_gold':'A','B_retrieved_non_gold':'B','C_hard_negative':'C','D_random_negative':'D'}
    counts={k:sum(condition_to_pool[r['condition']]==k for r in rows) for k in 'ABCD'}; assert counts==data['counts']
    split=json.loads((ROOT/'configs/splits/scifact_train_dev_v1.json').read_text()); train=set(map(str,split['train_ids']));dev=set(map(str,split['dev_ids']))
    assert all(r['source_split']=='TRAIN' and r['claim_id'] in train and r['claim_id'] not in dev for r in rows)
    assert sha256(ROOT/config['cross_validation']['fold_file'])==config['cross_validation']['fold_file_sha256']

def preflight(args):
    config=json.loads(args.config.read_text()); rows=read_jsonl(Path(config['training_data']['path']));validate(config,rows)
    model_config=json.loads((Path(config['local_model_path'])/'config.json').read_text()); params=3_085_938_688
    gib=2**30
    payload={'status':'passed_before_training','dataset_rows':len(rows),'dataset_sha256':config['training_data']['sha256'],'all_parameters_targeted':True,'lora_active':False,'max_sequence_tokens':None,'parameter_count_expected':params,'memory_estimate_gib':{'fp32_master_parameters':params*4/gib,'bf16_compute_parameters_transient':params*2/gib,'fp32_gradients':params*4/gib,'adam_first_moment_fp32':params*4/gib,'adam_second_moment_fp32':params*4/gib,'persistent_unsharded_subtotal':params*16/gib,'activation_estimate_with_checkpointing':4.0,'available_per_gpu':46068/1024},'conclusion':'one GPU expected to OOM before activation/temporary overhead; controlled fit required','model_architecture':model_config['architectures'],'config_sha256':sha256(args.config),'benchmark_loaded':False}
    tok=AutoTokenizer.from_pretrained(config['local_model_path'],local_files_only=True)
    enc=[encode_training_row(r,tok,config['optimization']['max_sequence_length']) for r in rows]
    payload['max_sequence_tokens']=max(x['length'] for x in enc);payload['max_answer_tokens']=max(x['answer_tokens'] for x in enc)
    out=Path.home()/'rag/runs/posttraining/full-ft-v1';out.mkdir(parents=True);(out/'preflight.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))

def make_model(config,local,world):
    model=AutoModelForCausalLM.from_pretrained(config['local_model_path'],local_files_only=True,torch_dtype=torch.float32).to(local)
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    assert all(p.requires_grad for p in model.parameters())
    parameter_count=sum(p.numel() for p in model.parameters())
    if world==1:
        return model,parameter_count,False
    policy=functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={Qwen2DecoderLayer})
    mp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
    wrapped=FSDP(model,sharding_strategy=ShardingStrategy.FULL_SHARD,auto_wrap_policy=policy,mixed_precision=mp,device_id=local,sync_module_states=True,forward_prefetch=False,limit_all_gathers=True,use_orig_params=True)
    return wrapped,parameter_count,True

def save_full(model,tokenizer,path,rank,manifest):
    cfg=FullStateDictConfig(offload_to_cpu=True,rank0_only=True)
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,cfg): state=model.state_dict()
    if rank==0:
        path.mkdir(parents=True);model.module.save_pretrained(path,state_dict=state,safe_serialization=True,max_shard_size='4GB');tokenizer.save_pretrained(path)
        (path.parent/'checkpoint-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dist.barrier()

def train(args):
    rank,local,world=setup();seed_all(args.seed)
    try:
        config=json.loads(args.config.read_text());rows=read_jsonl(Path(config['training_data']['path']));validate(config,rows)
        folds=json.loads((ROOT/config['cross_validation']['fold_file']).read_text())['folds']
        if args.fold is not None:
            valid=set(folds[args.fold]['validation_claim_ids']); rows=[r for r in rows if r['claim_id'] not in valid]
        micro=config['optimization']['micro_batch_size_per_gpu'];global_batch=config['optimization']['global_batch_size']
        if global_batch%(micro*world): raise ValueError('Global batch is not divisible by microbatch*world')
        accumulation=global_batch//(micro*world)
        if args.expected_accumulation is not None and accumulation!=args.expected_accumulation: raise ValueError('Accumulation mismatch')
        output=Path(args.output);checkpoint=Path(args.checkpoint) if args.checkpoint else None
        exists=torch.tensor(int(output.exists()),device=local);dist.all_reduce(exists,op=dist.ReduceOp.MAX)
        if exists.item(): raise FileExistsError(output)
        if rank==0: output.mkdir(parents=True)
        dist.barrier()
        tokenizer=AutoTokenizer.from_pretrained(config['local_model_path'],local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
        encoded=[encode_training_row(r,tokenizer,config['optimization']['max_sequence_length']) for r in rows]
        sampler=DistributedSampler(Encoded(encoded),num_replicas=world,rank=rank,shuffle=True,seed=args.seed,drop_last=False)
        loader=DataLoader(Encoded(encoded),batch_size=micro,sampler=sampler,collate_fn=lambda b:collate(b,tokenizer.pad_token_id),num_workers=0)
        updates_epoch=math.ceil(len(loader)/accumulation);horizon=updates_epoch*config['optimization']['scheduler_horizon_epochs']
        warmup=math.ceil(horizon*config['optimization']['warmup_ratio'])
        torch.cuda.reset_peak_memory_stats();model,param_count,is_fsdp=make_model(config,local,world)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimization']['learning_rate'],weight_decay=config['optimization']['weight_decay'],betas=tuple(config['optimization']['betas']),eps=config['optimization']['epsilon'])
        scheduler=get_linear_schedule_with_warmup(optimizer,warmup,horizon)
        steps=0;tokens=0;samples=0;trace=[];training_seconds=0.0;stop=False
        save_epochs=set(args.save_epochs)
        for epoch in range(1,args.epochs+1):
            sampler.set_epoch(epoch);model.train();optimizer.zero_grad(set_to_none=True);group=[];group_tokens=0;group_samples=0
            iterator=iter(loader);batch_index=0;epoch_start=time.perf_counter()
            while batch_index<len(loader):
                group_start=time.perf_counter();group=[];group_tokens=0;group_samples=0
                remaining=min(accumulation,len(loader)-batch_index)
                for inner in range(remaining):
                    batch=next(iterator);batch_index+=1;batch={k:v.to(local,non_blocking=True) for k,v in batch.items()}
                    sync=inner==remaining-1
                    context=model.no_sync() if is_fsdp and not sync else __import__('contextlib').nullcontext()
                    with context:
                        if is_fsdp:
                            loss=model(**batch).loss
                        else:
                            with torch.autocast(device_type='cuda',dtype=torch.bfloat16): loss=model(**batch).loss
                        loss.backward()
                    group.append(float(loss.detach()));group_tokens+=int(batch['attention_mask'].sum());group_samples+=batch['input_ids'].shape[0]
                scale=1/remaining
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(scale)
                grad=float(model.clip_grad_norm_(config['optimization']['max_gradient_norm']) if is_fsdp else torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimization']['max_gradient_norm']));optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);steps+=1
                local_counts=torch.tensor([group_tokens,group_samples],device=local,dtype=torch.long);dist.all_reduce(local_counts)
                tokens+=int(local_counts[0]);samples+=int(local_counts[1]);torch.cuda.synchronize();elapsed=time.perf_counter()-group_start
                max_elapsed=torch.tensor(elapsed,device=local);dist.all_reduce(max_elapsed,op=dist.ReduceOp.MAX)
                if rank==0:trace.append({'step':steps,'epoch':epoch,'loss':sum(group)/len(group),'gradient_norm':grad,'learning_rate':scheduler.get_last_lr()[0],'step_seconds':float(max_elapsed),'tokens':int(local_counts[0]),'samples':int(local_counts[1])})
                if args.max_steps and steps>=args.max_steps:stop=True;break
            torch.cuda.synchronize();epoch_elapsed=time.perf_counter()-epoch_start;max_epoch=torch.tensor(epoch_elapsed,device=local);dist.all_reduce(max_epoch,op=dist.ReduceOp.MAX);training_seconds+=float(max_epoch)
            if checkpoint and epoch in save_epochs and not args.max_steps:
                manifest={'epoch':epoch,'seed':args.seed,'fold':args.fold,'world_size':world,'optimizer_steps':steps,'config_sha256':sha256(args.config),'dataset_sha256':config['training_data']['sha256'],'model_revision':config['model_revision'],'tokenizer_revision':config['tokenizer_revision'],'fsdp':config['fsdp'],'git_commit':config['git_commit']}
                save_full(model,tokenizer,checkpoint/f'epoch-{epoch}'/'model',rank,manifest)
            if stop:break
        if checkpoint and args.save_at_end:
            manifest={'epoch':epoch,'seed':args.seed,'fold':args.fold,'world_size':world,'optimizer_steps':steps,'config_sha256':sha256(args.config),'dataset_sha256':config['training_data']['sha256'],'model_revision':config['model_revision'],'tokenizer_revision':config['tokenizer_revision'],'fsdp':config['fsdp'],'git_commit':config['git_commit'],'smoke_checkpoint':True}
            save_full(model,tokenizer,checkpoint/'smoke-model',rank,manifest)
        peak=torch.tensor(torch.cuda.max_memory_allocated()/2**20,device=local);peaks=[torch.zeros_like(peak) for _ in range(world)];dist.all_gather(peaks,peak)
        measured=[x for x in trace if x['step']>args.measure_after_step]
        payload={'status':'complete','mode':args.mode,'seed':args.seed,'fold':args.fold,'world_size':world,'gpu_model':torch.cuda.get_device_name(local),'rows':len(rows),'epochs_requested':args.epochs,'epochs_completed':epoch,'optimizer_steps':steps,'micro_batch_size_per_gpu':micro,'gradient_accumulation':accumulation,'global_batch_size':global_batch,'training_seconds':training_seconds,'gpu_hours':training_seconds*world/3600,'tokens_processed':tokens,'samples_processed':samples,'tokens_per_second':tokens/training_seconds,'samples_per_second':samples/training_seconds,'mean_step_seconds':statistics.fmean(x['step_seconds'] for x in trace) if trace else None,'measurement':{'after_step':args.measure_after_step,'steps':len(measured),'seconds':sum(x['step_seconds'] for x in measured),'tokens':sum(x['tokens'] for x in measured),'samples':sum(x['samples'] for x in measured),'tokens_per_second':sum(x['tokens'] for x in measured)/sum(x['step_seconds'] for x in measured) if measured else None,'samples_per_second':sum(x['samples'] for x in measured)/sum(x['step_seconds'] for x in measured) if measured else None},'peak_memory_mib_per_gpu':[float(x) for x in peaks],'parameter_count':param_count,'trainable_parameter_count':param_count,'all_parameters_trainable':True,'lora_active':False,'config_sha256':sha256(args.config),'dataset_sha256':config['training_data']['sha256'],'environment':{'python':platform.python_version(),'pytorch':torch.__version__,'cuda_runtime':torch.version.cuda,'nccl':torch.cuda.nccl.version()},'distributed':{'backend':dist.get_backend(),'sharding_strategy':'FULL_SHARD' if is_fsdp else 'NO_SHARD_NATIVE_SINGLE_GPU','cpu_offload':False,'auto_wrap':'Qwen2DecoderLayer','mixed_precision':'BF16 compute/reduce/buffers; FP32 master parameters'},'timing_trace':trace}
        if rank==0:(output/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:payload[k] for k in ['status','mode','world_size','optimizer_steps','training_seconds','gpu_hours','tokens_per_second','peak_memory_mib_per_gpu']},indent=2))
        dist.barrier()
    finally:
        if dist.is_initialized():dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('preflight');a.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    a=sub.add_parser('train');a.add_argument('--config',type=Path,default=DEFAULT_CONFIG);a.add_argument('--mode',choices=['scaling','cv','final'],required=True);a.add_argument('--output',required=True);a.add_argument('--checkpoint');a.add_argument('--seed',type=int,required=True);a.add_argument('--epochs',type=int,default=3);a.add_argument('--fold',type=int);a.add_argument('--save-epochs',type=int,nargs='*',default=[]);a.add_argument('--max-steps',type=int);a.add_argument('--measure-after-step',type=int,default=2);a.add_argument('--expected-accumulation',type=int);a.add_argument('--save-at-end',action='store_true')
    args=p.parse_args();preflight(args) if args.command=='preflight' else train(args)
if __name__=='__main__':
    import statistics
    main()
