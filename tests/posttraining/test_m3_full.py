import json,os,socket
from pathlib import Path
import pytest,torch,torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,FullStateDictConfig,StateDictType
from posttraining.training.m1_gold import encode_training_row,read_jsonl,sha256
from posttraining.evaluation.m3_benchmark import validate as validate_m3_benchmark

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/'configs/posttraining/train-m3-full-ft-v1.json'

def config(): return json.loads(CFG.read_text())

def test_m3_exact_frozen_mix2_and_no_split_leakage():
 c=config();rows=read_jsonl(Path(c['training_data']['path']));split=json.loads((ROOT/'configs/splits/scifact_train_dev_v1.json').read_text())
 assert len(rows)==573 and sha256(Path(c['training_data']['path']))==c['training_data']['sha256']=='7ca1013d3ff393eddb805f0e3ce5b5dc85b18055775c264dcdc073867514dfeb'
 names={'A_gold':'A','B_retrieved_non_gold':'B','C_hard_negative':'C','D_random_negative':'D'}
 assert {k:sum(names[r['condition']]==k for r in rows) for k in 'ABCD'}=={'A':344,'B':86,'C':86,'D':57}
 ids={r['claim_id'] for r in rows};assert ids<=set(map(str,split['train_ids'])) and not ids&set(map(str,split['dev_ids']))

def test_all_parameters_and_no_lora_are_frozen_requirements():
 c=config();assert c['optimization']['all_parameters_trainable'] is True and c['optimization']['lora_active'] is False
 assert 'lora' not in c and c['fsdp']['sharding_strategy']=='FULL_SHARD'

def test_loss_masking_reuses_m2_encoder():
 from transformers import AutoTokenizer
 c=config();row=read_jsonl(Path(c['training_data']['path']))[0];tok=AutoTokenizer.from_pretrained(c['local_model_path'],local_files_only=True)
 encoded=encode_training_row(row,tok,c['optimization']['max_sequence_length'])
 first=next(i for i,x in enumerate(encoded['labels']) if x!=-100)
 assert all(x==-100 for x in encoded['labels'][:first]) and all(x!=-100 for x in encoded['labels'][first:])

def test_scaling_holds_global_batch_and_only_accumulation_varies():
 c=json.loads((ROOT/'configs/posttraining/fsdp-scaling-v1.json').read_text());assert c['global_batch_size']==16 and c['micro_batch_size_per_gpu']==2
 assert c['gradient_accumulation']=={'1':8,'2':4,'4':2}
 assert c['only_vary']==['gpu_count','gradient_accumulation'] and c['cpu_offload'] is False
 for n,a in c['gradient_accumulation'].items(): assert int(n)*c['micro_batch_size_per_gpu']*a==16

def test_completed_output_guard_is_present():
 text=(ROOT/'posttraining/training/m3_full.py').read_text();assert 'if exists.item(): raise FileExistsError(output)' in text

def test_environment_and_hashes_recorded():
 c=config();assert c['environment'] and c['git_commit'] and c['training_data']['sha256'] and c['cross_validation']['fold_file_sha256']
 assert sha256(ROOT/c['cross_validation']['fold_file'])==c['cross_validation']['fold_file_sha256']


def test_m3_benchmark_runner_cannot_generate_m2_and_requires_freeze(tmp_path):
 c={'status':'not_frozen','systems':{'M2':{}},'run_root':str(tmp_path/'run')}
 p=tmp_path/'config.json';p.write_text(json.dumps(c))
 with pytest.raises(ValueError,match='not frozen'):
  validate_m3_benchmark(p,c)
 text=(ROOT/'posttraining/evaluation/m3_benchmark.py').read_text()
 assert 'set(config["systems"]) != {"M3"}' in text
 assert 'M2_predictions_reused' in text and 'main_benchmark_rerun' in text

def test_fsdp_full_state_dict_roundtrip_on_tiny_model(tmp_path):
 if not torch.cuda.is_available(): pytest.skip('CUDA required for FSDP smoke')
 if dist.is_initialized(): pytest.skip('process group already active')
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
 os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  original=torch.nn.Linear(4,3).cuda();wrapped=FSDP(original,device_id=0,use_orig_params=True)
  with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
   state=wrapped.state_dict()
  restored=torch.nn.Linear(4,3);restored.load_state_dict(state)
  assert torch.equal(restored.weight,state['weight'].cpu()) and torch.equal(restored.bias,state['bias'].cpu())
 finally: dist.destroy_process_group()
