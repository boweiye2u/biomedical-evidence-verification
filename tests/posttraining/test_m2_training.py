import json
from pathlib import Path
import pytest
from posttraining.training.m2_mixed import make_shared_folds,select_ranked,validate_mixture,train_run,encode_training_row
from posttraining.training.m1_gold import encode_training_row as encode_m1
from tests.posttraining.test_m1_gold import TinyTokenizer,row as gold_row

def mixed_row(claim='1',condition='A_gold',label='SUPPORT'):
 r=gold_row(claim,label,condition);r.update({'mixture_name':'mix-1','sampling_seed':7})
 if condition!='A_gold':r.update({'pair_annotation':None,'context_label':'INSUFFICIENT','rationale_sentences':[]})
 return r

def test_shared_folds_keep_claims_together_across_mixtures():
 mixtures={'mix-1':[mixed_row(str(i)) for i in range(1,21)],'mix-2':[mixed_row(str(i),'D_random_negative') for i in range(1,21)]}
 folds=make_shared_folds(mixtures,5,17,[str(i) for i in range(1,21)])
 sets=list(map(set,folds));assert len(set().union(*sets))==sum(map(len,sets))==20

def test_mixture_validation_rejects_dev_and_semantic_errors():
 spec={'rows':1,'counts':{'A':1},'sampling_seed':7};r=mixed_row()
 validate_mixture('mix-1',[r],spec,{'1'},{'2'})
 with pytest.raises(ValueError,match='leakage'):validate_mixture('mix-1',[r],spec,{'1'},{'1'})
 bad=mixed_row(condition='C_hard_negative');bad['rationale_sentences']=[0]
 spec={'rows':1,'counts':{'C':1},'sampling_seed':7}
 with pytest.raises(ValueError,match='negative'):validate_mixture('mix-1',[bad],spec,{'1'},set())

def test_frozen_ranking_uses_macro_f1_then_simplicity():
 cfg={'cross_validation':{'tie_margin':.005},'mixtures':{'mix-1':{'rows':430},'mix-2':{'rows':573}}}
 xs=[{'mixture':'mix-2','epoch':3,'mean_macro_f1':.700},{'mixture':'mix-1','epoch':2,'mean_macro_f1':.697},{'mixture':'mix-2','epoch':2,'mean_macro_f1':.68}]
 ranked=select_ranked(xs,cfg);assert (ranked[0]['mixture'],ranked[0]['epoch'])==('mix-1',2)

def test_output_only_masking_is_identical_to_m1():
 t=TinyTokenizer();r=gold_row();assert encode_training_row(r,t,10)==encode_m1(r,t,10)

def test_m2_config_pins_m1_hyperparameters_and_revisions():
 root=Path(__file__).resolve().parents[2];m1=json.loads((root/'configs/posttraining/train-m1-lora-gold-v1.json').read_text());m2=json.loads((root/'configs/posttraining/train-m2-lora-mixed-v1.json').read_text())
 for key in ('model_revision','tokenizer_revision','lora'):assert m2[key]==m1[key]
 for key in ('micro_batch_size','gradient_accumulation','learning_rate','weight_decay','scheduler','warmup_ratio','candidate_epochs','max_gradient_norm','precision','gradient_checkpointing','max_sequence_length','prompt_loss_mask'):assert m2['optimization'][key]==m1['optimization'][key]
 assert m2['final_training']['seeds']==[0,1,2] and m2['evaluation']['fixed_benchmark_access_allowed'] is False

def test_m2_run_directories_cannot_be_overwritten(tmp_path):
 out=tmp_path/'run';out.mkdir()
 with pytest.raises(FileExistsError):train_run({},[],[],0,0,out,tmp_path/'ckpt',set(),tmp_path/'cfg','x',{})
 ckpt=tmp_path/'ckpt2';ckpt.mkdir()
 with pytest.raises(FileExistsError):train_run({},[],[],0,0,tmp_path/'run2',ckpt,set(),tmp_path/'cfg','x',{})
