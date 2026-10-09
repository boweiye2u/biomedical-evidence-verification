import json
from pathlib import Path

import pytest

from posttraining.training.m1_gold import (
    collate, encode_training_row, make_grouped_folds, target_text, validate_gold,
)


class TinyTokenizer:
    eos_token_id=0
    pad_token_id=0
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        prompt=[1,2,3]
        return prompt if add_generation_prompt else prompt+[4,5,0]


def row(claim_id="1", label="SUPPORT", condition="A_gold"):
    return {"claim_id":claim_id,"doc_id":"10","condition":condition,"source_split":"TRAIN","claim":"c","evidence":{"title":"t","abstract":["a","b"]},"claim_gold_label":label,"pair_annotation":label,"context_label":label,"rationale_sentences":[0]}


def test_output_only_masking_and_structured_target():
    encoded=encode_training_row(row(),TinyTokenizer(),10)
    assert encoded["labels"][:3]==[-100,-100,-100]
    assert encoded["labels"][3:]==[4,5,0]
    assert json.loads(target_text(row()))=={"decision":"SUPPORT","rationale_sentences":[0]}


def test_grouped_folds_have_zero_claim_overlap_and_are_reproducible():
    rows=[row(str(i),"SUPPORT" if i%2 else "CONTRADICT") for i in range(30)]+[row("1")]
    first=make_grouped_folds(rows,5,17);second=make_grouped_folds(rows,5,17)
    assert first==second
    assert set().union(*map(set,first))=={str(i) for i in range(30)}
    assert sum(len(x) for x in first)==30


def test_gold_validation_rejects_non_a_and_dev():
    cfg={"training_data":{"rows":1,"support":1,"contradict":0,"condition":"A_gold"}}
    validate_gold([row()],cfg,{"1"},{"2"})
    bad=row(condition="B_retrieved_non_gold")
    with pytest.raises(ValueError,match="Condition A"):
        validate_gold([bad],cfg,{"1"},{"2"})
    with pytest.raises(ValueError,match="leakage"):
        validate_gold([row()],cfg,{"1"},{"1"})


def test_collator_preserves_mask():
    batch=collate([{"input_ids":[1,2],"labels":[-100,2]},{"input_ids":[1],"labels":[-100]}],0)
    assert batch["labels"].tolist()==[[-100,2],[-100,-100]]


def test_frozen_config_is_gold_only_pinned_and_records_seeds():
    root=Path(__file__).resolve().parents[2];cfg=json.loads((root/"configs/posttraining/train-m1-lora-gold-v1.json").read_text())
    assert cfg["training_data"]["condition"]=="A_gold"
    assert len(cfg["model_revision"])==len(cfg["tokenizer_revision"])==40
    assert cfg["final_training"]["seeds"]==[0,1,2]
    assert cfg["lora"]["target_modules"]==["q_proj","v_proj"]
    assert cfg["evaluation"]["fixed_benchmark_access_allowed"] is False


def test_completed_output_directory_is_never_overwritten(tmp_path):
    from posttraining.training.m1_gold import train_run
    output=tmp_path/"complete";output.mkdir()
    with pytest.raises(FileExistsError):
        train_run({},[],[],0,0,output,tmp_path/"checkpoints",set())


def test_validation_rejects_pair_mismatch_and_bad_rationale():
    cfg={"training_data":{"rows":1,"support":1,"contradict":0,"condition":"A_gold"}}
    bad=row();bad["pair_annotation"]="CONTRADICT"
    with pytest.raises(ValueError,match="Pair/context"):
        validate_gold([bad],cfg,{"1"},set())
    bad=row();bad["rationale_sentences"]=[2]
    with pytest.raises(ValueError,match="rationale"):
        validate_gold([bad],cfg,{"1"},set())


def test_validation_rejects_non_train_and_non_binary_rows():
    cfg={"training_data":{"rows":1,"support":0,"contradict":1,"condition":"A_gold"}}
    bad=row(label="CONTRADICT");bad["source_split"]="DEV"
    with pytest.raises(ValueError,match="Condition A"):
        validate_gold([bad],cfg,{"1"},set())
    bad=row(label="INSUFFICIENT")
    cfg["training_data"].update({"support":0,"contradict":0})
    with pytest.raises(ValueError):
        validate_gold([bad],cfg,{"1"},set())


def test_real_frozen_fold_file_has_disjoint_claim_groups():
    root=Path(__file__).resolve().parents[2]
    payload=json.loads((root/"configs/posttraining/m1-gold-cv-folds-v1.json").read_text())
    sets=[set(fold["validation_claim_ids"]) for fold in payload["folds"]]
    assert len(sets)==5
    assert sum(map(len,sets))==len(set().union(*sets))==407


def test_config_freezes_optimizer_candidates_and_data_hash():
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/"configs/posttraining/train-m1-lora-gold-v1.json").read_text())
    assert cfg["optimization"]["candidate_epochs"]==[2,3]
    assert cfg["optimization"]["effective_batch_size"]==16
    assert cfg["optimization"]["prompt_loss_mask"] is True
    assert len(cfg["training_data"]["sha256"])==64


def test_timing_metrics_contract_and_gpu_hours():
    from posttraining.evaluation.finalize_m1 import REQUIRED_TIMING_FIELDS,validate_training_metrics
    metrics={key:1 for key in REQUIRED_TIMING_FIELDS};metrics.update({"gpu_count":1,"wall_clock_hours":2,"gpu_hours":2})
    validate_training_metrics(metrics)
    del metrics["tokens_processed"]
    with pytest.raises(ValueError,match="tokens_processed"):
        validate_training_metrics(metrics)


def test_completed_checkpoint_directory_is_never_overwritten(tmp_path):
    from posttraining.training.m1_gold import train_run
    checkpoint=tmp_path/"checkpoints";checkpoint.mkdir()
    with pytest.raises(FileExistsError):
        train_run({},[],[],0,0,tmp_path/"run",checkpoint,set())


def test_finalist_was_frozen_before_dev_for_all_predefined_seeds():
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/"configs/posttraining/m1-gold-finalist-v1.json").read_text())
    assert cfg["status"]=="frozen_after_cv_before_dev"
    assert cfg["selected_epochs"]==3
    assert cfg["final_seeds"]==[0,1,2]
    assert cfg["dev_selection_prohibited"] is True
    assert cfg["fixed_benchmark_access_allowed"] is False
