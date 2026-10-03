"""Replicate the frozen 4B pilot for seeds 20261003 and 20261004 only."""
import csv, hashlib, json, os, random, time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from retrieval.metrics import evaluate
from retrieval.training import embed, explicit_loss

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("RAG_ROOT", str(Path.home() / "rag")))
SEEDS = [20261003, 20261004]
PILOT_CONFIG_SHA256 = "035c39c5c0409836d5c6415cd31f48a668b402b7930e87579d06316a56073e04"
ORIGINAL_CONFIG_SHA256 = "5043b73f81383488ba0e0d2b95a7263268784548db421cb529943713cb7cb225"
PILOT_SUMMARY_SHA256 = "c5fdac0a897a7cdd23d37c8ff836d67791c3b726343728f1c4f2b8253a97ad42"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def rows(path, key):
    return {str(value[key]): value for value in map(json.loads, path.read_text().splitlines())}


def state_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode())
        digest.update(parameter.cpu().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def encode(model, tokenizer, texts, batch=32):
    return np.concatenate([
        embed(
            model,
            tokenizer(
                texts[start:start + batch], padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to("cuda"),
        ).cpu().numpy()
        for start in range(0, len(texts), batch)
    ])


def main():
    pilot_config = ROOT / "configs/training-4b-pilot-v1.json"
    original_config = ROOT / "configs/training-random-vs-hard-v1.json"
    pilot_summary = ROOT / "docs/milestone4b-pilot-results.json"
    assert sha(pilot_config) == PILOT_CONFIG_SHA256
    assert sha(original_config) == ORIGINAL_CONFIG_SHA256
    assert sha(pilot_summary) == PILOT_SUMMARY_SHA256
    config = json.loads(pilot_config.read_text())
    training = config["training"]
    assert config["arms"] == ["random", "hard"]
    assert training["final_seeds"] == [20261002, 20261003, 20261004]
    assert training["epochs"] == 3 and training["total_optimizer_steps"] == 141
    assert training["gradient_accumulation"] == 1 and training["precision"] == "float32"
    assert not config["loss"]["in_batch_negatives"]
    assert torch.cuda.device_count() == 1

    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    output = ART / "runs/milestone4b-replication-v1"
    output.mkdir(parents=True, exist_ok=False)
    save(output / "frozen-config.json", config)
    split_path = ROOT / config["split"]
    assert sha(split_path) == config["split_sha256"]
    split = json.loads(split_path.read_text())
    data = ART / "data/scifact"
    for filename, expected in split["input_sha256"].items():
        assert sha(data / filename) == expected

    corpus = rows(data / "beir/corpus.jsonl", "_id")
    claims = rows(data / "original/claims_train.jsonl", "id")
    document_ids = sorted(corpus, key=int)
    document_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in document_ids]
    text_by_id = dict(zip(document_ids, document_texts))
    dev_ids = split["dev_ids"]
    prefix = config["model"]["query_prefix"]

    # Load only training qrels, then filter them to the frozen DEV IDs.
    qrels = {query_id: {} for query_id in dev_ids}
    qrels_path = data / "beir/qrels/train.tsv"
    assert qrels_path.name == "train.tsv"
    for row in csv.DictReader(qrels_path.open(), delimiter="\t"):
        if row["query-id"] in qrels:
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    assert set(qrels) == set(dev_ids) and all(qrels.values())

    arms = {}
    for arm in config["arms"]:
        path = ART / config["data"][arm]["path"]
        assert sha(path) == config["data"][arm]["sha256"]
        arms[arm] = list(map(json.loads, path.read_text().splitlines()))
        assert {row["query_id"] for row in arms[arm]} == set(split["train_ids"])
    assert [(row["query_id"], row["positive_id"]) for row in arms["random"]] == [
        (row["query_id"], row["positive_id"]) for row in arms["hard"]
    ]

    summary = {
        "seeds": SEEDS,
        "frozen_config_sha256": sha(pilot_config),
        "original_config_sha256": sha(original_config),
        "pilot_summary_sha256_before": sha(pilot_summary),
        "qrels_path": str(qrels_path.relative_to(ART)),
        "test_qrels_loaded": False,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "runs": {},
    }
    canonical_base_hash = None

    for seed in SEEDS:
        seed_summary = {"arms": {}}
        for arm in config["arms"]:
            seed_all(seed)
            model = AutoModel.from_pretrained(
                config["base_model"], revision=config["revision"], local_files_only=True,
            ).cuda()
            tokenizer = AutoTokenizer.from_pretrained(
                config["base_model"], revision=config["revision"], local_files_only=True,
            )
            initial_hash = state_hash(model)
            if canonical_base_hash is None:
                canonical_base_hash = initial_hash
            assert initial_hash == canonical_base_hash
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            no_decay = training["no_decay_parameters"]
            parameter_groups = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(term in n for term in no_decay)],
                    "weight_decay": training["weight_decay"],
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(term in n for term in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(
                parameter_groups, lr=training["learning_rate"],
                betas=tuple(training["betas"]), eps=training["epsilon"],
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer, training["warmup_steps"], training["total_optimizer_steps"],
            )
            seed_all(seed)
            arm_output = output / f"seed-{seed}" / arm
            arm_output.mkdir(parents=True)
            results = []
            step = 0
            parameter_probe = model.embeddings.word_embeddings.weight.detach().clone()

            for epoch in range(1, 4):
                model.train()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                order = list(range(len(arms[arm])))
                random.Random(seed + epoch).shuffle(order)
                losses, norms, learning_rates, traces = [], [], [], []

                for offset in range(0, len(order), training["batch_size"]):
                    batch = [arms[arm][index] for index in order[offset:offset + training["batch_size"]]]
                    query_texts = [prefix + claims[row["query_id"]]["claim"] for row in batch]
                    candidate_texts = [
                        text_by_id[doc_id]
                        for row in batch
                        for doc_id in [row["positive_id"]] + row["negative_ids"]
                    ]
                    query_tokens = tokenizer(
                        query_texts, padding=True, truncation=True, max_length=512,
                        return_tensors="pt",
                    ).to("cuda")
                    document_tokens = tokenizer(
                        candidate_texts, padding=True, truncation=True, max_length=512,
                        return_tensors="pt",
                    ).to("cuda")
                    optimizer.zero_grad(set_to_none=True)
                    loss, query_embeddings, document_embeddings = explicit_loss(
                        model, query_tokens, document_tokens, config["loss"]["temperature"],
                    )
                    if step == 0:
                        query_embeddings.retain_grad()
                        document_embeddings.retain_grad()
                    assert torch.isfinite(loss)
                    loss.backward()
                    if step == 0:
                        assert query_embeddings.grad.abs().sum() > 0
                        assert document_embeddings.grad.abs().sum() > 0
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), training["max_gradient_norm"], error_if_nonfinite=True,
                    )
                    learning_rate = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    if step == 2:
                        assert not torch.equal(parameter_probe, model.embeddings.word_embeddings.weight)
                    losses.append((float(loss.detach()), len(batch)))
                    norms.append(float(norm))
                    learning_rates.append(learning_rate)
                    traces.append({
                        "step": step, "loss": losses[-1][0], "batch_size": len(batch),
                        "gradient_norm_before_clip": norms[-1], "learning_rate": learning_rate,
                    })
                    if step % 10 == 0:
                        print(seed, arm, "epoch", epoch, "step", step, "loss", losses[-1][0], flush=True)

                torch.cuda.synchronize()
                training_seconds = time.perf_counter() - started
                assert step == epoch * training["steps_per_epoch"]
                peak_memory = torch.cuda.max_memory_allocated() / 2**20
                checkpoint = ART / "checkpoints/milestone4b-replication-v1" / f"seed-{seed}" / arm / f"epoch-{epoch}"
                checkpoint.mkdir(parents=True, exist_ok=False)
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                torch.save({
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "step": step, "seed": seed, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                }, checkpoint / "training_state.pt")

                model.eval()
                evaluation_started = time.perf_counter()
                corpus_embeddings = encode(model, tokenizer, document_texts)
                assert corpus_embeddings.shape == (5183, 768) and np.isfinite(corpus_embeddings).all()
                index = faiss.IndexFlatIP(768)
                index.add(corpus_embeddings)
                query_embeddings = encode(
                    model, tokenizer, [prefix + claims[query_id]["claim"] for query_id in dev_ids],
                )
                _, indices = index.search(query_embeddings, 100)
                rankings = {
                    query_id: [document_ids[index] for index in positions]
                    for query_id, positions in zip(dev_ids, indices)
                }
                metrics, per_query = evaluate(qrels, rankings)
                result = {
                    "seed": seed, "arm": arm, "epoch": epoch, "step": step,
                    "metrics": metrics,
                    "train_loss": sum(value * size for value, size in losses) / sum(size for _, size in losses),
                    "gradient_norm_mean": float(np.mean(norms)),
                    "learning_rate_first": learning_rates[0], "learning_rate_last": learning_rates[-1],
                    "train_seconds": training_seconds,
                    "eval_seconds": time.perf_counter() - evaluation_started,
                    "peak_allocated_mib": peak_memory,
                    "checkpoint": str(checkpoint.relative_to(ART)),
                    "row_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                    "model_sha256": sha(checkpoint / "model.safetensors"),
                }
                save(arm_output / f"epoch-{epoch}.json", result)
                save(arm_output / f"epoch-{epoch}-steps.json", traces)
                save(arm_output / f"epoch-{epoch}-per-query.json", per_query)
                save(arm_output / f"epoch-{epoch}-rankings.json", rankings)
                results.append(result)
                seed_summary["arms"][arm] = {"initial_state_sha256": initial_hash, "epochs": results}
                summary["runs"][str(seed)] = seed_summary
                save(output / "summary.json", summary)
                print("EPOCH_RESULT", json.dumps(result), flush=True)

            best = max(results, key=lambda result: result["metrics"]["NDCG@10"])
            seed_summary["arms"][arm]["selected_epoch"] = best["epoch"]
            seed_summary["arms"][arm]["selected_checkpoint"] = best["checkpoint"]
            summary["runs"][str(seed)] = seed_summary
            save(output / "summary.json", summary)
            del model, optimizer, scheduler, parameter_groups, parameter_probe
            del query_embeddings, document_embeddings, corpus_embeddings, loss
            torch.cuda.empty_cache()

        assert seed_summary["arms"]["random"]["initial_state_sha256"] == seed_summary["arms"]["hard"]["initial_state_sha256"]
        assert all(
            seed_summary["arms"]["random"]["epochs"][index]["row_order_sha256"]
            == seed_summary["arms"]["hard"]["epochs"][index]["row_order_sha256"]
            for index in range(3)
        )
    summary["canonical_base_state_sha256"] = canonical_base_hash
    summary["pilot_summary_sha256_after"] = sha(pilot_summary)
    assert summary["pilot_summary_sha256_after"] == PILOT_SUMMARY_SHA256
    save(output / "summary.json", summary)
    print("REPLICATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
