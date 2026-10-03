# Retrieval environment

Run from the repository root in Bash. All installed packages and caches stay under
`~/rag`; the Conda base environment is not modified. No models or datasets are
needed for setup or verification.

## Recreate

```bash
source scripts/env.sh
conda env create --prefix "$RAG_ROOT/envs/retrieval" -f environments/retrieval.yml
conda activate "$RAG_ROOT/envs/retrieval"
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r environments/retrieval-requirements.txt
python -m pip check
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/check_retrieval_env.py
```

GPU 0 was idle during setup; choose an available allocated GPU for future runs.
The visibility setting above applies only to that command.

The YAML and requirements file describe portable intent. Exact installed Python
packages are recorded in `retrieval-pip-freeze.txt`; use that file instead of
`retrieval-requirements.txt` to replay those versions, adding
`--extra-index-url https://download.pytorch.org/whl/cu126` to resolve the CUDA wheel.
`retrieval-conda-linux-64.explicit.txt` records exact Conda package URLs for the
Linux bootstrap; recreate with `conda create --prefix ... --file ...`, then install
the pip freeze. Neither exact snapshot promises portability to other platforms.

PyTorch 2.9.1 CUDA 12.6 was selected from the official installation matrix:
https://pytorch.org/get-started/previous-versions/
Its wheel supplies the CUDA runtime; the system nvcc toolkit is not replaced.
FAISS is CPU-only. `pytrec-eval-terrier` supplies the `pytrec_eval` import used by
BEIR. Accelerate supports later Sentence Transformers training.

## Daily activation

```bash
source scripts/env.sh
conda activate "$RAG_ROOT/envs/retrieval"
```

vLLM belongs in a separate future serving environment.
