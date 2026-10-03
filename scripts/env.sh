# Source from Bash before installing dependencies or running project commands.
# Override RAG_ROOT before sourcing to use another external storage location.
export RAG_ROOT="${RAG_ROOT:-$HOME/rag}"
export HF_HOME="$RAG_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$RAG_ROOT/cache/torch"
export XDG_CACHE_HOME="$RAG_ROOT/cache"
export PIP_CACHE_DIR="$RAG_ROOT/cache/pip"
export TMPDIR="$RAG_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1

# Keep Conda package downloads outside Dropbox and the base installation.
export CONDA_PKGS_DIRS="$RAG_ROOT/cache/conda/pkgs"
