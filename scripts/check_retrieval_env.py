"""Offline dependency and single-GPU smoke check; no model/data downloads."""
import importlib
import importlib.metadata
import json
import os
import platform

import numpy as np
import torch
import faiss
import pytrec_eval
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

packages = {
    'torch': 'torch', 'transformers': 'transformers',
    'sentence-transformers': 'sentence_transformers', 'datasets': 'datasets',
    'beir': 'beir', 'faiss-cpu': 'faiss', 'rank-bm25': 'rank_bm25',
    'pytrec-eval-terrier': 'pytrec_eval', 'numpy': 'numpy', 'pandas': 'pandas',
    'scipy': 'scipy', 'pytest': 'pytest', 'accelerate': 'accelerate',
}
versions = {}
for distribution, module in packages.items():
    importlib.import_module(module)
    versions[distribution] = importlib.metadata.version(distribution)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 1, 'Select exactly one GPU with CUDA_VISIBLE_DEVICES'
x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
y = x.cuda() @ x.cuda().T
torch.cuda.synchronize()
torch.testing.assert_close(y.cpu(), x @ x.T)
index = faiss.IndexFlatIP(2)
index.add(np.eye(2, dtype=np.float32))
_, ids = index.search(np.array([[1, 0]], dtype=np.float32), 1)
assert ids[0, 0] == 0
bm25 = BM25Okapi([['gene', 'evidence'], ['other'], ['unrelated']])
assert int(np.argmax(bm25.get_scores(['gene']))) == 0
scores = pytrec_eval.RelevanceEvaluator({'q': {'d': 1}}, {'ndcg_cut_10', 'recall_10', 'recip_rank'}).evaluate({'q': {'d': 1.0}})
assert all(value == 1.0 for value in scores['q'].values())
print(json.dumps({
    'python': platform.python_version(), 'packages': versions,
    'cuda_runtime': torch.version.cuda,
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'visible_gpu_count': torch.cuda.device_count(),
    'gpu': torch.cuda.get_device_name(0),
    'checks': ['imports', 'CUDA matrix multiplication', 'FAISS search', 'BM25 ranking', 'IR metrics toy case'],
    'status': 'passed',
}, indent=2))
