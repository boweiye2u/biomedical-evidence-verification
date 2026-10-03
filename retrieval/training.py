"""Differentiable shared-encoder training with per-row explicit negatives only."""
import torch
from torch.nn import functional as F

def embed(model, tokens):
    return F.normalize(model(**tokens).last_hidden_state[:, 0], p=2, dim=-1)

def explicit_loss(model, query_tokens, document_tokens, temperature=0.05):
    q = embed(model, query_tokens)
    d = embed(model, document_tokens).reshape(q.shape[0], 6, -1)
    logits = torch.einsum('bd,bkd->bk', q, d) / temperature
    return F.cross_entropy(logits, torch.zeros(q.shape[0], dtype=torch.long, device=q.device)), q, d
