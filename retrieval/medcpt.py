"""Asymmetric CLS encoders; unnormalized inner-product scoring."""
import torch
from transformers import AutoModel, AutoTokenizer

class MedCPT:
    def __init__(self, revisions, device='cuda:0'):
        self.device=device
        self.models={}; self.tokenizers={}
        for kind in ['Query','Article']:
            name=f'ncbi/MedCPT-{kind}-Encoder'
            self.tokenizers[kind]=AutoTokenizer.from_pretrained(name, revision=revisions[name])
            self.models[kind]=AutoModel.from_pretrained(name, revision=revisions[name]).to(device).eval()
    @torch.inference_mode()
    def encode_queries(self, queries):
        tokens=self.tokenizers['Query'](queries,padding=True,truncation=True,max_length=64,return_tensors='pt').to(self.device)
        return self.models['Query'](**tokens).last_hidden_state[:,0].cpu().numpy()
    @torch.inference_mode()
    def encode_articles(self, documents):
        pairs=[[d['title'],d['text']] for d in documents]
        tokens=self.tokenizers['Article'](pairs,padding=True,truncation=True,max_length=512,return_tensors='pt').to(self.device)
        return self.models['Article'](**tokens).last_hidden_state[:,0].cpu().numpy()
