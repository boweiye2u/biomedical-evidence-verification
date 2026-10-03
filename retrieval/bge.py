import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = 'BAAI/bge-base-en-v1.5'
PREFIX = 'Represent this sentence for searching relevant passages: '

class BGE:
    def __init__(self, revision, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
        self.model = AutoModel.from_pretrained(MODEL, revision=revision).to(device).eval()
    @torch.inference_mode()
    def encode(self, texts, query=False, batch_size=32):
        outputs=[]
        for start in range(0,len(texts),batch_size):
            batch=texts[start:start+batch_size]
            if query: batch=[PREFIX+t for t in batch]
            tokens=self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            cls=self.model(**tokens).last_hidden_state[:,0]
            outputs.append(torch.nn.functional.normalize(cls,p=2,dim=1).cpu().numpy())
        return np.concatenate(outputs).astype('float32')
