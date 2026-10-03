"""Lexical baseline: Unicode word tokens, lowercase, no stemming/stopword removal."""
import re
import numpy as np
from rank_bm25 import BM25Okapi

def tokenize(text): return re.findall(r'\w+', text.lower())

class BM25:
    def __init__(self, documents):
        self.index = BM25Okapi([tokenize(d['title']+' '+d['text']) for d in documents], k1=1.5, b=0.75, epsilon=0.25)
    def search(self, query, k=100):
        scores = self.index.get_scores(tokenize(query))
        # Stable ties retain the recorded corpus ID order.
        order = np.argsort(-scores, kind='stable')[:k]
        return order, scores[order]
