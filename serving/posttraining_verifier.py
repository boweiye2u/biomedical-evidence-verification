from __future__ import annotations
import time
import httpx
from transformers import AutoTokenizer
from posttraining.evaluation.baselines import build_messages, parse_output, prompt_sha256

class PosttrainingVerifier:
    def __init__(self, config: dict, timeout: float = 180.0):
        self.config=config
        if config['prompt_sha256'] != prompt_sha256(): raise ValueError('frozen posttraining prompt hash mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(config['local_path'],local_files_only=True)
        self.client=httpx.AsyncClient(base_url=config['base_url'],timeout=timeout)
    async def reachable(self):
        try:return (await self.client.get('/health')).status_code==200
        except httpx.HTTPError:return False
    def build_prompt(self,claim,document):
        source={'doc_id':document['document_id'],'title':document['title'],'abstract':document['abstract']}
        prompt=self.tokenizer.apply_chat_template(build_messages(claim,source,[]),tokenize=False,add_generation_prompt=True)
        tokens=len(self.tokenizer.encode(prompt,add_special_tokens=False))
        if tokens+self.config['max_new_tokens']>self.config['max_model_len']:raise ValueError('rendered prompt exceeds frozen model-token limit')
        return prompt,{'prompt_tokens':tokens,'sentence_count':len(source['abstract'])}
    async def generate(self,prompt,sentence_count):
        started=time.perf_counter();response=await self.client.post('/v1/completions',json={'model':self.config['served_model_name'],'prompt':prompt,'temperature':self.config['temperature'],'repetition_penalty':self.config['repetition_penalty'],'max_tokens':self.config['max_new_tokens'],'seed':self.config['seed'],'stream':False});response.raise_for_status();payload=response.json();raw=payload['choices'][0]['text'].strip();parsed=parse_output(raw,sentence_count);usage=payload.get('usage') or {}
        return {'raw_output':raw,'parsed':parsed['value'] if parsed['decision_valid'] and parsed['rationale_index_valid'] and parsed['schema_valid'] else None,'decision':parsed['decision'],'errors':parsed['errors'],'json_valid':parsed['json_valid'],'schema_valid':parsed['schema_valid'],'decision_valid':parsed['decision_valid'],'rationale_index_valid':parsed['rationale_index_valid'],'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),'generation_ms':(time.perf_counter()-started)*1000}
    async def close(self):await self.client.aclose()
