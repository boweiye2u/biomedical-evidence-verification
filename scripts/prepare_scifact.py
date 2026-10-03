"""Download minimal SciFact files and validate training mappings; never load test qrels."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('RAG_ROOT', str(Path.home() / 'rag'))) / 'data/scifact'
SOURCES = {
    'beir.zip': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
    'original.tar.gz': 'https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path, key):
    result = {}
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        ident = str(obj[key])
        assert ident not in result, f'Duplicate ID {ident}'
        result[ident] = obj
    return result

def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))

def groups_for(claims):
    """Connected components: shared source-cited/evidence document or near-duplicate text."""
    ids = sorted(claims, key=int)
    parent = {q: q for q in ids}
    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q
    def union(a, b):
        parent[find(b)] = find(a)
    owner = {}
    for q in ids:
        c = claims[q]
        keys = [('doc', str(d)) for d in set(map(str, c['cited_doc_ids'])) | set(c['evidence'])]
        keys.append(('text', normalized(c['claim'])))
        for key in keys:
            if key in owner: union(q, owner[key])
            else: owner[key] = q
    # Explicit lexical rule, not a guarantee of semantic deduplication.
    tokens = {q: set(normalized(claims[q]['claim']).split()) for q in ids}
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            union_tokens = tokens[a] | tokens[b]
            if union_tokens and len(tokens[a] & tokens[b]) / len(union_tokens) >= 0.8:
                union(a, b)
    groups = {}
    for q in ids: groups.setdefault(find(q), []).append(q)
    return sorted(groups.values(), key=lambda g: int(g[0]))

def split_groups(groups, seed=20261001, fraction=0.2):
    shuffled = [list(g) for g in groups]
    random.Random(seed).shuffle(shuffled)
    target = round(sum(map(len, groups)) * fraction)
    dev, train = [], []
    for group in shuffled:
        if abs(len(dev) + len(group) - target) < abs(len(dev) - target): dev.extend(group)
        else: train.extend(group)
    return sorted(train, key=int), sorted(dev, key=int)

def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {'sources': [], 'files': [], 'test_policy': 'No test qrels or original dev/test files extracted or read. Combined BEIR queries are filtered to training IDs before text validation.'}
    for name, url in SOURCES.items():
        path = DATA / 'archives' / name
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            temp = path.with_suffix(path.suffix + '.part')
            urllib.request.urlretrieve(url, temp)
            temp.replace(path)
        manifest['sources'].append({'url': url, 'path': str(path.relative_to(DATA)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    selected = {}
    with zipfile.ZipFile(DATA / 'archives/beir.zip') as archive:
        for suffix in ['corpus.jsonl', 'queries.jsonl', 'qrels/train.tsv']:
            matches = [n for n in archive.namelist() if n == 'scifact/' + suffix]
            assert len(matches) == 1, (suffix, matches)
            selected['beir/' + suffix] = archive.read(matches[0])
    with tarfile.open(DATA / 'archives/original.tar.gz') as archive:
        for name in ['corpus.jsonl', 'claims_train.jsonl']:
            matches = [m for m in archive.getmembers() if m.name == 'data/' + name]
            assert len(matches) == 1, (name, [m.name for m in archive.getmembers()])
            selected['original/' + name] = archive.extractfile(matches[0]).read()
    for name, content in selected.items():
        path = DATA / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): assert path.read_bytes() == content, f'Existing file differs: {path}'
        else: path.write_bytes(content)
        manifest['files'].append({'path': name, 'bytes': len(content), 'sha256': digest(path)})
    qrels = {}
    for row in csv.DictReader(io.StringIO(selected['beir/qrels/train.tsv'].decode()), delimiter='\t'):
        assert int(row['score']) > 0
        qrels.setdefault(row['query-id'], set()).add(row['corpus-id'])
    original = rows(DATA / 'original/claims_train.jsonl', 'id')
    assert set(qrels) <= set(original)
    queries = {}
    # Combined file contains held-out queries: read IDs only for nontraining records.
    for line in selected['beir/queries.jsonl'].decode().splitlines():
        obj = json.loads(line)
        if str(obj['_id']) in qrels:
            assert str(obj['_id']) not in queries
            queries[str(obj['_id'])] = obj['text']
    assert set(queries) == set(qrels)
    corpus = rows(DATA / 'beir/corpus.jsonl', '_id')
    originals = rows(DATA / 'original/corpus.jsonl', 'doc_id')
    assert set(corpus) == set(originals)
    for d, doc in originals.items():
        assert corpus[d]['title'] == doc['title']
        assert ' '.join(corpus[d]['text'].split()) == ' '.join(' '.join(doc['abstract']).split())
    label_counts = {}
    cited_non_evidence = []
    empty_evidence = []
    for q in sorted(qrels, key=int):
        c = original[q]
        assert queries[q] == c['claim'], q
        evidence = set(c['evidence'])
        # Diagnose actual qrel semantics before assigning verification labels.
        assert qrels[q] == set(map(str, c['cited_doc_ids'])), q
        assert evidence <= qrels[q], q
        if not evidence: empty_evidence.append(q)
        if set(map(str, c['cited_doc_ids'])) - evidence: cited_non_evidence.append(q)
        for d, rationales in c['evidence'].items():
            assert d in originals
            for rationale in rationales:
                assert rationale['label'] in {'SUPPORT', 'CONTRADICT'}
                assert rationale['sentences']
                assert all(0 <= i < len(originals[d]['abstract']) for i in rationale['sentences'])
                label_counts[rationale['label']] = label_counts.get(rationale['label'], 0) + 1
    summary = {'beir_training_queries': len(qrels), 'original_training_claims': len(original), 'corpus_documents': len(corpus), 'qrel_pairs': sum(map(len, qrels.values())), 'qrel_evidence_mismatches': sum(qrels[q] != set(original[q]['evidence']) for q in qrels), 'training_queries_without_annotated_evidence': len(empty_evidence), 'training_queries_with_cited_non_evidence': len(cited_non_evidence), 'rationale_label_counts': label_counts}
    print(json.dumps(summary, indent=2))
    manifest['summary'] = summary
    summary['qrel_semantics'] = 'All training qrels equal cited_doc_ids; these are retrieval targets, not necessarily verification evidence.'
    summary['corpus_whitespace_differences'] = sum(corpus[d]['text'] != ' '.join(originals[d]['abstract']) for d in corpus)
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    claims = {q: original[q] for q in qrels}
    groups = groups_for(claims)
    train, dev = split_groups(groups)
    split = {'version': 'scifact_train_dev_v1', 'seed': 20261001, 'target_dev_fraction': 0.2, 'group_rule': 'Connected components of shared cited/evidence document IDs, normalized exact claims, or token-set Jaccard >= 0.8. Training population only.', 'group_count': len(groups), 'largest_group': max(map(len, groups)), 'train_count': len(train), 'dev_count': len(dev), 'train_ids': train, 'dev_ids': dev, 'groups': groups, 'input_sha256': {f['path']: f['sha256'] for f in manifest['files']}, 'test_policy': manifest['test_policy']}
    assert train and dev
    assert not set(train) & set(dev)
    assert set(train) | set(dev) == set(qrels)
    dest = ROOT / 'configs/splits/scifact_train_dev_v1.json'
    text = json.dumps(split, indent=2) + '\n'
    if dest.exists(): assert dest.read_text() == text, 'Refusing to change frozen split'
    else: dest.write_text(text)
    (ROOT / 'docs/scifact-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Training-only examples include both labels and cited-but-not-evidence cases.
    chosen = []
    categories = [
        (lambda c: not c['evidence'], 2),
        (lambda c: c['evidence'] and set(map(str,c['cited_doc_ids']))-set(c['evidence']), 2),
        (lambda c: any(r['label']=='CONTRADICT' for rs in c['evidence'].values() for r in rs), 3),
        (lambda c: any(r['label']=='SUPPORT' for rs in c['evidence'].values() for r in rs), 3),
    ]
    for condition, count in categories:
        candidates = [q for q in train if q not in chosen and condition(original[q])]
        chosen.extend(candidates[:count])
    assert len(chosen) == 10
    examples = []
    for q in chosen[:10]:
        c = original[q]
        examples.append({'query_id': q, 'claim': c['claim'], 'qrel_document_ids': sorted(qrels[q]), 'cited_document_ids': c['cited_doc_ids'], 'evidence': [{'doc_id':d, 'title': originals[d]['title'], 'label': r['label'], 'sentence_ids':r['sentences'], 'sentences':[originals[d]['abstract'][i] for i in r['sentences']]} for d, rs in c['evidence'].items() for r in rs]})
    (ROOT / 'docs/scifact-mapping-examples.json').write_text(json.dumps(examples, indent=2) + '\n')
    print('SPLIT',len(train),len(dev),'groups',len(groups),'largest',max(map(len,groups)))

if __name__ == '__main__': prepare()
