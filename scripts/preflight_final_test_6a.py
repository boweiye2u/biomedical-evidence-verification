"""Create the immutable pre-TEST manifest and source snapshot without loading TEST records."""
from __future__ import annotations
import hashlib,json,os,platform,shutil,subprocess,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ART=Path(os.environ.get("RAG_ROOT",str(Path.home()/"rag"))); RUN=ART/"runs/milestone6a-test-v1"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
 return h.hexdigest()
def main():
 if RUN.exists(): raise FileExistsError(f"refusing to overwrite {RUN}")
 RUN.mkdir(parents=True); source=RUN/"source"; source.mkdir()
 files=["configs/final-test-6a-v1.json","configs/verification-5a-v1.json","configs/splits/scifact_train_dev_v1.json","retrieval/bge.py","retrieval/final_test.py","retrieval/metrics.py","retrieval/verification.py","retrieval/reranking.py","scripts/preflight_final_test_6a.py","scripts/run_final_test_retrieval_6a.py","scripts/run_final_test_verification_6a.py","scripts/evaluate_final_test_6a.py","scripts/run_verification_5a.py","scripts/evaluate_verification_5a.py","tests/test_final_test.py","environments/final-test-6a-pip-freeze.txt"]
 for rel in files:
  src=ROOT/rel; dst=source/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
 prior=["configs/verification-5a-v1.json","docs/milestone5a-verification-dev-report.md","docs/milestone5b-reranking-dev-report.md","docs/milestone5c-passage-localization-dev-report.md"]
 prior_hashes={rel:sha(ROOT/rel) for rel in prior}; external=[ART/"runs/scifact_dev_v1/bge-rankings.json",ART/"runs/milestone5a-verification-v1/final/D1/evaluation.json",ART/"runs/milestone5a-verification-v1/final/D3/evaluation.json"]
 prior_hashes.update({str(p):sha(p) for p in external})
 try: commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
 except Exception: commit=None
 manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha(ROOT/"configs/final-test-6a-v1.json"),"source_files":{rel:sha(ROOT/rel) for rel in files},"prior_dev_artifact_hashes":prior_hashes,"git_commit":commit,"git_status":"not_a_git_repository" if commit is None else "recorded","test_records_loaded":False,"new_run_directory_verified":True,"random_seeds":{"generation":20261009,"ndcg_bootstrap":20261016,"accuracy_bootstrap":20261017,"macro_f1_bootstrap":20261018},"no_dev_dynamic_parameters_recomputed_from_test":True}
 (RUN/"pre-run-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 import torch,transformers,numpy
 (source/"environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"transformers":transformers.__version__,"numpy":numpy.__version__},indent=2)+"\n")
 archive=source/"pre-test-source-config-dependencies.tar.gz"
 with tarfile.open(archive,"w:gz") as tar:
  for p in sorted(source.rglob("*")):
   if p.is_file() and p!=archive: tar.add(p,arcname=p.relative_to(source))
 print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
