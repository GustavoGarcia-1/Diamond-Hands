"""Restore pinned runtime inputs without committing models or checkpoints."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import argparse,json,os,tempfile
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1]/'src/g1_receiver/_vendor/g1_runtime'
def fetch(root:Path,check:bool=False):
 manifest=json.loads((ROOT/'ASSET-MANIFEST.json').read_text());records=manifest['files']
 def one(r):
  relative=Path(r['path'])
  if relative.is_absolute()or'..'in relative.parts:raise ValueError('unsafe manifest path')
  p=root/relative
  if p.is_file()and sha256(p.read_bytes()).hexdigest()==r['sha256']:return p
  if check:raise ValueError('Missing or changed runtime input: '+r['path'])
  with urlopen(r['url'],timeout=90)as response:data=response.read()
  if sha256(data).hexdigest()!=r['sha256']:raise ValueError('Upstream hash mismatch: '+r['path'])
  if len(data)!=r['bytes']:raise ValueError('Upstream size mismatch: '+r['path'])
  p.parent.mkdir(parents=True,exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=p.parent,delete=False)as stream:stream.write(data);temporary=stream.name
  os.replace(temporary,p);return p
 with ThreadPoolExecutor(max_workers=6)as pool:paths=list(pool.map(one,records))
 return {'verified_files':len(paths),'verified_bytes':sum(r['bytes']for r in records),'upstream_repository':manifest['repository'],'upstream_revision':manifest['revision']}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,default=ROOT);p.add_argument('--check',action='store_true');a=p.parse_args();print(json.dumps(fetch(a.destination,a.check),indent=2))
