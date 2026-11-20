"""Preserve the costly generated translations; source code and CSVs stay in Git."""
from pathlib import Path
import json,hashlib,zipfile,argparse
ROOT=Path(__file__).resolve().parent

def main(destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    names=['translations_es.jsonl','translations_fr.jsonl']
    manifest={name:{'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),'rows':len((ROOT/name).read_text().splitlines())} for name in names}
    assert all(v['rows']==3500 for v in manifest.values())
    message='''Generated translations for Biniam Habte, COSC 4371 Project 1 final checkpoint.

Each file has all 3,500 frozen test IDs, inherited labels, source SHA-256 hashes,
and final reviewed translations. Output-limit repairs and the disclosed
targeted meaning corrections have already been applied.

Code, model revisions, required CSVs, review records and the final-checkpoint
checker output are in:
https://github.com/nrcisst/homework-solutions/tree/main/cosc4371/difraud-final

To reuse this cache, place both translations_*.jsonl files in that directory.
The pipeline verifies source hashes before scoring. Raw original data and
model weights are not bundled. The source dataset's MIT license is included.
'''
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in names:archive.write(ROOT/name,name)
        archive.writestr('README.txt',message)
        archive.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
        archive.write(ROOT.parent/'difraud-checkpoint'/'source_LICENSE.txt','DIFrauD_LICENSE.txt')
    with zipfile.ZipFile(destination) as archive:assert archive.testzip() is None
    print(destination)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');main(p.parse_args().destination)
