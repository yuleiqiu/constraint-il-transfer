"""Keep progress and video navigation current while the two workers execute."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    root=args.out.resolve()
    previous=-1
    while True:
        rows=list(root.glob('shard_*/episodes/ep*/*/result.json'))
        complete=len(list(root.glob('shard_*/complete.json')))==2
        if len(rows)!=previous or complete:
            cmd=[sys.executable,str(Path(__file__).with_name('analyze.py')),'--out',str(root)]
            if not complete:
                cmd.append('--partial')
            result=subprocess.run(cmd,text=True,capture_output=True)
            if result.returncode:
                (root/'analysis_error.log').write_text(result.stdout+result.stderr)
                raise RuntimeError('Offline analysis failed; inspect analysis_error.log')
            previous=len(rows)
        progress={'utc':datetime.now(timezone.utc).isoformat(),'formal_completed':len(rows),
            'formal_total':100,'status':'complete' if complete else 'running',
            'shards':{p.name:len(list(p.glob('episodes/ep*/*/result.json'))) for p in root.glob('shard_*')}}
        tmp=root/'progress.json.tmp';tmp.write_text(json.dumps(progress,indent=2)+'\n');tmp.replace(root/'progress.json')
        print(json.dumps(progress),flush=True)
        if complete:
            return
        time.sleep(30)


if __name__=='__main__':
    main()
