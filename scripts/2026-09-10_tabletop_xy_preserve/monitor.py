"""Update Markdown results while workers run; validate the completed matrix."""
import json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[2]/'outputs/tabletop_xy_preserve_2026-09-10'
last=-1
while True:
 n=len(list(root.glob('dev_*/ep*/*/result.json')));done=len(list(root.glob('dev_*/complete.json')))==2
 if n!=last or done:
  cmd=[sys.executable,str(Path(__file__).with_name('analyze.py')),'--out',str(root)]
  if done:cmd.append('--final')
  r=subprocess.run(cmd,capture_output=True,text=True)
  if r.returncode:
   (root/'analysis_error.log').write_text(r.stdout+r.stderr);raise RuntimeError('analysis failed')
  last=n
 p={'utc':datetime.now(timezone.utc).isoformat(),'completed':n,'total':30,'status':'complete' if done else 'running'}
 (root/'progress.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p),flush=True)
 if done:break
 time.sleep(30)
