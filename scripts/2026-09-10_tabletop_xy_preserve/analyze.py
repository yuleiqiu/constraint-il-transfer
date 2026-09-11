"""Offline validation and Markdown video navigation for the bounded ablation."""
import argparse,json,math,subprocess
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--final',action='store_true');args=ap.parse_args();root=args.out.resolve()
 rows=[]
 for p in sorted(root.glob('dev_*/ep*/*/result.json')):
  r=json.loads(p.read_text());trace=json.loads((p.parent/'trace.json').read_text());chunks=json.loads((p.parent/'chunks.json').read_text())
  assert len(trace)==r['steps']+1
  start=None
  for t in trace:
   eligible=t['inside_goal'] and t['no_gripper_contact']
   if eligible:
    if start is None:start=t['step']
   else:start=None
   hold=(t['step']-start)/20 if start is not None else 0.
   assert abs(t['hold_seconds']-hold)<1e-9 and t['success']==(hold>=.5)
  assert r['success']==trace[-1]['success'] and r['collision_any']==any(t['contact'] for t in trace)
  if not r['success']:assert r['steps']==600
  with np.load(p.parent/'trajectory.npz') as d:
   assert len(d['actions'])==r['steps'] and len(d['states'])==r['steps']+1
   if r['condition']=='protected':
    assert np.array_equal(d['source_chunks'][:,:,2:],d['output_chunks'][:,:,2:]);assert r['protected_non_xy_max_error']==0
    assert np.array_equal(d['guided_chunks'][:,:,:2],d['output_chunks'][:,:,:2])
  if args.final:
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(p.parent/'rollout.mp4')],text=True))
   assert int(probe['streams'][0]['nb_read_frames'])==r['video_frames']
  r['folder']=str(p.parent.relative_to(root));rows.append(r)
 by={(r['episode'],r['condition']):r for r in rows};episodes=sorted(e for e in range(0,50,5) if all((e,c) in by for c in ['baseline','guided','protected']))
 if args.final:assert len(rows)==30 and len(episodes)==10
 for ep in episodes:
  a=by[ep,'baseline']
  for c in ['guided','protected']:
   assert by[ep,c]['reset_audit']['initial_obs_hashes']==a['reset_audit']['initial_obs_hashes']
 summary={'status':'complete' if args.final else 'partial','formal_completed':len(rows),'complete_layouts':len(episodes),'conditions':{},'paired':{}}
 keys=['success','safe_success','collision_any','ncr']
 for c in ['baseline','guided','protected']:
  rs=[by[e,c] for e in episodes]
  if rs:summary['conditions'][c]={'n':len(rs),**{k:sum(r[k] for r in rs) for k in keys},'four_way':{k:sum(r[k] for r in rs) for k in ['safe_success','success_with_collision','collision_failure','ncr']}}
 for ref in ['baseline','guided']:
  out={}
  for k in ['success','safe_success']:
   resc=sum(not by[e,ref][k] and by[e,'protected'][k] for e in episodes);reg=sum(by[e,ref][k] and not by[e,'protected'][k] for e in episodes)
   out[k]={'rescues':resc,'regressions':reg,'net':resc-reg}
  summary['paired']['protected_vs_'+ref]=out
 if args.final:
  s=summary['conditions'];summary['expansion_gate_pass']=s['protected']['safe_success']-s['guided']['safe_success']>=2 and s['protected']['success']>=max(s['guided']['success'],s['baseline']['success'])
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 lines=['# Development results','',f"{summary['status']}: {len(rows)}/30 branches; {len(episodes)}/10 complete layouts.",'','| Condition | Task | Safe | Contact | NCR |','|---|---:|---:|---:|---:|']
 for c,s in summary['conditions'].items():lines.append('| '+c+' | '+' | '.join(f"{s[k]}/{s['n']}" for k in keys)+' |')
 lines+=['','Source/guided/protected all use released0.5s placement stopping. Pilot excluded.','The development bank was previously examined; this is not a held-out performance claim.','']
 if args.final:lines.append(f"Preregistered expansion gate: {summary['expansion_gate_pass']}.")
 (root/'report.md').write_text('\n'.join(lines)+'\n')
 index=['# Video index for VS Code','','Actual execution videos, not policy replays.','','| Layout | Baseline | Original guidance | Protected non-XY |','|---|---|---|---|']
 def label(r):return 'safe success' if r['safe_success'] else 'success + contact' if r['success'] else 'collision failure' if r['collision_any'] else 'NCR'
 for ep in episodes:index.append(f'| {ep:02d} | '+' | '.join(f"[{label(by[ep,c])}]({by[ep,c]['folder']}/rollout.mp4)" for c in ['baseline','guided','protected'])+' |')
 index+=['','## Separate layout28 pilot','']
 for p in sorted(root.glob('pilot/r*/result.json')):
  r=json.loads(p.read_text());index.append(f"- [{p.parent.name}: {label(r)}]({p.parent.relative_to(root)}/rollout.mp4)")
 (root/'video_index.md').write_text('\n'.join(index)+'\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
