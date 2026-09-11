"""Fixed-observation paired denoising, with original RNG-hash and action checks."""
import importlib.util,json,random,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reeval',ROOT/'scripts/2026-09-09_tabletop_guidance_reeval/run.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
P=R.P
BASE=ROOT/'outputs/tabletop_guidance_reeval_2026-09-09'
FOLDER=BASE/'shard_25_49/episodes/ep28/guided';OUT=BASE/'ep28_denoising_audit'

def main():
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 manifest=json.loads((BASE/'shard_25_49/manifest.json').read_text())
 policy,_=P.paper_guided_policy_from_checkpoint(ckpt_path=manifest['checkpoint'],device=torch.device('cuda'))
 obsfile=np.load(FOLDER/'observations.npz');actual=np.load(FOLDER/'trajectory.npz')['actions'];chunks=json.loads((FOLDER/'chunks.json').read_text());trace=json.loads((FOLDER/'trace.json').read_text())
 R.seed(828000028);policy.start_episode();rows=[];rng_audit=[]
 direction=np.asarray(trace[0]['goal_xy'])-np.asarray(trace[0]['can'][:2]);direction/=np.linalg.norm(direction)
 for i in range(33):
  step=i*8;obs={k[len('chunks_'):]:obsfile[k][i].copy() for k in obsfile.files if k.startswith('chunks_')}
  cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();ns=np.random.get_state();ps=random.getstate()
  audit={'step':step,'cpu_hash':R.sha_array(cpu.numpy()),'cuda_hash':R.sha_array(gpu.cpu().numpy())}
  audit['matches_saved']=audit['cpu_hash']==chunks[i]['torch_rng_hash'] and audit['cuda_hash']==chunks[i]['cuda_rng_hash'];rng_audit.append(audit)
  R.dump(OUT/'rng_reconstruction.json',rng_audit)
  if not audit['matches_saved']:raise RuntimeError(f'Original RNG not reconstructed at {step}; stop exact-replay claim')
  baseline,_=P.sample_raw_chunk(policy,obs,None)
  after_cpu=torch.get_rng_state();after_gpu=torch.cuda.get_rng_state();after_np=np.random.get_state();after_py=random.getstate()
  if step in (224,232,240,248,256):
   torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);np.random.set_state(ns);random.setstate(ps)
   cloud=obsfile[f'obstacle_cloud_{i:03d}'];current=np.asarray(obs['agent_pos'])[-1,:3]
   context=P.paper_lan_guidance_context_from_rollout_policy(policy,current_eef_pos=current,obstacle_points=cloud,guidance_scale=45.,safety_distance=.08,xy_only=True,obstacle_cost_type='huber_hinge',obstacle_cost_smoothing=.01,tangent_ratio=0.)
   guided,diagnostics=P.sample_raw_chunk(policy,obs,context)
   original_enabled=bool(chunks[i]['guidance_context'])
   original_replay=guided if original_enabled else baseline
   error=float(np.max(np.abs(original_replay-actual[step:step+8])))
   assert torch.equal(torch.cuda.get_rng_state(),after_gpu),'guidance altered random draws'
   assert error<1e-5,f'Original guided actions not reproduced: {step}, error {error}'
   eef=np.asarray(trace[step]['eef']);a=(baseline[:,:2]-eef[:2])@direction;b=(guided[:,:2]-eef[:2])@direction
   row={'step':step,'original_guidance_enabled':original_enabled,'can_z':trace[step]['can'][2],'original_action_max_error':error,'source_mean_forward_m':float(a.mean()),'guided_mean_forward_m':float(b.mean()),'guidance_change_forward_m':float((b-a).mean()),'source_forward_targets':int((a>0).sum()),'guided_forward_targets':int((b>0).sum()),'selected_point':diagnostics[0]['closest_obstacle_point'],'source_actions':baseline,'forced_guidance_actions':guided}
   rows.append(row);R.dump(OUT/f'step{step}_diagnostics.json',diagnostics)
   np.savez_compressed(OUT/f'step{step}_rng.npz',cpu=cpu.numpy(),cuda=gpu.cpu().numpy())
   R.dump(OUT/'results.json',rows)
   print(json.dumps(R.safe({k:v for k,v in row.items() if 'actions' not in k})),flush=True)
  torch.set_rng_state(after_cpu);torch.cuda.set_rng_state(after_gpu);np.random.set_state(after_np);random.setstate(after_py)
 R.dump(OUT/'manifest.json',{'mode':'saved-observation denoising only; no simulator or rollout','episode':28,'policy_seed':828000028,'steps':[224,232,240,248,256],'original_checkpoint_sha256':manifest['checkpoint_sha256'],'device':torch.cuda.get_device_name(0),'script_sha256':R.sha_file(Path(__file__)),'rng_check_count':len(rng_audit),'all_rng_hashes_match':all(x['matches_saved'] for x in rng_audit)})
if __name__=='__main__':main()
