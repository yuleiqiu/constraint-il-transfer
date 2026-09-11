"""Offline contact reconstruction; never advance physics or execute a policy."""
import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import mujoco
from robosuite.utils.binding_utils import MjSim

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/tabletop_guidance_reeval_2026-09-09'
OUT=BASE/'placement_rescore'

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    bank=pickle.loads((ROOT/'outputs/tabletop_wide_support/raised_milk_state_bank_seed828_n50/initial_state_bank.pkl').read_bytes())
    rows=[json.loads(l) for l in (BASE/'results.jsonl').read_text().splitlines()]
    by={(r['episode'],r['condition']):r for r in rows}; results=[]
    for ep in range(50):
        state=bank['states'][f'PickPlaceCanTabletopRaisedMilk:828:{ep}']['state']
        sim=MjSim.from_xml_string(state['model']);model=sim.model
        names={i:model.geom_id2name(i) or '' for i in range(model.ngeom)}
        target={i for i,n in names.items() if n.startswith('Can_') and not n.endswith('_visual')}
        grip={i for i,n in names.items() if n.startswith('gripper') and (model.geom_contype[i] or model.geom_conaffinity[i])}
        left={i for i in grip if 'l_fingerpad' in names[i]};right={i for i in grip if 'r_fingerpad' in names[i]}
        assert target and grip and left and right
        can_id=model.body_name2id('Can_main')
        assert int(model.opt.integrator)==0, 'Only verified Euler snapshot inversion is supported'
        eef_id=next(i for i in range(model.nsite) if model.site_id2name(i)=='gripper0_right_grip_site')
        for cond in ('baseline','guided'):
            r=by[ep,cond];folder=BASE/r['folder'];trace=json.loads((folder/'trace.json').read_text())
            data=np.load(folder/'trajectory.npz');states=data['states']
            max_position_error=0.;contact_steps=[];run_start=None;success_step=None;grasp_seen=False
            for i,(saved,t) in enumerate(zip(states,trace)):
                sim.set_state_from_flattened(saved)
                # Recorded derived fields/contact refer to the final Euler substep before
                # qpos integration. Undo only that kinematic integration, never step dynamics.
                if i:
                    mujoco.mj_integratePos(model._model,sim.data.qpos,sim.data.qvel,-model.opt.timestep)
                sim.forward()
                assert np.max(np.abs(sim.data.site_xpos[eef_id]-t['eef']))<1e-7
                error=float(np.max(np.abs(sim.data.body_xpos[can_id]-np.asarray(t['can']))))
                max_position_error=max(error,max_position_error);assert error<1e-8
                touched=set()
                for c in sim.data.contact[:sim.data.ncon]:
                    a,b=int(c.geom1),int(c.geom2)
                    if a in target and b in grip:touched.add(b)
                    if b in target and a in grip:touched.add(a)
                grasp=bool(touched&left and touched&right);grasp_seen|=grasp
                in_goal=bool(t['goal_xy_ok'] and t['goal_z_ok'])
                released=not touched
                eligible=in_goal and released
                if eligible:
                    if run_start is None:run_start=i
                else:run_start=None
                duration=float(saved[0]-states[run_start][0]) if run_start is not None else 0.
                confirmed=duration>=.5-1e-9
                if confirmed and success_step is None:success_step=i
                contact_steps.append({'step':i,'time':float(saved[0]),'in_goal':in_goal,
                    'gripper_contact_geoms':[names[j] for j in sorted(touched)],'bilateral_fingerpad_grasp':grasp,
                    'released_no_gripper_contact':released,'continuous_eligible_seconds':duration,'placement_confirmed':confirmed})
            status='success' if success_step is not None else 'failure' if r['horizon']==600 else 'censored'
            end=success_step if success_step is not None else r['horizon']
            contact=bool(trace[end]['collision_any'])
            out={'episode':ep,'condition':cond,'status':status,'success_step':success_step,
                 'safe_success':status=='success' and not contact,'contact_until_new_stop_or_record_end':contact,
                 'recorded_steps':r['horizon'],'old_success':r['success'],'old_safe_success':r['safe_success'],
                 'final_eligible_seconds':contact_steps[-1]['continuous_eligible_seconds'],
                 'max_reconstructed_can_position_error':max_position_error,'bilateral_grasp_seen':grasp_seen,
                 'video':'../'+r['folder']+'/rollout.mp4'}
            dump(OUT/f'ep{ep:02d}_{cond}_contacts.json',contact_steps);results.append(out);data.close()
        print(json.dumps({'episode':ep,'results':[{k:x[k] for k in ('condition','status','success_step','safe_success')} for x in results[-2:]]}),flush=True)
    dump(OUT/'results.json',results)
    summary={}
    for cond in ('baseline','guided'):
        rs=[r for r in results if r['condition']==cond];counts=Counter(r['status'] for r in rs)
        safe=sum(r['safe_success'] for r in rs);uncertain_safe=sum(r['status']=='censored' and not r['contact_until_new_stop_or_record_end'] for r in rs)
        summary[cond]={'n':50,'status_counts':dict(counts),'task_sr_bounds':[counts['success']/50,(counts['success']+counts['censored'])/50],
            'safe_success':safe,'safe_sr_bounds':[safe/50,(safe+uncertain_safe)/50],
            'observed_contact_count':sum(r['contact_until_new_stop_or_record_end'] for r in rs),
            'four_way_confirmed':{'safe_success':safe,'success_with_collision':sum(r['status']=='success' and r['contact_until_new_stop_or_record_end'] for r in rs),
                'collision_failure':sum(r['status']=='failure' and r['contact_until_new_stop_or_record_end'] for r in rs),
                'ncr':sum(r['status']=='failure' and not r['contact_until_new_stop_or_record_end'] for r in rs)},
            'bilateral_grasp_seen_rollouts':sum(r['bilateral_grasp_seen'] for r in rs)}
    dump(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
