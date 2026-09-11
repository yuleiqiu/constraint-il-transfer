"""Validate saved executions, aggregate pairs, and create a local video index."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [c-h, c+h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    root = args.out.resolve()
    rows = []
    for f in sorted(root.glob('shard_*/episodes/ep*/*/result.json')):
        row = read(f)
        folder = f.parent
        trace = read(folder/'trace.json')
        with np.load(folder/'trajectory.npz') as tr:
            assert len(tr['actions']) == row['horizon']
            assert len(tr['states']) == len(trace) == row['horizon']+1
            assert len(tr['video_steps']) == row['video_frames']
        assert all(t['success'] == (t['goal_xy_ok'] and t['goal_z_ok'] and t['retreat_ok']) for t in trace)
        for t in trace:
            can=np.asarray(t['can']); eef=np.asarray(t['eef'])
            assert bool(np.all(np.abs(can[:2]-np.asarray(t['goal_xy'])) < np.asarray(t['goal_half_size']))) == t['goal_xy_ok']
            # This Tabletop implementation fixes table_offset z at 0.8 m.
            assert bool(.8 < can[2] < .9) == t['goal_z_ok']
            # EEF helper records float32; predicate distance is computed in float64.
            assert np.isclose(np.linalg.norm(eef-can),t['eef_can_distance'],atol=1e-7,rtol=0)
            assert bool(1.-np.tanh(10.*t['eef_can_distance']) < .6) == t['retreat_ok']
        assert trace[-1]['success'] == row['success']
        assert trace[-1]['collision_any'] == row['collision_any']
        assert row['safe_success'] == (row['success'] and not row['collision_any'])
        assert row['reset_audit']['restored_state_hash'] == row['initial_state_hash']
        assert (folder/'rollout.mp4').stat().st_size > 1000
        rows.append(row)
    by = {(r['episode'],r['condition']):r for r in rows}
    assert len(by) == len(rows)
    episodes = sorted(ep for ep in range(50) if all((ep,c) in by for c in ('baseline','guided')))
    if not args.partial:
        assert episodes == list(range(50)) and len(rows) == 100
        assert len(list(root.glob('shard_*/complete.json'))) == 2
        assert len(list(root.glob('shard_*/preflight_pass.json'))) == 2
    for ep in episodes:
        a,b = by[ep,'baseline'], by[ep,'guided']
        assert a['policy_seed'] == b['policy_seed']
        assert a['initial_state_hash'] == b['initial_state_hash']
        assert a['reset_audit']['initial_obs_hashes'] == b['reset_audit']['initial_obs_hashes']
        pair = root / a['folder'] / '..' / 'pair_audit.json'
        if not args.partial:
            assert read(pair)['initial_observations_equal']
    report = {'status':'partial' if args.partial else 'complete', 'completed_rollouts':len(rows),
              'paired_layouts':len(episodes), 'conditions':{}, 'paired':{}}
    fields = ('success','safe_success','collision_any','ncr')
    for cond in ('baseline','guided'):
        rs=[by[ep,cond] for ep in episodes]
        if not rs:
            continue
        report['conditions'][cond] = {'n':len(rs), **{k:{'count':sum(bool(r[k]) for r in rs),
            'rate':float(np.mean([r[k] for r in rs])),
            'wilson95':wilson(sum(bool(r[k]) for r in rs),len(rs))} for k in fields},
            'four_way':{k:sum(bool(r[k]) for r in rs) for k in
                        ('safe_success','success_with_collision','collision_failure','ncr')}}
    rng = np.random.default_rng(20260909)
    for field in ('success','safe_success'):
        if not episodes:
            continue
        a=np.array([by[ep,'baseline'][field] for ep in episodes],dtype=int)
        b=np.array([by[ep,'guided'][field] for ep in episodes],dtype=int)
        rescue=int(np.sum((a==0)&(b==1))); regression=int(np.sum((a==1)&(b==0)))
        discordant=rescue+regression
        p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(rescue,regression)+1))/2**discordant) if discordant else 1.
        diff=b-a
        boots=diff[rng.integers(0,len(diff),size=(20000,len(diff)))].mean(axis=1)
        report['paired'][field]={'rescues':rescue,'regressions':regression,
            'both_success':int(np.sum((a==1)&(b==1))), 'both_failure':int(np.sum((a==0)&(b==0))),
            'net_rate_difference':float(diff.mean()), 'layout_bootstrap95':np.quantile(boots,[.025,.975]).tolist(),
            'mcnemar_exact_two_sided':p}
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    lines=['# Corrected Tabletop guidance comparison','',f"Status: {report['status']}; {len(rows)}/100 formal rollouts, {len(episodes)}/50 paired layouts.",'',
        'Frozen wide-support epoch-50 policy; H600; original target-lifted point guidance.',
        'All videos are captured inside their evaluated rollouts. Preflight repetitions are separate.', '',
        '| Condition | Task SR | Safe SR | CR | NCR |','|---|---:|---:|---:|---:|']
    for c,v in report['conditions'].items():
        lines.append('| '+c+' | '+' | '.join(f"{v[k]['count']}/{v['n']} ({v[k]['rate']:.2f})" for k in fields)+' |')
    lines += ['', 'Four-way counts and paired transitions / uncertainty are in `summary.json`.',
              'Layout bootstrap intervals describe this empirical bank; the bank was previously examined,',
              'not a new held-out sample. An inconclusive difference is not proof of no effect.', '',
              '## Paired task transitions','']
    for k,v in report['paired'].items():
        lines.append(f"- {k}: {v['rescues']} rescues / {v['regressions']} regressions; net {v['net_rate_difference']:+.3f}; paired bootstrap95 {v['layout_bootstrap95']}; exact McNemar p={v['mcnemar_exact_two_sided']:.4g}.")
    lines += ['', '## Layouts','', '| Layout | Source | Guidance |','|---|---|---|']
    def outcome(r):
        return ('safe success' if r['safe_success'] else 'success + contact' if r['success'] else 'collision failure' if r['collision_any'] else 'no contact, incomplete')+f" ({r['horizon']} steps)"
    index=['# Video index', '', f"Status: {report['status']}; {len(rows)}/100 formal rollouts.", '',
           'Open individual MP4 files directly in VS Code. Videos were recorded during the evaluated execution.', '',
           '| Layout | Source video | Guidance video |', '|---|---|---|']
    for ep in episodes:
        a,b=by[ep,'baseline'],by[ep,'guided']
        lines.append(f"| {ep:02d} | {outcome(a)} | {outcome(b)} |")
        index.append(f"| {ep:02d} | [{outcome(a)}]({a['folder']}/rollout.mp4) | [{outcome(b)}]({b['folder']}/rollout.mp4) |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'video_index.md').write_text('\n'.join(index)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
