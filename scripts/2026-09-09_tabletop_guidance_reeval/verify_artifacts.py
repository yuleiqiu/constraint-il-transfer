"""Verify video frame counts and write an integrity inventory after completion."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.out.resolve()
    results=sorted(root.glob('shard_*/episodes/ep*/*/result.json'))
    assert len(results)==100
    records=[]
    for p in results:
        result=json.loads(p.read_text());folder=p.parent
        video=folder/'rollout.mp4'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_read_frames,r_frame_rate',
            '-of','json',str(video)],text=True))['streams'][0]
        assert int(probe['nb_read_frames'])==result['video_frames']
        assert probe['width']==384 and probe['height']==432
        with np.load(folder/'observations.npz') as obs:
            assert obs.files and all(np.isfinite(obs[k]).all() for k in obs.files)
        files={f.name:{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(folder.iterdir()) if f.is_file()}
        records.append({'folder':str(folder.relative_to(root)),'video':probe,'files':files})
    assert not list(root.glob('shard_*/episodes/**/*.partial.mp4'))
    assert not list(root.glob('shard_*/episodes/**/*.tmp'))
    summary={'formal_rollouts':len(records),'videos_decoded_and_counted':len(records),
        'total_branch_bytes':sum(v['bytes'] for r in records for v in r['files'].values()),'records':records}
    (root/'artifact_integrity.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}))


if __name__=='__main__':
    main()
