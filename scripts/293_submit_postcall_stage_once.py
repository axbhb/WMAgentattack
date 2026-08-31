"""Remote exclusive submission receipt. Never retries an uncertain submission."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['v42','v43'],required=True);args=ap.parse_args()
    repo=Path('/share/guozhix/WMagentattack-postcall-recovery-sep1')
    if Path.cwd().resolve()!=repo:raise RuntimeError('Run in the designated remote worktree only')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise RuntimeError('Tracked worktree changes need inspection')
    cfg=repo/'configs/0901_postcall_recovery_v42_v43_protocol.json'
    p=json.loads(cfg.read_text())
    if args.stage=='v43':
        previous=Path(p['v42']['archive'])
        cfg=previous/'preregistered_protocol.json';p=json.loads(cfg.read_text())
        g=json.loads((previous/'gate.json').read_text())
        if not (previous/'COMPLETE').exists() or g['decision']!='GO_POSTCALL_RECOVERY' or not g['selected_arm']:raise RuntimeError('V43 not authorized')
    audit=json.loads((Path(p['audit_archive'])/'audit.json').read_text())
    assert audit['decision']=='DESCRIPTIVE_AUDIT_COMPLETE_V41'
    root=Path(p[args.stage]['archive']);root.mkdir(parents=True,exist_ok=True)
    if (root/'submission.json').exists():
        print('ALREADY_RECORDED', (root/'submission.json').read_text());return
    if any(root.iterdir()):raise RuntimeError('Nonempty archive without receipt; inspect, never blindly resubmit')
    name='wma-v42-postcall' if args.stage=='v42' else 'wma-v43-confirm'
    queue=subprocess.check_output(['squeue','-h','-u','guozhix','-o','%i %j'],text=True)
    if name in queue:raise RuntimeError('Existing matching job; inspect before any submission')
    with (root/'submission.lock').open('x') as f:f.write(datetime.now(timezone.utc).isoformat())
    shutil.copy2(cfg,root/'preregistered_protocol.json')
    script=f'scripts/server/run_0901_postcall_recovery_{args.stage}.sbatch'
    receipt={'stage':args.stage,'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'submitted_at_utc':datetime.now(timezone.utc).isoformat(),'planned_episodes':p[args.stage]['episodes'],'gpus':1,'batch_script':script}
    if args.stage=='v43':receipt['selected_arm']=g['selected_arm']
    (root/'pre_submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    run=subprocess.run(['sbatch','--parsable',script],capture_output=True,text=True)
    receipt.update(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
    if run.returncode==0:receipt['job_id']=run.stdout.strip().split(';')[0]
    (root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));run.check_returncode()


if __name__=='__main__':main()
