"""Run all methods using generated synthetic data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from generate_synthetic import generate
from paired import analyze as paired_analysis
from appraisal import analyze as appraisal_analysis
from incremental import update
from ic_solver import A


def run(output, epochs=160, skip_echo=False):
    output=output.resolve(); inputs=generate(output/'synthetic_inputs'); results=output/'results'
    results.mkdir(parents=True, exist_ok=True)
    here=Path(__file__).resolve().parent
    def script(name,*args):
        subprocess.run([sys.executable,str(here/name),*map(str,args)],check=True)
    script('aggregate_publications.py','--labels',inputs/'publications.csv','--output',results/'aggregation')
    script('calendar_contrasts.py','--layers',results/'aggregation/monthly_layers.csv',
           '--events',inputs/'events.csv','--output',results/'calendar_contrasts.csv')
    paired_analysis(pd.read_csv(inputs/'paired.csv')).to_csv(results/'paired.csv',index=False)
    appraisal_analysis(pd.read_csv(inputs/'appraisal_articles.csv'),pd.read_csv(inputs/'appraisal_ratings.csv'),results/'appraisal')
    data=np.load(inputs/'patterns.npz'); n=data['population'].astype(float); h=data['pattern_counts'].astype(float)
    masks=A.T[1:]; y=h@masks; a=h.sum(axis=1)
    prior=np.r_[1-a[-1]/n[-1],h[-1]/n[-1]]
    old=A@prior; new=old+np.linspace(-.004,.004,9)
    update_result=update(prior,old,new,np.full(9,.002),1.)
    error=float(np.max(np.abs(update_result['corrected']['p']-update_result['cold']['p'])))
    np.savetxt(results/'ic_probabilities.csv', update_result['corrected']['p'],delimiter=',',header='probability',comments='')
    (results/'ic_diagnostics.json').write_text(json.dumps({'corrected':update_result['corrected']['diagnostic'],
        'max_difference_from_cold_start':error},indent=2),encoding='utf-8')
    if not skip_echo:
        from echo import train_model
        import torch
        torch.set_num_threads(1)
        pred, logp, info=train_model('ECHO',20261012,n,y,h,a,train_end=36,val_end=42,lam=.15,epochs=epochs)
        np.savez_compressed(results/'echo_predictions.npz',predicted_counts=pred,log_pattern_probabilities=logp)
        (results/'echo_fit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (results/'run_manifest.json').write_text(json.dumps({'input':'synthetic','echo_run':not skip_echo,
        'echo_epochs':epochs if not skip_echo else None,'seed':20261020},indent=2),encoding='utf-8')
    print('Synthetic example complete:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('demo_output'))
    p.add_argument('--epochs',type=int,default=160)
    p.add_argument('--skip-echo',action='store_true',help='Run only methods that do not require PyTorch')
    args=p.parse_args()
    if args.epochs<1: p.error('--epochs must be positive')
    run(args.output,args.epochs,args.skip_echo)
