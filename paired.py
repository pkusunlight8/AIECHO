"""Paired article-level information gain and reader agreement."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


def analyze(frame, seed=20261020, replicates=2000):
    if frame.case_id.duplicated().any():
        raise ValueError('Expected one row per article')
    items = [c[:-10] for c in frame.columns if c.endswith('__metadata')]
    rows = []
    for item in items:
        a = frame[item+'__metadata'].to_numpy()
        b = frame[item+'__body'].to_numpy()
        if not np.isin(a, [0,1]).all() or not np.isin(b, [0,1]).all():
            raise ValueError('Paired indicators must be binary')
        n = len(a)
        if not n:
            raise ValueError('No paired articles')
        d = b-a
        rng = np.random.default_rng(seed)
        draws = d[rng.integers(0,n,size=(replicates,n))].mean(axis=1)*100
        gained = int(((a == 0) & (b == 1)).sum())
        lost = int(((a == 1) & (b == 0)).sum())
        low, high = np.quantile(draws, [.025,.975])
        rows.append(dict(item=item,paired_n=n,metadata_percent=100*a.mean(),body_percent=100*b.mean(),
                         difference_pp=100*d.mean(),ci_low_pp=low,ci_high_pp=high,gained_n=gained,
                         lost_n=lost,exact_mcnemar_p=binomtest(gained,gained+lost,.5).pvalue if gained+lost else 1.))
    out = pd.DataFrame(rows)
    out['benjamini_hochberg_q'] = multipletests(out.exact_mcnemar_p, method='fdr_bh')[1]
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    analyze(pd.read_csv(a.input)).to_csv(a.output,index=False)
