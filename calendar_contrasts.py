"""Compare fixed calendar windows using monthly publication-layer counts."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected', type=Path)
    p.add_argument('--complete-through', default='2026-06')
    arg = p.parse_args()
    g = pd.read_csv(arg.layers)
    e = pd.read_csv(arg.events)
    g = g.rename(columns={'month': 'publication_month'})
    g = g[g.publication_month.le(arg.complete_through)].copy()
    g['ordinal'] = g.publication_month.map(lambda x: pd.Period(x, freq='M').ordinal)
    for de in ['ai', 'strict', 'llm']:
        g[de + '_percent'] = 100 * g[de] / g.n
    rng = np.random.default_rng(20261001)

    def boot(x, y):

        def block(v):
            starts = rng.integers(0, len(v), size=(2000, int(np.ceil(len(v) / 2))))
            ix = (starts[:, :, None] + np.arange(2)) % len(v)
            return v[ix.reshape(2000, -1)[:, :len(v)]].mean(axis=1)
        return np.quantile(block(x) - block(y), [0.025, 0.975])
    rows = []
    for _, event in e.iterrows():
        anchor = pd.Period(event.release_month, freq='M').ordinal
        for h in [3, 6, 12]:
            if event.complete_post_months < h:
                continue
            for mode in ['common_followup', 'nonoverlap']:
                if mode == 'nonoverlap' and event.complete_nonoverlap_months < h:
                    continue
                for lyr, d in g.groupby('layer'):
                    pre = d[d.ordinal.between(anchor - h, anchor - 1)]
                    post = d[d.ordinal.between(anchor + 1, anchor + h)]
                    if len(pre) != h or len(post) != h:
                        continue
                    for de in ['ai', 'strict', 'llm']:
                        x = post[de + '_percent'].to_numpy()
                        y = pre[de + '_percent'].to_numpy()
                        low, high = boot(x, y)
                        rows.append({'event': event.canonical_label, 'release_date': event.event_date, 'horizon_months': h, 'mode': mode, 'layer': lyr, 'definition': de, 'pre_n': int(pre.n.sum()), 'post_n': int(post.n.sum()), 'mean_pre_percent': y.mean(), 'mean_post_percent': x.mean(), 'difference_pp': x.mean() - y.mean(), 'interval_low': low, 'interval_high': high})
    out = pd.DataFrame(rows)
    arg.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(arg.output, index=False)
    report = {'contrasts': len(out), 'complete_month_end': arg.complete_through, 'seed': 20261001, 'replicates': 2000, 'block_months': 2}
    if arg.expected:
        expected = pd.read_csv(arg.expected)
        keys = ['event', 'release_date', 'horizon_months', 'mode', 'layer', 'definition']
        merged = out.merge(expected, on=keys, suffixes=('_new', '_ref'), validate='one_to_one')
        failures = []
        for col in ['pre_n', 'post_n', 'mean_pre_percent', 'mean_post_percent', 'difference_pp', 'interval_low', 'interval_high']:
            if not np.allclose(merged[col + '_new'], merged[col + '_ref'], rtol=1e-11, atol=1e-11):
                failures.append(col)
        if len(merged) != len(out) or len(merged) != len(expected):
            failures.append('row coverage')
        report.update(reference_match=not failures, failures=failures)
        if failures:
            raise ValueError('Calendar reference comparison failed: ' + ', '.join(failures))
    arg.output.with_suffix('.settings.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))
if __name__ == '__main__':
    main()
