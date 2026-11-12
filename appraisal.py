"""Agreement and proportional-odds analysis of article ratings."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests


def kappa(a, b, quadratic=False):
    if not len(a):
        return np.nan
    tab = np.zeros((4, 4))
    np.add.at(tab, (np.asarray(a, int), np.asarray(b, int)), 1)
    tab /= tab.sum()
    expected = np.outer(tab.sum(1), tab.sum(0))
    weight = (np.arange(4)[:, None]-np.arange(4)[None, :])**2 if quadratic else 1-np.eye(4)
    den = (weight*expected).sum()
    return 1-(weight*tab).sum()/den if den > 1e-14 else np.nan


def analyze(articles, ratings, output, seed=20261020, replicates=2000):
    output.mkdir(parents=True, exist_ok=True)
    if articles.case_id.duplicated().any() or ratings.duplicated(['case_id','domain','reader']).any():
        raise ValueError('Duplicate article or rating')
    valid = ratings.status.eq('assessed')
    if not ratings.loc[valid,'score'].isin([0,1,2,3]).all() or ratings.loc[~valid,'score'].notna().any():
        raise ValueError('Scores must be 0–3 when assessed and missing otherwise')
    rows=[]
    for domain, part in ratings.groupby('domain'):
        a=part[part.reader.eq('A')]; b=part[part.reader.eq('B')]
        pairs=a.merge(b,on=['case_id','domain'],suffixes=('_A','_B'),validate='one_to_one')
        good=pairs.status_A.eq('assessed') & pairs.status_B.eq('assessed')
        x=pairs.loc[good,'score_A'].to_numpy(); y=pairs.loc[good,'score_B'].to_numpy()
        rng=np.random.default_rng(seed); draws=[]
        for _ in range(replicates):
            sampled=pairs.iloc[rng.integers(0,len(pairs),len(pairs))]
            assessed=sampled.status_A.eq('assessed') & sampled.status_B.eq('assessed')
            xx=sampled.loc[assessed,'score_A'].to_numpy(); yy=sampled.loc[assessed,'score_B'].to_numpy()
            draws.append([np.mean(xx==yy) if len(xx) else np.nan, kappa(xx,yy), kappa(xx,yy,True)])
        row=dict(domain=domain,paired_articles=len(pairs),both_assessed=int(good.sum()),
                 state_agreement=float((pairs.status_A==pairs.status_B).mean()),
                 exact_agreement=float(np.mean(x==y)) if len(x) else np.nan,
                 kappa=kappa(x,y),quadratic_kappa=kappa(x,y,True))
        for i,name in enumerate(['agreement','kappa','quadratic_kappa']):
            values=np.asarray(draws)[:,i]; values=values[np.isfinite(values)]
            row[name+'_ci_low'],row[name+'_ci_high'] = np.quantile(values,[.025,.975]) if len(values) else (np.nan,np.nan)
            row[name+'_valid_bootstraps']=len(values)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'reader_agreement.csv',index=False)
    adjudicated=ratings[(ratings.reader=='C') & (ratings.status=='assessed')]
    adjudicated.groupby('domain').score.agg(['count','median','min','max']).to_csv(output/'domain_summary.csv')
    outcome=adjudicated[adjudicated.domain.eq('ai_method_fit_validation')][['case_id','score']]
    d=articles.merge(outcome,on='case_id',validate='one_to_one').dropna(subset=['score','metadata_item_count','year','source_words','field'])
    d=d[d.source_words.gt(0)]
    counts=d.score.value_counts(); result={'status':'not_estimable','n':len(d),'effects':[]}
    if len(d)>=80 and len(counts)>=3 and counts.min()>=5:
        x=pd.DataFrame({'metadata_count':d.metadata_item_count.astype(float),
            'publication_year':d.year-d.year.mean(),
            'log_source_words':np.log(d.source_words)-np.log(d.source_words).mean(),
            'oncology_field':d.field.astype(str).str.casefold().eq('oncology').astype(float)})
        x=x.loc[:,x.nunique()>1]
        try:
            fit=OrderedModel(d.score.astype(int),x,distr='logit').fit(method='bfgs',disp=False,maxiter=1000)
            if fit.mle_retvals.get('converged') and np.isfinite(fit.bse).all() and (fit.bse>0).all():
                ci=fit.conf_int(); result.update(status='estimated',log_likelihood=float(fit.llf))
                result['effects']=[dict(predictor=k,log_odds=float(fit.params[k]),standard_error=float(fit.bse[k]),
                    odds_ratio=float(np.exp(fit.params[k])),ci_low=float(np.exp(ci.loc[k,0])),
                    ci_high=float(np.exp(ci.loc[k,1])),p_value=float(fit.pvalues[k])) for k in x]
                for effect,q in zip(result['effects'],multipletests([e['p_value'] for e in result['effects']],method='fdr_bh')[1]):
                    effect['q_value']=float(q)
        except (ValueError,np.linalg.LinAlgError):
            pass
    (output/'ordinal_model.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--articles',type=Path,required=True); p.add_argument('--ratings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    analyze(pd.read_csv(a.articles),pd.read_csv(a.ratings),a.output)
