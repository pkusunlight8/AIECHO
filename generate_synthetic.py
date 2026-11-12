"""Generate synthetic inputs for the examples."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

QUALITY = ['quality_model_version', 'quality_prompt_reporting', 'quality_code_available',
           'quality_data_available', 'quality_calibration', 'quality_external_validation',
           'quality_bias_fairness', 'quality_reporting_guideline']
ITEMS = ['Model version', 'Prompt/input', 'Code access', 'Data access', 'Calibration',
         'External validation', 'Bias/fairness', 'Reporting guideline']


def generate(output, seed=20261020):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    months = pd.period_range('2022-01', '2026-06', freq='M')
    records = []
    for month in months:
        for layer in ['Empirical', 'Clinical trials', 'Interpretive', 'Synthesis', 'Other']:
            for _ in range(20):
                ai = int(rng.random() < .25)
                llm = int(ai and rng.random() < .28)
                date = str(month.start_time.date())
                row = dict(primary_pmid=len(records)+10000000, publication_date=date,
                           publication_date_upper=date, month_known=True,
                           prediction_eligible=bool(ai and rng.random() < .4),
                           era='early' if month.year < 2025 else 'late', layer=layer,
                           field=rng.choice(['Oncology', 'General']), broad_ai_flag=ai,
                           strict_ai_flag=int(ai and rng.random() < .65), llm_flag=llm,
                           multimodal_flag=int(ai and rng.random() < .18))
                row.update({name: int(rng.random() < .3) for name in QUALITY})
                records.append(row)
    pd.DataFrame(records).to_csv(output/'publications.csv', index=False)
    pd.DataFrame([dict(canonical_label='Example release', event_date='2024-01-15',
                       release_month='2024-01', complete_post_months=12,
                       complete_nonoverlap_months=12)]).to_csv(output/'events.csv', index=False)
    paired = pd.DataFrame({'case_id': np.arange(300)})
    for item in ITEMS:
        meta = rng.binomial(1, .32, len(paired))
        body = np.where(rng.random(len(paired)) < .25, 1-meta, meta)
        paired[item+'__metadata'] = meta
        paired[item+'__body'] = body
    paired.to_csv(output/'paired.csv', index=False)
    n = 160
    covariates = pd.DataFrame(dict(case_id=np.arange(n), metadata_item_count=rng.integers(0, 9, n),
        year=rng.integers(2022, 2027, n), source_words=rng.integers(800, 10000, n),
        field=rng.choice(['Oncology','Other'], n)))
    ratings=[]
    for case in range(n):
        for domain in ['ai_method_fit_validation', 'data_code_access', 'result_interpretation', 'task_relevant_reporting']:
            score = int(rng.integers(0,4))
            for reader in ['A','B','C']:
                value = score if reader != 'B' or rng.random() < .7 else int(rng.integers(0,4))
                ratings.append(dict(case_id=case,domain=domain,reader=reader,score=value,status='assessed'))
    covariates.to_csv(output/'appraisal_articles.csv', index=False)
    pd.DataFrame(ratings).to_csv(output/'appraisal_ratings.csv', index=False)
    masks = np.array([[(i >> j) & 1 for j in range(9)] for i in range(1,512)])
    counts = np.zeros((len(months),511), dtype=int)
    population = np.full(len(months), 1000, dtype=int)
    for t in range(len(months)):
        fields = np.linspace(-2.8,-1.3,9)+.006*t
        logits = masks @ fields
        probs = np.exp(logits-logits.max()); probs /= probs.sum()
        total = rng.binomial(population[t], .1+.001*t)
        counts[t] = rng.multinomial(total, probs)
    np.savez_compressed(output/'patterns.npz', population=population, pattern_counts=counts)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261020)
    args = parser.parse_args()
    generate(args.output, args.seed)
