"""Aggregate publication counts and applicable reporting indicators."""
from pathlib import Path
import argparse, json, re, zipfile
import numpy as np
import pandas as pd
QUALITY = {'Model version': 'quality_model_version', 'Prompt/input': 'quality_prompt_reporting', 'Code access': 'quality_code_available', 'Data access': 'quality_data_available', 'Calibration': 'quality_calibration', 'External validation': 'quality_external_validation', 'Bias/fairness': 'quality_bias_fairness', 'Reporting guideline': 'quality_reporting_guideline'}

def reader(path):
    options = {'chunksize': 100000, 'dtype': {'primary_pmid': str}, 'low_memory': False}
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith('.csv')]
            if len(names) != 1:
                raise ValueError('Expected exactly one CSV in the label archive')
            with archive.open(names[0]) as stream:
                with pd.read_csv(stream, **options) as chunks:
                    yield from chunks
    else:
        with pd.read_csv(path, **options) as chunks:
            yield from chunks


def boolean_values(values, column):
    tokens = values.astype(str).str.strip().str.casefold()
    allowed = ['true', 'false', '1', '0', '1.0', '0.0']
    if not tokens.isin(allowed).all():
        raise ValueError('Expected Boolean or binary values: ' + column)
    return tokens.isin(['true', '1', '1.0'])


def finish(parts, keys):
    result = pd.concat(parts).groupby(keys, dropna=False, as_index=False, sort=True).sum(numeric_only=True)
    for c in ['ai', 'strict', 'llm', 'multimodal']:
        if c in result:
            result[c + '_percent'] = 100 * result[c] / result['n']
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-monthly', type=Path)
    p.add_argument('--publication-cutoff', default='2026-06-30')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    seen = set()
    monthly = []
    layer = []
    field = []
    report = []
    count = 0
    temporal_count = 0
    inventory_flags = {c: 0 for c in ['ai', 'strict', 'llm', 'multimodal']}
    for frame in reader(a.labels):
        required = {'primary_pmid', 'publication_date', 'publication_date_upper', 'month_known', 'prediction_eligible', 'era', 'layer', 'field', *QUALITY.values()}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError('Missing label columns: ' + ', '.join(sorted(missing)))
        ids = frame.primary_pmid.fillna('').astype(str)
        if not ids.str.fullmatch('\\d+').all():
            raise ValueError('Invalid primary PMID')
        if ids.duplicated().any() or any((x in seen for x in ids)):
            raise ValueError('Duplicate primary PMID')
        seen.update(ids)
        count += len(frame)
        datecol = 'publication_date'
        dates = pd.to_datetime(frame[datecol], errors='raise')
        if not dates.between('2022-01-01', a.publication_cutoff).all():
            raise ValueError('Date outside observation period')
        if 'publication_date_upper' in frame and (not pd.to_datetime(frame.publication_date_upper).le(a.publication_cutoff).all()):
            raise ValueError('Publication interval exceeds the endpoint')
        frame['month'] = dates.dt.strftime('%Y-%m')
        mappings = {'ai': 'broad_ai_flag' if 'broad_ai_flag' in frame else 'ai_flag_final', 'strict': 'strict_ai_flag', 'llm': 'llm_flag', 'multimodal': 'multimodal_flag'}
        for short, long in mappings.items():
            if long not in frame:
                raise ValueError('Missing required derived indicator ' + long)
            values = pd.to_numeric(frame[long], errors='raise')
            if not values.isin([0, 1]).all():
                raise ValueError('Nonbinary indicator ' + long)
            frame[short] = values
            inventory_flags[short] += int(values.sum())
        frame['n'] = 1
        known = boolean_values(frame.month_known, 'month_known')
        temporal = frame[known].copy()
        temporal_count += len(temporal)
        monthly.append(temporal.groupby('month', as_index=False)[['n', *mappings]].sum())
        layer.append(temporal.groupby(['month', 'layer'], as_index=False)[['n', *mappings]].sum())
        field.append(temporal.groupby(['month', 'field', 'layer'], as_index=False)[['n', *mappings]].sum())
        empirical = frame.layer.isin(['Empirical', 'Clinical trials'])
        prediction = boolean_values(frame.prediction_eligible, 'prediction_eligible')
        for definition, flag in [('broad', 'ai'), ('strict', 'strict')]:
            for item, col in QUALITY.items():
                if col not in frame:
                    raise ValueError('Missing reporting indicator ' + col)
                eligible = empirical.copy()
                if item == 'Prompt/input':
                    eligible &= frame.llm.eq(1)
                if item in ['Calibration', 'External validation']:
                    if prediction is None:
                        raise ValueError('Prediction applicability indicator is required')
                    eligible = prediction.copy()
                s = frame[eligible & frame[flag].eq(1)].copy()
                s['numerator'] = pd.to_numeric(s[col], errors='raise')
                s['denominator'] = 1
                keys = [k for k in ['era', 'layer', 'field'] if k in s]
                grouped = s.groupby(keys, as_index=False)[['numerator', 'denominator']].sum()
                grouped['item'] = item
                grouped['definition'] = definition
                report.append(grouped)
        print('Processed', count, flush=True)
    if not count:
        raise ValueError('Label table is empty')
    m = finish(monthly, ['month'])
    l = finish(layer, ['month', 'layer'])
    f = finish(field, ['month', 'field', 'layer'])
    r = pd.concat(report).groupby(['era', 'layer', 'field', 'item', 'definition'], as_index=False)[['numerator', 'denominator']].sum()
    r['detected_percent'] = 100 * r.numerator / r.denominator
    for name, frame in [('monthly', m), ('monthly_layers', l), ('monthly_field_layers', f), ('applicable_reporting', r)]:
        frame.to_csv(a.output / (name + '.csv'), index=False)
    summary = {'unique_primary_pmids': len(seen), 'article_rows': count, 'month_known_articles': temporal_count, 'unspecified_month_articles': count - temporal_count, 'publication_cutoff': a.publication_cutoff, 'complete_month_end': a.publication_cutoff, 'screen_counts': {c: int(m[c].sum()) for c in ['ai', 'strict', 'llm', 'multimodal']}, 'inventory_screen_counts': inventory_flags, 'rate_definition': 'screen-positive records divided by all eligible records', 'date_definition': 'source publication date'}
    if a.expected_monthly:
        expected = pd.read_csv(a.expected_monthly)
        month = 'publication_month' if 'publication_month' in expected else 'month'
        expected = expected.rename(columns={month: 'month'})
        z = m.merge(expected, on='month', suffixes=('_new', '_reference'), validate='one_to_one')
        errors = []
        for c in ['n', 'ai', 'strict', 'llm', 'multimodal']:
            if c + '_reference' in z and (not np.array_equal(z[c + '_new'], z[c + '_reference'])):
                errors.append(c)
        if len(z) != len(m) or len(z) != len(expected):
            errors.append('monthly coverage')
        summary['monthly_reference_exact_match'] = not errors
        summary['differences'] = errors
        if errors:
            raise ValueError('Monthly reference comparison failed: ' + ', '.join(errors))
    (a.output / 'publication_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
