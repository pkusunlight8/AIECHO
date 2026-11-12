# Model-family evidence in medical publications

This repository contains selected analysis methods for the accompanying study of AI and language-model publications in medical journals. It provides a runnable example using generated data; the example does not reproduce the study's reported estimates.

## Setup and example

Use Python 3.11 or 3.12. Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python run_demo.py --output demo_output
python -m unittest discover -s tests
```

The example generates all inputs at runtime with a fixed seed, then writes results beneath the requested output directory. The numeric identifiers are simulated and do not identify study articles. `--epochs 5` provides a quick execution check; the default is 160, as in the study implementation. `--skip-echo` omits PyTorch training while running the remaining methods. The run manifest records whether ECHO was run.

## Methods and inputs

| Code | Purpose | Example input |
| --- | --- | --- |
| `aggregate_publications.py` | Monthly publication counts and applicable reporting indicators | One simulated record per publication |
| `calendar_contrasts.py` | Equal-duration pre/post comparisons, excluding the release month; circular two-month bootstrap | Monthly layer counts and event definitions |
| `paired.py` | Paired article-level information gain, exact McNemar test, article bootstrap and Benjamini–Hochberg adjustment | Binary metadata and body indicators for the same article |
| `echo.py` | Exposure-aware GRU and distribution over 511 nonempty family patterns | Monthly population and pattern counts |
| `ic_solver.py`, `incremental.py` | ECHO-IC convex reconstruction over 512 patterns and implicit predictor followed by Newton correction | Prior distribution, source marginals and variances |
| `appraisal.py` | Domain summaries, reader agreement, article bootstrap intervals and proportional-odds model | Article covariates and A/B/C ratings |

`generate_synthetic.py` defines the input columns and creates the example files. `run_demo.py` is an executable reference for calling each method. Paired and appraisal modules extract the corresponding computations from the numerical replay; incremental updating follows the archived predictor–corrector calculation.

The example uses nine family dimensions. ECHO is fitted on the first 36 months and selects an epoch using months 37–42; the remaining months are held out. ECHO-IC's example prior is constructed from synthetic counts to demonstrate calibration; it is not a saved neural forecast. The source-reading process, literature acquisition, label generation, full selection design, original figures, and the complete publication analysis are outside this selected code package. In particular, appraisal ratings are inputs to statistical analysis, not generated judgments.

For independent use, inspect the example schemas and method assumptions before replacing simulated inputs. Individual command-line interfaces for aggregation, calendar comparisons, paired analysis, and appraisal expose input and output paths through `--help`. ECHO and ECHO-IC are also available as Python functions.

## License

MIT License. Copyright (c) 2026 Ning Shen. See [LICENSE](LICENSE).
