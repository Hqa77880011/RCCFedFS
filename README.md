# RCCFedFS

Implementation of **Influence-Calibrated Counterfactual Stress Testing for Robust Federated Feature Selection**.

The server selects features from client relevance intervals, candidate sets, redundancy sketches, label priors and marginal summaries. Credibility combines multi-view agreement with leave-one-client-out influence on the generated stress domain. Effective trustworthy support controls the extrapolation radius. Selection combines observed and local worst-case margins with a redundancy penalty.

## Install

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11 or 3.12. Activate the virtual environment in your shell.

## Prepare client data

Download [Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset), [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), or a labeled feature-selection dataset such as Gisette. Use sensor batch, subject or institution identities as clients. Preserve chronological batches for future-client evaluation where they exist.

Create `data/clients.csv` with `client,split,label` plus numeric feature columns. `split` is `train` or `test`; test clients are wholly unseen during selection. Labels are consecutive nonnegative integers and every client belongs to one split. Features must be finite; imputation and normalization must be fitted using training data only. Optional binary `malicious` and `extreme` fields annotate known corrupted and prespecified honest-extreme clients for diagnostic metrics. They are excluded from selector inputs.

At least three training clients and two features are required. The client-summary function performs only local label-association/correlation calculations. The server's `select` entry consumes these compact summaries, without accessing raw examples. A pooled logistic probe is trained after feature selection to measure predictive utility; that evaluation stage does not alter selection or credibility.

## Select and evaluate

```bash
rccfedfs data/clients.csv --k 10 --output outputs/clean
rccfedfs data/clients.csv --k 10 --attack mixed --attack-fraction 0.2 --output outputs/attacked
```

`--k` is subset size; `--candidates 30` is each client's local candidate budget. `--trim 0.1` defines robust interval endpoints. `--influence-weight` weights normalized leave-one-out domain displacement. `--iterations 20` bounds credibility updates, with early stopping at L1 weight change below 1e-5. `--radius-min`, `--radius-max` set the support-calibrated stress budget. `--alpha` mixes observed/counterfactual margins; `--redundancy-weight` penalizes already selected correlations. `--grid 17` controls the bounded local stress grid. Each tested shift changes one feature group, giving sparsity budget one.

The attack modes are `sign`, `magnitude`, `stuffing`, `redundancy`, `steering`, `collusion`, or seeded `mixed`. Attacks modify uploaded summaries rather than raw examples. `--target-features 5` identifies the final five feature columns as explicit attack targets; choose actual nuisance columns before export if attack-target selection is to have a substantive meaning. Colluders share high relevance for these targets. Steering moves marginal summaries and relevance bounds. Attack flags are retained solely for evaluation.

## Baselines and ablations

```bash
rccfedfs data/clients.csv --variant uniform --output outputs/uniform
rccfedfs data/clients.csv --variant no_influence --output outputs/no_influence
rccfedfs data/clients.csv --variant one_shot --output outputs/one_shot
rccfedfs data/clients.csv --variant fixed_radius --output outputs/fixed_radius
rccfedfs data/clients.csv --variant no_guard --output outputs/no_guard
rccfedfs data/clients.csv --variant no_worst --output outputs/no_worst
```

Apply the same attack/seed flags to every attacked comparison. Uniform aggregation is the summary-level control. The other variants remove influence feedback, repeated credibility refinement, support-adaptive radius, interval guards or worst-case stress scoring.

## Outputs and repeated seeds

`selected.json` lists chosen feature names. `credibility.csv` records client weights/influences, `iterations.csv` records radius/support updates, and `stress.csv` exposes every tested local shift and margin. `clients.csv` contains future-client macro-F1. `credibility.png` plots weights and the diagnostic cutoff.

`metrics.json` reports pooled macro-F1, worst-client macro-F1, balanced accuracy, attack-target selection rate, malicious-summary detection AUROC, honest-extreme false-positive rate, future relevance-domain coverage, effective support, radius, server seconds and numeric upload payload bytes. Honest FPR uses `--credibility-cutoff 0.5` relative to uniform weight, fixed before the test evaluation. Detection diagnostics with a missing class or annotation population are `null`.

Repeat the selection with paired seeds and identical splits/attacks for every method. Summarize actual saved runs:

```bash
python compare_runs.py outputs/seed0 outputs/seed1 outputs/seed2 --baseline outputs/base0 outputs/base1 outputs/base2 --output outputs/summary.json
```

This computes average pairwise subset Jaccard, mean/sample SD, paired Wilcoxon p-value and matched rank-biserial effect size. Use one matched directory per seed in the same order. For a four-dataset family, apply Holm adjustment to the four saved p-values rather than testing a selected best dataset.

## Summary-space conventions

Signed association is the strongest absolute one-versus-rest Pearson correlation; intervals use a sample-size-scaled approximation. Weighted medians provide robust centers, with weighted quantiles defining trimmed interval bounds. Redundancy is symmetrized and projected to a correlation matrix with nonnegative eigenvalues. Domain influence uses endpoint Hausdorff distance in normalized summary coordinates. Stress operators implement label-skew attenuation, redundancy-coupled nuisance shift and sensor association drift on a bounded grid. A single-feature shift satisfies the distance/sparsity budget; generated relevance is clipped to the trustworthy interval unless guards are disabled. These are summary stress configurations, not generated raw clients or physical interventions. Future-domain coverage in the output measures the declared relevance-space ball, separately from predictive utility.

## Selected reproducible setting

Use client-disjoint tabular classification, three or more training clients, raw
numeric features, a training-fitted StandardScaler and logistic-regression readout.
The finite summary-level stress operators are the chosen method: label skew,
nuisance correlation and sensor drift; their names do not imply retraining a
physical simulator. Full, uniform, no_influence, fixed_radius, no_guard and
no_worst controls share the same feature panel and classifier protocol.

```bash
rccfedfs data/clients.csv --output runs/full --variant full --k 10 --candidates 30 --iterations 20 --trim 0.1 --radius-min 0.02 --radius-max 0.3 --alpha 0.5 --grid 17 --influence-weight 1 --redundancy-weight 0.2 --attack mixed --attack-fraction 0.2 --target-features 5 --seed 13
```

The CSV supplies `client,split,label` and at least two finite numeric feature
columns (`split=train|test`, nonnegative integer shared-class labels). Optional
malicious/extreme flags are evaluator annotations. Honest inputs can use
`--attack none`; attack AUROC is then undefined if only one status is present.
`k` and candidate count are clipped to available features. Raw client data for
this chosen protocol are the required run input; no named external baseline or
pretrained model is required.

## Generate all inputs locally

The selected input source is an analytic client-shift task. Two Gaussian causal
features determine the binary label through the same noisy linear threshold in
all clients. Other measurements have client-specific shifts and a class
correlation that reverses on held-out clients. Their correlation is part of the
generative data model; inference receives only feature columns. Whole clients
belong to train or test. This is generated data, not measured client records.

```bash
python -m rccfedfs_data --output data/generated_clients.csv --train-clients 12 --test-clients 4 --samples-per-client 128 --features 20 --seed 13
rccfedfs data/generated_clients.csv --data-source analytic_client_shift --output runs/generated --variant full --k 10 --candidates 20 --iterations 20 --grid 17 --attack mixed --attack-fraction 0.2 --seed 13
```

The generator writes the actual CSV consumed by rccfedfs and a small JSON input
specification containing the seed, role counts and stable causal feature names.
No normalization or selection uses test clients. No download or external weights
are required. Training/selection and resulting metrics run only when the second
command is invoked; no such result is bundled.
