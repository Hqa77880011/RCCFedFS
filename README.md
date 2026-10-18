# RCCFedFS

Implementation of **Influence-Calibrated Counterfactual Stress Testing for Robust Federated Feature Selection**.

## Installation

Use Python 3.11 or 3.12. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Generate an analytic client-shift classification task. Two Gaussian causal features determine labels while nuisance correlations reverse on held-out clients; these are generated data. Measured alternatives include [Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) and [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). The CSV needs `client,split,label` and finite numeric feature columns, whole-client `train|test` roles, shared zero-based labels and training-only preprocessing. Optional `malicious,extreme` are evaluator annotations.

```bash
python -m rccfedfs_data --output data/generated_clients.csv --train-clients 12 --test-clients 4 --samples-per-client 128 --features 20 --seed 13
```

## Run and evaluate

```bash
rccfedfs data/generated_clients.csv --data-source analytic_client_shift --output runs/generated --variant full --k 10 --candidates 20 --iterations 20 --grid 17 --attack mixed --attack-fraction 0.2 --seed 13
```

## Ablations

Supported controls: `uniform`, `no_influence`, `one_shot`, `fixed_radius`, `no_guard`, `no_worst`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
rccfedfs data/generated_clients.csv --data-source analytic_client_shift --output outputs/uniform --variant uniform --k 10 --candidates 20 --iterations 20 --grid 17 --attack mixed --attack-fraction 0.2 --seed 13
```

## Outputs

The generator writes its CSV and adjacent input-specification JSON. Selection writes `selected.json`, `clients.csv`, `credibility.csv`, `iterations.csv`, `stress.csv`, `metrics.json` and `credibility.png` under `runs/generated/`.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
