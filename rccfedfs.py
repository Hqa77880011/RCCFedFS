import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def weighted_quantile(values, weights, q):
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    expanded = np.broadcast_to(weights.reshape((-1,) + (1,) * (values.ndim - 1)), values.shape)
    sorted_weights = np.take_along_axis(expanded, order, axis=0)
    ids = (sorted_weights.cumsum(0) < q * sorted_weights.sum(0)).sum(0).clip(0, len(values) - 1)
    return np.take_along_axis(sorted_values, ids[None], axis=0)[0]


def summaries(frame, features):
    records = []
    for client, group in frame.groupby("client", sort=True):
        x = group[features].to_numpy(float)
        y = group.label.to_numpy()
        classes = np.unique(y)
        signs = []
        for label in classes:
            binary = (y == label).astype(float)
            denominator = x.std(0) * binary.std()
            signs.append(np.divide(((x - x.mean(0)) * (binary - binary.mean())[:, None]).mean(0), denominator, out=np.zeros(len(features)), where=denominator > 1e-10))
        correlations = np.stack(signs)
        winner = np.argmax(np.abs(correlations), axis=0)
        relevance = correlations[winner, np.arange(len(features))]
        redundancy = np.nan_to_num(np.corrcoef(x, rowvar=False))
        np.fill_diagonal(redundancy, 1)
        records.append(dict(client=str(client), n=len(x), relevance=relevance, lower=np.maximum(-1, relevance - 1.96 / np.sqrt(max(4, len(x)))), upper=np.minimum(1, relevance + 1.96 / np.sqrt(max(4, len(x)))), redundancy=redundancy, marginal=x.mean(0), prior=float(np.max(np.bincount(y.astype(int))) / len(y)), malicious=bool(group.malicious.iloc[0]) if "malicious" in group else False, extreme=bool(group.extreme.iloc[0]) if "extreme" in group else False))
    return records


def attack(records, args, rng):
    ids = rng.choice(len(records), int(len(records) * args.attack_fraction), replace=False)
    d = len(records[0]["relevance"])
    targets = np.arange(max(0, d - args.target_features), d)
    for i in ids:
        r = records[i]
        r["malicious"] = True
        mode = args.attack if args.attack != "mixed" else rng.choice(["sign", "magnitude", "stuffing", "redundancy", "steering", "collusion"])
        if mode == "sign":
            r["relevance"] = -r["relevance"]
        elif mode in ("magnitude", "stuffing", "collusion"):
            r["relevance"][targets] = .99
        elif mode == "redundancy":
            r["redundancy"][targets, :] = 0
            r["redundancy"][:, targets] = 0
            np.fill_diagonal(r["redundancy"], 1)
        elif mode == "steering":
            r["marginal"][targets] += 3
            r["lower"][targets] = -.9
            r["upper"][targets] = .99
        if mode != "steering":
            r["lower"] = np.maximum(-1, r["relevance"] - .05)
            r["upper"] = np.minimum(1, r["relevance"] + .05)
    return targets if len(ids) else np.array([], dtype=int)


def anchor(records, weights, args):
    r = np.stack([v["relevance"] for v in records])
    lo = np.stack([v["lower"] for v in records])
    hi = np.stack([v["upper"] for v in records])
    lower = weighted_quantile(lo, weights, args.trim)
    upper = weighted_quantile(hi, weights, 1 - args.trim)
    center = weighted_quantile(r, weights, .5)
    mad = weighted_quantile(np.abs(r - center), weights, .5)
    red = weighted_quantile(np.stack([v["redundancy"] for v in records]), weights, .5)
    red = (red + red.T) / 2
    eig, vec = np.linalg.eigh(red)
    red = (vec * np.maximum(eig, 1e-6)) @ vec.T
    diagonal = np.sqrt(np.diag(red))
    red /= diagonal[:, None] * diagonal[None, :]
    priors = np.array([v["prior"] for v in records])[:, None]
    prior = np.array([weighted_quantile(priors, weights, q)[0] for q in (args.trim, 1 - args.trim)])
    marginal = weighted_quantile(np.stack([v["marginal"] for v in records]), weights, .5)
    dispersion = float(np.sum(weights[:, None] * np.abs(r - center)) / r.shape[1])
    support = 1 / (len(weights) * np.sum(weights ** 2)) * np.exp(-dispersion)
    radius = args.radius_min + (args.radius_max - args.radius_min) * support
    if args.variant == "fixed_radius":
        radius = args.radius_max
    descriptor = np.r_[lower, upper, prior, red[np.triu_indices(len(center), 1)], marginal]
    domain = np.stack([descriptor - radius, descriptor + radius])
    return dict(center=center, mad=mad, lower=np.minimum(lower, upper), upper=np.maximum(lower, upper), redundancy=red, prior=prior, marginal=marginal, radius=radius, support=support, domain=domain)


def hausdorff(first, second):
    distance = np.linalg.norm(first[:, None] - second[None, :], axis=-1) / np.sqrt(first.shape[1])
    return max(distance.min(0).max(), distance.min(1).max())


def select(records, args):
    relevance = np.stack([r["relevance"] for r in records])
    red = np.stack([r["redundancy"] for r in records])
    d = relevance.shape[1]
    candidates = np.zeros_like(relevance, dtype=bool)
    candidates[np.arange(len(records))[:, None], np.argsort(-np.abs(relevance), axis=1)[:, :min(args.candidates, d)]] = True
    consensus = candidates.mean(0) >= .5
    median = np.median(relevance, axis=0)
    agreement = []
    for i, record in enumerate(records):
        jac = np.sum(candidates[i] & consensus) / max(1, np.sum(candidates[i] | consensus))
        tau = kendalltau(np.abs(relevance[i]), np.abs(median)).statistic
        tau = 0 if not np.isfinite(tau) else tau
        sign = np.mean(np.sign(relevance[i]) == np.sign(median))
        redundancy = 1 - np.abs(red[i] - np.median(red, axis=0)).mean()
        agreement.append(jac + .5 * (tau + sign) + redundancy + .05 * np.log1p(record["n"]))
    q = np.asarray(agreement)
    weights = softmax(q) if args.variant != "uniform" else np.ones(len(q)) / len(q)
    trajectory = []
    for step in range(args.iterations):
        current = anchor(records, weights, args)
        influence = []
        for e in range(len(records)):
            keep = np.arange(len(records)) != e
            other = anchor([r for i, r in enumerate(records) if keep[i]], weights[keep] / weights[keep].sum(), args)
            influence.append(hausdorff(current["domain"], other["domain"]))
        influence = np.asarray(influence)
        influence /= max(np.median(influence), 1e-8)
        updated = softmax(q - args.influence_weight * influence)
        trajectory.append(dict(iteration=step, radius=current["radius"], support=current["support"], weight_change=float(np.abs(updated - weights).sum())))
        if args.variant in ("no_influence", "uniform"):
            break
        converged = np.abs(updated - weights).sum() < 1e-5
        weights = updated
        if converged or args.variant == "one_shot":
            break
    state = anchor(records, weights, args)
    real = np.abs(state["center"]) - state["mad"]
    worst = real.copy()
    stress = []
    for j in range(d):
        for operator in ("label_skew", "nuisance", "sensor_drift"):
            for shift in np.linspace(-state["radius"], state["radius"], args.grid):
                candidate = state["center"].copy()
                if operator == "label_skew":
                    candidate[j] *= np.clip(1 - abs(shift), 0, 1)
                elif operator == "nuisance":
                    candidate[j] -= shift * state["redundancy"][j].mean()
                else:
                    candidate[j] += shift
                if args.variant != "no_guard":
                    candidate[j] = np.clip(candidate[j], state["lower"][j], state["upper"][j])
                margin = abs(candidate[j]) - state["mad"][j]
                worst[j] = min(worst[j], margin)
                stress.append([j, operator, float(shift), float(candidate[j]), float(margin)])
    if args.variant == "no_worst":
        worst = real.copy()
    score = (1 - args.alpha) * real + args.alpha * worst
    chosen = []
    for _ in range(min(args.k, d)):
        penalty = np.abs(state["redundancy"][:, chosen]).mean(1) if chosen else np.zeros(d)
        current = score - args.redundancy_weight * penalty
        current[chosen] = -np.inf
        chosen.append(int(np.argmax(current)))
    return chosen, weights, influence, state, trajectory, stress


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--data-source", default="user_supplied")
    p.add_argument("--output", default="outputs/full")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=30)
    p.add_argument("--attack", choices=["none", "mixed", "sign", "magnitude", "stuffing", "redundancy", "steering", "collusion"], default="none")
    p.add_argument("--attack-fraction", type=float, default=.2)
    p.add_argument("--target-features", type=int, default=5)
    p.add_argument("--variant", choices=["full", "uniform", "no_influence", "one_shot", "fixed_radius", "no_guard", "no_worst"], default="full")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--influence-weight", type=float, default=1.)
    p.add_argument("--trim", type=float, default=.1)
    p.add_argument("--radius-min", type=float, default=.02)
    p.add_argument("--radius-max", type=float, default=.3)
    p.add_argument("--alpha", type=float, default=.5)
    p.add_argument("--redundancy-weight", type=float, default=.2)
    p.add_argument("--grid", type=int, default=17)
    p.add_argument("--credibility-cutoff", type=float, default=.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    df = pd.read_csv(args.data)
    meta = {"client", "split", "label", "malicious", "extreme"}
    features = [c for c in df if c not in meta]
    if len(features) < 2 or df.groupby("client").split.nunique().max() != 1:
        raise ValueError("Use at least two features and client-disjoint splits")
    train = df[df.split == "train"]
    future = df[df.split == "test"]
    if train.client.nunique() < 3 or future.empty or not np.isfinite(df[features]).all().all():
        raise ValueError("Need >=3 training clients, future test clients and finite features")
    rng = np.random.default_rng(args.seed)
    records = summaries(train, features)
    targets = attack(records, args, rng) if args.attack != "none" else np.array([], int)
    started = time.perf_counter()
    chosen, weights, influence, state, trajectory, stress = select(records, args)
    elapsed = time.perf_counter() - started
    scaler = StandardScaler().fit(train[features].iloc[:, chosen])
    classifier = LogisticRegression(max_iter=1000, random_state=args.seed).fit(scaler.transform(train[features].iloc[:, chosen]), train.label)
    predictions = classifier.predict(scaler.transform(future[features].iloc[:, chosen]))
    scores = []
    for client in future.client.unique():
        ids = (future.client == client).to_numpy()
        scores.append(dict(client=str(client), macro_f1=float(f1_score(future.label.to_numpy()[ids], predictions[ids], average="macro", zero_division=0))))
    malicious = np.array([r["malicious"] for r in records])
    extreme = np.array([r["extreme"] and not r["malicious"] for r in records])
    normalized_weight = weights * len(weights)
    future_records = summaries(future, features)
    covered = [np.linalg.norm(r["relevance"] - state["center"]) / np.sqrt(len(features)) <= state["radius"] for r in future_records]
    metrics = dict(data_source=args.data_source, macro_f1=float(f1_score(future.label, predictions, average="macro", zero_division=0)), worst_client_macro_f1=min(s["macro_f1"] for s in scores), balanced_accuracy=float(balanced_accuracy_score(future.label, predictions)), attack_target_selection=float(np.isin(chosen, targets).mean()) if len(targets) else None, malicious_detection_auroc=float(roc_auc_score(malicious, -weights)) if len(np.unique(malicious)) == 2 else None, honest_extreme_fpr=float((normalized_weight[extreme] < args.credibility_cutoff).mean()) if extreme.any() else None, future_domain_coverage=float(np.mean(covered)), server_seconds=elapsed, upload_bytes=int(sum(sum(v.nbytes for v in r.values() if isinstance(v, np.ndarray)) + 16 for r in records)), radius=state["radius"], effective_support=state["support"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "selected.json").write_text(json.dumps([features[i] for i in chosen], indent=2))
    pd.DataFrame(scores).to_csv(out / "clients.csv", index=False)
    pd.DataFrame(dict(client=[r["client"] for r in records], weight=weights, influence=influence, malicious=malicious)).to_csv(out / "credibility.csv", index=False)
    pd.DataFrame(trajectory).to_csv(out / "iterations.csv", index=False)
    pd.DataFrame(stress, columns=["feature", "operator", "shift", "association", "margin"]).to_csv(out / "stress.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(weights)), normalized_weight, color=np.where(malicious, "#bc6969", "#4d8796"))
    plt.axhline(args.credibility_cutoff, color="black", linestyle="--")
    plt.xlabel("Training client")
    plt.ylabel("Credibility relative to uniform")
    plt.tight_layout()
    plt.savefig(out / "credibility.png", dpi=180)


if __name__ == "__main__":
    main()
