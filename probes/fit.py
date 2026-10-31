"""Nested two-fold probes: python -m probes.fit RUN --family essay|mmlu|supergpqa|combined|hinted.

Outputs are separate per family/site under RUN/fits. Input rows and capture chunks remain on disk;
only a single layer's selected training matrix is materialized. --smoke checks the numerical objective.
"""

import argparse
import hashlib
import json
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.optimize import check_grad, minimize
from scipy.special import logsumexp
from scipy.stats import rankdata

LAYERS = [32, 40, 48, 52, 56, 60]
SITES = ["user_end", "response_start"]
REGULARIZATION = 1.0
SEED = 0


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def auc(y, score):
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        raise ValueError("AUROC needs both classes in every essay validation/evaluation subset")
    ranks = rankdata(score)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def capped(rows, cap):
    """Select complete question/essay groups; source-family sampling is independent of the site."""
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    groups = np.random.default_rng(SEED).permutation(sorted(by_group))
    selected = []
    for group in groups:
        members = by_group[group]
        if len(selected) + len(members) <= cap:
            selected.extend(members)
    return sorted(selected, key=lambda row: row["row_index"])


def inner_folds(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    groups = np.random.default_rng(SEED).permutation(sorted(by_group))
    counts, assignment = [0, 0], {}
    for group in groups:
        fold = int(counts[1] < counts[0])
        assignment[group] = fold
        counts[fold] += len(by_group[group])
    return np.array([assignment[row["group"]] for row in rows])


def mcq_inner_assignment(rows, outer):
    """One pre-cap partition per outer fold, shared by every MCQ source family and position."""
    groups = sorted({row["group"] for row in rows if row["family"] != "essay" and row["outer_fold"] != outer})
    shuffled = np.random.default_rng(SEED).permutation(groups)
    return {group: index % 2 for index, group in enumerate(shuffled)}


def standardize(x):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x - mean) / std, mean, std


def logistic_objective(parameters, x, labels, n_options, regularization=REGULARIZATION):
    n, d = x.shape
    weights = parameters[: d * 10].reshape(d, 10)
    bias = parameters[d * 10 :]
    logits = x @ weights + bias
    logits[np.arange(10)[None, :] >= n_options[:, None]] = -np.inf
    log_z = logsumexp(logits, axis=1)
    value = float(np.mean(log_z - logits[np.arange(n), labels]) + 0.5 * regularization * np.square(weights).sum())
    gradient_logits = np.exp(logits - log_z[:, None])
    gradient_logits[np.arange(n), labels] -= 1.0
    gradient_logits /= n
    gradient = np.concatenate(((x.T @ gradient_logits + regularization * weights).ravel(), gradient_logits.sum(axis=0)))
    return value, gradient


def fit_model(x, rows, essay):
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite training activations")
    z, mean, std = standardize(x)
    y = np.array([row["label"] for row in rows], dtype=float if essay else int)
    if essay:
        y_mean = float(y.mean())
        # Mean squared error + lambda ||w||^2 is sum-loss ridge with alpha=n*lambda.
        kernel = z @ z.T
        kernel.flat[:: len(rows) + 1] += len(rows) * REGULARIZATION
        dual = solve(kernel, y - y_mean, assume_a="pos")
        weights = z.T @ dual
        return {"weights": weights, "bias": np.array(y_mean), "mean": mean, "std": std}, {"alpha": len(rows) * REGULARIZATION}
    n_options = np.array([row["n_options"] for row in rows])
    if np.any(y < 0) or np.any(y >= n_options):
        raise ValueError("A training target is outside its question's options")
    result = minimize(
        logistic_objective,
        np.zeros(z.shape[1] * 10 + 10),
        args=(z, y, n_options),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "gtol": 1e-6, "ftol": 1e-10, "maxls": 40},
    )
    gradient_max = float(np.abs(result.jac).max())
    if not result.success and gradient_max > 2e-5:
        raise RuntimeError(f"Logistic fit did not converge: {result.message}; gradient={gradient_max}")
    model = {
        "weights": result.x[: z.shape[1] * 10].reshape(z.shape[1], 10),
        "bias": result.x[z.shape[1] * 10 :],
        "mean": mean,
        "std": std,
    }
    return model, {
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "gradient_max": gradient_max,
        "converged": bool(result.success),
    }


def predict(model, x, rows, essay):
    logits = ((x - model["mean"]) / model["std"]) @ model["weights"] + model["bias"]
    if essay:
        return logits
    n_options = np.array([row["n_options"] for row in rows])
    logits[np.arange(10)[None, :] >= n_options[:, None]] = -np.inf
    return np.exp(logits - logsumexp(logits, axis=1)[:, None])


class Activations:
    """Resolve saved IDs to mmap chunks; works with contiguous or length-balanced capture shards."""

    def __init__(self, run, rows, root=None):
        self.run = run
        self.root = root or run / "activations"
        self.by_id = {}
        self.arrays = OrderedDict()
        self.essay = {}
        self.rows = rows

    def index(self):
        if self.by_id:
            return
        for path in sorted(self.root.rglob("logp.jsonl")):
            records = read_jsonl(path)
            for index, record in enumerate(records):
                if record["id"] in self.by_id:
                    raise ValueError(f"Duplicate capture ID {record['id']}")
                self.by_id[record["id"]] = (path.parent, index)
        if not self.by_id:
            # Capture-only runners may save ids.jsonl instead of unnecessary next-token logits.
            for path in sorted(self.root.rglob("ids.jsonl")):
                for index, record in enumerate(read_jsonl(path)):
                    key = record if isinstance(record, str) else record["id"]
                    if key in self.by_id:
                        raise ValueError(f"Duplicate capture ID {key}")
                    self.by_id[key] = (path.parent, index)
        if not self.by_id:
            raise FileNotFoundError(f"No completed capture chunks under {self.root}")

    def get(self, rows, site, layer_index):
        if rows[0]["family"] == "essay":
            if site not in self.essay:
                self.essay[site] = np.load(self.run / "essay_activations" / f"{site}.npy", mmap_mode="r")
            indices = [row["activation_index"] for row in rows]
            return np.asarray(self.essay[site][indices, layer_index], dtype=np.float64)
        self.index()
        grouped = defaultdict(list)
        for output_index, row in enumerate(rows):
            directory, input_index = self.by_id[row["id"]]
            grouped[directory].append((output_index, input_index))
        output = None
        for directory, indices in grouped.items():
            key = (directory, site)
            if key not in self.arrays:
                layers_path = directory / "layers.json"
                if layers_path.exists() and json.loads(layers_path.read_text()) != LAYERS:
                    raise ValueError(f"Unexpected layer order in {directory}")
                self.arrays[key] = np.load(directory / f"{site}.npy", mmap_mode="r")
                if len(self.arrays) > 32:
                    self.arrays.popitem(last=False)
            self.arrays.move_to_end(key)
            array = self.arrays[key]
            if output is None:
                output = np.empty((len(rows), array.shape[2]), dtype=np.float64)
            target, source = zip(*indices, strict=True)
            output[list(target)] = array[list(source), layer_index]
        return output


def evaluate(rows, prediction, essay):
    if essay:
        y = np.array([row["label"] for row in rows])
        observed = np.array([row["observed"] for row in rows])
        truth = np.array([row["truth"] for row in rows])
        shift = prediction - observed
        return {
            "n": len(rows),
            "groups": len({row["group"] for row in rows}),
            "auroc": auc(truth, shift),
            "baseline_auroc": auc(truth, -observed),
            "grade_rmse": float(np.sqrt(np.mean(np.square(prediction - y)))),
            "shift_rmse": float(np.sqrt(np.mean(np.square(shift - (y - observed))))),
            "grade_mae": float(np.mean(np.abs(prediction - y))),
        }
    y = np.array([row["label"] for row in rows])
    gold = np.array([row["gold"] for row in rows])
    correct = prediction.argmax(axis=1) == y
    nll = -np.log(np.maximum(prediction[np.arange(len(rows)), y], np.finfo(float).tiny))
    result = {}
    for name, mask in (("all", np.ones(len(rows), dtype=bool)), ("qwen_right", y == gold), ("qwen_wrong", y != gold)):
        result[name] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
            "nll": float(nll[mask].mean()) if mask.any() else None,
        }
    return result


def training_rows(rows, family, fold):
    available = [row for row in rows if row["outer_fold"] != fold]
    if family == "combined":
        sources = [[row for row in available if row["family"] == name] for name in ("mmlu", "supergpqa")]
        cap = min(500, *(len(source) for source in sources))
        return sorted([row for source in sources for row in capped(source, cap)], key=lambda row: row["row_index"])
    return capped([row for row in available if row["family"] == family], 1000)


def run_fit(run, family, site, activations, rows):
    started = time.monotonic()
    essay = family == "essay"
    targets = [family] if family in ("essay", "hinted") else ["mmlu", "supergpqa"]
    out = run / "fits" / family / site
    out.mkdir(parents=True, exist_ok=True)
    (out / "fit_source.py").write_text(Path(__file__).read_text())
    predictions, fold_records, pooled = [], [], defaultdict(list)
    for outer in range(2):
        train = training_rows(rows, family, outer)
        test = [row for row in rows if row["family"] in targets and row["outer_fold"] == outer]
        if not train or not test:
            raise ValueError("An outer fold is empty")
        train_groups = {row["group"] for row in train}
        assert train_groups.isdisjoint(row["group"] for row in test)
        if essay:
            assignment = None
            inner = inner_folds(train)
        else:
            assignment = mcq_inner_assignment(rows, outer)
            inner = np.array([assignment[row["group"]] for row in train])
        layer_scores: list[dict] = []
        print(f"{family}/{site} outer={outer} train={len(train)} test={len(test)}", flush=True)
        for layer_index, layer in enumerate(LAYERS):
            x = activations.get(train, site, layer_index)
            validation = []
            for held in range(2):
                fitting = [row for row, fold in zip(train, inner, strict=True) if fold != held]
                scoring = [row for row, fold in zip(train, inner, strict=True) if fold == held]
                assert {row["group"] for row in fitting}.isdisjoint(row["group"] for row in scoring)
                model, diagnostics = fit_model(x[inner != held], fitting, essay)
                prediction = predict(model, x[inner == held], scoring, essay)
                metrics = evaluate(scoring, prediction, essay)
                score = metrics["auroc"] if essay else metrics["all"]["accuracy"]
                validation.append({"fold": held, "n_train": len(fitting), "n_validation": len(scoring), "score": score, "fit": diagnostics})
            average = float(np.mean([record["score"] for record in validation]))
            layer_scores.append({"layer": layer, "mean_score": average, "folds": validation})
            print(f"  L{layer}: inner score={average:.5f}", flush=True)
        selected = max(layer_scores, key=lambda record: (record["mean_score"], -record["layer"]))["layer"]
        layer_index = LAYERS.index(selected)
        model, diagnostics = fit_model(activations.get(train, site, layer_index), train, essay)
        np.savez_compressed(out / f"model-fold-{outer}.npz", **model)
        scored = []
        for start in range(0, len(test), 512):
            batch = test[start : start + 512]
            prediction = predict(model, activations.get(batch, site, layer_index), batch, essay)
            scored.extend(prediction)
        scored = np.asarray(scored)
        metrics = {}
        for target in targets:
            mask = np.array([row["family"] == target for row in test])
            target_rows = [row for row in test if row["family"] == target]
            metrics[target] = evaluate(target_rows, scored[mask], essay)
            pooled[target].extend(zip(target_rows, scored[mask], strict=True))
        for row, prediction in zip(test, scored, strict=True):
            entry = {key: row[key] for key in ("id", "family", "item", "group", "outer_fold", "label")}
            entry |= {"probe_family": family, "site": site, "layer": selected}
            if essay:
                entry |= {
                    "predicted_grade": float(prediction),
                    "predicted_shift": float(prediction - row["observed"]),
                    "observed": row["observed"],
                    "truth": row["truth"],
                }
            else:
                entry |= {
                    "probabilities": prediction.tolist(),
                    "predicted_label": int(prediction.argmax()),
                    "gold": row["gold"],
                    "n_options": row["n_options"],
                }
            predictions.append(entry)
        fold_records.append(
            {
                "fold": outer,
                "selected_layer": selected,
                "n_train": len(train),
                "n_train_groups": len(train_groups),
                "train_ids": [row["id"] for row in train],
                "inner_fold": inner.tolist(),
                "inner_partition": "essay_groups" if essay else "all_eligible_mcq_training_groups_before_caps",
                "inner_assignment_sha256": (
                    hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest() if assignment is not None else None
                ),
                "layer_scores": layer_scores,
                "fit": diagnostics,
                "metrics": metrics,
            }
        )
        write_json(out / f"fold-{outer}.json", fold_records[-1])
        with (out / "predictions.jsonl").open("w") as stream:
            for entry in predictions:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(f"  selected L{selected}, outer metrics={metrics}", flush=True)
    overall = {
        target: evaluate([row for row, _ in pairs], np.asarray([prediction for _, prediction in pairs]), essay)
        for target, pairs in pooled.items()
    }
    write_json(
        out / "summary.json",
        {
            "family": family,
            "site": site,
            "layers": LAYERS,
            "regularization": REGULARIZATION,
            "seed": SEED,
            "outer_folds": 2,
            "inner_folds": 2,
            "metrics": overall,
            "folds": fold_records,
            "seconds": time.monotonic() - started,
            "manifest_sha256": hashlib.sha256((run / "manifest.jsonl").read_bytes()).hexdigest(),
        },
    )
    print(f"DONE {family}/{site}: {overall}", flush=True)


def smoke():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(40, 7))
    n_options = rng.integers(3, 11, len(x))
    y = np.array([rng.integers(n) for n in n_options])
    parameters = rng.normal(scale=0.1, size=7 * 10 + 10)
    error = check_grad(lambda p: logistic_objective(p, x, y, n_options)[0], lambda p: logistic_objective(p, x, y, n_options)[1], parameters)
    assert error < 1e-5, error
    rows = [
        {"label": int(label), "n_options": int(n), "group": str(i // 2), "row_index": i}
        for i, (label, n) in enumerate(zip(y, n_options, strict=True))
    ]
    model, _ = fit_model(x, rows, False)
    probability = predict(model, x, rows, False)
    assert np.allclose(probability.sum(axis=1), 1)
    assert np.all(probability[np.arange(10)[None, :] >= n_options[:, None]] == 0)
    folds = inner_folds(rows)
    assert {row["group"] for row, fold in zip(rows, folds, strict=True) if fold == 0}.isdisjoint(
        row["group"] for row, fold in zip(rows, folds, strict=True) if fold == 1
    )
    assert len(capped(rows, 11)) <= 11
    essay_rows = [row | {"label": float(x[i, 0] * 2 + 3)} for i, row in enumerate(rows)]
    ridge, _ = fit_model(x, essay_rows, True)
    z, _, _ = standardize(x)
    centered = np.array([row["label"] for row in essay_rows]) - np.mean([row["label"] for row in essay_rows])
    primal = solve(z.T @ z + len(x) * np.eye(x.shape[1]), z.T @ centered, assume_a="pos")
    assert np.allclose(ridge["weights"], primal)
    assert auc([0, 0, 1, 1], [1, 1, 2, 3]) == 1
    print(f"smoke passed: gradient error={error:.3g}; masked softmax, grouped splits, ridge equivalence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--family", choices=["essay", "mmlu", "supergpqa", "combined", "hinted"])
    parser.add_argument("--site", choices=SITES, action="append")
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    if args.run is None or args.family is None:
        parser.error("RUN and --family are required unless --smoke")
    rows = read_jsonl(args.run / "manifest.jsonl")
    assert len({row["id"] for row in rows}) == len(rows), "Duplicate manifest IDs"
    group_folds = defaultdict(set)
    for row in rows:
        group_folds[row["group"]].add(row["outer_fold"])
    assert all(len(folds) == 1 for folds in group_folds.values()), "Question group crosses outer folds"
    activations = Activations(args.run, rows, args.activations)
    for site in args.site or SITES:
        run_fit(args.run, args.family, site, activations, rows)


if __name__ == "__main__":
    main()
