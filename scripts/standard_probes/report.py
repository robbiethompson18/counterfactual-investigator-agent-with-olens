"""Validate and summarize the standardized probe cells; outputs belong to the tracked run."""

import argparse
import json
from pathlib import Path

from scripts.standard_probes.fit import mcq_inner_assignment, read_jsonl

FAMILIES = ("essay", "hinted", "mmlu", "supergpqa", "combined")
SITES = ("user_end", "response_start")


def report(run: Path):
    rows = read_jsonl(run / "manifest.jsonl")
    manifest = {row["id"]: row for row in rows}
    assert len(manifest) == len(rows)
    summaries = {}
    for family in FAMILIES:
        targets = (family,) if family in ("essay", "hinted") else ("mmlu", "supergpqa")
        expected = {row["id"] for row in rows if row["family"] in targets}
        for site in SITES:
            directory = run / "fits" / family / site
            summary = json.loads((directory / "summary.json").read_text())
            predictions = read_jsonl(directory / "predictions.jsonl")
            assert len(predictions) == len(expected), (family, site, len(predictions), len(expected))
            assert {row["id"] for row in predictions} == expected
            for fold in summary["folds"]:
                training = [manifest[key] for key in fold["train_ids"]]
                testing = [manifest[key] for key in expected if manifest[key]["outer_fold"] == fold["fold"]]
                assert {row["group"] for row in training}.isdisjoint(row["group"] for row in testing)
                assert all(row["outer_fold"] != fold["fold"] for row in training)
                assert len(training) <= 1000
                if family == "combined":
                    assert all(sum(row["family"] == source for row in training) <= 500 for source in targets)
                else:
                    assert all(row["family"] == family for row in training)
                inner = fold["inner_fold"]
                assert len(inner) == len(training)
                if family != "essay":
                    shared = mcq_inner_assignment(rows, fold["fold"])
                    assert inner == [shared[row["group"]] for row in training]
                assert {row["group"] for row, k in zip(training, inner, strict=True) if k == 0}.isdisjoint(
                    row["group"] for row, k in zip(training, inner, strict=True) if k == 1
                )
            summaries[family, site] = summary
        assert [f["train_ids"] for f in summaries[family, SITES[0]]["folds"]] == [
            f["train_ids"] for f in summaries[family, SITES[1]]["folds"]
        ]
    lines = [
        "# Standard probes: nested two-fold results",
        "",
        "Both positions reported separately. Layers selected only in inner grouped two-fold validation.",
        "MCQs keep >=7/8 consistency without correctness or disagreement selection. All reported predictions are outer-held-out.",
        "Training is capped at 1,000 questions (combined: 500 per benchmark); all eligible questions are evaluated.",
        "",
        "| Training family | Evaluation | Position | n | Accuracy | Qwen-wrong accuracy (n) | NLL | Selected layers |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for family in FAMILIES[1:]:
        for site in SITES:
            summary = summaries[family, site]
            selected = ", ".join(str(f["selected_layer"]) for f in summary["folds"])
            for target, metrics in summary["metrics"].items():
                overall, wrong = metrics["all"], metrics["qwen_wrong"]
                lines.append(
                    f"| {family} | {target} | {site} | {overall['n']} | {overall['accuracy']:.2%} | "
                    f"{wrong['accuracy']:.2%} ({wrong['n']}) | {overall['nll']:.3f} | {selected} |"
                )
    lines.extend(
        [
            "",
            "| Essay position | n | Essays | AUROC | Minus-observed baseline | Grade RMSE | Selected layers |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for site in SITES:
        summary = summaries["essay", site]
        metrics = summary["metrics"]["essay"]
        selected = ", ".join(str(f["selected_layer"]) for f in summary["folds"])
        lines.append(
            f"| {site} | {metrics['n']} | {metrics['groups']} | {metrics['auroc']:.4f} | "
            f"{metrics['baseline_auroc']:.4f} | {metrics['grade_rmse']:.3f} | {selected} |"
        )
    hinted = [row for row in rows if row["family"] == "hinted"]
    hint_baseline = sum(row["label"] == row["gold"] for row in hinted) / len(hinted)
    lines.extend(["", f"Hinted MCQ baseline: always choose the hinted gold letter = {hint_baseline:.2%} ({len(hinted)} questions)."])
    lines.extend(
        [
            "",
            "MCQ capture pins Qwen3.6-27B revision in provenance.json. Historical essay activations identify the same model",
            "but did not record a checkpoint revision. Hinted MCQ uses the canonical gold-letter hint (`hint+d1`) and predicts",
            "the no-hint modal answer; it does not condition on whether Qwen follows that hint. SuperGPQA's original",
            "non-calculation sampling restriction remains. These are standardized reruns, not an untouched confirmation set.",
            "",
            "Validation: every expected held-out ID appears once, outer and inner groups are disjoint, training caps hold,",
            "and both positions use identical training questions. Per-fold metrics, training IDs, layer-selection scores,",
            "model weights, preprocessing parameters and all predictions are saved under fits/.",
        ]
    )
    (run / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    report(parser.parse_args().run)
