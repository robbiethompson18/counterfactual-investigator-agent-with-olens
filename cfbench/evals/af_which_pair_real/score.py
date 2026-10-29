"""Score af_which_pair_real runs: af_which_pair's scorer (Brier, AUROC, pick accuracy, order consistency).

uv run python -m cfbench.evals.af_which_pair_real.score data/runs/cfbench/<date>_<slug> [...]
"""

from cfbench.evals.af_which_pair.score import main

if __name__ == "__main__":
    main()
