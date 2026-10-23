"""CLI for src/counterfactual_investigator_agent_with_olens/runs.py (usage in its docstring). The re-exports keep old
`from runs import ...` callers in scripts/ working; import from the package in new code."""

from counterfactual_investigator_agent_with_olens.runs import finish, main, new_run, upload  # noqa: F401

if __name__ == "__main__":
    main()
