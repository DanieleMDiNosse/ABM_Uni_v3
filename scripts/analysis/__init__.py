"""scripts/analysis — Post-processing analyses for paper-quality figures.

Each module exposes functions that accept the raw dict returned by
``scripts.run.simulate()`` (single-run or list-of-dicts for multi-seed).

Quick start::

    python -m scripts.analysis.run_paper_figures \\
        --config abm_results/scenarios/test.yml \\
        --runs 10 --max-workers 4 \\
        --output-dir paper/images/analysis
"""
