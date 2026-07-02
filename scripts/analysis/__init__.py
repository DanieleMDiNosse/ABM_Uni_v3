"""scripts/analysis — Post-processing analyses for paper-quality figures.

Each module exposes functions that accept the raw dict returned by
``scripts.run.simulate()`` (single-run or list-of-dicts for multi-seed).

Quick start::

    python -m scripts.analysis.run_paper_figures \\
        --config configs/scenarios/section4_microstructure_model0_static.yml \\
        --runs 10 --max-workers 4 \\
        --output-dir paper/images/analysis
"""
