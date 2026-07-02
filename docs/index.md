---
title: Home
nav_order: 1
---

# ABM Uni v3 Documentation

Documentation for the current `ABM_Uni_v3` codebase: a Python simulator for a Uniswap v3-style pool, plus experiment-design utilities, diagnostics scripts, and a live Dash webapp.

## Quick Start

```bash
conda activate main
python -m scripts.run --config configs/scenarios/section4_microstructure_model0_static.yml
pytest -q
```

Main outputs land under `abm_results/scenarios/<scenario_name>/`. The newest CLI run for each scenario is recorded in `abm_results/scenarios/<scenario_name>/latest_run.json`.

## Documentation Map

- [Model Overview](README.md): simulator scope, preferred scenario knobs, output layout, and main entry points.
- [Agent Behaviour Details](agents_spec.md): execution ordering, mempool semantics, LP behavior, arbitrage, and JIT logic as implemented in `scripts/run.py`.
- [Loss-Versus-Rebalancing](LVR_explanation.md): theory and the repo’s discrete-time benchmark construction.
- [LP PnL](LP_PnL.md): unhedged vs hedged LP accounting, fees, LVR, and reported series.
- [Fee Schedules](fee_schedules.md): the five fee modes and the shared controller logic.
- [Asymmetric Dynamic Fees Paper Summary](asymmetric_dynamic_fees_paper_summary.md): reading note for arXiv:2506.02869v1 and its proposed direction-specific fee schedules.
- [Sigma Calibration](sigma_calibration.md): turning Binance 1-second ETH/USDC data into `cex_sigma` inputs.
- [Stress Tests](stress_tests.md): YAML-only stress scenarios and recommended diagnostics.
- [Webapp](webapp.md): Dash architecture, persistence, crash recovery, and live telemetry.
- [LP Width Mint Signals](lp_width_mint_signals.md): research notes on alternative active-LP width signals.
- [nD Sampling Designs](nd_grid_sampling_methods.md): grid, sampled, and sequential experiment workflows.

## Output Conventions

- Run config YAMLs live under `configs/` (`configs/scenarios/`, `configs/experiments/`, and `configs/paper/`).
- Each CLI run writes a new record under `abm_results/scenarios/<scenario_name>/runs/<run_id>/`.
- Experiment-design caches write immutable tagged folders under `abm_results/experiments_runs/`.
- ND grid sweeps write global caches under `abm_results/grid_search/dashboard_nd/`.
- Webapp runs are isolated under `abm_results/web_runs/<run_id>/`.

## Pages

{% assign nav_pages = site.html_pages | sort: "nav_order" %}
<ul>
  {% for p in nav_pages %}
    {% if p.nav_order and p.title %}
      {% unless p.nav_exclude %}
        {% if p.url != page.url %}
          {% assign href = p.url %}
          {% if href == "/" %}
            {% assign href = "index.html" %}
          {% else %}
            {% assign href = href | remove_first: "/" %}
          {% endif %}
          <li><a href="{{ href }}">{{ p.title | default: p.name }}</a></li>
        {% endif %}
      {% endunless %}
    {% endif %}
  {% endfor %}
</ul>
