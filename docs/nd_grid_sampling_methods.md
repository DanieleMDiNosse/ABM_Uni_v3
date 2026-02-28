# nD Sampling Designs for ABM Parameter Sweeps


- [<span class="toc-section-number">1</span> 1) What Are We
  Sampling?](#1-what-are-we-sampling)
- [<span class="toc-section-number">2</span> 2) Parameter-Space Sampling
  Designs (What “Grid / LHS / Sobol / Saltelli”
  Mean)](#2-parameter-space-sampling-designs-what-grid--lhs--sobol--saltelli-mean)
  - [<span class="toc-section-number">2.1</span> 2.1 Full-factorial grid
    (Cartesian product)](#21-full-factorial-grid-cartesian-product)
  - [<span class="toc-section-number">2.2</span> 2.2 Latin Hypercube
    Sampling (LHS): stratified, space-filling
    screening](#22-latin-hypercube-sampling-lhs-stratified-space-filling-screening)
  - [<span class="toc-section-number">2.3</span> 2.3 Sobol sequence:
    quasi-random low-discrepancy
    coverage](#23-sobol-sequence-quasi-random-low-discrepancy-coverage)
  - [<span class="toc-section-number">2.4</span> 2.4 Sobol–Saltelli
    design: structured samples for Sobol sensitivity
    indices](#24-sobolsaltelli-design-structured-samples-for-sobol-sensitivity-indices)
  - [<span class="toc-section-number">2.5</span> 2.5 Sequential designs:
    adaptive refinement and Bayesian
    optimization](#25-sequential-designs-adaptive-refinement-and-bayesian-optimization)
- [<span class="toc-section-number">3</span> 3) Mapping a Design Point
  to Actual ABM Parameters (Mixed Continuous +
  Discrete)](#3-mapping-a-design-point-to-actual-abm-parameters-mixed-continuous--discrete)
  - [<span class="toc-section-number">3.1</span> 3.1 Continuous
    parameters](#31-continuous-parameters)
  - [<span class="toc-section-number">3.2</span> 3.2 Discrete parameters
    (explicit value
    lists)](#32-discrete-parameters-explicit-value-lists)
  - [<span class="toc-section-number">3.3</span> 3.3 ABM-specific
    “scale” tips (choosing
    bounds/transforms)](#33-abm-specific-scale-tips-choosing-boundstransforms)
- [<span class="toc-section-number">4</span> 4) Replication / Seed
  Sampling (What “runs_per_point” and “common_seeds”
  Do)](#4-replication--seed-sampling-what-runs_per_point-and-common_seeds-do)
  - [<span class="toc-section-number">4.1</span> 4.1 Independent seeds
    per point (default)](#41-independent-seeds-per-point-default)
  - [<span class="toc-section-number">4.2</span> 4.2 Common seeds across
    points (paired comparisons / common random
    numbers)](#42-common-seeds-across-points-paired-comparisons--common-random-numbers)
- [<span class="toc-section-number">5</span> 5) Which Design Should You
  Use?](#5-which-design-should-you-use)
- [<span class="toc-section-number">6</span> 6) Reproduction Recipes
  (Minimal)](#6-reproduction-recipes-minimal)
- [<span class="toc-section-number">7</span> Appendix A) A Note on Grid
  Indexing (Only If You Slice Jobs by
  Index)](#appendix-a-a-note-on-grid-indexing-only-if-you-slice-jobs-by-index)

This note explains, at a conceptual level, the **sampling methods** used
in this repo to explore an *n*-dimensional ABM parameter space:

- **Parameter-space sampling**: how we pick parameter vectors (grid
  vs. LHS vs. Sobol vs. Saltelli vs. sequential designs).
- **Replication/seed sampling**: how we allocate RNG seeds at each
  parameter vector to estimate noisy outcomes.

Relevant runners: - Grid (dashboard):
`scripts/run_parameter_surface_nd_pnl_fee_dashboard.py` - General
experiment designs (grid/LHS/Sobol/Saltelli/adaptive_refine/bayesopt):
`scripts/run_experiment_design.py` + `core/experiment_design.py`

Assumptions (make them explicit when interpreting results): - The
simulator is **stochastic**: a single run returns a noisy draw of your
metric(s). - “Better sampling” means “more informative given a fixed
compute budget”, not “more true”. - For non-grid designs, points are
sampled in a **unit hypercube** and then **transformed** into parameter
values (details below).

## 1) What Are We Sampling?

Let: - $\theta \in \Theta \subseteq \mathbb{R}^d$ be the parameter
vector (with some coordinates possibly discrete). - $Y = f(\theta, s)$
be an output metric from one simulation run, where $s$ is the RNG seed.

In practice we care about quantities like: - a conditional expectation
$\mu(\theta) = \mathbb{E}[Y \mid \theta]$, - quantiles of
$Y \mid \theta$, - or rankings/comparisons between settings.

This gives you two independent “sampling” choices: 1) **Design points**:
choose $\theta_1, \dots, \theta_N$ (parameter-space sampling). 2)
**Replicates per point**: for each $\theta_i$, run $R$ seeds to
estimate the distribution of $Y \mid \theta_i$ (replication sampling).

Compute budget bookkeeping: 
$$
\text{total simulator calls} ;\approx; N \times R.
$$ 
For a fixed budget you are trading off *more points* (larger $N$)
vs. *less Monte Carlo noise per point* (larger $R$).

## 2) Parameter-Space Sampling Designs (What “Grid / LHS / Sobol / Saltelli” Mean)

The repo supports these design families:

### 2.1 Full-factorial grid (Cartesian product)

Concept: - You choose an explicit discrete list of values for each
parameter. - You evaluate **every combination** of those values (a
Cartesian product).

If parameter $k$ has $n_k$ candidate values, the number of grid points
is: 
$$
N_{\text{grid}} = \prod_{k=1}^d n_k.
$$

What it’s good for: - Low-dimensional surfaces you want to **visualize**
(heatmaps, slices). - Controlled comparisons at specific “knots” you
care about (thresholds, regimes).

Limitations: - **Curse of dimensionality**: $N_{\text{grid}}$ grows
exponentially in $d$. - Resolution is only as good as your chosen value
lists (you can miss behavior between knots).

Where it’s used here: - The dashboard runner
`scripts/run_parameter_surface_nd_pnl_fee_dashboard.py` evaluates a grid
defined by explicit value lists.

### 2.2 Latin Hypercube Sampling (LHS): stratified, space-filling screening

Concept (in the unit hypercube $[0,1)^d$): - Choose a target number of
points $N$. - In each dimension separately, split $[0,1)$ into $N$
equal-width strata. - Sample **one value from each stratum** per
dimension. - Randomly *permute* strata assignments across dimensions so
that, in every dimension, each stratum is used exactly once.

Key intuition: - A small LHS design typically covers each coordinate
axis much more evenly than i.i.d. random sampling. - This is often a
strong default for **screening** (which parameters matter?) when $d$ is
moderate and $N$ is limited. - In 2D: if you draw an $N\times N$ grid
of strata in the unit square, each **row** and each **column** contains
exactly one point.

Limitations: - LHS guarantees good **1D marginal** coverage; it does
*not* guarantee perfect high-dimensional uniformity. - If some
parameters are discrete with only a few allowed values, many LHS strata
can map to the same discrete value after transformation (see Section
3). - LHS is not inherently “nested”: if you decide later that you need
more points, generating a fresh LHS with larger $N$ does not usually
preserve the original points.

Where it’s used here: - Static design type `lhs` in
`scripts/run_experiment_design.py` (implemented via SciPy’s
`scipy.stats.qmc.LatinHypercube` in `core/experiment_design.py`).

### 2.3 Sobol sequence: quasi-random low-discrepancy coverage

Concept: - A Sobol sequence is a deterministic (optionally “scrambled”)
sequence of points in $[0,1)^d$ designed to have **low discrepancy**. -
“Low discrepancy” means the points fill space more evenly than standard
random draws, especially when estimating integrals/averages over
$\Theta$.

Key intuition: - Compared to i.i.d. random sampling, Sobol points
usually produce fewer clusters and fewer large holes for the same $N$. -
Many quasi-Monte Carlo methods (including Sobol) work best when $N$ is a
**power of two**. - Sobol sequences are naturally **nested**: increasing
$N$ can be thought of as taking a longer prefix of the same sequence
(when seed/scramble are held fixed).

Where it’s used here: - Static design type `sobol` in
`scripts/run_experiment_design.py` (implemented via
`scipy.stats.qmc.Sobol(..., scramble=True, seed=...)` in
`core/experiment_design.py`).

### 2.4 Sobol–Saltelli design: structured samples for Sobol sensitivity indices

Goal: - Estimate **variance-based global sensitivity** measures (Sobol
indices), i.e. “how much of $\mathrm{Var}(Y)$ is attributable to
parameter $i$?”

Definitions (for an input vector $X = (X_1,\dots,X_d)$ and scalar
output $Y$): 
$$
S_i = \frac{\mathrm{Var}_{X_i}(\,\mathbb{E}[Y\mid X_i]\,)}{\mathrm{Var}(Y)}
\qquad
S_{T_i} = 1 - \frac{\mathrm{Var}_{X_{\sim i}}(\,\mathbb{E}[Y\mid X_{\sim i}]\,)}{\mathrm{Var}(Y)}.
$$ 
- $S_i$ (first-order) measures the contribution of $X_i$ alone. 
- $S_{T_i}$ (total-order) measures the contribution of $X_i$ including
interactions.

Concept (Saltelli’s construction): 
- Generate two base matrices of
points $A$ and $B$ (each has $N_{\text{base}}$ rows). 
- For each parameter $i$, create a hybrid matrix $AB_i$ that uses all columns from
$A$ except column $i$, which comes from $B$. 
- Total design size is:
$$
N_{\text{Saltelli}} = N_{\text{base}}\,(d + 2).
$$

Practical note for stochastic ABMs: 
- Sobol indices assume the output
variance is driven by *inputs*, but simulation noise also adds
variance. 
- In practice, you often need **more replications per point**
(larger $R$) than you would for a purely deterministic model, or you
accept higher uncertainty in the indices.

Where it’s used here: 
- Design type `sobol_saltelli` in
`scripts/run_experiment_design.py` with analysis support in
`scripts/analyze_experiment_design.py`.

### 2.5 Sequential designs: adaptive refinement and Bayesian optimization

These designs decide where to sample next *based on previous results*
(so point order matters).

**Adaptive refine (`adaptive_refine`)** 
- Starts with an initial
space-filling set (an LHS). 
- Fits a cheap surrogate to existing
results. 
- Proposes new points where either: 
  - the surrogate uncertainty
  is high (to “learn the surface”), or 
  - a regime boundary is likely (if
  you provide a `regime_threshold` for a metric).

Use it when: 
- you suspect **phase transitions / steep regions** and
want to concentrate samples there.

**Bayesian optimization (`bayesopt`)** 
- Starts with an initial LHS. 
- Fits a surrogate model for a chosen metric (target) and proposes points
that trade off: 
  - exploiting regions with high predicted value, 
  - exploring uncertain regions.

Use it when: 
- your primary goal is to **maximize/minimize** one metric
(not to map the whole surface).

Important caveat: 
- With noisy objectives (stochastic ABM metrics),
sequential methods can chase noise unless $R$ is large enough or you use
variance-reduction techniques (Section 4).

Where it’s used here: - Design types `adaptive_refine` and `bayesopt`
are implemented in `scripts/run_experiment_design.py` (these are not
generated by `core/experiment_design.generate_design_points`, because
they are sequential).

## 3) Mapping a Design Point to Actual ABM Parameters (Mixed Continuous + Discrete)

For non-grid designs (LHS/Sobol/Saltelli/sequential), the runner samples
a vector: 
$$
u \in [0,1)^d
$$ 
and maps it into concrete parameter values using the space definition
(see `core/experiment_design.py`).

### 3.1 Continuous parameters

Given bounds $(\ell, h)$:

- Linear transform: $$
  x = \ell + u\,(h - \ell)
  $$

- Log transform (log-uniform in $x$): $$
  x = \ell\,(h/\ell)^u
  \;=\;
  \exp\left(\log \ell + u\,(\log h - \log \ell)\right).
  $$

Notes: 
- “Uniform in $u$” means uniform in the *chosen transform space*.
If the natural scale is multiplicative (rates, sigmas), log transforms
are often more appropriate than linear spacing.

### 3.2 Discrete parameters (explicit value lists)

Given a value list `values = (v_0, ..., v_{k-1})`: 
- Convert $u$ to an
index $j = \lfloor u\,k \rfloor$ (clipped to $0..k-1$). 
- Use value
$v_j$ (optionally cast to `int`).

Important consequence: 
- If a discrete parameter has only a few allowed
values, then even with many $N$ you can only ever explore those few
values. 
- LHS/Sobol stratify **$u$**, not the discrete indices directly;
so discrete parameters can repeat heavily when $k \ll N$.

### 3.3 ABM-specific “scale” tips (choosing bounds/transforms)

These are not rules, but they prevent common sampling mistakes:

- **Rates/intensities** (arrivals per second, review clocks, etc.) often
  vary over orders of magnitude; prefer **log** bounds when you truly
  mean “uniform in log-space”.
- **Fractions/probabilities** in $[0,1]$ are usually fine with
  **linear** bounds (but avoid putting hard mass exactly at 0/1 unless
  you intend corner cases).
- **Counts / integer knobs** should be treated as discrete:
  - In grid sweeps, build an explicit integer list (the grid runner
    includes a small helper `linspace_int(...)`) and/or cast via its
    `INT_PARAMS`.
  - In experiment designs, define the parameter as `kind: discrete` and
    set `cast: int` if needed.
- **Arrival-rate semantics in this repo**:
  - `*_per_second` are Poisson intensities per **micro-step** (1 second)
    and override legacy `*_per_block` knobs when provided.
  - Expected arrivals per block scale with `block_time`:
    $\mathbb{E}[N_{\text{block}}] = \text{block\_time}\cdot\lambda_{\text{second}}$.
  - If you sweep arrival rates, pick one parameterization (per-second
    *or* per-block) to avoid confusing interpretation.

## 4) Replication / Seed Sampling (What “runs_per_point” and “common_seeds” Do)

At each design point $\theta_i$ you typically run $R$ replicates with
different seeds.

Two conceptual seed modes are used across runners:

### 4.1 Independent seeds per point (default)

Each point gets its own seed stream (so different points are
statistically independent, conditional on their parameters).

Use it when: 
- you want an unbiased “map of the surface” with
independent Monte Carlo noise across points.

### 4.2 Common seeds across points (paired comparisons / common random numbers)

All points reuse the same seed sequence for replicate $r$.

Why this can help: 
- When comparing two points $\theta$ and $\theta'$,
using the *same* underlying randomness often reduces the variance of the
**difference** in estimated metrics (a standard variance-reduction
trick).

Tradeoff: 
- It introduces correlation across points (which is fine for
paired comparisons, but changes how you should think about
independence).

## 5) Which Design Should You Use?

Quick heuristics (for a fixed budget $N \times R$):

- Want dashboards / clean slices in 1–3 dimensions: use a **grid** and
  increase resolution only where needed.
- Want broad screening in moderate/high $d$ (which knobs matter?): start
  with **LHS** (simple) or **Sobol** (more uniform).
- Want variance-based global sensitivity indices: use **Sobol–Saltelli**
  (and plan for its $N_{\text{base}}(d+2)$ cost).
- Want to find a maximizer/minimizer of a metric: use **Bayesian
  optimization**.
- Want to map regime boundaries/phase transitions: use **adaptive
  refinement** with a clear threshold metric.

In all cases: 
- If your metric is noisy, increasing **replicates per
point** ($R$) can matter more than increasing $N$. 
- If you change the
scenario YAML or the design spec, treat the result folder as a new
experiment record (do not overwrite).

## 6) Reproduction Recipes (Minimal)

Grid dashboard runner (full-factorial grid; cached CSV + meta under
`abm_results/grid_search/dashboard_nd/data/`):

``` bash
conda activate main
python -m scripts.run_parameter_surface_nd_pnl_fee_dashboard --config abm_results/scenarios/test.yml --dry-run
```

``` bash
python -m scripts.run_parameter_surface_nd_pnl_fee_dashboard \
  --config abm_results/scenarios/test.yml \
  --index-start 0 --index-stop 50 \
  --runs-per-point 10 \
  --max-workers 8
```

Experiment design runner (LHS/Sobol/Saltelli/adaptive/bayesopt; cached
CSV + meta under `abm_results/experiments_runs/<tag>/data/`):

``` bash
conda activate main

# LHS example
python -m scripts.run_experiment_design --experiment abm_results/experiments/example_lhs_screening.yml --dry-run
python -m scripts.run_experiment_design --experiment abm_results/experiments/example_lhs_screening.yml

# Sobol–Saltelli example (for Sobol indices)
python -m scripts.run_experiment_design --experiment abm_results/experiments/example_sobol_saltelli.yml
```

Build dashboards:

``` bash
python -m scripts.build_parameter_surface_nd_pnl_fee_dashboard --cache abm_results/grid_search/dashboard_nd/data/grid_<tag>.csv --meta abm_results/grid_search/dashboard_nd/data/meta_<tag>.json
python -m scripts.build_experiment_design_dashboard --cache abm_results/experiments_runs/<tag>/data/points_<tag>.csv --meta abm_results/experiments_runs/<tag>/data/meta_<tag>.json
```

## Appendix A) A Note on Grid Indexing (Only If You Slice Jobs by Index)

In the grid dashboard runner, points are enumerated by iterating the
Cartesian product in the parameter order as written in the sweep
definition (Python insertion order of the mapping).

Consequences: - The **rightmost** parameter changes **fastest** in the
enumeration. - `--index-start/--index-stop` slices refer to this
enumeration order, so reordering sweep keys changes which points fall
into a slice.
