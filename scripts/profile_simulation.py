#!/usr/bin/env python3
"""
Profile a single simulation run to identify performance bottlenecks.
Uses cProfile + pstats for detailed function-level timing.
"""
import cProfile
import pstats
import io
import inspect
from pathlib import Path
import yaml

# Silence tqdm for cleaner profiling output
from scripts import run as run_module
def _silent_tqdm(iterable=None, **kwargs):
    if iterable is None:
        class DummyTqdm:
            def update(self, n=1): pass
            def close(self): pass
        return DummyTqdm()
    return iterable
run_module.tqdm = _silent_tqdm

simulate = run_module.simulate

def load_params_from_yaml(config_path: Path):
    """Load simulation parameters from YAML, avoiding relative import issues."""
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)
    
    params = dict(config_data.get("simulate", {}))
    scenario_fee_mode = config_data.get("fee_mode")
    if scenario_fee_mode is not None:
        params["fee_mode"] = scenario_fee_mode
    
    # Fill in defaults from simulate signature
    signature = inspect.signature(simulate)
    for name, param in signature.parameters.items():
        if name not in params and param.default is not inspect._empty:
            params[name] = param.default
    
    return params

def run_profiled_simulation():
    """Run a short simulation matching grid search settings."""
    # Load base config
    config_path = Path("abm_results/scenarios/test.yml")
    params = load_params_from_yaml(config_path)
    
    # Override for profiling: shorter run, light_mode like grid search
    params["T"] = 2000  # Shorter for faster profiling
    params["light_mode"] = True
    params["visualize"] = False
    params["verbose"] = False
    params["skip_step"] = 100
    
    # Run simulation
    result = run_module.simulate(**params)
    return result

def main():
    print("=" * 70)
    print("PROFILING ABM SIMULATION (T=2000, light_mode=True)")
    print("=" * 70)
    
    # Profile the simulation
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = run_profiled_simulation()
    
    profiler.disable()
    
    # Analyze results
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    
    # Sort by cumulative time and print top 50 functions
    print("\n" + "=" * 70)
    print("TOP 50 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 70)
    stats.sort_stats('cumulative')
    stats.print_stats(50)
    print(stream.getvalue())
    
    # Reset stream for next analysis
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    
    # Sort by total time (time spent in function itself, not callees)
    print("\n" + "=" * 70)
    print("TOP 50 FUNCTIONS BY TOTAL TIME (self time)")
    print("=" * 70)
    stats.sort_stats('tottime')
    stats.print_stats(50)
    print(stream.getvalue())
    
    # Reset stream for caller analysis
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    
    # Show callers for key functions we suspect are bottlenecks
    print("\n" + "=" * 70)
    print("CALLER ANALYSIS FOR KEY FUNCTIONS")
    print("=" * 70)
    
    # Look for specific bottleneck candidates
    key_patterns = [
        '_ensure',           # BoundaryIndex rebuild
        'active_liquidity',  # Prefix sum lookups
        'recompute_active',  # Full L_active recompute
        'swap_x_to_y',       # Swap functions
        'swap_y_to_x',
        'quote_x_to_y',      # Quote functions
        'quote_y_to_x',
        '_rebalance',        # LP rebalancing
        'allocate_fees',     # Fee allocation
        'lp_token0_exposure', # LP exposure calc
        'lp_wealth_y',       # LP wealth calc
        'current_amounts',   # Position amounts
        'poisson',           # RNG
        'normal',            # RNG
        'lognormal',         # RNG
    ]
    
    stats.sort_stats('cumulative')
    for pattern in key_patterns:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(pattern, 5)
        output = stream.getvalue()
        if "function calls" in output and "0 function calls" not in output:
            print(f"\n--- {pattern} ---")
            print(output)
    
    # Save full profile for later analysis
    profiler.dump_stats("profile_results.prof")
    print("\n" + "=" * 70)
    print("Full profile saved to: profile_results.prof")
    print("View with: python -m pstats profile_results.prof")
    print("Or: snakeviz profile_results.prof")
    print("=" * 70)

if __name__ == "__main__":
    main()
