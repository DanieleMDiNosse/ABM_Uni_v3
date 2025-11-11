#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abm_mempool_blockviz.py

Reads your existing `abm_mempool.py`, applies the block-time + visualization fixes,
and writes a fully-patched script `abm_mempool_blockviz_gen.py`.

What this applies:
  1) Inner loop = exact `range(block_size)` and **calls** `ref.diffuse_only()`.
  2) Records **start-of-block** arbitrage target band (lo/hi) and plots it (dotted)
     alongside your usual post-block band (grey), so you see what arb actually targets.
  3) Adds a **micro-time** price panel (DEX flat inside blocks; CEX diffuses each micro-step).
  4) Boundary order switched to **A(LPs) → B(mempool) → C(arb)** (backrun-style),
     and updates the micro series’ last point after boundary execution.

No changes to your Uniswap V3 math or fee accounting.
"""

import re
from pathlib import Path

SRC = Path("abm_mempool.py")
DST = Path("abm_mempool_blockviz_gen.py")

if not SRC.exists():
    raise SystemExit("ERROR: abm_mempool.py not found next to this script.")

text = SRC.read_text(encoding="utf-8", errors="replace")
orig = text
notes = []

def sub1(pat, repl, flags=0):
    global text
    new, n = re.subn(pat, repl, text, count=1, flags=flags)
    text = new
    return n

def subN(pat, repl, flags=0):
    global text
    new, n = re.subn(pat, repl, text, flags=flags)
    text = new
    return n

# ---------------------------------------------------------------------
# 0) Safety: ensure arb uses block-start CEX (if you already did this, no-op)
#    def arbitrage_to_target() -> ...  -> arbitrage_to_target(arb_ref_m: float)
#    and use arb_ref_m for lo/hi/target; pass arb_ref_m from schedule.
# ---------------------------------------------------------------------
sub1(r"def\s+arbitrage_to_target\s*\(\s*\)\s*->", "def arbitrage_to_target(arb_ref_m: float) ->")
text = re.sub(r"^\s*arb_ref_m\s*=\s*ref\.m\s*$",
              "# patched: arb_ref_m provided by caller (block-start CEX)", text, flags=re.MULTILINE)
text = re.sub(r"lo\s*,\s*hi\s*=\s*ref\.m\s*\*\s*r\s*,\s*ref\.m\s*/\s*r",
              "lo, hi = arb_ref_m * r, arb_ref_m / r", text)
text = re.sub(r"target\s*=\s*ref\.m", "target = arb_ref_m", text)
text = text.replace("= arbitrage_to_target()", "= arbitrage_to_target(arb_ref_m)")

# ---------------------------------------------------------------------
# 1) Recorders: add start-of-block band & micro-time series
# ---------------------------------------------------------------------
if "band_lo_target" not in text or "micro_steps" not in text:
    n = sub1(
        r"(delta_a_cex_series\s*=\s*\[\]\s*\n)",
        r"\1"
        r"    # --- Block-start target band (arb_ref_m) ---\n"
        r"    band_lo_target, band_hi_target = [], []\n"
        r"    # --- Micro-time traces (for block_size > 1 visualization) ---\n"
        r"    micro_steps, M_micro, P_micro = [], [], []\n"
    )
    notes.append(f"[recorders] inserted: {n}")

# ---------------------------------------------------------------------
# 2) Non-block branch: append band (uses current ref.m)
# ---------------------------------------------------------------------
n = sub1(
    r"(if\s+block_size\s*==\s*1\s*:\s*\n)(\s*)(_enable\(bucketA\))",
    r"\1\2# target band uses current ref.m in non-block mode\n"
    r"\2band_lo_target.append(ref.m * r)\n\2band_hi_target.append(ref.m / r)\n\2\3"
)
notes.append(f"[non-block] target-band append: {n}")

# ---------------------------------------------------------------------
# 3) Block branch: after arb_ref_m snapshot, append band & init micro arrays
# ---------------------------------------------------------------------
m = re.search(r"(else:\s*\n)(\s*)arb_ref_m\s*=\s*ref\.m[^\n]*\n", text)
if m and "micro_steps.extend([t * block_size + k" not in text:
    indent = m.group(2)
    inject = (
        f"{indent}# record target band for this block (start-of-block CEX)\n"
        f"{indent}band_lo_target.append(arb_ref_m * r)\n"
        f"{indent}band_hi_target.append(arb_ref_m / r)\n"
        f"{indent}# prepare micro-time arrays: keep DEX price stale within the block\n"
        f"{indent}_micro_start = len(P_micro)\n"
        f"{indent}P_micro.extend([pool.price] * block_size)\n"
        f"{indent}micro_steps.extend([t * block_size + k for k in range(block_size)])\n"
    )
    text = text[:m.end()] + inject + text[m.end():]
    notes.append("[block] band + micro arrays appended after arb_ref_m snapshot")

# ---------------------------------------------------------------------
# 4) Inner loop: exact block_size; call diffuse_only(); record M_micro
# ---------------------------------------------------------------------
# a) off-by-one → exact block_size
fixed1 = subN(
    r"for\s+_k\s+in\s+range\(\s*max\(0\s*,\s*block_size\s*-\s*1\)\s*\)\s*:",
    "for _k in range(block_size):"
)
# b) ensure parentheses on diffuse_only
fixed2 = subN(r"\bref\.diffuse_only\b(?!\()", "ref.diffuse_only()")
# c) add M_micro.append(ref.m) right after diffuse_only() in the inner loop (first occurrence only)
text, n_m = re.subn(
    r"(for _k in range\(block_size\):\s*\n(\s*)"
    r"maybe_enqueue_smart_router_intent\(ref\.m\)\s*\n\s*"
    r"maybe_enqueue_noise_trader_intent\(ref\.m\)\s*\n\s*"
    r"ref\.diffuse_only\(\)\s*)",
    r"\1\n\2M_micro.append(ref.m)\n",
    text, count=1
)
# safety: if some concatenation occurred, fix
text = text.replace("M_micro.append(ref.m)_enable(bucketA)", "M_micro.append(ref.m)\n            _enable(bucketA)")

notes += [
    f"[block] inner off-by-one fixed across occurrences: {fixed1}",
    f"[block] diffuse_only() call normalized: {fixed2}",
    f"[block] M_micro append injected (first loop): {n_m}",
]

# ---------------------------------------------------------------------
# 5) Boundary order: A → B(mempool) → C(arb); update last micro sample
# ---------------------------------------------------------------------
pattern_with_Lpre = re.compile(
    r"_enable\(bucketA\)\s*\n\s*act_LPs\(\)\s*\n\s*\n"
    r"\s*_enable\(bucketC\)\s*\n\s*act_arbitrageur\(\)\s*\n\s*act_LPs\(\)\s*\n\s*\n"
    r"\s*_enable\(bucketB\)\s*\n\s*L_pre_trader_this[^\n]*\n\s*execute_mempool_orders\(\)\s*\n\s*act_LPs\(\)"
)
repl_common = (
    "_enable(bucketA)\n            act_LPs()\n\n"
    "            _enable(bucketB)\n            L_pre_trader_this = pool.L_active  # active L before trader fill\n"
    "            execute_mempool_orders()\n            act_LPs()\n\n"
    "            _enable(bucketC)\n            act_arbitrageur()\n            act_LPs()\n"
    "            # update last micro-sample of DEX price to reflect boundary execution\n"
    "            if block_size > 1:\n"
    "                P_micro[_micro_start + block_size - 1] = pool.price"
)
text, n_with = pattern_with_Lpre.subn(repl_common, text, count=1)

if not n_with:
    pattern_no_Lpre = re.compile(
        r"_enable\(bucketA\)\s*\n\s*act_LPs\(\)\s*\n\s*\n"
        r"\s*_enable\(bucketC\)\s*\n\s*act_arbitrageur\(\)\s*\n\s*act_LPs\(\)\s*\n\s*\n"
        r"\s*_enable\(bucketB\)\s*\n\s*execute_mempool_orders\(\)\s*\n\s*act_LPs\(\)"
    )
    text, n_no = pattern_no_Lpre.subn(
        "_enable(bucketA)\n            act_LPs()\n\n"
        "            _enable(bucketB)\n            execute_mempool_orders()\n            act_LPs()\n\n"
        "            _enable(bucketC)\n            act_arbitrageur()\n            act_LPs()\n"
        "            # update last micro-sample of DEX price to reflect boundary execution\n"
        "            if block_size > 1:\n"
        "                P_micro[_micro_start + block_size - 1] = pool.price",
        text, count=1
    )
    notes.append(f"[boundary] swapped (no L_pre): {n_no}")
else:
    notes.append(f"[boundary] swapped (with L_pre): {n_with}")

# ---------------------------------------------------------------------
# 6) Price panel: plot start-of-block band (dotted)
# ---------------------------------------------------------------------
if "Target band (start-of-block)" not in text:
    m = re.search(r"(# ----- 1\) Price panel -----[\s\S]+?)(ax\.legend\([^\)]*\))", text)
    if m:
        pre, leg = m.group(1), m.group(2)
        inject = (
            "        # Show target band (start-of-block) as dotted lines so you can see what arb targets\n"
            "        band_lo_target_v = np.array(band_lo_target)[s0:]\n"
            "        band_hi_target_v = np.array(band_hi_target)[s0:]\n"
            "        ax.plot(steps_v, band_lo_target_v, linestyle='--', linewidth=1.0, alpha=0.6,\n"
            "                label='Target band (start-of-block) lo')\n"
            "        ax.plot(steps_v, band_hi_target_v, linestyle='--', linewidth=1.0, alpha=0.6,\n"
            "                label='Target band (start-of-block) hi')\n"
        )
        text = text.replace(pre + leg, pre + inject + leg, 1)
        notes.append("[plot] start-of-block band added to price panel")

# ---------------------------------------------------------------------
# 7) Micro-time figure (1b) before Notionals
# ---------------------------------------------------------------------
if "_save_fig(fig1b, \"1b_price_micro\")" not in text:
    idx = text.find("# ----- 2) Notionals -----")
    if idx != -1:
        panel = (
            "        # ----- 1b) Micro-time price panel (only meaningful if block_size>1) -----\n"
            "        if block_size > 1 and len(M_micro) == len(P_micro) == len(micro_steps) and len(micro_steps) > 0:\n"
            "            fig1b, ax = plt.subplots(figsize=(15, 3.2))\n"
            "            ax.plot(micro_steps, P_micro, label=\"DEX price (micro)\", linewidth=1.2)\n"
            "            ax.plot(micro_steps, M_micro, \"--\", label=\"CEX price (micro)\", linewidth=1.0)\n"
            "            ax.set_title(\"Micro-time CEX vs DEX (within blocks)\", fontsize=TITLE_FONT_SIZE-2)\n"
            "            ax.set_xlabel(\"Micro step\", fontsize=LABEL_FONT_SIZE-1)\n"
            "            ax.set_ylabel(\"Price\", fontsize=LABEL_FONT_SIZE-1)\n"
            "            ax.grid(True, alpha=0.3)\n"
            "            ax.legend(fontsize=LEGEND_FONT_SIZE-1)\n"
            "            _save_fig(fig1b, \"1b_price_micro\")\n"
        )
        text = text[:idx] + panel + text[idx:]
        notes.append("[plot] micro-time panel inserted")

# ---------------------------------------------------------------------
# 8) Write the full patched file
# ---------------------------------------------------------------------
DST.write_text(text, encoding="utf-8")

print(f"✅ Wrote: {DST.name}")
print("Patch summary:")
for n in notes:
    print(" -", n)
print("\nNext:")
print("  python abm_mempool_blockviz.py")
print("  python abm_mempool_blockviz_gen.py   # run your sim with the new visuals/logic")
