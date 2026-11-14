#!/usr/bin/env bash
# Recompute every analysis output in verify_out/ from the raw per-run files in results/runs/ (CPU only, no GPU,
# no checkpoints needed). Run from recfreq/ after installing the pinned environment:
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#   bash scripts/reproduce_analysis.sh            # writes verify_out/ (overwrites it)
#   .venv/bin/python scripts/compare_outputs.py <reference verify_out> verify_out
# To check against the committed numbers without touching them, run this in a copy of recfreq/ with verify_out/ emptied.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
P=${PROCS:-$(getconf _NPROCESSORS_ONLN)}
mkdir -p verify_out figs_region figs_validity/slides
run() { echo "== $*"; "$PY" -m "$@"; }

run recfreq.verify.prop2_exact --out verify_out                      # Q1: corrected formula vs exact Bayes
run recfreq.verify.law_scope --out verify_out --procs "$P"           # per-model Bayes fits (cached in law_scope_fits.csv)
run recfreq.verify.region_map --out verify_out --figs figs_region --procs "$P"   # Q2: law error / own-fit error
run recfreq.verify.clock_summary --out verify_out                    # Q3: the clock (survival probes)
run recfreq.verify.evidence_weight --out verify_out                  # Q3: evidence weight w
run recfreq.verify.switch_points --out verify_out
run recfreq.verify.clock_fixed --out verify_out
run recfreq.verify.natural_law --exp E6 --out verify_out
run recfreq.verify.surgery_score --match E6_ | tee verify_out/surgery_score.txt     # slope surgery (H-tie)
run recfreq.verify.dose | tee verify_out/dose.txt                    # extra conflict data
run recfreq.verify.long_train | tee verify_out/long_train.txt        # E10
run recfreq.verify.mass_score | tee verify_out/mass_score.txt        # round 8 (H-mass)
run recfreq.verify.validity baseline --out verify_out --procs "$P"   # run 8: band validation
run recfreq.verify.validity score --out verify_out --procs "$P" | tee verify_out/validity_score.txt
run recfreq.verify.registry --out verify_out                         # 31 headline numbers of the first report
echo "done"
