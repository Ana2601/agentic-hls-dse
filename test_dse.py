#!/usr/bin/env python3
"""Verifies the DSE agent recovers the true Pareto front (oracle check)."""
import sys

from dse import DSEAgent, true_front

N, K = 8, 8

truth = {(p[0], p[1]) for p in true_front(N, K)}

agent = DSEAgent(N, K, seed=42)
agent.search(300)
found = {(p[0], p[1]) for p in agent.front}

missing = truth - found
extra = found - truth

print(f"truth front : {sorted(truth)}")
print(f"agent front : {sorted(found)}")
print(f"missing     : {sorted(missing)}")
print(f"extra       : {sorted(extra)}")

ok = not missing and not extra
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
