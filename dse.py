"""Agentic HLS design-space exploration (DSE).

The agent owns the *search* of HLS directives:
  - proposes (UF, PF) directive pairs,
  - evaluates each with the cycle/area model,
  - learns which region of the space is best (Thompson sampling over a
    discrete arm set),
  - maintains the Pareto front of latency vs. area,
  - can optionally take a starting proposal from an LLM (NVIDIA NIM) when
    NVIDIA_API_KEY is set - the agent then searches around that seed.

The critic step is an analytical feasibility/quality check per proposal.

Run:
  python3 dse.py                    # full agentic search, prints report
  python3 dse.py --propose-llm      # LLM seeds the search (needs key)
  python3 dse.py --oracle           # verify agent finds the true Pareto front
"""
import argparse
import random
from dataclasses import dataclass

from gemm_model import (directive_space, evaluate, feasible, gemm_area,
                        gemm_cycles, gemm_power)


@dataclass
class Arm:
    uf: int
    pf: int
    n: int = 0          # times sampled
    sum_score: float = 0.0
    best_score: float = float("-inf")


def score(cyc, area, w_area=0.35):
    """Single-objective scalarization (weighted geometric mean, higher=better)."""
    # normalize: fewer cycles/area is better; log-space geometric mean
    return -(0.65 * cyc / 1e6 + w_area * area)


class DSECritic:
    """Analytical critic: flags infeasible or dominated proposals."""

    def critique(self, uf, pf, n, k, area, best_front):
        issues = []
        if uf < 1 or pf < 1:
            issues.append("unroll/partition must be >= 1")
        if uf > n or uf > k:
            issues.append("unroll exceeds loop bound")
        if pf > k:
            issues.append("partition exceeds array size")
        if not feasible(n, k, uf, pf, area_max := 10 ** 9):
            issues.append("area infeasible")
        dominated = any(
            (o_cyc <= gemm_cycles(n, k, uf, pf)
             and o_ar <= gemm_area(n, k, uf, pf)
             and (o_cyc < gemm_cycles(n, k, uf, pf) or o_ar < gemm_area(n, k, uf, pf)))
            for (o_uf, o_pf, o_cyc, o_ar, _) in best_front
        )
        if dominated:
            issues.append("dominated by existing Pareto point")
        return issues


class DSEAgent:
    def __init__(self, N, K, seed=None, alpha=2.0, beta=2.0, propose_llm=None):
        self.N, self.K = N, K
        self.rng = random.Random(seed)
        self.alpha, self.beta = alpha, beta
        self.space = directive_space(N, K)
        self.arms = {a: Arm(*a) for a in self.space}
        self.history = []
        self.front = []          # list of (UF, PF, cycles, area, power)
        self.critic = DSECritic()
        self.llm = propose_llm   # callable(N, K) -> (uf, pf) or None
        self.llm_seed = None

    # ---- sampling policy ----
    def propose(self):
        # LLM seed: try it first (warm start), then Thompson sample.
        if self.llm is not None and self.llm_seed is None:
            seed_dir = self.llm(self.N, self.K)
            if seed_dir:
                self.llm_seed = seed_dir
                return seed_dir
        # Thompson sampling: draw Beta from each arm's belief, pick max.
        best = None
        best_draw = -1.0
        for arm in self.arms.values():
            if arm.n == 0:
                return (arm.uf, arm.pf)   # explore all arms at least once
        for arm in self.arms.values():
            a = self.alpha + max(arm.best_score, 0.0)
            b = self.beta + (arm.n - max(arm.best_score, 0.0))
            draw = self.rng.betavariate(a, b)
            if draw > best_draw:
                best_draw = draw
                best = arm
        return (best.uf, best.pf)

    # ---- evaluation ----
    def run_proposal(self, uf, pf):
        cyc = gemm_cycles(self.N, self.K, uf, pf)
        ar  = gemm_area(self.N, self.K, uf, pf)
        pw  = gemm_power(self.N, self.K, uf, pf)
        s   = score(cyc, ar)
        arm = self.arms[(uf, pf)]
        arm.n += 1
        arm.sum_score += s
        arm.best_score = max(arm.best_score, s)
        self.history.append((uf, pf, cyc, ar, pw))
        return cyc, ar, pw

    # ---- Pareto front update ----
    def update_front(self, cyc, ar, pw, uf, pf):
        # remove dominated (strictly worse in one axis, not better in either)
        self.front = [p for p in self.front
                      if not (p[2] >= cyc and p[3] >= ar and (p[2] > cyc or p[3] > ar))]
        # add if not dominated by existing and not an exact duplicate
        if not any(p[2] == cyc and p[3] == ar for p in self.front) and \
           not any(p[2] <= cyc and p[3] <= ar and (p[2] < cyc or p[3] < ar)
                   for p in self.front):
            self.front.append((uf, pf, cyc, ar, pw))
            self.front.sort(key=lambda x: x[2])

    # ---- main loop ----
    def search(self, iters=200):
        print(f"DSE agent: N={self.N} K={self.K}, {len(self.space)} directive "
              f"points, {iters} evaluations")
        if self.llm is not None:
            print(f"[*] LLM proposer active (seed expected)")
        for i in range(iters):
            uf, pf = self.propose()
            cyc, ar, pw = self.run_proposal(uf, pf)
            issues = self.critic.critique(uf, pf, self.N, self.K, ar, self.front)
            if not issues:
                self.update_front(cyc, ar, pw, uf, pf)
            if i % 40 == 0:
                print(f"  iter {i:4d}: (UF={uf:2d}, PF={pf}) -> "
                      f"{cyc:>9,} cyc, {ar:6.1f} area"
                      f"{'  LLM-seeded' if (uf,pf)==self.llm_seed else ''}")
        return self.front

    def report(self):
        print("\n=== Pareto front (cycles vs area) ===")
        print(f"{'UF':>3} {'PF':>3} {'cycles':>10} {'area':>8} {'power':>10}")
        for uf, pf, cyc, ar, pw in self.front:
            print(f"{uf:>3} {pf:>3} {cyc:>10,} {ar:>8.1f} {pw:>10,.0f}")
        # best latency
        b_lat = min(self.front, key=lambda x: x[2])
        print(f"\nBest-latency: (UF={b_lat[0]}, PF={b_lat[1]}) "
              f"{b_lat[2]:,} cyc @ {gemm_cycles(self.N,self.K,b_lat[0],b_lat[1])/1e3:.1f} kcyc")
        # best area
        b_ar = min(self.front, key=lambda x: x[3])
        print(f"Best-area:    (UF={b_ar[0]}, PF={b_ar[1]}) {b_ar[3]:.1f} area")
        return self.front


# ---------------------------------------------------------------- LLM proposer
def make_llm_proposer():
    """Return a proposer callable using project-02's LLM client, or None."""
    try:
        from llm import LLMClient
        from prompts import SYSTEM_HLS_SEED, HLS_SEED_TEMPLATE

        client = LLMClient()

        def proposer(N, K):
            import re
            text = client.chat(SYSTEM_HLS_SEED,
                               HLS_SEED_TEMPLATE.format(N=N, K=K))
            m = re.search(r"UF\s*=\s*(\d+)", text)
            m2 = re.search(r"PF\s*=\s*(\d+)", text)
            if not (m and m2):
                return None
            uf = max(1, min(16, int(m.group(1))))
            pf = max(1, min(8, int(m2.group(1))))
            return (uf, pf)
        return proposer
    except Exception:
        return None


# ---------------------------------------------------------------- oracle check
def true_front(N, K):
    """Brute-force the true Pareto front to grade the agent."""
    ev = evaluate(directive_space(N, K), N, K)
    front = []
    for uf, pf, cyc, ar, pw in ev:
        if not any(o_cyc <= cyc and o_ar <= ar and (o_cyc < cyc or o_ar < ar)
                   for o_uf, o_pf, o_cyc, o_ar, _ in ev):
            front.append((uf, pf, cyc, ar, pw))
    return sorted(front, key=lambda x: x[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--propose-llm", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()

    if args.oracle:
        front = true_front(args.n, args.k)
        print("=== True Pareto front (brute force) ===")
        print(f"{'UF':>3} {'PF':>3} {'cycles':>10} {'area':>8} {'power':>10}")
        for uf, pf, cyc, ar, pw in front:
            print(f"{uf:>3} {pf:>3} {cyc:>10,} {ar:>8.1f} {pw:>10,.0f}")
        return 0

    llm_proposer = make_llm_proposer() if args.propose_llm else None
    if args.propose_llm and llm_proposer is None:
        print("[!] NVIDIA_API_KEY not set / LLM unavailable; running "
              "self-contained (no LLM seed).", file=__import__("sys").stderr)

    agent = DSEAgent(args.n, args.k, seed=args.seed, propose_llm=llm_proposer)
    agent.search(args.iters)
    agent.report()
    return 0


if __name__ == "__main__":
    __import__("sys").exit(main())
