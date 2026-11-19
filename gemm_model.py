"""Cycle-accurate + area model of an HLS-synthesized GEMM kernel.

The kernel computes C[N][N] = A[N][K] * B[K][N] with two HLS directives:

  UF   : unroll factor of the inner product loop  (parallel MACs)
  PF   : factor by which the A/B arrays are partitioned into banks
         (partition PF K-words of A-row and B-col in parallel)

These mirror real HLS pragmas:
  #pragma HLS UNROLL factor=UF
  #pragma HLS ARRAY_PARTITION variable=A factor=PF

Scheduling model (single-issue loop pipeline):
  - Each output element C[i][j] needs K multiply-accumulates.
  - With UF MACs in parallel, each C[i][j] takes ceil(K/UF) * II cycles,
    where II = pipeline initiation interval (1 if pipelined, else the
    combinational latency of one MAC iteration).
  - The loop nests add a small overhead (loop trip counts) we model as a
    constant factor + the fill/drain of the pipeline.

This is the *cost model* the DSE agent evaluates. It is intentionally
simplified (no loop-carry dependency analysis, fixed II) but captures the
real tradeoff: more parallelism (UF, PF) -> fewer cycles but more area.

Latency (cycles):
  T = ceil(N / TT) * ceil(N / TT) * (ceil(K / UF) * II + PI_OVH)
      where TT is the tile size (assume 1 for a flat GEMM here)
    = N * N * (ceil(K / UF) * II + PIPELINE_DEPTH)

Area (normalized units):
  AREA = MAC_UF * UF + BANK_BITS * PF + FLOP_REG * N * PF
  MAC_UF  = 1 unit per MAC
  BANK_BITS = 1 unit per partitioned word
  FLOP_REG = small term for the extra write ports of partitioned arrays
"""


# ---------------------------------------------------------------- model
MAC_UNIT_AREA   = 1.0
BANK_UNIT_AREA  = 0.25
PORT_UNIT_AREA  = 0.5
PIPELINE_DEPTH  = 4   # pipeline fill/drain cycles

CLOCK_NS        = 5.0  # 200 MHz


def gemm_cycles(N: int, K: int, UF: int, PF: int, pipelined: bool = True) -> int:
    """Cycle count for the NxNxK GEMM under the given directives.

    Real HLS behavior: with array partitioning, the K-loop reads are spread
    across PF banks, so each MAC can fetch a new K element every cycle as
    long as UF <= PF (enough banks). If UF > PF, bank conflicts serialize the
    reads -> the effective unroll is capped by PF for memory throughput.
    """
    II = 1 if pipelined else (PIPELINE_DEPTH + 1)  # non-pipelined: long II
    # memory-limited unroll: can't feed more MACs than banks
    eff_uf = min(UF, PF)
    per_out = (K + eff_uf - 1) // eff_uf          # ceil(K / eff_uf) iterations
    return N * N * (per_out * II + PIPELINE_DEPTH)


def gemm_area(N: int, K: int, UF: int, PF: int) -> float:
    """Normalized area: MACs + partitioned memory banks + write ports."""
    macs    = MAC_UNIT_AREA * UF
    banks   = BANK_UNIT_AREA * min(PF, K) * (N + N)   # A + B arrays
    ports   = PORT_UNIT_AREA * (2 * min(PF, K) // max(1, min(PF, K))) * N
    # simpler port term: each partitioned word adds a read port on both arrays
    ports   = PORT_UNIT_AREA * 2 * min(PF, K) * N
    return macs + banks + ports


def gemm_power(N: int, K: int, UF: int, PF: int) -> float:
    """Normalized dynamic power ~ area * switching * cycles."""
    return gemm_area(N, K, UF, PF) * 0.5 * (N * N * K)


def feasible(N: int, K: int, UF: int, PF: int, max_area: float) -> bool:
    return gemm_area(N, K, UF, PF) <= max_area and UF >= 1 and PF >= 1


# ---------------------------------------------------------------- search space
def directive_space(N: int, K: int, max_uf: int = 16, max_pf: int = 8):
    """Grid of (UF, PF) directives to explore."""
    return [(uf, pf)
            for uf in (1, 2, 4, 8, 16) if uf <= max_uf
            for pf in (1, 2, 4, 8) if pf <= max_pf]


def evaluate(space, N, K, pipelined=True):
    """Return list of (UF, PF, cycles, area, power)."""
    out = []
    for uf, pf in space:
        cyc = gemm_cycles(N, K, uf, pf, pipelined)
        ar  = gemm_area(N, K, uf, pf)
        pw  = gemm_power(N, K, uf, pf)
        out.append((uf, pf, cyc, ar, pw))
    return out
