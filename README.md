# Agentic HLS Design-Space Explorer

[![CI](https://github.com/Ana2601/agentic-hls-dse/actions/workflows/ci.yml/badge.svg)](https://github.com/Ana2601/agentic-hls-dse/actions/workflows/ci.yml)

An AI-for-EDA project: an autonomous agent that explores HLS synthesis
directives for a GEMM kernel and recovers the Pareto-optimal front of
**latency vs. area**. Self-contained, pure Python, runs anywhere — including
Apple Silicon — so it is a portable, always-runnable demonstration of agentic
hardware design.

## What the agent does

1. **Proposes** `(UF, PF)` directive pairs — unroll factor and array
   partition factor (the two pragmas that most move GEMM QoR).
2. **Evaluates** each proposal with a cycle-accurate + area model of a
   pipelined GEMM datapath (memory-bank-limited throughput, pipeline
   fill/drain, MAC + bank + port area).
3. **Learns** via Thompson sampling over the discrete directive space.
4. **Critiques** each proposal (feasibility, dominance) like an HLS expert.
5. **Keeps the Pareto front** and reports best-latency / best-area points.

Optional **LLM seeding**: when `NVIDIA_API_KEY` is set, an NVIDIA NIM LLM
(reused from [agentic-rtl-design](https://github.com/Ana2601/agentic-rtl-design)) proposes a warm-start directive; the agent searches
around it. Without a key the agent runs fully self-contained.

## Files

```
gemm_model.py    cycle/area/power model of the HLS-scheduled GEMM
dse.py           the agent: Thompson-sampling search + Pareto front
test_dse.py      self-test: agent front == brute-force oracle front
llm.py           NVIDIA NIM client (reused from 10)
prompts.py       HLS seeding prompts
```

## Run

```
make test       # PASS means the agent recovered the true Pareto front
make demo       # 300-eval search, prints the Pareto front + best points
make oracle     # brute-force ground truth for comparison
python3 dse.py --propose-llm   # LLM-seeded search (needs NVIDIA_API_KEY)
```

## Result (N=K=8)

```
   UF   PF   cycles    area     power
    8    8      320   104.0    26,624   <-- fastest
    4    4      384    52.0    13,312
    2    2      512    26.0     6,656
    1    1      768    13.0     3,328   <-- smallest
```

The agent finds the same front as the brute-force oracle — a tractable
stand-in for the full `vitis_hls` flow, which this pattern drives remotely on
an FPGA host.

## Setup

- Python packages: `pip install -r requirements.txt`

```bash
python3 test_dse.py
```

The same commands run in CI on every push (`.github/workflows/ci.yml`).
