# Environment

Every result in `results/` was produced here. `results/final.json` also records the machine, commit and seeds.

| Item | Value |
|---|---|
| Machine | Apple M5 Pro, 18 cores (6 Super, 12 Performance), 48 GB RAM |
| Operating system | macOS 26.6.2 (25G83), Darwin 25.6.0 |
| Python | 3.12.7, pinned in `.python-version` |
| Package manager | uv 0.11.24, locked in `uv.lock` |
| C compiler | Apple clang 21.0.0 |
| Direct packages | numpy 2.2.6, scikit-learn 1.6.1 |
| Dev and plot packages | pytest 8.4.1, pytest-cov 6.2.1, matplotlib 3.10.3 |
| Threads | `OMP_NUM_THREADS=1`, set in `src/marketreplay/__init__.py` |
| Workers | 8 processes for day-parallel stages |

## What shared the machine

Other projects ran on this Mac during the study, with a load average near 40 during profiling. That affects time, not results, because every stage is seeded and single-threaded per worker. `results/profile.json` records the load around each measurement.

`elapsed_s` in `results/final.json` is 37,835 s. That covers a run with the thread problem in [experiments.md](experiments.md), entry 7, and two periods when it was paused for other work. Do not read it as the cost of the final test. After the fix, 19 stress scenarios on the same 34 days took 641 s.

## Reproduction

Checked on 2026-11-05 from a fresh clone on this Mac:

| Check | Result |
|---|---|
| Rebuilt feed, 11,648,571 events | Same input hash |
| `results/raw/validation.csv` after refit and tune | Byte identical |
| `results/raw/test.csv`, final test with committed models | Byte identical, in 40 s |
| Refit models against committed models | Identical predictions, max difference 0.0 on 5,000 random rows |
| `results/fitted/fitted.pkl` after refit | 23 bytes different, so the freeze guard refuses it |
| `report`, `demo` and `mreplay manifest --check` | All files match |
| Tests | 140 passing |

The feed is checked by SHA-256. Pickled models and floating point results can change across chips, operating systems or library versions, so a different machine may need its own `fit`, `tune` and `freeze`.
