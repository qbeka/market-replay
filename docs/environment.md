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

## Hash stability

The synthetic feed comes from a seeded generator and is checked by SHA-256. Hashes of pickled models and floating point results can change across chips, operating systems or library versions, so a different machine may need its own `fit`, `tune` and `freeze`.
