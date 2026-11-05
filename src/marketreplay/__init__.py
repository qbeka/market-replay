"""Event replay engine and learned execution policy. Every order is simulated."""

import os

# Must run before scikit-learn loads. Each worker process already gets its own core, and a
# per-row OpenMP model call on top of that spins idle threads: one day took 113 s instead of 3 s.
os.environ.setdefault("OMP_NUM_THREADS", "1")
