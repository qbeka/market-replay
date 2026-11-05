# Experiment log

Failed experiments and the changes they caused, in order. All numbers come from training or validation days. Test days stayed locked throughout.

## 1. The first task had no decision in it

- **Setup:** 40 lots in 300 s, about 1% of volume.
- **Result:** `passive_only` filled every lot as a maker at -1.04 bps with no forced lots. `urgency` matched it exactly, because it never fell behind.
- **Cause:** at that size a resting order always finishes, so nothing needs deciding.
- **Change:** a sweep on 12 validation days showed `passive_only` moving from best to worst as size grew and the horizon shrank. It reached +5.30 bps at 200 lots in 60 s. The study now uses 120 lots in 60 s, between those extremes.

## 2. One-second snapshots hid the latency being studied

- **Found in:** planning the latency stress test, before any run.
- **Cause:** with snapshots on whole seconds and decisions on whole seconds, any latency from 1 ms to 999 ms showed the same stale snapshot.
- **Change:** the snapshot grid is offset by the observation latency, so a decision sees a snapshot exactly that old. A test enforces this.

## 3. The learned policy threw away its own queue

- **Setup:** first validation run of the learned policy, on 24 days and 642 parents.
- **Result:** 1.67 bps at best, against 0.03 bps for the urgency rule, and only 24.6% maker lots against 79.2%. The fill model was fine: Brier 0.144 against 0.192 for the base rate.
- **Cause:** a decision count on 4 validation days showed `wait` chosen on 1,215 of 2,731 model decisions. The rule was `wait = move + min(cross, rest)`, so `wait` beat `rest` whenever the predicted move was below zero. That prediction is noise around zero (10th to 90th percentile -0.53 to +0.60 ticks). A coin toss was cancelling resting orders.
- **Change:** waiting now pays `wait_margin_ticks`, and a parent behind schedule may not wait. The tuning grid includes a setting that never waits.
- **Lesson:** a good model does not rescue a bad decision rule. The rule charged nothing for leaving a queue, and the fill model cannot see that cost.

## 4. Waiting turned out to have no value

- **Setup:** the validation grid after experiment 3 crossed behind limits, catch-up costs, final windows and three wait margins, one of them meaning never wait.
- **Result:** the five cheapest settings by mean cost all used margins that rarely or never wait. The chosen setting never waits. Compared with urgency it had a similar mean (-0.014 against 0.030 bps), a better p95 (4.78 against 5.52 bps) and less forced completion (3.2% against 9.4%).
- **Meaning:** on this feed, the three-way question reduces to two: rest or cross. That is one finding about one synthetic market, not a general rule.
- **Stopped here:** no more policy changes after this grid. More tuning on the same 24 days would fit noise.

## 5. The reinforcement learning comparator could not see the deadline

- **Setup:** first validation run of `fitted_q`, at 2.18 bps with 23.6% forced lots.
- **Cause:** fitted Q-iteration ran 8 iterations with no discounting. Each one carries cost back one decision, and the horizon is 30 decisions, so most states never learned the deadline penalty. The behavior policy also crossed too often to reach deadlines.
- **Change:** 30 iterations, and a lower range of behavior cross rates. Reporting the first version as "reinforcement learning lost" would have been unfair.
- **After the fix:** p95 fell from 14.02 to 11.17 bps, but forced lots stayed at 20.9% and the mean stayed at 2.18 bps. It remains worse than the simple rules on validation days. It trained on 27,765 transitions with no tuning of its own, so treat this as one modest comparator, not a verdict on reinforcement learning.

## 6. A review agent found three biases before the freeze

A separate AI agent, with no access to the author's reasoning, got the acceptance criteria and a commit hash, then tried to break the engine. No human reviewed this code. Each finding was reproduced as a failing test before its fix, in `tests/test_review_findings.py`. These numbers are the reviewer's, from 30 small non-test days.

- **Hidden prints burned visible queue.** A hidden trade removes no visible lots, but it reduced `queue_ahead`. Passive cost was understated by 0.052 bps and urgency by 0.027 bps, a bias that favored resting. Fixed in both engines.
- **Own order state leaked ahead of the observation clock.** `rest_price` read the live order, so a strategy could learn of a full fill early. It was harmless at base latency but changed 15 of 40 parents at 904 ms. That would have biased the high-latency stress tests. The causality test now compares every field of the view, which would have caught this.
- **The arrival midpoint in the view came from the live book.** Only `fitted_q` read it. The ledger keeps the true midpoint, and strategies now get the delayed one.

Off the study path, the review also fixed the deadline price on an empty far side, priority rules missing from the compiled replay, and a snapshot grid that drifted when a feed started off-interval. Lots per share was removed from the LOBSTER adapter, because flooring odd lots to zero broke the book.

What held up: the book against a brute-force FIFO over 183,621 steps, conservation and no double fills over 1,500 fuzz seeds, ledger isolation, shortfall signs, and compiled against Python under ASan and UBSan.

## 7. A thread setting that did nothing cost 36 times the runtime

- **Setup:** the final test used 19 CPU hours on a day-parallel run that should have taken minutes.
- **Result:** one test day took 113 s of wall time and 841 s of CPU. With `OMP_NUM_THREADS=1` it took 3.1 s, 36 times faster.
- **Cause:** each worker process carried 19 threads. Every model call opened an OpenMP region for a single row, and the idle threads spun. `threadpool_limits(1)` looked like it handled this, but it did not stop them. CPU per worker read as 1.2 cores, so `top` showed nothing alarming.
- **Change:** the package sets `OMP_NUM_THREADS=1` before scikit-learn loads, and the `threadpool_limits` call is gone.
- **Check:** the stress run's base scenario, run after the change, matched the final test on all 5,508 shared rows, so the fix changed speed and nothing else.
- **Lesson:** the worker count looked parallel and the CPU looked busy. Only timing a single day alone showed the waste.

## 8. The first speedup figure was mostly the machine's load

- **Setup:** the first profile, five runs each with the load average near 40, reported a book speedup of 23 times from medians.
- **Result:** Python's five runs ranged from 0.074 s to 0.498 s while the compiled replay stayed within 6%. Python, at about a tenth of a second per run, took the hit from contention and the short compiled run did not.
- **Change:** 15 runs each, and speed now comes from the best run, because other work can only slow a run. Median and range stay in the record.
- **Lesson:** a ratio of two medians under load measures the load. The reported number would have been about 3 times too high.
