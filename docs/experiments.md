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

## 6. The independent review found three biases before the freeze

A separate reviewer got the acceptance criteria and a commit hash, then tried to break the engine. Each finding was reproduced as a failing test before its fix, in `tests/test_review_findings.py`. These numbers are the reviewer's, from 30 small non-test days.

- **Hidden prints burned visible queue.** A hidden trade removes no visible lots, but it reduced `queue_ahead`. Passive cost was understated by 0.052 bps and urgency by 0.027 bps, a bias that favored resting. Fixed in both engines.
- **Own order state leaked ahead of the observation clock.** `rest_price` read the live order, so a strategy could learn of a full fill early. It was harmless at base latency but changed 15 of 40 parents at 904 ms. That would have biased the high-latency stress tests. The causality test now compares every field of the view, which would have caught this.
- **The arrival midpoint in the view came from the live book.** Only `fitted_q` read it. The ledger keeps the true midpoint, and strategies now get the delayed one.

Off the study path, the review also fixed the deadline price on an empty far side, priority rules missing from the compiled replay, and a snapshot grid that drifted when a feed started off-interval. Lots per share was removed from the LOBSTER adapter, because flooring odd lots to zero broke the book.

What held up: the book against a brute-force FIFO over 183,621 steps, conservation and no double fills over 1,500 fuzz seeds, ledger isolation, shortfall signs, and compiled against Python under ASan and UBSan.
