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
