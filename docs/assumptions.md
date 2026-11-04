# Assumptions and limits

Every result in this project depends on the rules below. Each configurable rule has a stress test, listed in the report.

## What this replay cannot tell you

- **It cannot measure market impact.** The feed is fixed. A simulated order never moves a price, and nobody reacts to it. A real order that crosses 120 lots would change what happens next. Every cost here is a lower bound for an order large enough to be noticed.
- **It does not show a real market.** The study feed comes from `synthetic.py`. A model trained on it learns the generator's rules. A gain here shows the method can use structure that exists. It does not show that the structure exists on any venue.
- **The market is partly the author's design.** The generator has hidden drift that order flow leaks, and this project set how strong it is. A policy doing well on it partly reflects that choice.

## Matching rules

| Rule | Default | Stress |
|---|---|---|
| Queue position at arrival | All visible lots at the price are ahead | `queue_ahead_pct` up to 250% |
| Cancels ahead of you | Advance you by the exact lots, because the feed shows which orders were ahead | `cancel_ahead_credit_pct` 50 and 0 |
| Trades at your price | Burn the queue ahead first, and the rest fills you | `trade_through_only` removes these fills |
| Trades through your price | Clear the queue and fill you at your limit | Always on |
| A recorded add that lands on your price | Fills you at your limit | Always on |
| Passive fill cost | Maker fee only | `adverse_fill_mticks` adds a cost per lot |
| Participation | At most 30% of trailing 60 s tape, plus 16 free lots | 10% and 60% |

Other rules:

- A passive order that would cross on arrival is rejected, like a post-only order.
- A cross is immediate-or-cancel, limited to the observed touch plus 1 tick.
- A cross walks at most 20 levels. Fills use displayed depth only.
- A fill does not reduce the size of the recorded order it replaced. That order stays in the book, which is optimistic for later simulated orders at that price, within one ledger only.
- One historical trade can pay a ledger at most its own size. Lots a ledger crossed are marked consumed, so it cannot take them twice.
- Consumed liquidity never returns within a ledger, and ledgers never share it.

## Clocks

| Clock | Rule |
|---|---|
| Observation | A strategy at time t sees the snapshot from t minus the observation latency. Its own fills arrive just as late |
| Order | An order sent at t arrives at t plus the order latency |
| Cancel | A cancel sent at t lands at t plus the cancel latency. Fills keep happening until then |
| Ties | A feed event at time t runs before a simulated action at time t |

Observation latency is the 95th percentile of feed delay on training days. Order entry latency cannot be seen in a feed, so it is that value plus an assumed 5 ms gateway delay. Decisions run every 2 s and snapshots every 1 s.

## Overfill

A strategy can only send lots it knows are not already working. It learns of a fill or a finished order late, so `sendable` counts every order not yet confirmed done. An in-flight cancel therefore delays the next order by up to one decision, as it would in practice.

## Deadline rule

Every order completes. At the deadline, any resting order is removed at once and the rest is bought from the visible book at any price, paying the taker fee plus 2 ticks per lot. If the book runs out, the rest pays the last visible price plus 10 ticks. Leaving an order unfinished therefore always costs money and can never improve a score.

The instant removal is an assumption. It ignores cancel latency at the deadline.

## What gets excluded

A parent is excluded for every strategy, so pairs stay intact, if any of these happen between its start and deadline:

- the book is invalid, from a sequence gap or an impossible event, until resynchronized
- a halt or auction is in progress
- the book is one-sided at arrival, so no midpoint exists

## Numeric note

On this machine numpy's Accelerate backend raises false floating point warnings in matrix products. A direct check against pure Python arithmetic matched to 1.8e-15 with no NaNs. The CLI silences that one warning.
