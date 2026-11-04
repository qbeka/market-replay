# Research note

Read on 4 November 2026. Each entry records what was adopted and where this project differs.

## Databento MBO schema

Source: [MBO schema docs](https://databento.com/docs/schemas-and-data-formats/mbo), current page.

- **Fields:** `ts_event` (matching engine), `ts_recv` (capture server), venue `sequence`, `order_id`, integer `price` in units of 1e-9, and `action` of add, cancel, modify, clear, trade, fill or none.
- **Adopted:** integer prices, one normalized event per book change, and a CLEAR action that rebuilds the book. The two timestamps became `ts_ns` and `recv_ns`, and `recv_ns - ts_ns` is how observation latency gets fitted.
- **Different:** no Databento adapter exists. Trade and fill semantics vary by venue, so one needs venue-specific work and a paid order.

## LOBSTER data structure

Source: secondary quotes of the [format page](https://lobsterdata.com/info/DataStructure.php). See [data_rights.md](data_rights.md).

- **Adopted:** LOBSTER is the first adapter, because its orderbook file supplies a separate check on reconstruction.
- **Different:** tested on a hand-written fixture and on synthetic windows exported to the format. No real file has been read.

## HftBacktest fill models

Source: [order fill docs](https://hftbacktest.readthedocs.io/en/latest/order_fill.html), current page.

- **Limits it names:** replay cannot react to your orders, queue position must be modeled when data lacks order ids, liquidity-taking fills can be unrealistic, and results need checking against live trading.
- **Adopted:** its risk-averse queue model, where cancels never advance you, is `cancel_ahead_credit_pct=0`. A trade through your price fills you, as in its exchange models. Each assumption is a config value and a stress test.
- **Different:** order-by-order data shows exactly which cancels were ahead, so the default is exact FIFO, not probabilistic. Partial fills are always on.

## ABIDES

Source: Byrd, Hybinette and Balch, [arXiv:1904.12066v1](https://arxiv.org/abs/1904.12066), April 2019.

- **Point taken:** models built on historical data cannot repeat history with a change and let later events respond to that change. In ABIDES, agents react to each other through an exchange agent.
- **Adopted:** this project says plainly that it cannot measure market impact. Consumed liquidity never comes back within a ledger, and that is a limit on double counting, not an impact model.
- **Different:** no agent simulation and no impact sensitivity experiment. The synthetic market generates the feed once, and it does not respond to simulated orders.

## Hafsi and Vittori

Source: [arXiv:2411.06389v2](https://arxiv.org/abs/2411.06389), revised 1 November 2025. Licensed CC BY 4.0.

- **Their task:** buy 20,000 shares in 30 minutes in ABIDES with a DQN. Their state uses remaining holdings, remaining time, volume imbalance over five levels, and best bid and ask. Their actions are market orders of four sizes, or nothing. Their reward is shortfall less a depth penalty, with a terminal penalty on residual inventory.
- **Adopted:** the finite-horizon inventory shape, a similar state, a random baseline, and a terminal cost on residual inventory.
- **Different:** fitted Q-iteration on replay data, not DQN in an interactive simulator. Actions are wait, rest and cross, because passive orders are the question here. They report one-sided t-tests on 50 runs per strategy. This project bootstraps days instead. Their numbers are not comparable with these and none are reused.

## Upstream code

No upstream code is copied. numpy, scikit-learn and threadpoolctl are dependencies under BSD-3-Clause. Versions are pinned in `pyproject.toml` and `uv.lock`.
