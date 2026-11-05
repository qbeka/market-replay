# Market replay and execution

A limit order book replay engine in Python and C, with a learned execution policy tested the way a research result should be: rules frozen first, one final test, and every assumption stressed.

It answers one question. You have a small order to finish within a minute. Should a simulated trader rest in the queue, cross the spread, or wait?

**Every order here is simulated. The market is a seeded synthetic feed. Nothing connects to a real venue, and these results say nothing about real trading profit.**

| | |
|---|---|
| **Result** | The learned policy passed all three frozen checks against the best simple rule on 34 held-out days and 909 paired orders. |
| **Forced completion** | 1.2% of lots, against 7.9% for the urgency rule and 15.4% for resting only |
| **Mean cost** | 0.064 bps lower than the urgency rule, 95% interval 0.014 to 0.110, resampling whole days. 3.27 bps lower than crossing at once. |
| **Tail cost** | P95 of 2.69 bps, against 3.07 for the urgency rule |
| **Exactness** | 0 ledger or book divergences between C and Python across 100,000 fuzz cases, 51.5M events and 5.9M fills, under ASan and UBSan |
| **Speed** | 9.5M events per second on one core in C, 7.8 times the Python reference. Peak memory 21 MB against 184 MB. |
| **Tests** | 140 tests, with 99% line coverage of the engine and 81% of the package |
| **Stress** | 18 moved assumptions. The result reverses in 1 and loses its clear interval in 2. |

Measured on 2026-11-05 on one Apple M5 Pro Mac. Other projects shared the machine, so speed comes from the best of 15 runs. Replay cannot measure market impact, and the mean gain is small. [What the numbers leave out](#what-the-numbers-leave-out) has the details.

## See it

Open [`site/index.html`](site/index.html) in a browser. It is one file and makes no network requests. Two simulated traders work the same order on the same recorded market, with the book, queue position, decisions, fills, inventory and cost shown side by side.

## Run it

You need [uv](https://docs.astral.sh/uv/) and clang. On a Mac, clang comes with the Xcode command line tools.

```bash
xcode-select --install
```

```bash
brew install uv
```

Install the pinned dependencies. uv fetches Python 3.12 if you need it.

```bash
uv sync
```

Run the smoke test. In about a second, it checks the hand-worked trace, runs the baselines on a tiny study and compares C with Python.

```bash
uv run mreplay smoke
```

Print the full story of one simulated order. You need the study data first, about 2 minutes and 534 MB.

```bash
uv run mreplay generate
```

```bash
uv run mreplay trace --day 100 --parent 0 --strategy urgency
```

## Check it

Run the correctness suite.

```bash
uv run pytest
```

Check that every saved result still matches its recorded hash.

```bash
uv run mreplay manifest --check
```

Run the full benchmark. It takes about 45 minutes on 8 cores, and you need `generate` first.

```bash
uv run mreplay benchmark
```

The stages run in this order. Each one saves to `results/` or `docs/`.

| Command | What it does | Time |
|---|---|---:|
| `mreplay generate` | Writes 120 seeded days, 11.6M events | 2 min |
| `mreplay fit` | Fits fill models and the comparator on days 0 to 59 | 5 min |
| `mreplay tune` | Picks settings on days 61 to 84 | 4 min |
| `mreplay freeze` | Writes the hypothesis, split and rule. Commit this file before testing. | 1 s |
| `mreplay final` | The one evaluation on days 86 to 119 | 1 min |
| `mreplay stress` | Reruns frozen strategies under 18 moved assumptions | 11 min |
| `mreplay profile` | Measures speed and memory | 2 min |
| `mreplay fuzz` | Runs 100,000 cases of C against Python under sanitizers | 23 min |
| `mreplay evidence` | Records the test count and coverage | 1 min |
| `mreplay report` | Writes `docs/report.md`, `docs/resume.md` and the plots | 5 s |
| `mreplay demo` | Writes `site/index.html` from a recorded run | 10 s |
| `mreplay manifest` | Hashes every saved result | 1 s |

`mreplay final` refuses to run unless `results/freeze.json` is committed and the fitted models and inputs still match it. Hashes of pickled models can differ on another chip or library version. If yours do, run your own `fit`, `tune` and `freeze`.

## How it works

```
seeded feed -> order book -> three clocks -> strategies -> isolated ledgers -> cost
   synthetic.py   book.py     execution.py   policy.py        ledger.py      metrics.py
```

- **Feed.** 48-byte market-by-order records in integer ticks, lots and nanoseconds. It carries halts, sequence gaps and resets. [`events.py`](src/marketreplay/events.py) has the format.
- **Book.** A FIFO limit order book. A guard stops trusting it after a sequence gap, until a snapshot rebuilds it.
- **Three clocks.** A strategy sees the market late, its orders arrive late, and its cancels land late. Fills keep happening in between.
- **Queue position.** Order ids show exactly which orders ahead of you left, so position is tracked and never drawn at random.
- **Isolated ledgers.** One pass over a feed runs every strategy, because simulated orders never touch the book. Cash is in integer milliticks, and every ledger must conserve exactly.
- **Policy.** An isotonic-calibrated gradient-boosted fill model prices resting against crossing inside hard limits on deadline and schedule. [`policy.py`](src/marketreplay/policy.py) is 92 lines.
- **Evaluation.** Strategies trade the same parents, and days are the unit of replication. The interval comes from a paired day bootstrap with 10,000 draws.
- **Compiled replay.** [`compiled/replay.c`](compiled/replay.c) is a port with no shared code. It is the independent implementation that the Python reference has to agree with, byte for byte.

## What the numbers leave out

- **Market impact.** The feed is fixed, so a simulated order never moves a price and nobody reacts to it. No setting fixes this. Costs are a lower bound for any order large enough to be noticed.
- **A real market.** The feed comes from a generator, and this project set how strong its signal is. Results show the method can use structure that exists. They do not show the structure exists on any venue.
- **The mean gain is small.** 0.064 bps is a narrow win. The larger change is forced completion and finishing early: 85.6% of orders finished before the deadline, against 37.7%.
- **Where it stops.** At a 10% participation cap the result reverses and resting alone wins. With 50 ms or 250 ms of added latency the interval reaches zero. On the 6 sparse days the difference is not clear either.
- **Sparse days are extreme.** Some show spreads near 50 ticks, so their largest gains and losses are not realistic.
- **Fills are a model.** The policy trains on labels from the queue model, so calibration shows it learned the simulator. The stress tests show how much the answer depends on that model.
- **No real data.** The LOBSTER adapter is tested against the format and has never read a real file. [`docs/data_rights.md`](docs/data_rights.md) explains why.
- **The review was not human.** A separate AI agent reviewed the code before the freeze. It found three biases, and each became a failing test, then a fix.

## Find your way around

| Read | For |
|---|---|
| [`docs/report.md`](docs/report.md) | Every result, generated from saved files |
| [`docs/assumptions.md`](docs/assumptions.md) | Every matching rule and how it is stressed |
| [`docs/experiments.md`](docs/experiments.md) | What failed and what changed because of it |
| [`docs/walkthrough.md`](docs/walkthrough.md) | The project in five minutes, out loud |
| [`docs/resume.md`](docs/resume.md) | Resume lines tied to the exact run |
| [`docs/research_note.md`](docs/research_note.md) | Sources, and where this project differs from them |
| [`docs/environment.md`](docs/environment.md) | The machine and versions used |
| [`CHECKPOINT.md`](CHECKPOINT.md) | What passed, what failed and the next command |
| [`results/manifest.json`](results/manifest.json) | Every saved result with its hash and the command that made it |

## License

MIT. Geist Serif, embedded in the page, is under the SIL Open Font License, in [`site/fonts/`](site/fonts/GeistSerif-OFL.txt).
