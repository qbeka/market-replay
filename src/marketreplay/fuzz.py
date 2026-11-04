"""Differential fuzz: the compiled replay against the Python reference.

Each case builds a random feed and random configuration, runs random seeded commands
through the Python engine, then replays the same commands in C. The ledgers and the book
checksum must match exactly. The C side runs under ASan and UBSan, so a memory error or
undefined behavior also fails the case. Every Python ledger must also conserve.

Feeds are hostile on purpose: sequence gaps, impossible events, halts, mid-feed
resynchronization, negative prices, and orders landing exactly on deadlines.
The generator's approach came from the review in docs/experiments.md.
"""

import json
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .book import Book, PriorityRules
from .compiled import build, by_run, python_book_checksum, reference_lines, run_book, run_exec, write_tape
from .events import Event, Kind, Side, to_array
from .execution import ExecConfig, Intent, Parent, run_session

CHUNK = 250


class FeedBuilder:
    def __init__(self, rng: random.Random, rules: PriorityRules, base: int):
        self.rng, self.base = rng, base
        self.book = Book(rules)
        self.events: list[Event] = []
        self.seq = self.ts = self.next_id = 0

    def push(self, kind: Kind, side=Side.NONE, order_id=0, price=0, qty=0, apply=True) -> None:
        self.seq += 1
        event = Event(self.seq, self.ts, kind, side, order_id, price, qty, self.ts)
        if apply:
            self.book.apply(event)
        self.events.append(event)

    def add(self, side: Side, price: int, qty: int) -> None:
        self.next_id += 1
        self.push(Kind.ADD, side, self.next_id, price, qty)

    def open(self) -> None:
        self.push(Kind.CLEAR)
        for depth in range(1, self.rng.randint(2, 6)):
            for side in (Side.BID, Side.ASK):
                self.add(side, self.base - side * depth, self.rng.randint(1, 9))
        self.push(Kind.SNAPSHOT_END)

    def random_order(self) -> tuple[int, Side, int, int] | None:
        if not self.book.order_index:
            return None
        order_id = self.rng.choice(list(self.book.order_index))
        side, price = self.book.order_index[order_id]
        return order_id, Side(side), price, self.book.order_qty(order_id)

    def legal_step(self, wide: bool) -> None:
        rng, book = self.rng, self.book
        bid, ask = book.best(Side.BID), book.best(Side.ASK)
        roll = rng.random()
        picked = self.random_order()
        if bid is None or ask is None or roll < 0.36 or picked is None:
            side = rng.choice((Side.BID, Side.ASK))
            opposite = ask if side == Side.BID else bid
            top = opposite - side if opposite is not None else self.base
            self.add(side, top - side * rng.randint(0, 40 if wide else 5), rng.randint(1, 9))
        elif roll < 0.54:
            order_id, side, price, qty = picked
            self.push(Kind.CANCEL, side, order_id, price, qty if rng.random() < 0.6 else rng.randint(1, qty))
        elif roll < 0.68:
            order_id, side, price, qty = picked
            choice = rng.random()
            if choice < 0.35 and qty > 1:
                self.push(Kind.MODIFY, side, order_id, price, rng.randint(1, qty - 1))
            elif choice < 0.7:
                self.push(Kind.MODIFY, side, order_id, price, qty + rng.randint(1, 4))
            else:
                self.push(Kind.MODIFY, side, order_id, price - side * rng.randint(1, 3), qty)
        elif roll < 0.94:
            side = rng.choice((Side.BID, Side.ASK))
            lots = rng.randint(1, 12)
            while lots > 0 and book.best(side) is not None:
                price = book.best(side)
                front = book.queue(side, price)[0]
                take = min(lots, book.order_qty(front))
                self.push(Kind.EXECUTE, side, front, price, take)
                lots -= take
        else:
            side = rng.choice((Side.BID, Side.ASK))
            if book.best(side) is not None:
                self.push(Kind.TRADE_HIDDEN, side, 0, book.best(side), rng.randint(1, 6))

    def hostile_step(self) -> None:
        rng, book = self.rng, self.book
        roll = rng.random()
        if roll < 0.3:
            self.seq += rng.randint(1, 3)  # A sequence gap.
            self.push(Kind.ADD, Side.BID, 10**7 + self.seq, self.base - 9, 1, apply=False)
        elif roll < 0.5 and (picked := self.random_order()):
            order_id, side, price, _ = picked  # An event the book must refuse.
            self.push(Kind.CANCEL, side, order_id, price + 1, 1, apply=False)
        elif roll < 0.75:
            self.push(Kind.HALT_START, apply=False)
            self.ts += rng.choice((0, 1, 5, 30))
            self.push(Kind.HALT_END, apply=False)
        else:
            orders = [
                (side, price, order_id, qty)
                for side in (Side.BID, Side.ASK)
                for price, level in sorted(book.levels[side].items())
                for order_id, qty in level.items()
            ]
            self.push(Kind.CLEAR)
            for side, price, order_id, qty in orders:
                self.push(Kind.ADD, side, order_id, price, qty)
            self.push(Kind.SNAPSHOT_END)


def random_feed(rng: random.Random, rules: PriorityRules) -> list[Event]:
    builder = FeedBuilder(rng, rules, rng.choice((0, 7, -50, 100, 10_001)))
    hostile = rng.choice((0.0, 0.005, 0.02, 0.05))
    wide = rng.random() < 0.3
    builder.open()
    for _ in range(rng.choice((100, 300, 800))):
        builder.ts += rng.choice((0, 0, 1, 1, 2, 3, 5, 9))
        if rng.random() < hostile:
            builder.hostile_step()
        else:
            builder.legal_step(wide)
    return builder.events


def random_config(rng: random.Random, rules: PriorityRules) -> ExecConfig:
    return ExecConfig(
        obs_latency_ns=rng.choice((0, 1, 3, 7, 30)),
        order_latency_ns=rng.choice((0, 1, 5, 7, 11, 20)),
        cancel_latency_ns=rng.choice((0, 1, 5, 7, 17, 20)),
        decision_interval_ns=rng.choice((1, 3, 7, 10, 25)),
        snapshot_interval_ns=rng.choice((1, 2, 5, 10)),
        taker_fee_mticks=rng.choice((0, 300, 900)),
        maker_fee_mticks=rng.choice((-100, 0, 250)),
        adverse_fill_mticks=rng.choice((0, 700)),
        participation_cap_pct=rng.choice((None, None, 0, 10, 30, 60, 100, 250)),
        participation_window_ns=rng.choice((1, 50, 200, 600)),
        participation_free_lots=rng.choice((0, 5, 16)),
        cancel_ahead_credit_pct=rng.choice((100, 100, 50, 33, 0)),
        queue_ahead_pct=rng.choice((100, 100, 150, 175, 250, 0)),
        trade_through_only=rng.random() < 0.15,
        book_levels=rng.choice((1, 2, 5, 5)),
        cross_slippage_ticks=rng.choice((0, 1, 3)),
        completion_penalty_ticks=rng.choice((0, 2, 5)),
        unfilled_extra_ticks=rng.choice((0, 3, 10)),
        priority=rules,
        record_trace=True,
    )


class RandomCommands:
    name = "random_commands"

    def __init__(self, rng: random.Random, max_lots: int):
        self.rng, self.max_lots = rng, max_lots

    def decide(self, view) -> Intent | None:
        roll = self.rng.random()
        lots = self.rng.randint(1, self.max_lots)
        if roll < 0.45:
            return Intent.rest(lots)
        if roll < 0.7:
            return Intent.cross(lots)
        return Intent.wait() if roll < 0.85 else None


def random_work(rng: random.Random, config: ExecConfig, end_ns: int) -> list[tuple[Parent, RandomCommands]]:
    work = []
    for index in range(rng.randint(1, 30)):
        start = rng.randint(0, max(1, end_ns))
        length = rng.randint(0, max(2, end_ns // 2))
        roll = rng.random()
        if roll < 0.3:  # An order arrives exactly on the deadline.
            length = config.decision_interval_ns * rng.randint(1, 10) + config.order_latency_ns
        elif roll < 0.4:  # A cancel lands exactly on the deadline.
            length = config.decision_interval_ns * rng.randint(1, 10) + config.cancel_latency_ns
        side = rng.choice((Side.BID, Side.ASK))
        commands = RandomCommands(random.Random(rng.getrandbits(32)), rng.choice((3, 30, 150)))
        work.append((Parent(index, side, rng.randint(1, 150), start, start + length), commands))
    return work


def run_case(seed: int, folder: Path, tally: Counter) -> None:
    rng = random.Random(seed)
    rules = rng.choice(
        (PriorityRules(), PriorityRules(False, False), PriorityRules(True, True), PriorityRules(False, True))
    )
    config = random_config(rng, rules)
    feed = random_feed(rng, rules)
    runs = run_session(feed, random_work(rng, config, feed[-1].ts_ns), config)
    for run in runs:
        if run.ledger is not None:
            run.ledger.check_conservation()
    events_path, tape_path = folder / "events.bin", folder / "tape.txt"
    events_path.write_bytes(to_array(feed).tobytes())
    write_tape(runs, tape_path)
    compiled = run_exec(events_path, tape_path, config, checked=True).lines
    reference = by_run(reference_lines(runs))
    tally["cases"] += 1
    tally["events"] += len(feed)
    tally["parents"] += len(runs)
    tally["fills"] += sum(1 for line in reference if line.startswith("F"))
    tally["excluded_parents"] += sum(1 for run in runs if run.excluded)
    tally["ledger_divergences"] += compiled != reference
    tally["book_divergences"] += run_book(events_path, checked=True)["checksum"] != python_book_checksum(feed)


def run_chunk(seeds: list[int]) -> Counter:
    tally: Counter = Counter()
    with tempfile.TemporaryDirectory() as folder:
        for seed in seeds:
            run_case(seed, Path(folder), tally)
    return tally


def run_fuzz(cases: int, output: Path | None = None, workers: int = 8) -> dict:
    build(checked=True)
    started = time.perf_counter()
    chunks = [list(range(i, min(i + CHUNK, cases))) for i in range(0, cases, CHUNK)]
    total: Counter = Counter()
    with ProcessPoolExecutor(workers) as pool:
        for tally in pool.map(run_chunk, chunks):
            total.update(tally)  # update keeps zero counts. Adding Counters would drop them.
    result = {**total, "seconds": round(time.perf_counter() - started, 1), "sanitizers": ["address", "undefined"]}
    if output is not None:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


