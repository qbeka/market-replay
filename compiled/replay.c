/* Compiled replay of the book and the simulated execution engine.
 *
 * This is a port of book.py and execution.py, and those stay the reference. It only
 * exists for speed, and it only counts if tests/test_compiled.py finds its ledgers
 * byte-identical to theirs. Strategies stay in Python: this program reads a tape of
 * the commands they sent (see src/marketreplay/compiled.py).
 *
 * Usage:
 *   replay book <events.bin>                         replay only, prints a checksum
 *   replay exec <events.bin> <tape.txt> [key=value]  replay and execute, prints ledgers
 *
 * Units match the Python: ticks, lots, nanoseconds, millitick-lots.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int64_t i64;

/* The 48-byte on-disk record. Must match EVENT_DTYPE in events.py. */
typedef struct {
    i64 seq, ts_ns, recv_ns, order_id, price;
    int32_t qty;
    int16_t kind, side;
} Event;

enum { ADD = 1, CANCEL, MODIFY, EXECUTE, TRADE_HIDDEN, CLEAR, SNAPSHOT_END, HALT_START, HALT_END };
enum { MAKER = 1, TAKER = 2, FORCED = 3 };
#define BID 1
#define ASK (-1)
#define SIDE_INDEX(s) ((s) == BID ? 0 : 1)
#define MAX_LEVELS 4096
#define MAX_ACTIVE 4096
#define MTICKS_PER_TICK 1000

/* Book ------------------------------------------------------------------- */

typedef struct Watch { struct Child *child; struct Watch *next; } Watch;

typedef struct Order {
    i64 id, price, qty;
    int side;
    struct Order *prev, *next;
    struct Level *level;
    Watch *watchers;
} Order;

typedef struct Level { i64 price; Order *head, *tail; struct Level *free_next; } Level;

#define TABLE_BITS 21
#define TABLE_SIZE (1u << TABLE_BITS)
static Order *table[TABLE_SIZE];
static Level *sorted[2][MAX_LEVELS]; /* Ascending by price*side, so best is last. */
static int level_count[2];
static Order *order_pool; static size_t pool_used, pool_size; static Order *free_orders;
static Level *free_levels;
static Watch *free_watches;

static inline uint32_t slot_of(i64 id) {
    return (uint32_t)(((uint64_t)id * 0x9E3779B97F4A7C15ull) >> (64 - TABLE_BITS));
}

static Order *find_order(i64 id) {
    uint32_t i = slot_of(id);
    while (table[i]) {
        if (table[i]->id == id) return table[i];
        i = (i + 1) & (TABLE_SIZE - 1);
    }
    return NULL;
}

static void table_insert(Order *o) {
    uint32_t i = slot_of(o->id);
    while (table[i]) i = (i + 1) & (TABLE_SIZE - 1);
    table[i] = o;
}

/* Linear probing deletion: shift back any entry that would become unreachable. */
static void table_remove(i64 id) {
    uint32_t i = slot_of(id);
    while (table[i]->id != id) i = (i + 1) & (TABLE_SIZE - 1);
    uint32_t j = i;
    for (;;) {
        j = (j + 1) & (TABLE_SIZE - 1);
        if (!table[j]) break;
        uint32_t home = slot_of(table[j]->id);
        if (((j - home) & (TABLE_SIZE - 1)) >= ((j - i) & (TABLE_SIZE - 1))) {
            table[i] = table[j];
            i = j;
        }
    }
    table[i] = NULL;
}

static Order *new_order(void) {
    if (free_orders) { Order *o = free_orders; free_orders = o->next; return o; }
    if (pool_used == pool_size) { fprintf(stderr, "order pool exhausted\n"); exit(2); }
    return &order_pool[pool_used++];
}

/* Index of the first level whose key is >= key. */
static int lower_bound(int s, i64 key, int side) {
    int lo = 0, hi = level_count[s];
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (sorted[s][mid]->price * side < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static Level *find_level(int side, i64 price) {
    int s = SIDE_INDEX(side), i = lower_bound(s, price * side, side);
    return (i < level_count[s] && sorted[s][i]->price == price) ? sorted[s][i] : NULL;
}

static inline int has_best(int side) { return level_count[SIDE_INDEX(side)] > 0; }
static inline i64 best(int side) { int s = SIDE_INDEX(side); return sorted[s][level_count[s] - 1]->price; }

static i64 level_total(const Level *l) {
    i64 total = 0;
    for (const Order *o = l->head; o; o = o->next) total += o->qty;
    return total;
}

static void drop_watchers(Order *o) {
    while (o->watchers) { Watch *w = o->watchers; o->watchers = w->next; w->next = free_watches; free_watches = w; }
}

static void remove_order(Order *o) {
    Level *l = o->level;
    if (o->prev) o->prev->next = o->next; else l->head = o->next;
    if (o->next) o->next->prev = o->prev; else l->tail = o->prev;
    table_remove(o->id);
    drop_watchers(o);
    if (!l->head) {
        int s = SIDE_INDEX(o->side), i = lower_bound(s, l->price * o->side, o->side);
        memmove(&sorted[s][i], &sorted[s][i + 1], sizeof(Level *) * (size_t)(level_count[s] - i - 1));
        level_count[s]--;
        l->free_next = free_levels;
        free_levels = l;
    }
    o->next = free_orders;
    free_orders = o;
}

static void book_clear(void) {
    for (int s = 0; s < 2; s++) {
        while (level_count[s]) {
            Level *l = sorted[s][level_count[s] - 1];
            while (l->head) {
                Order *o = l->head;
                int was_last = o->next == NULL;
                remove_order(o);
                if (was_last) break;
            }
        }
    }
}

/* Each returns 0 on success, or 1 for an event the book cannot accept (BookError). */
static int book_add(int side, i64 id, i64 price, i64 qty) {
    if (find_order(id) || qty <= 0) return 1;
    if (has_best(-side) && (price - best(-side)) * side >= 0) return 1;
    int s = SIDE_INDEX(side);
    Level *l = find_level(side, price);
    if (!l) {
        if (level_count[s] == MAX_LEVELS) { fprintf(stderr, "too many levels\n"); exit(2); }
        if (free_levels) { l = free_levels; free_levels = l->free_next; } else l = malloc(sizeof(Level));
        l->price = price; l->head = l->tail = NULL;
        int i = lower_bound(s, price * side, side);
        memmove(&sorted[s][i + 1], &sorted[s][i], sizeof(Level *) * (size_t)(level_count[s] - i));
        sorted[s][i] = l;
        level_count[s]++;
    }
    Order *o = new_order();
    o->id = id; o->price = price; o->qty = qty; o->side = side;
    o->prev = l->tail; o->next = NULL; o->level = l; o->watchers = NULL;
    if (l->tail) l->tail->next = o; else l->head = o;
    l->tail = o;
    table_insert(o);
    return 0;
}

static int book_reduce(const Event *e) {
    Order *o = find_order(e->order_id);
    if (!o || e->price != o->price || e->side != o->side) return 1;
    i64 left = o->qty - e->qty;
    if (left < 0 || e->qty <= 0) return 1;
    if (left == 0) remove_order(o); else o->qty = left;
    return 0;
}

static int size_up_loses = 1, size_down_loses = 0;

static inline int loses_priority(const Event *e, const Order *o) {
    if (e->price != o->price) return 1;
    if (e->qty > o->qty) return size_up_loses;
    if (e->qty < o->qty) return size_down_loses;
    return 0;
}

static int book_modify(const Event *e) {
    Order *o = find_order(e->order_id);
    if (!o || e->side != o->side) return 1;
    if (loses_priority(e, o)) {
        int side = o->side;
        remove_order(o);
        return book_add(side, e->order_id, e->price, e->qty);
    }
    if (e->qty <= 0) return 1;
    o->qty = e->qty;
    return 0;
}

static int book_apply(const Event *e) {
    switch (e->kind) {
        case ADD: return book_add(e->side, e->order_id, e->price, e->qty);
        case CANCEL: case EXECUTE: return book_reduce(e);
        case MODIFY: return book_modify(e);
        case CLEAR: book_clear(); return 0;
        default: return 0;
    }
}

/* Feed guard, as in ReplayedBook --------------------------------------------- */

static int valid, rebuilding, have_seq;
static i64 last_seq;

static void guard_apply(const Event *e) {
    if (e->kind == CLEAR) {
        valid = 0;
        book_clear();
        rebuilding = 1;
        have_seq = 1; last_seq = e->seq;
        return;
    }
    if (have_seq && e->seq != last_seq + 1) { rebuilding = 0; valid = 0; }
    if (valid || rebuilding) {
        if (book_apply(e)) { rebuilding = 0; valid = 0; }
    }
    if (e->kind == SNAPSHOT_END && rebuilding) { rebuilding = 0; valid = 1; }
    have_seq = 1; last_seq = e->seq;
}

/* Execution ------------------------------------------------------------------ */

typedef struct {
    i64 taker_fee, maker_fee, credit_pct, ahead_pct, adverse, cap_pct, cap_window, free_lots;
    i64 penalty_ticks, extra_ticks, take_levels;
    int trade_through_only;
} Config;

static Config cfg = {300, -100, 100, 100, 0, -1, 60000000000LL, 0, 2, 10, 20, 0};

typedef struct { i64 id, lots; } Consumed;
typedef struct { i64 ts, qty; } Stamp;
typedef struct { Stamp *data; size_t head, tail, cap; } Ring;

typedef struct Run {
    int index, side, live, finished, activated;
    i64 target, start, deadline;
    i64 filled, notional, fees, penalties, cash, mid_x2;
    const char *excluded;
    struct Child **children; int child_count, child_cap;
    Consumed *consumed; int consumed_count, consumed_cap;
    Ring window; i64 window_lots;
    i64 budget, budget_stamp;
} Run;

typedef struct Child {
    i64 id, price, lots, filled, ahead;
    int side, immediate, live, finished;
    Run *run;
} Child;

static Run *runs; static int run_count;
static Child *active[2][MAX_ACTIVE]; static int active_count[2];
static Run *live_runs[MAX_ACTIVE]; static int live_count;
static int halted;
static Ring cap_tape; static i64 cap_tape_lots;
static i64 feed_index; /* Identifies one historical trade within a ledger's budget. */
static i64 last_best[2]; static int have_last[2];

static void ring_push(Ring *r, i64 ts, i64 qty) {
    if (r->tail == r->cap) {
        size_t n = r->tail - r->head;
        if (n * 2 < r->cap) memmove(r->data, r->data + r->head, n * sizeof(Stamp));
        else { r->cap = r->cap ? r->cap * 2 : 64; Stamp *d = malloc(r->cap * sizeof(Stamp)); if (n) memcpy(d, r->data + r->head, n * sizeof(Stamp)); free(r->data); r->data = d; }
        r->head = 0; r->tail = n;
    }
    r->data[r->tail++] = (Stamp){ts, qty};
}

static i64 consumed_of(const Run *r, i64 id) {
    for (int i = 0; i < r->consumed_count; i++) if (r->consumed[i].id == id) return r->consumed[i].lots;
    return 0;
}

static void consumed_set(Run *r, i64 id, i64 lots) {
    for (int i = 0; i < r->consumed_count; i++) if (r->consumed[i].id == id) { r->consumed[i].lots = lots; return; }
    if (r->consumed_count == r->consumed_cap) {
        r->consumed_cap = r->consumed_cap ? r->consumed_cap * 2 : 32;
        r->consumed = realloc(r->consumed, sizeof(Consumed) * (size_t)r->consumed_cap);
    }
    r->consumed[r->consumed_count++] = (Consumed){id, lots};
}

static inline i64 min64(i64 a, i64 b) { return a < b ? a : b; }
static inline i64 max64(i64 a, i64 b) { return a > b ? a : b; }

/* Python floor division. C truncates toward zero, which differs for a negative cap. */
static inline i64 floor_div(i64 a, i64 b) { i64 q = a / b; return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q; }

static void record(Run *r, i64 ts, i64 child, i64 price, i64 qty, int role, i64 fee, i64 penalty, i64 seq) {
    r->filled += qty;
    r->notional += price * qty;
    r->fees += fee;
    r->penalties += penalty;
    r->cash -= r->side * price * qty * MTICKS_PER_TICK + fee + penalty;
    printf("F %d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %d %" PRId64 " %" PRId64 " %" PRId64 "\n",
           r->index, ts, child, price, qty, role, fee, penalty, seq);
}

static i64 capped(Run *r, i64 now, i64 qty) {
    if (cfg.cap_pct < 0) return qty;
    i64 horizon = now - cfg.cap_window;
    while (cap_tape.head < cap_tape.tail && cap_tape.data[cap_tape.head].ts <= horizon) cap_tape_lots -= cap_tape.data[cap_tape.head++].qty;
    Ring *w = &r->window;
    while (w->head < w->tail && w->data[w->head].ts <= horizon) r->window_lots -= w->data[w->head++].qty;
    i64 allowed = floor_div(cap_tape_lots * cfg.cap_pct, 100) + cfg.free_lots - r->window_lots;
    return max64(0, min64(qty, allowed));
}

static void note_own(Run *r, i64 now, i64 qty) {
    if (cfg.cap_pct >= 0) { ring_push(&r->window, now, qty); r->window_lots += qty; }
}

static void finish_child(Child *c) {
    c->finished = 1;
    if (c->live) {
        c->live = 0;
        int s = SIDE_INDEX(c->side);
        for (int i = 0; i < active_count[s]; i++) {
            if (active[s][i] == c) {
                memmove(&active[s][i], &active[s][i + 1], sizeof(Child *) * (size_t)(active_count[s] - i - 1));
                active_count[s]--;
                break;
            }
        }
    }
}

static void fill_passive(Child *c, i64 now, i64 qty, i64 seq) {
    Run *r = c->run;
    record(r, now, c->id, c->price, qty, MAKER, cfg.maker_fee * qty, cfg.adverse * qty, seq);
    c->filled += qty;
    note_own(r, now, qty);
    if (c->lots - c->filled == 0) finish_child(c);
}

static void historical_trade(const Event *e) {
    int s = SIDE_INDEX(e->side);
    static Child *snapshot[MAX_ACTIVE];
    int n = active_count[s];
    memcpy(snapshot, active[s], sizeof(Child *) * (size_t)n);
    for (int i = 0; i < n; i++) {
        Child *c = snapshot[i];
        i64 improvement = (c->price - e->price) * e->side;
        if (improvement < 0) continue;
        Run *r = c->run;
        i64 passing = e->qty;
        if (improvement > 0) c->ahead = 0;
        else if (cfg.trade_through_only) continue;
        else if (e->kind == TRADE_HIDDEN) { if (c->ahead > 0) continue; }
        else { i64 burned = min64(c->ahead, passing); c->ahead -= burned; passing -= burned; }
        if (r->budget_stamp != feed_index) { r->budget_stamp = feed_index; r->budget = e->qty; }
        i64 qty = min64(min64(passing, c->lots - c->filled), min64(r->budget, r->target - r->filled));
        if (qty > 0) qty = capped(r, e->ts_ns, qty);
        if (qty > 0) { r->budget -= qty; fill_passive(c, e->ts_ns, qty, e->seq); }
    }
}

static void cancel_ahead(Order *o, i64 qty, int gone) {
    i64 credit = qty * cfg.credit_pct / 100;
    for (Watch *w = o->watchers; w; w = w->next) w->child->ahead = max64(0, w->child->ahead - credit);
    if (gone) drop_watchers(o);
}

static void modify_ahead(const Event *e) {
    Order *o = find_order(e->order_id);
    if (!o || !o->watchers) return;
    if (loses_priority(e, o)) cancel_ahead(o, o->qty, 1);
    else if (e->qty < o->qty) cancel_ahead(o, o->qty - e->qty, 0);
    else for (Watch *w = o->watchers; w; w = w->next) w->child->ahead += e->qty - o->qty;
}

static void crossing_add(const Event *e) {
    int s = SIDE_INDEX(-e->side);
    static Child *snapshot[MAX_ACTIVE];
    int n = active_count[s];
    memcpy(snapshot, active[s], sizeof(Child *) * (size_t)n);
    for (int i = 0; i < n; i++) {
        Child *c = snapshot[i];
        Run *r = c->run;
        if ((c->price - e->price) * c->side < 0) continue;
        c->ahead = 0;
        i64 used = consumed_of(r, e->order_id);
        i64 qty = min64(min64(e->qty - used, c->lots - c->filled), r->target - r->filled);
        if (qty > 0) qty = capped(r, e->ts_ns, qty);
        if (qty > 0) { consumed_set(r, e->order_id, used + qty); fill_passive(c, e->ts_ns, qty, e->seq); }
    }
}

static void on_feed(const Event *e) {
    int kind = e->kind;
    if (kind == EXECUTE || kind == TRADE_HIDDEN) {
        if (cfg.cap_pct >= 0) { ring_push(&cap_tape, e->ts_ns, e->qty); cap_tape_lots += e->qty; }
        if (active_count[SIDE_INDEX(e->side)]) historical_trade(e);
    } else if (kind == CANCEL) {
        Order *o = find_order(e->order_id);
        if (o && o->watchers) cancel_ahead(o, e->qty, e->qty == o->qty);
    } else if (kind == MODIFY) {
        modify_ahead(e);
    } else if (kind == ADD && active_count[SIDE_INDEX(-e->side)]) {
        crossing_add(e);
    }
}

static void exclude_run(Run *r, const char *reason) {
    r->excluded = reason;
    r->finished = 1;
    for (int i = 0; i < r->child_count; i++) if (!r->children[i]->finished) finish_child(r->children[i]);
    for (int i = 0; i < live_count; i++) {
        if (live_runs[i] == r) { memmove(&live_runs[i], &live_runs[i + 1], sizeof(Run *) * (size_t)(live_count - i - 1)); live_count--; break; }
    }
}

static void after_feed(const Event *e) {
    if (e->kind == HALT_START) halted = 1; else if (e->kind == HALT_END) halted = 0;
    if (live_count) {
        if (has_best(BID)) { last_best[0] = best(BID); have_last[0] = 1; }
        if (has_best(ASK)) { last_best[1] = best(ASK); have_last[1] = 1; }
    }
    if ((halted || !valid) && live_count) {
        const char *reason = halted ? "halt" : "invalid book";
        while (live_count) exclude_run(live_runs[0], reason);
    }
}

/* Scheduled actions ------------------------------------------------------------- */

enum { DO_ACTIVATE, DO_ARRIVE, DO_CANCEL_LANDS, DO_DEADLINE };
typedef struct { i64 ts; i64 ticket; int what; void *target; } Action;
static Action *heap; static size_t heap_len, heap_cap; static i64 next_ticket;

static inline int action_before(const Action *a, const Action *b) {
    return a->ts != b->ts ? a->ts < b->ts : a->ticket < b->ticket;
}

static void schedule(i64 ts, int what, void *target) {
    if (heap_len == heap_cap) { heap_cap = heap_cap ? heap_cap * 2 : 1024; heap = realloc(heap, heap_cap * sizeof(Action)); }
    size_t i = heap_len++;
    Action a = {ts, next_ticket++, what, target};
    while (i > 0) {
        size_t up = (i - 1) / 2;
        if (!action_before(&a, &heap[up])) break;
        heap[i] = heap[up]; i = up;
    }
    heap[i] = a;
}

static Action pop_action(void) {
    Action top = heap[0], last = heap[--heap_len];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = l;
        if (l >= heap_len) break;
        if (r < heap_len && action_before(&heap[r], &heap[l])) m = r;
        if (!action_before(&heap[m], &last)) break;
        heap[i] = heap[m]; i = m;
    }
    heap[i] = last;
    return top;
}

static i64 available_at(const Run *r, const Level *l, i64 wanted) {
    i64 available = 0;
    for (const Order *o = l->head; o; o = o->next) {
        available += max64(0, o->qty - consumed_of(r, o->id));
        if (available >= wanted) return wanted;
    }
    return available;
}

static void commit_consumption(Run *r, const Level *l, i64 qty) {
    for (const Order *o = l->head; o && qty; o = o->next) {
        i64 used = consumed_of(r, o->id), take = min64(qty, max64(0, o->qty - used));
        if (take) { consumed_set(r, o->id, used + take); qty -= take; }
    }
}

static void take_liquidity(i64 now, Child *c) {
    Run *r = c->run;
    int side = -c->side, s = SIDE_INDEX(side);
    i64 wanted = min64(c->lots, r->target - r->filled);
    /* The depth list is fixed before fills, as in Python. Fills never change the book. */
    int top = level_count[s] - 1;
    for (int k = 0; k < cfg.take_levels && top - k >= 0; k++) {
        const Level *l = sorted[s][top - k];
        if (wanted == 0 || (l->price - c->price) * c->side > 0) break;
        i64 qty = available_at(r, l, wanted);
        if (qty) qty = capped(r, now, qty);
        if (qty == 0) continue;
        commit_consumption(r, l, qty);
        record(r, now, c->id, l->price, qty, TAKER, cfg.taker_fee * qty, 0, 0);
        c->filled += qty;
        wanted -= qty;
        note_own(r, now, qty);
    }
}

static void join_queue(Child *c) {
    Level *l = find_level(c->side, c->price);
    i64 ahead = 0;
    if (l) {
        for (Order *o = l->head; o; o = o->next) {
            ahead += o->qty;
            Watch *w = free_watches ? free_watches : malloc(sizeof(Watch));
            if (w == free_watches) free_watches = w->next;
            w->child = c; w->next = o->watchers; o->watchers = w;
        }
    }
    Run *r = c->run;
    for (int i = 0; i < r->child_count; i++) {
        Child *other = r->children[i];
        if (other != c && other->live && other->price == c->price) ahead += other->lots - other->filled;
    }
    c->ahead = ahead * cfg.ahead_pct / 100;
    c->live = 1;
    int s = SIDE_INDEX(c->side);
    if (active_count[s] == MAX_ACTIVE) { fprintf(stderr, "too many active children\n"); exit(2); }
    active[s][active_count[s]++] = c;
}

static void arrive(i64 now, Child *c) {
    if (c->run->finished || c->finished) return;
    if (c->immediate) { take_liquidity(now, c); finish_child(c); return; }
    if (has_best(-c->side) && (c->price - best(-c->side)) * c->side >= 0) { finish_child(c); return; }
    join_queue(c);
}

static void force(i64 now, Run *r, i64 price, i64 qty) {
    record(r, now, 0, price, qty, FORCED, cfg.taker_fee * qty, cfg.penalty_ticks * MTICKS_PER_TICK * qty, 0);
}

static void deadline(i64 now, Run *r) {
    if (r->finished) return;
    r->finished = 1;
    for (int i = 0; i < r->child_count; i++) if (!r->children[i]->finished) finish_child(r->children[i]);
    int side = -r->side, s = SIDE_INDEX(side), have_worst = 0;
    i64 worst = 0;
    for (int k = level_count[s] - 1; k >= 0 && r->target - r->filled > 0; k--) {
        const Level *l = sorted[s][k];
        i64 qty = available_at(r, l, r->target - r->filled);
        worst = l->price; have_worst = 1;
        if (qty) { commit_consumption(r, l, qty); force(now, r, l->price, qty); }
    }
    if (r->target - r->filled > 0) {
        i64 base = have_worst ? worst : have_last[s] ? last_best[s] : floor_div(r->mid_x2 + 1, 2);
        force(now, r, base + r->side * cfg.extra_ticks, r->target - r->filled);
    }
    for (int i = 0; i < live_count; i++) {
        if (live_runs[i] == r) { memmove(&live_runs[i], &live_runs[i + 1], sizeof(Run *) * (size_t)(live_count - i - 1)); live_count--; break; }
    }
}

static void activate(i64 now, Run *r) {
    r->activated = 1;
    if (!valid || halted) { r->excluded = halted ? "halt" : "invalid book"; r->finished = 1; return; }
    if (!has_best(BID) || !has_best(ASK)) { r->excluded = "one-sided book at arrival"; r->finished = 1; return; }
    r->mid_x2 = best(BID) + best(ASK);
    last_best[0] = best(BID); last_best[1] = best(ASK); have_last[0] = have_last[1] = 1;
    if (live_count == MAX_ACTIVE) { fprintf(stderr, "too many live parents\n"); exit(2); }
    live_runs[live_count++] = r;
    (void)now;
}

static void run_until(i64 ts) {
    while (heap_len && heap[0].ts < ts) {
        Action a = pop_action();
        switch (a.what) {
            case DO_ACTIVATE: activate(a.ts, a.target); break;
            case DO_ARRIVE: arrive(a.ts, a.target); break;
            case DO_CANCEL_LANDS: { Child *c = a.target; if (!c->finished) finish_child(c); break; }
            case DO_DEADLINE: deadline(a.ts, a.target); break;
        }
    }
}

/* Entry points ------------------------------------------------------------------- */

static Event *load_events(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    Event *events = malloc((size_t)bytes);
    if (fread(events, 1, (size_t)bytes, f) != (size_t)bytes) { fprintf(stderr, "short read\n"); exit(2); }
    fclose(f);
    *count = (size_t)bytes / sizeof(Event);
    return events;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* FNV-1a over the best prices and touch sizes after every event. */
static uint64_t mix(uint64_t h, i64 v) { h ^= (uint64_t)v; return h * 1099511628211ull; }

static int book_mode(const char *path, int repeats) {
    size_t n;
    Event *events = load_events(path, &n);
    uint64_t hash = 14695981039346656037ull;
    double start = seconds();
    for (int r = 0; r < repeats; r++) {
        hash = 14695981039346656037ull;
        have_seq = 0; valid = 0; rebuilding = 0;
        for (size_t i = 0; i < n; i++) {
            guard_apply(&events[i]);
            hash = mix(hash, has_best(BID) ? best(BID) : -1);
            hash = mix(hash, has_best(ASK) ? best(ASK) : -1);
            hash = mix(hash, has_best(BID) ? level_total(sorted[0][level_count[0] - 1]) : 0);
            hash = mix(hash, has_best(ASK) ? level_total(sorted[1][level_count[1] - 1]) : 0);
            hash = mix(hash, valid);
        }
    }
    double elapsed = seconds() - start;
    printf("events %zu repeats %d seconds %.6f checksum %016" PRIx64 "\n", n, repeats, elapsed, hash);
    return 0;
}

static void set_config(const char *key, i64 v) {
    if (!strcmp(key, "taker_fee")) cfg.taker_fee = v;
    else if (!strcmp(key, "maker_fee")) cfg.maker_fee = v;
    else if (!strcmp(key, "credit_pct")) cfg.credit_pct = v;
    else if (!strcmp(key, "ahead_pct")) cfg.ahead_pct = v;
    else if (!strcmp(key, "adverse")) cfg.adverse = v;
    else if (!strcmp(key, "cap_pct")) cfg.cap_pct = v;
    else if (!strcmp(key, "cap_window")) cfg.cap_window = v;
    else if (!strcmp(key, "free_lots")) cfg.free_lots = v;
    else if (!strcmp(key, "penalty_ticks")) cfg.penalty_ticks = v;
    else if (!strcmp(key, "extra_ticks")) cfg.extra_ticks = v;
    else if (!strcmp(key, "take_levels")) cfg.take_levels = v;
    else if (!strcmp(key, "trade_through_only")) cfg.trade_through_only = (int)v;
    else if (!strcmp(key, "size_up_loses")) size_up_loses = (int)v;
    else if (!strcmp(key, "size_down_loses")) size_down_loses = (int)v;
    else if (!strcmp(key, "order_latency") || !strcmp(key, "cancel_latency")) { }
    else { fprintf(stderr, "unknown option %s\n", key); exit(2); }
}

static i64 order_latency, cancel_latency;

static Child *child_with_id(Run *r, i64 id) {
    for (int i = r->child_count - 1; i >= 0; i--) if (r->children[i]->id == id) return r->children[i];
    fprintf(stderr, "unknown child %" PRId64 "\n", id);
    exit(2);
}

static void read_tape(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(2); }
    char tag;
    if (fscanf(f, " R %d", &run_count) != 1) { fprintf(stderr, "bad tape\n"); exit(2); }
    runs = calloc((size_t)run_count, sizeof(Run));
    while (fscanf(f, " %c", &tag) == 1) {
        int index; i64 ts;
        if (tag == 'P') {
            int side; i64 target, start, end;
            if (fscanf(f, "%d %d %" SCNd64 " %" SCNd64 " %" SCNd64, &index, &side, &target, &start, &end) != 5) exit(2);
            Run *r = &runs[index];
            r->index = index; r->side = side; r->target = target; r->start = start; r->deadline = end;
            r->budget_stamp = -1;
            schedule(start, DO_ACTIVATE, r);
            /* Every P line precedes every S line, so at equal times the deadline runs
             * before an arrival, as it does in Python. */
            schedule(end, DO_DEADLINE, r);
        } else if (tag == 'S') {
            i64 id, price, lots; int immediate;
            if (fscanf(f, "%d %" SCNd64 " %" SCNd64 " %" SCNd64 " %" SCNd64 " %d", &index, &ts, &id, &price, &lots, &immediate) != 6) exit(2);
            Run *r = &runs[index];
            Child *c = calloc(1, sizeof(Child));
            c->id = id; c->run = r; c->side = r->side; c->price = price; c->lots = lots; c->immediate = immediate;
            if (r->child_count == r->child_cap) { r->child_cap = r->child_cap ? r->child_cap * 2 : 16; r->children = realloc(r->children, sizeof(Child *) * (size_t)r->child_cap); }
            r->children[r->child_count++] = c;
            schedule(ts + order_latency, DO_ARRIVE, c);
        } else if (tag == 'C') {
            i64 id;
            if (fscanf(f, "%d %" SCNd64 " %" SCNd64, &index, &ts, &id) != 3) exit(2);
            schedule(ts + cancel_latency, DO_CANCEL_LANDS, child_with_id(&runs[index], id));
        } else { fprintf(stderr, "bad tape tag %c\n", tag); exit(2); }
    }
    fclose(f);
}

static int exec_mode(int argc, char **argv) {
    size_t n;
    Event *events = load_events(argv[2], &n);
    for (int i = 4; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) { fprintf(stderr, "bad option %s\n", argv[i]); return 2; }
        *eq = 0;
        i64 v = strtoll(eq + 1, NULL, 10);
        if (!strcmp(argv[i], "order_latency")) order_latency = v;
        if (!strcmp(argv[i], "cancel_latency")) cancel_latency = v;
        set_config(argv[i], v);
    }
    read_tape(argv[3]);
    double start = seconds();
    for (size_t i = 0; i < n; i++) {
        const Event *e = &events[i];
        feed_index = (i64)i;
        if (heap_len && heap[0].ts < e->ts_ns) run_until(e->ts_ns);
        on_feed(e);
        guard_apply(e);
        after_feed(e);
    }
    run_until(INT64_MAX);
    double elapsed = seconds() - start;
    for (int i = 0; i < run_count; i++) {
        Run *r = &runs[i];
        if (r->excluded) printf("E %d %s\n", i, r->excluded);
        else printf("L %d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "\n", i, r->filled, r->notional, r->fees, r->penalties, r->cash);
    }
    fprintf(stderr, "events %zu seconds %.6f\n", n, elapsed);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: replay book|exec <events.bin> ...\n"); return 2; }
    pool_size = 1u << 20;
    order_pool = malloc(pool_size * sizeof(Order));
    if (!strcmp(argv[1], "book")) return book_mode(argv[2], argc > 3 ? atoi(argv[3]) : 1);
    if (!strcmp(argv[1], "exec") && argc >= 4) return exec_mode(argc, argv);
    fprintf(stderr, "unknown mode\n");
    return 2;
}
