"""Reports and the page must word results by the evidence, never by hope."""

from marketreplay.demo import choose_scenes, compact, verdict_sentence
from marketreplay.evaluate import Row, read_rows, write_rows
from marketreplay.report import comparison_table, ordered_scenarios, reversals, strategy_table
from marketreplay.smoke import run_smoke
from marketreplay.tracing import format_entry


def entry(low, high, changes=False, cheapest="learned"):
    return {
        "learned_vs_baseline": {"mean_diff_bps": (low + high) / 2, "ci_low": low, "ci_high": high},
        "conclusion_changes": changes,
        "cheapest": cheapest,
        "summaries": {cheapest: {"mean_bps": 1.0}},
    }


STRESS = {
    "scenarios": {
        "base": entry(-0.1, -0.01),
        "kept": entry(-0.2, -0.1),
        "lost": entry(-0.1, 0.1, changes=True),
        "flipped": entry(0.01, 0.05, changes=True, cheapest="passive_only"),
    }
}


def test_the_base_case_leads_the_stress_table():
    assert ordered_scenarios(STRESS) == ["base", "flipped", "kept", "lost"]


def test_reversals_keep_a_flip_apart_from_a_lost_interval():
    text = reversals(STRESS)
    assert "interval above zero, in 1 of 3 moved scenarios: flipped." in text
    assert "reaches zero in 1 more, so the evidence for a difference is lost there: lost." in text
    assert "It loses that place in 1 scenarios: flipped." in text


def summary(mean):
    return {
        "parents": 10, "mean_bps": mean, "median_bps": mean, "p95_bps": mean, "voluntary_completion_pct": 100.0,
        "forced_lot_pct": 0.0, "maker_lot_pct": 50.0,
    }  # fmt: skip


def test_the_strategy_table_runs_cheapest_first():
    table = strategy_table({"immediate": summary(2.0), "learned": summary(-1.0)})
    assert table.index("Learned policy") < table.index("Immediate cross")


def test_the_comparison_table_prints_each_interval():
    row = {"b": "urgency", "mean_diff_bps": -0.0635, "ci_low": -0.11, "ci_high": -0.014, "days": 34}
    assert "| Urgency rule | -0.064 | -0.110 to -0.014 | 34 |" in comparison_table([row])


def test_a_failed_rule_is_called_negative_on_the_page():
    verdict = {
        "improved": False,
        "paired": {"mean_diff_bps": 0.1, "ci_low": -0.1, "ci_high": 0.3, "days": 34},
        "checks": {"mean_cost_lower": False, "tail_not_worse": True},
    }
    sentence = verdict_sentence(verdict, "urgency")
    assert "did not pass" in sentence and "mean cost lower" in sentence


def test_compact_numbers_keep_one_decimal():
    assert [compact(11_648_571), compact(100_000), compact(0)] == ["11.6M", "100.0K", "0"]


def row(day, strategy, cost):
    return Row(day, 1, 1, strategy, "", cost, 0, 0, 0, 0, 0, 1.0)


def test_scenes_come_from_the_rule_and_not_from_taste():
    rows = []
    for day, learned in enumerate([-3.0, -1.0, 0.0, 0.5, 2.0]):
        rows += [row(day, "learned", learned), row(day, "urgency", 0.0)]
    rows += [Row(9, 1, 1, "learned", "halt", float("nan"), 0, 0, 0, 0, 0, float("nan"))]
    typical, best, worst = choose_scenes(rows, "urgency")
    assert best["key"] == (0, 1)
    assert worst["key"] == (4, 1)
    assert typical["key"] in {(1, 1), (3, 1)}  # Day 2 had no difference, so it cannot be typical.
    assert "80%" in typical["rule"]


def test_rows_survive_a_trip_through_csv(tmp_path):
    rows = [row(1, "learned", -0.125), Row(2, 3, -1, "urgency", "halt", float("nan"), 0, 0, 0, 0, 0, float("nan"))]
    write_rows(rows, tmp_path / "rows.csv")
    back = read_rows(tmp_path / "rows.csv")
    assert back[0] == rows[0]
    assert back[1].excluded == "halt" and back[1].side == -1


def test_trace_lines_show_seconds_since_the_parent_began():
    line = format_entry(1_000_000_000, {"ts_ns": 3_500_000_000, "what": "fill", "qty": 4})
    assert line.strip().startswith("2.500s") and "fill" in line and "'qty': 4" in line


def test_the_smoke_run_passes(capsys):
    run_smoke()
    assert "smoke passed" in capsys.readouterr().out
