"""Command line entry point. Run `uv run mreplay --help`."""

import argparse
import json
import warnings

# Apple's Accelerate sets false floating point flags on this chip during matrix products.
# Results matched a pure Python calculation to 2e-15 with no NaNs, see docs/assumptions.md.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def main() -> None:
    parser = argparse.ArgumentParser(prog="mreplay", description="Event replay and execution research. All orders are simulated.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in [
        ("smoke", "one-minute check: hand trace, baselines and compiled parity"),
        ("generate", "write the seeded synthetic study data"),
        ("fit", "fit models and schedules on training days"),
        ("tune", "pick settings on validation days"),
        ("freeze", "write the frozen hypothesis and decision rule"),
        ("final", "evaluate once on test days, after a committed freeze"),
        ("stress", "evaluate frozen strategies under moved assumptions"),
        ("profile", "measure speed and memory, Python against compiled"),
        ("report", "write docs/report.md and the plots from saved results"),
        ("demo", "write the offline website demo from a recorded run"),
        ("benchmark", "run final, stress, profile, report and demo in order"),
    ]:
        commands.add_parser(name, help=text)
    fuzz = commands.add_parser("fuzz", help="differential fuzz of C against Python under sanitizers")
    fuzz.add_argument("--cases", type=int, default=100_000)
    trace = commands.add_parser("trace", help="print one parent's full event trace")
    trace.add_argument("--day", type=int, required=True)
    trace.add_argument("--parent", type=int, required=True, help="index of the parent within the day")
    trace.add_argument("--strategy", default="learned")
    args = parser.parse_args()

    # Imports stay here so `--help` is instant.
    from .pipeline import final, fit, freeze, stress, tune
    from .study import Study, ensure_data

    study = Study()
    if args.command == "smoke":
        from .smoke import run_smoke

        run_smoke()
    elif args.command == "fuzz":
        from .fuzz import run_fuzz
        from .pipeline import RESULTS

        print(json.dumps(run_fuzz(args.cases, RESULTS / "fuzz.json"), indent=2))
    elif args.command == "generate":
        index = ensure_data(study.plan)
        print(f"{index['events']:,} events over {study.plan.days} days. Input hash {index['sha256'][:16]}")
    elif args.command == "fit":
        ensure_data(study.plan)
        report = fit(study)
        print(f"Winner {report['fill_winner']}. Saved results/fit.json")
    elif args.command == "tune":
        report = tune(study)
        print(f"Chose {report['chosen_policy']} and {report['chosen_urgency']}. Best baseline {report['best_baseline']}")
    elif args.command == "freeze":
        freeze(study)
        print("Saved results/freeze.json. Commit it, then run `mreplay final`.")
    elif args.command == "final":
        report = final(study)
        print(json.dumps(report["verdict"], indent=2))
    elif args.command == "stress":
        report = stress(study)
        print(f"Saved results/stress.json with {len(report['scenarios'])} scenarios")
    elif args.command == "profile":
        from .profile import profile

        print(json.dumps(profile(study), indent=2))
    elif args.command == "report":
        from .report import write_report

        print(write_report(study))
    elif args.command == "demo":
        from .demo import write_demo

        print(write_demo(study))
    elif args.command == "trace":
        from .tracing import print_trace

        print_trace(study, args.day, args.parent, args.strategy)
    elif args.command == "benchmark":
        from .demo import write_demo
        from .profile import profile
        from .report import write_report

        final(study)
        stress(study)
        profile(study)
        print(write_report(study))
        print(write_demo(study))
