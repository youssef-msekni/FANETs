"""PDR (default) or max delay vs DATA interval (fixed-count scenario).

The Source sends exactly --num-packets (default 1000) DATA packets for each
transmission interval (default 10,20,30,40,50 ms). Every (interval, run) pair
runs both systems (LLM+RL and Q-FANET) on identical traffic/mobility/links.
Error bars = standard deviation over --runs independent runs.

    python data_interval_study.py --runs 5 --backend ollama --uavs 10
    python data_interval_study.py --intervals 20 --runs 1      # single 20 ms scenario
"""
import argparse, json, os, random, statistics, sys, time

import main as sim

OUT_JSON = "data_interval_results.json"
OUT_PNG = os.path.join("charts", "pdr_vs_data_interval.png")


def _out(msg):
    # run_simulation(verbose=False) silences print(); write to the real stdout.
    sys.__stdout__.write(msg)
    sys.__stdout__.flush()


def run_one(interval_ms, run_idx, args):
    seed = 10_000 * (run_idx + 1) + int(interval_ms)
    t0 = time.time()

    def progress(step, steps):
        # live progress + ETA for this run (one line, updated in place)
        el = time.time() - t0
        eta = el / max(step, 1) * (steps - step)
        _out(f"\r    tick {step}/{steps} | elapsed {el/60:5.1f} min | "
             f"ETA this run {eta/60:5.1f} min      ")
    result = sim.run_simulation(
        num_uavs=args.uavs,
        verbose=args.verbose,
        persist_q_table=False,
        export_json=False,
        force_backend=None if args.backend == "mixed" else args.backend,
        packet_rate_pps=1000.0 / interval_ms,
        relay_capacity_pps=args.relay_capacity_pps,
        mean_speed_mps=args.speed_mps,
        num_packets=args.num_packets,
        entry_uavs=args.entry_uavs,
        battery_drain_pct=args.battery_drain_pct,
        time_profile=args.time_profile, tick_ms=args.tick_ms,
        deadlines_ms=[float(x) for x in args.deadlines_ms.split(",")] if args.deadlines_ms else None,
        llm_ttl_ms=args.llm_ttl_ms,
        seed=seed,
        progress_callback=progress,
    )
    _out("\n")
    s = result["summary"] if isinstance(result, dict) and "summary" in result else None
    return s


def plot(agg, path, metric="pdr", series_label="LLM+RL"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = sorted(agg, key=float)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    style = {
        "llm": (series_label, dict(color="blue", linestyle="-")),
        "qfanet": ("Q-FANET", dict(color="green", linestyle="-.")),
    }
    for key, (label, kw) in style.items():
        field = key if metric == "delay" else key + "_pdr"
        scale = 1.0 if metric == "delay" else 0.01   # PDR shown as a ratio in [0,1]
        ys = [agg[x][field]["mean"] * scale for x in xs]
        es = [agg[x][field]["std"] * scale for x in xs]
        ax.errorbar([float(x) for x in xs], ys, yerr=es, label=label,
                    linewidth=2, capsize=3, elinewidth=2, **kw)
    ax.set_xlabel("Data interval (ms)")
    ax.set_ylabel("Max end-to-end delay (ms)" if metric == "delay"
                  else "Packet delivery ratio")
    ax.set_xlim(min(map(float, xs)) - 1, max(map(float, xs)) + 1)
    if metric == "delay":
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(top=1.0)
    lo, hi = int(min(map(float, xs))), int(max(map(float, xs)))
    ax.set_xticks(list(range(lo, hi + 1, 5)))      # 10, 15, 20, 25, ...
    ax.legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _stat(v):
    return {"mean": statistics.fmean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0}


def _save_partial(raw, args):
    """Save JSON + figure after every finished run, so Ctrl+C loses nothing."""
    done = {k: v for k, v in raw.items() if v["llm"]}
    agg = {k: {"llm": _stat(v["llm"]), "qfanet": _stat(v["qfanet"]),
               "llm_pdr": _stat(v["llm_pdr"]), "qfanet_pdr": _stat(v["qfanet_pdr"])}
           for k, v in done.items()}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"series_label": args.series_label, "num_packets": args.num_packets,
                   "runs": args.runs, "uavs": args.uavs, "entry_uavs": args.entry_uavs,
                   "battery_drain_pct": args.battery_drain_pct, "raw": done, "aggregate": agg}, f, indent=2)
    plot(agg, args.output_png, args.metric, args.series_label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", type=float, nargs="+", default=[10, 15, 20, 25, 30, 35, 40, 45, 50])
    ap.add_argument("--num-packets", type=int, default=1000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--uavs", type=int, default=10)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "groq", "mixed"])
    ap.add_argument("--time-profile", default="legacy", choices=["legacy", "article"],
                    help="legacy: 250 ms tick / 1-3 s deadlines; article: 10 ms tick / 100-300 ms deadlines")
    ap.add_argument("--tick-ms", type=float, default=None)
    ap.add_argument("--deadlines-ms", default=None, help="e.g. 100,150,200,250,300")
    ap.add_argument("--llm-ttl-ms", type=float, default=None,
                    help="reuse LLM guidance for this many ms (0/None = every decision)")
    ap.add_argument("--entry-uavs", type=int, default=1,
                    help="Source injects into the k nearest UAVs (round-robin)")
    ap.add_argument("--battery-drain-pct", default=None,
                    help="number (%% per tick) or 'scaled'; default = original 3 %%/tick")
    ap.add_argument("--relay-capacity-pps", type=float, default=sim.DEFAULT_RELAY_CAPACITY_PPS)
    ap.add_argument("--speed-mps", type=float, default=sim.DEFAULT_MEAN_SPEED_MPS)
    ap.add_argument("--metric", default="pdr", choices=["pdr", "delay"],
                    help="y-axis of the figure (both are always saved in the JSON)")
    ap.add_argument("--series-label", default="LLM+RL",
                    help="legend label of the proposed system, e.g. 'RL + prior (no LLM)' for ablation runs")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--output-json", default=OUT_JSON)
    ap.add_argument("--output-png", default=OUT_PNG)
    args = ap.parse_args()

    raw = {}
    total = len(args.intervals) * args.runs
    _out(f"{len(args.intervals)} interval(s) x {args.runs} run(s) = {total} simulations, "
         f"{args.num_packets} packets each, {args.uavs} UAVs, backend={args.backend}\n"
         "Each decision calls the LLM (2-4 calls with ReAct), so a single run can take a long time.\n"
         "Tip: first test with  --intervals 20 --runs 1 --uavs 10\n")
    done = 0
    t_start = time.time()
    for interval in args.intervals:
        key = f"{interval:g}"
        raw[key] = {"llm": [], "qfanet": [], "llm_pdr": [], "qfanet_pdr": []}
        for r in range(args.runs):
            _out(f"[{done+1}/{total}] interval={key} ms, run {r+1}/{args.runs}\n")
            s = run_one(interval, r, args)
            done += 1
            raw[key]["llm"].append(s["llm"]["max_e2e_delay_ms"])
            raw[key]["qfanet"].append(s["qfanet"]["max_e2e_delay_ms"])
            raw[key]["llm_pdr"].append(s["llm"]["pdr_pct"])
            raw[key]["qfanet_pdr"].append(s["qfanet"]["pdr_pct"])
            sys.__stdout__.write(
                f"interval={key}ms run={r+1}/{args.runs} "
                f"PDR {args.series_label}={raw[key]['llm_pdr'][-1]:.1f}% Q-FANET={raw[key]['qfanet_pdr'][-1]:.1f}% "
                f"(max delay {raw[key]['llm'][-1]:.0f}/{raw[key]['qfanet'][-1]:.0f} ms)\n")
            sys.__stdout__.flush()
            _save_partial(raw, args)

    _save_partial(raw, args)   # final JSON + figure (same code path as partial saves)
    print(f"saved {args.output_json} and {args.output_png}")


if __name__ == "__main__":
    main()