"""Ablation: what does each part of the LLM+RL design add?  (same scenario, same seeds)

Configurations (all run in the same process, each on seeds 1..R):
  noLLM        RL + topology prior only                (LLMRL_LLM=0)
  single_call  1-call LLM scorer, no ReAct, no shaping (LLMRL_REACT=0, LLMRL_SHAPING=0)
  react        ReAct+CoT guidance, no reward shaping   (LLMRL_SHAPING=0)
  react_shaped ReAct+CoT guidance + bounded shaping    (full proposed model)
Q-FANET is run inside every simulation, so it is reported once as the pooled baseline.

    python ablation_study.py --interval 20 --num-packets 300 --runs 10 --uavs 25 \
        --entry-uavs 3 --battery-drain-pct scaled --backend ollama
Results: ablation_results.json and charts/ablation_pdr.png.
"""
import argparse, json, os, random, statistics, sys, time

import main as sim

CONFIGS = {
    "noLLM":        dict(llm=False, react=False, shaping=False),
    "single_call":  dict(llm=True,  react=False, shaping=False),
    "react":        dict(llm=True,  react=True,  shaping=False),
    "react_shaped": dict(llm=True,  react=True,  shaping=True),
}
LABELS = {"noLLM": "RL + prior\n(no LLM)", "single_call": "LLM single call",
          "react": "ReAct\n(no shaping)", "react_shaped": "ReAct + shaping\n(proposed)"}


def _out(m):
    sys.__stdout__.write(m); sys.__stdout__.flush()


def stat(v):
    return {"mean": statistics.fmean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0, "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--interval", type=float, default=20.0, help="data interval in ms")
    ap.add_argument("--num-packets", type=int, default=300)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--uavs", type=int, default=25)
    ap.add_argument("--time-profile", default="legacy", choices=["legacy", "article"],
                    help="legacy: 250 ms tick / 1-3 s deadlines; article: 10 ms tick / 100-300 ms deadlines")
    ap.add_argument("--tick-ms", type=float, default=None)
    ap.add_argument("--deadlines-ms", default=None, help="e.g. 100,150,200,250,300")
    ap.add_argument("--llm-ttl-ms", type=float, default=None,
                    help="reuse LLM guidance for this many ms (0/None = every decision)")
    ap.add_argument("--entry-uavs", type=int, default=1)
    ap.add_argument("--battery-drain-pct", default=None)
    ap.add_argument("--relay-capacity-pps", type=float, default=sim.DEFAULT_RELAY_CAPACITY_PPS)
    ap.add_argument("--speed-mps", type=float, default=sim.DEFAULT_MEAN_SPEED_MPS)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "groq", "mixed"])
    ap.add_argument("--output-json", default="ablation_results.json")
    ap.add_argument("--output-png", default=os.path.join("charts", "ablation_pdr.png"))
    a = ap.parse_args()

    raw = {c: {"llm_pdr": [], "qf_pdr": [], "llm_drops": [], "qf_drops": []} for c in a.configs}
    total = len(a.configs) * a.runs
    k = 0
    for cfg in a.configs:
        c = CONFIGS[cfg]
        sim.LLMRL_LLM_ENABLED = c["llm"]
        sim.LLMRL_REACT_ENABLED = c["react"]
        sim.LLMRL_SHAPING_ENABLED = c["shaping"]
        for r in range(a.runs):
            k += 1
            _out(f"[{k}/{total}] {cfg} run {r+1}/{a.runs} ... ")
            seed = 1000 + r            # same seed list for every configuration
            t0 = time.time()
            res = sim.run_simulation(
                num_uavs=a.uavs, verbose=True, persist_q_table=False, export_json=False,
                force_backend=None if a.backend == "mixed" else a.backend,
                packet_rate_pps=1000.0 / a.interval, num_packets=a.num_packets,
                relay_capacity_pps=a.relay_capacity_pps, mean_speed_mps=a.speed_mps,
                entry_uavs=a.entry_uavs, battery_drain_pct=a.battery_drain_pct,
                time_profile=a.time_profile, tick_ms=a.tick_ms,
                deadlines_ms=[float(x) for x in a.deadlines_ms.split(",")] if a.deadlines_ms else None,
                llm_ttl_ms=a.llm_ttl_ms, seed=seed)
            s = res["summary"]
            raw[cfg]["llm_pdr"].append(s["llm"]["pdr_pct"])
            raw[cfg]["qf_pdr"].append(s["qfanet"]["pdr_pct"])
            raw[cfg]["llm_drops"].append(s["llm"]["dropped_deadline"])
            raw[cfg]["qf_drops"].append(s["qfanet"]["dropped_deadline"])
            _out(f"PDR proposed-side={s['llm']['pdr_pct']:.1f}%  Q-FANET={s['qfanet']['pdr_pct']:.1f}%  "
                 f"({time.time()-t0:.0f}s)\n")
            save(raw, a)
    save(raw, a)
    _out(f"saved {a.output_json} and {a.output_png}\n")


def save(raw, a):
    done = {c: v for c, v in raw.items() if v["llm_pdr"]}
    agg = {c: {"pdr": stat(v["llm_pdr"]), "drops": stat(v["llm_drops"])} for c, v in done.items()}
    qf_all = [x for v in done.values() for x in v["qf_pdr"]]
    qf = {"pdr": stat(qf_all), "drops": stat([x for v in done.values() for x in v["qf_drops"]])} if qf_all else None
    # paired difference vs Q-FANET inside the same simulation (same traffic/mobility/links)
    paired = {c: stat([l - q for l, q in zip(v["llm_pdr"], v["qf_pdr"])]) for c, v in done.items()}
    with open(a.output_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "raw": raw, "aggregate": agg, "qfanet": qf,
                   "paired_diff_pdr_vs_qfanet": paired}, f, indent=2)
    plot(agg, qf, paired, a)


def plot(agg, qf, paired, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Q-FANET"] + [LABELS[c] for c in agg]
    means = ([qf["pdr"]["mean"]] if qf else [0]) + [agg[c]["pdr"]["mean"] for c in agg]
    stds = ([qf["pdr"]["std"]] if qf else [0]) + [agg[c]["pdr"]["std"] for c in agg]
    colors = ["#d62728"] + ["#1f77b4"] * len(agg)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(range(len(names)), means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Packet delivery ratio (%)")
    n = max((agg[c]["pdr"]["n"] for c in agg), default=0)
    ax.set_title(f"Ablation: {a.num_packets} packets, {a.interval:g} ms interval, {a.uavs} UAVs, {n} runs", fontsize=9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.output_png) or ".", exist_ok=True)
    fig.savefig(a.output_png, dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()