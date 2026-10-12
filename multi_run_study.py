"""
multi_run_study.py
======================================================================
Repond a la question : "Est-ce que Q-FANET bat LLM+RL a chaque run, ou est-ce
juste ce run-la ?" -- lance N simulations INDEPENDANTES (topologie, trafic
et connectivite tires a nouveau a chaque fois), collecte le PDR final de
chaque systeme pour chaque run, puis :

  1. Affiche un tableau recapitulatif (moyenne, ecart-type, min, max, taux
     de victoire de chaque systeme)
  2. Sauvegarde les resultats bruts dans multi_run_results.json
  3. Genere un boxplot PNG comparant les distributions de PDR

IMPORTANT -- cout API : chaque run fait jusqu'a (steps x num_uavs) appels
LLM. Avec les valeurs par defaut (N_RUNS=6, steps=15, 6 UAV), ca fait
jusqu'a 6 x 15 x 6 = 540 appels LLM au total -- largement de quoi epuiser
le quota gratuit Groq (200000 tokens/jour) en une seule etude.

Par defaut, ce script force TOUS les UAV sur Ollama (local, sans quota)
plutot que le partage habituel moitie-Groq/moitie-Ollama, justement pour
eviter ce probleme lors d'etudes a gros volume d'appels. Passez "groq" ou
"mixed" en 3e argument si vous voulez explicitement utiliser Groq (attention
au quota) ou le partage normal.

Chaque run est INDEPENDANT expres (persist_q_table=False) : le but ici est
de mesurer la variance intrinseque des deux algorithmes sur des topologies/
trafics differents, pas un effet d'apprentissage cumulatif d'un run a
l'autre (pour ca, relancez simplement main.py plusieurs fois normalement --
q_tables.json et la LTM continueront de s'accumuler comme d'habitude).

Usage:
    python multi_run_study.py [N_RUNS] [STEPS_PER_RUN] [BACKEND]
    python multi_run_study.py 10 20 ollama
    python multi_run_study.py 6 15 mixed
"""

import sys
import os
import json
import statistics as stats
import matplotlib.pyplot as plt

import main as sim  # attend que ce script soit place a cote de main.py

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS_PER_RUN = int(sys.argv[2]) if len(sys.argv) > 2 else 15
BACKEND_ARG = sys.argv[3] if len(sys.argv) > 3 else "ollama"  # "ollama" (defaut), "groq", ou "mixed"
FORCE_BACKEND = None if BACKEND_ARG == "mixed" else BACKEND_ARG
NUM_UAVS = sim.NUM_UAVS

COLOR_LLM = "#3d78b5"
COLOR_QFANET = "#c0392b"


def _make_progress_callback():
    """Ecrit directement sur le vrai stdout (sys.__stdout__), qui n'est jamais touche par
    le print() rendu muet dans run_simulation(verbose=False) -- donne un signe de vie
    tick par tick meme en mode silencieux, pour qu'un run lent ne ressemble pas a un blocage."""
    def _cb(step, total):
        sys.__stdout__.write(f"\r    tick {step}/{total}...")
        sys.__stdout__.flush()
        if step == total:
            sys.__stdout__.write("\n")
    return _cb


def run_study():
    estimated_calls = N_RUNS * STEPS_PER_RUN * NUM_UAVS
    print(f"===== Etude multi-run : {N_RUNS} runs independants, {STEPS_PER_RUN} ticks chacun =====")
    if FORCE_BACKEND == "ollama":
        print(f"Backend : Ollama (local) pour TOUS les UAV -- pas de quota, {estimated_calls} appels "
              f"locaux au total, aucun risque de rate-limit.")
    elif FORCE_BACKEND == "groq":
        print(f"Backend : Groq (cloud) pour TOUS les UAV -- ATTENTION, {estimated_calls} appels au total, "
              f"risque eleve d'epuiser le quota gratuit (200000 tokens/jour). Considerez 'ollama' a la place.")
    else:
        print(f"Backend : partage normal moitie-Groq/moitie-Ollama -- jusqu'a {estimated_calls // 2} appels "
              f"Groq, risque d'epuiser le quota gratuit sur une etude de cette taille.")
    print("Chaque run est independant (pas de Q-table/LTM partagee entre les runs de cette etude).\n")

    llm_pdrs, qfanet_pdrs = [], []
    llm_delivered, qfanet_delivered = [], []
    per_run_records = []
    llm_pdr_matrix, qfanet_pdr_matrix = [], []  # une ligne par run, une colonne par tick (PDR %, pas des comptes bruts)

    for i in range(1, N_RUNS + 1):
        print(f"--- Run {i}/{N_RUNS} en cours (silencieux, avec indicateur de progression) ---")
        results = sim.run_simulation(
            steps=STEPS_PER_RUN, num_uavs=NUM_UAVS,
            verbose=False, persist_q_table=False, export_json=False,
            progress_callback=_make_progress_callback(),
            force_backend=FORCE_BACKEND,
        )
        s = results["summary"]
        llm_pdr = s["llm"]["pdr_pct"]
        qfanet_pdr = s["qfanet"]["pdr_pct"]
        llm_pdrs.append(llm_pdr)
        qfanet_pdrs.append(qfanet_pdr)
        llm_delivered.append(s["llm"]["delivered"])
        qfanet_delivered.append(s["qfanet"]["delivered"])
        llm_pdr_matrix.append(results["llm_pdr_over_time"])
        qfanet_pdr_matrix.append(results["qfanet_pdr_over_time"])
        per_run_records.append({
            "run": i, "total_generated": s["total_generated"],
            "llm_pdr_pct": llm_pdr, "qfanet_pdr_pct": qfanet_pdr,
            "llm_delivered": s["llm"]["delivered"], "qfanet_delivered": s["qfanet"]["delivered"],
        })
        winner = "LLM+RL" if llm_pdr > qfanet_pdr else ("Q-FANET" if qfanet_pdr > llm_pdr else "egalite")
        print(f"    -> PDR LLM+RL={llm_pdr:.1f}%  PDR Q-FANET={qfanet_pdr:.1f}%  (gagnant: {winner})\n")

    return per_run_records, llm_pdrs, qfanet_pdrs, llm_pdr_matrix, qfanet_pdr_matrix


def summarize(llm_pdrs, qfanet_pdrs):
    n = len(llm_pdrs)
    llm_wins = sum(1 for l, g in zip(llm_pdrs, qfanet_pdrs) if l > g)
    qfanet_wins = sum(1 for l, g in zip(llm_pdrs, qfanet_pdrs) if g > l)
    ties = n - llm_wins - qfanet_wins

    def summary_stats(values):
        return {
            "mean": round(stats.mean(values), 2),
            "stdev": round(stats.stdev(values), 2) if len(values) > 1 else 0.0,
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    llm_stats = summary_stats(llm_pdrs)
    qfanet_stats = summary_stats(qfanet_pdrs)

    print("\n===== Resultats sur", n, "runs independants =====")
    print(f"{'':<12}{'Moyenne':>10}{'Ecart-type':>12}{'Min':>8}{'Max':>8}")
    print(f"{'LLM+RL':<12}{llm_stats['mean']:>9.1f}%{llm_stats['stdev']:>11.1f}%{llm_stats['min']:>7.1f}%{llm_stats['max']:>7.1f}%")
    print(f"{'Q-FANET':<12}{qfanet_stats['mean']:>9.1f}%{qfanet_stats['stdev']:>11.1f}%{qfanet_stats['min']:>7.1f}%{qfanet_stats['max']:>7.1f}%")
    print(f"\nTaux de victoire : LLM+RL {llm_wins}/{n} ({llm_wins/n*100:.0f}%)  |  "
          f"Q-FANET {qfanet_wins}/{n} ({qfanet_wins/n*100:.0f}%)  |  egalite {ties}/{n}")

    if llm_stats["mean"] > qfanet_stats["mean"]:
        print(f"\n=> En moyenne sur {n} runs, LLM+RL devance Q-FANET de {llm_stats['mean']-qfanet_stats['mean']:.1f} points de PDR.")
    elif qfanet_stats["mean"] > llm_stats["mean"]:
        print(f"\n=> En moyenne sur {n} runs, Q-FANET devance LLM+RL de {qfanet_stats['mean']-llm_stats['mean']:.1f} points de PDR.")
    else:
        print("\n=> Moyennes egales sur ces runs.")

    return llm_stats, qfanet_stats, llm_wins, qfanet_wins, ties


def plot_boxplot(llm_pdrs, qfanet_pdrs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    bp = ax.boxplot([llm_pdrs, qfanet_pdrs], labels=["LLM+RL", "Q-FANET"], patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [COLOR_LLM, COLOR_QFANET]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # overlay individual run points (jittered) so N is visible, not just the box
    import random as _random
    for x, values, color in [(1, llm_pdrs, COLOR_LLM), (2, qfanet_pdrs, COLOR_QFANET)]:
        jitter = [x + _random.uniform(-0.08, 0.08) for _ in values]
        ax.scatter(jitter, values, color=color, edgecolor="black", linewidth=0.5, zorder=3, s=40)

    ax.set_ylabel("PDR (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(f"Distribution du PDR sur {len(llm_pdrs)} runs independants")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "pdr_boxplot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_run_comparison(records, out_dir):
    runs = [r["run"] for r in records]
    llm = [r["llm_pdr_pct"] for r in records]
    qfanet = [r["qfanet_pdr_pct"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = range(len(runs))
    ax.bar([i - width/2 for i in x], llm, width, label="LLM+RL", color=COLOR_LLM)
    ax.bar([i + width/2 for i in x], qfanet, width, label="Q-FANET", color=COLOR_QFANET)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Run {r}" for r in runs])
    ax.set_ylabel("PDR (%)")
    ax.set_title("PDR par run -- LLM+RL vs Q-FANET")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "pdr_per_run.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def aggregate_over_time(cum_matrix):
    """cum_matrix : une ligne par run, une colonne par tick (meme longueur pour
    tous les runs puisque STEPS_PER_RUN est fixe pour toute l'etude). Retourne
    (moyenne, ecart-type) par tick, calcules a travers les runs."""
    n_runs = len(cum_matrix)
    n_ticks = len(cum_matrix[0]) if n_runs else 0
    means, stds = [], []
    for t in range(n_ticks):
        values = [run[t] for run in cum_matrix]
        means.append(stats.mean(values))
        stds.append(stats.stdev(values) if len(values) > 1 else 0.0)
    return means, stds


def plot_pdr_over_time(llm_pdr_matrix, qfanet_pdr_matrix, out_dir):
    """PDR (%) EN FONCTION DU TEMPS (tick), moyenne sur N runs independants avec
    barres d'erreur (ecart-type) -- meme principe que les graphiques de comparaison
    de protocoles de routage dans la litterature FANET (PDR vs parametre, plusieurs
    courbes, barres d'erreur par point)."""
    llm_means, llm_stds = aggregate_over_time(llm_pdr_matrix)
    qfanet_means, qfanet_stds = aggregate_over_time(qfanet_pdr_matrix)
    ticks = list(range(1, len(llm_means) + 1))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(ticks, llm_means, yerr=llm_stds, marker="o", linestyle="-", linewidth=2,
                capsize=3, color=COLOR_LLM, label="LLM+RL", markersize=6)
    ax.errorbar(ticks, qfanet_means, yerr=qfanet_stds, marker="s", linestyle="-.", linewidth=2,
                capsize=3, color=COLOR_QFANET, label="Q-FANET", markersize=6)

    ax.set_xlabel("Tick de simulation (temps)")
    ax.set_ylabel("Packet Delivery Ratio (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(f"PDR en fonction du temps -- moyenne sur {len(llm_pdr_matrix)} runs independants")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "pdr_over_time.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    records, llm_pdrs, qfanet_pdrs, llm_cum_matrix, qfanet_cum_matrix = run_study()
    llm_stats, qfanet_stats, llm_wins, qfanet_wins, ties = summarize(llm_pdrs, qfanet_pdrs)

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, "charts")
    os.makedirs(out_dir, exist_ok=True)

    p1 = plot_boxplot(llm_pdrs, qfanet_pdrs, out_dir)
    p2 = plot_per_run_comparison(records, out_dir)
    p3 = plot_pdr_over_time(llm_cum_matrix, qfanet_cum_matrix, out_dir)

    llm_ot_mean, llm_ot_std = aggregate_over_time(llm_cum_matrix)
    qfanet_ot_mean, qfanet_ot_std = aggregate_over_time(qfanet_cum_matrix)

    output = {
        "n_runs": len(records),
        "steps_per_run": STEPS_PER_RUN,
        "per_run": records,
        "llm_stats": llm_stats,
        "qfanet_stats": qfanet_stats,
        "win_rate": {"llm": llm_wins, "qfanet": qfanet_wins, "tie": ties},
        "over_time": {
            "ticks": list(range(1, STEPS_PER_RUN + 1)),
            "llm_mean": [round(v, 2) for v in llm_ot_mean],
            "llm_std": [round(v, 2) for v in llm_ot_std],
            "qfanet_mean": [round(v, 2) for v in qfanet_ot_mean],
            "qfanet_std": [round(v, 2) for v in qfanet_ot_std],
        },
    }
    out_json = os.path.join(base_dir, "multi_run_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGraphiques generes :\n  {p1}\n  {p2}\n  {p3}")
    print(f"Resultats bruts sauvegardes dans : {out_json}")


if __name__ == "__main__":
    main()