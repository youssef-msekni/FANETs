"""
learning_curve_study.py
======================================================================
Complement de multi_run_study.py : celui-ci mesurait la VARIANCE des deux
systemes sur des essais INDEPENDANTS (table Q remise a zero a chaque run).
Celui-ci mesure au contraire la COURBE D'APPRENTISSAGE : les tables Q de
LLM+RL (q_tables.json) et de Q-FANET (qfanet_q_tables.json) s'accumulent
reellement d'un run au suivant, exactement comme un usage repete normal de
main.py -- seule la topologie/le trafic changent a chaque run (mobilite
aleatoire), pas l'apprentissage.

Repond a la question : "Q-FANET s'ameliore-t-il vraiment avec l'experience,
ou reste-t-il mauvais indefiniment ?"

Par defaut, REINITIALISE q_tables.json et qfanet_q_tables.json au demarrage
(pour que la courbe parte d'un vrai etat vierge, interpretable) -- passez
--no-reset pour repartir de vos tables actuelles a la place.

Comme multi_run_study.py, force Ollama par defaut (pas de quota) -- passez
"groq" ou "mixed" en 4e argument pour changer.

Usage:
    python learning_curve_study.py [N_RUNS] [STEPS_PER_RUN] [--no-reset] [BACKEND]
    python learning_curve_study.py 10 15
    python learning_curve_study.py 10 15 --no-reset ollama
"""

import sys
import os
import json
import matplotlib.pyplot as plt

import main as sim  # attend que ce script soit place a cote de main.py

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_flags = [a for a in sys.argv[1:] if a.startswith("--")]

N_RUNS = int(_args[0]) if len(_args) > 0 else 10
STEPS_PER_RUN = int(_args[1]) if len(_args) > 1 else 15
RESET_AT_START = "--no-reset" not in _flags
BACKEND_ARG = _args[2] if len(_args) > 2 else "ollama"
FORCE_BACKEND = None if BACKEND_ARG == "mixed" else BACKEND_ARG
NUM_UAVS = sim.NUM_UAVS

COLOR_LLM = "#3d78b5"
COLOR_QFANET = "#c0392b"


def _make_progress_callback():
    def _cb(step, total):
        sys.__stdout__.write(f"\r    tick {step}/{total}...")
        sys.__stdout__.flush()
        if step == total:
            sys.__stdout__.write("\n")
    return _cb


def _base_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def reset_q_tables():
    for filename in (sim.Q_TABLE_FILENAME, sim.QFANET_Q_TABLE_FILENAME):
        path = os.path.join(_base_dir(), filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"[Reset] {filename} supprime -- redemarrage a partir d'une table vide.")


def run_study():
    print(f"===== Courbe d'apprentissage : {N_RUNS} runs SEQUENTIELS, "
          f"{STEPS_PER_RUN} ticks chacun =====")
    print("Contrairement a multi_run_study.py, les tables Q PERSISTENT reellement d'un "
          "run au suivant (comme un usage normal repete de main.py).")
    if RESET_AT_START:
        reset_q_tables()
    else:
        print("[Reset] Desactive (--no-reset) -- on repart des tables Q actuelles.")

    if FORCE_BACKEND == "ollama":
        print("Backend : Ollama (local) pour tous les UAV -- pas de quota.\n")
    elif FORCE_BACKEND == "groq":
        print("Backend : Groq pour tous les UAV -- ATTENTION au quota sur autant de runs.\n")
    else:
        print("Backend : partage normal Groq/Ollama.\n")

    llm_pdrs, qfanet_pdrs = [], []
    records = []

    for i in range(1, N_RUNS + 1):
        print(f"--- Run {i}/{N_RUNS} en cours (silencieux, tables Q conservees) ---")
        results = sim.run_simulation(
            steps=STEPS_PER_RUN, num_uavs=NUM_UAVS,
            verbose=True, persist_q_table=True, export_json=False,
            progress_callback=_make_progress_callback(),
            force_backend=FORCE_BACKEND,
        )
        s = results["summary"]
        llm_pdr = s["llm"]["pdr_pct"]
        qfanet_pdr = s["qfanet"]["pdr_pct"]
        llm_pdrs.append(llm_pdr)
        qfanet_pdrs.append(qfanet_pdr)
        records.append({
            "run": i, "total_generated": s["total_generated"],
            "llm_pdr_pct": llm_pdr, "qfanet_pdr_pct": qfanet_pdr,
            "llm_delivered": s["llm"]["delivered"], "qfanet_delivered": s["qfanet"]["delivered"],
        })
        print(f"    -> PDR LLM+RL={llm_pdr:.1f}%  PDR Q-FANET={qfanet_pdr:.1f}%\n")

    return records, llm_pdrs, qfanet_pdrs


def summarize(llm_pdrs, qfanet_pdrs):
    n = len(llm_pdrs)
    half = max(1, n // 2)
    llm_first_half = sum(llm_pdrs[:half]) / half
    llm_second_half = sum(llm_pdrs[half:]) / max(1, n - half)
    qfanet_first_half = sum(qfanet_pdrs[:half]) / half
    qfanet_second_half = sum(qfanet_pdrs[half:]) / max(1, n - half)

    print("\n===== Tendance sur", n, "runs sequentiels =====")
    print(f"{'':<12}{'1ere moitie':>14}{'2e moitie':>14}{'Evolution':>14}")
    print(f"{'LLM+RL':<12}{llm_first_half:>13.1f}%{llm_second_half:>13.1f}%"
          f"{llm_second_half - llm_first_half:>+13.1f}%")
    print(f"{'Q-FANET':<12}{qfanet_first_half:>13.1f}%{qfanet_second_half:>13.1f}%"
          f"{qfanet_second_half - qfanet_first_half:>+13.1f}%")

    if qfanet_second_half > qfanet_first_half + 5:
        print("\n=> Q-FANET progresse nettement avec l'accumulation d'experience "
              "(comme attendu pour un systeme purement Q-learning).")
    elif qfanet_second_half < qfanet_first_half - 5:
        print("\n=> Q-FANET ne montre pas d'amelioration nette sur ces runs -- la topologie "
              "aleatoire de chaque run peut dominer l'effet d'apprentissage sur un si petit "
              "nombre de runs. Essayez d'augmenter N_RUNS pour lisser la variance.")
    else:
        print("\n=> Pas de tendance nette sur ce nombre de runs -- essayez d'augmenter N_RUNS.")

    return {
        "llm_first_half_avg": round(llm_first_half, 2), "llm_second_half_avg": round(llm_second_half, 2),
        "qfanet_first_half_avg": round(qfanet_first_half, 2), "qfanet_second_half_avg": round(qfanet_second_half, 2),
    }


def plot_learning_curve(records, out_dir):
    runs = [r["run"] for r in records]
    llm = [r["llm_pdr_pct"] for r in records]
    qfanet = [r["qfanet_pdr_pct"] for r in records]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(runs, llm, marker="o", linewidth=2, color=COLOR_LLM, label="LLM+RL")
    ax.plot(runs, qfanet, marker="s", linewidth=2, color=COLOR_QFANET, label="Q-FANET")

    # ligne de tendance simple (regression lineaire) pour chaque systeme, pour visualiser la pente
    import statistics as stats
    def trend_line(values):
        n = len(values)
        if n < 2:
            return values
        mean_x = (n - 1) / 2
        mean_y = stats.mean(values)
        num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
        den = sum((i - mean_x) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0
        return [mean_y + slope * (i - mean_x) for i in range(n)]

    ax.plot(runs, trend_line(llm), linestyle="--", linewidth=1.2, color=COLOR_LLM, alpha=0.5)
    ax.plot(runs, trend_line(qfanet), linestyle="--", linewidth=1.2, color=COLOR_QFANET, alpha=0.5)

    ax.set_xlabel("Run (sequentiel, tables Q cumulees)")
    ax.set_ylabel("PDR (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(f"Courbe d'apprentissage sur {len(records)} runs sequentiels\n"
                  "(lignes pointillees = tendance lineaire)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "learning_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    records, llm_pdrs, qfanet_pdrs = run_study()
    trend = summarize(llm_pdrs, qfanet_pdrs)

    base_dir = _base_dir()
    out_dir = os.path.join(base_dir, "charts")
    os.makedirs(out_dir, exist_ok=True)
    p1 = plot_learning_curve(records, out_dir)

    output = {
        "n_runs": len(records),
        "steps_per_run": STEPS_PER_RUN,
        "reset_at_start": RESET_AT_START,
        "per_run": records,
        "trend": trend,
    }
    out_json = os.path.join(base_dir, "learning_curve_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGraphique genere : {p1}")
    print(f"Resultats bruts sauvegardes dans : {out_json}")


if __name__ == "__main__":
    main()