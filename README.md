# FANET Multi-UAV Relay Simulation — LLM+RL vs Q-FANET

Simulation of a Flying Ad Hoc Network (FANET): one fixed Source, one fixed Ground Station (GS) and N mobile UAV relays (default 6, tested up to 50). Two routing systems are compared on **identical** traffic, mobility and radio-link snapshots:

| System | Description |
|---|---|
| **LLM+RL (proposed)** | Neighbor-aware Q-learning. An LLM (with STM + LTM memory) scores every reachable next hop; the final action is chosen by Q-learning combined with the LLM scores and a topology/QoS prior. |
| **Q-FANET (baseline)** | QMR-inspired tabular Q-learning, no LLM, no STM/LTM. |

![Architecture](Architecure.png)

## Architecture

```text
Sensors (GPS, RSSI, Energy) ──observation──► LLM Core (CoT + ReAct) ◄── STM (session context)
                                                  │          ▲
                                                  │          └── LTM (LangSmith + FAISS, past experiences)
                                        neighbor scores / guidance
                                                  ▼
                                   RL Agent (neighbor-aware Q-learning)
                                                  │ action: HOLD | UAV-x | GS
                                                  ▼
                                         FANET Environment
                                                  │ reward R_t, next state S_(t+1)
                                                  ▼
                       Q-update + STM update + LTM save (+ feedback in next LLM prompt)
```

### What the LLM does (ReAct + CoT, current code)
The LLM **never picks the action**. For each UAV with a non-empty queue it runs a **ReAct loop** (`react_llm_core.py`), at most `REACT_MAX_STEPS` (default 4) replies:

```text
Thought (numbered CoT: 1 Situation, 2 Unknown, 3 Plan, then 4 Update, 5 Next)
  -> Action: read-only tool   -> Observation
  -> ... repeat ...           -> finish (guidance only)
```

Tools (read-only): `get_battery`, `get_distance`, `get_link_quality`, `get_queue_load`, `get_deadline_status`, `read_stm`, `search_ltm`. `finish` is refused until at least one observation succeeded. Invalid JSON, LLM errors or an exhausted budget fall back to neutral guidance (scores 0.5, shaping 0).

`finish` returns: a soft score per candidate and for HOLD, routing weights, a short rationale, and a **bounded shaping bonus** per action. The thoughts, tool calls and observations are exported as `react_trace` in `reward_trace`.

The **RL agent takes the final action**:

```text
Score(a) = 0.35 * Q(s,a) + 0.25 * LLM_score(a) + 0.40 * Topology/QoS_prior(a)
```

Hard masks apply first: direct GS is always taken when reachable; if a progress-making neighbor exists only progress neighbors are allowed; HOLD only when no useful link exists.

### Reward shaping (restored, bounded)
After the environment returns `R_env` for the action chosen by RL:

```text
R_learn = R_env + clip(bonus_LLM(a_chosen), -B, +B)       B = LLMRL_SHAPING_BOUND (default 0.25)
```

`R_learn` feeds the Q-update, STM and LTM. `reward_trace` logs `base_reward`, `llm_bonus` and `final_reward` per step. The bound keeps the environment reward dominant.

| Env variable | Effect | Default |
|---|---|---|
| `LLMRL_REACT` | `0` = single-call scorer instead of ReAct | 1 |
| `LLMRL_SHAPING` | `0` = no reward bonus (ablation: ReAct guidance only) | 1 |
| `LLMRL_SHAPING_BOUND` | bonus clamp `B` | 0.25 |
| `REACT_MAX_STEPS` | max replies per decision | 4 |

Cost: a ReAct decision needs 2 to 4 LLM calls instead of 1, so runs are slower. Q-FANET is unchanged.

## Network model

- Area **500 m × 500 m**, hard radio range **180 m** (3-D Euclidean distance).
- Tick = **250 ms**. A packet moves **at most one hop per tick** (synchronous commit).
- Source at `(250, 475, 150)`, GS at `(250, 25, 0)`.
- Gauss-Markov mobility, configurable mean speed (`--speed-mps`, default 12 m/s). UAVs start in a jittered Source→GS corridor.
- Links: in-range pairs follow a Poisson/exponential birth-death process (λ=1.4/s, μ=0.6/s) on **symmetric** edges (`SharedLinkGraph`).
- HELLO beacons every 100 ms (expire after 300 ms) maintain neighbor freshness.
- Source generates DATA at `--packet-rate-pps` (default 10 pps), deadlines drawn from {1000, 1500, 2000, 2500, 3000} ms.
- Each UAV has unlimited queues but a **finite forwarding capacity** (`--relay-capacity-pps`, default 40 pps) with EDF scheduling, so congestion shows up as deadline drops.
- Battery: random 50–100 % start, −3 % per tick, UAV cannot transmit below 1 %.
- Experiment = traffic phase + drain phase (so late packets can still be delivered).

### Fair comparison
Shared: traffic, deadlines, mobility, positions, link snapshot, initial batteries, exploration schedule.
Independent: queues, batteries, drops, Q-tables, decisions.

## Memory
- **STM**: per-UAV, per-session rolling window of the last 5 transitions.
- **LTM**: shared across the swarm. LangSmith dataset (`FANETs`) for persistence, SentenceTransformers + FAISS for semantic retrieval, re-ranked by outcome quality (successful deliveries preferred over deadline failures). Falls back to local lexical retrieval if LangSmith/FAISS is unavailable.

## Project structure

```text
.
├── main.py                    # simulation: LLM+RL, Q-FANET, environment, export
├── react_llm_core.py          # ReAct + CoT loop, tools, bounded shaping guidance
├── neighbor_discovery.py      # SharedLinkGraph, link/energy/distance helpers
├── plot_comparison.py         # sensitivity studies (flow / speed / nodes) -> experiment_results.json
├── multi_run_study.py         # independent runs: variance, boxplot, PDR over time
├── learning_curve_study.py    # sequential runs with persistent Q-tables
├── Dashboard.html             # interactive dashboard
├── run_project.bat            # Windows pipeline launcher
├── Architecure.png            # architecture diagram
├── CORRECTIONS.md             # change log of model corrections
└── requirements.txt
```
Generated: `comparison_results.json`, `exp_*.json`, `experiment_results.json`, `charts/`, `q_tables_*.json`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` (never commit it):

```env
GROQ_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT=
LANGSMITH_WORKSPACE_ID=
LTM_DATASET_NAME=FANETs
OLLAMA_MODEL=phi3
OLLAMA_BASE_URL=http://localhost:11434
```

Local LLM backend (default):

```bash
ollama serve
ollama pull phi3
```

## Run

`--uavs` is **required**.

```bash
python main.py --uavs 25 --steps 20 --packet-rate-pps 10 \
    --packet-size-bytes 128 --hello-interval-ms 100 --hello-size-bytes 32
```

Useful options:

| Option | Meaning | Default |
|---|---|---|
| `-n/--uavs` | number of UAV relays (required) | – |
| `-s/--steps` | total ticks (1 tick = 0.25 s) | 20 |
| `--backend` | `ollama`, `groq`, `mixed` | ollama |
| `--packet-rate-pps` | Source DATA rate | 10 |
| `--relay-capacity-pps` | per-UAV forwarding capacity | 40 |
| `--speed-mps` | mean UAV speed | 12 |
| `--output-json` | output file (never use shell `>` redirection) | comparison_results.json |
| `--fresh-learning` | ignore persisted Q-tables | off |

Output: `comparison_results.json` and `charts/pdr_vs_time.png`.

### Sensitivity studies

```bash
python plot_comparison.py --study all --runs 5 --fresh-learning
python -m http.server 8000     # then open http://localhost:8000/Dashboard.html
```

### Variance / learning studies

```bash
python multi_run_study.py 10 30 ollama
python learning_curve_study.py 10 30
```

## Exported data (per run)
`time_s`, `llm_pdr_over_time`, `qfanet_pdr_over_time`, `deadline_trace`, `packet_deadlines`, `packet_tracking` (full per-packet trajectory), `llm_delivery_records`, `qfanet_delivery_records`, `deadline_drop_events`, `battery_trace`, `reward_trace`, `summary`.

## Known limitations
See the "Limitations" discussion in the thesis; main points: ReAct latency (2-4 LLM calls per decision), small sample sizes in current sweeps, single entry UAV at the Source (bottleneck at high load), idealized MAC/PHY, and the absence of an LLM-ablation baseline.

## Security
`.env` is git-ignored. Rotate any key that was ever committed or shared.
