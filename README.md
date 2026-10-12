# FANET Multi-UAV Relay Simulation

This project simulates a Flying Ad Hoc Network (FANET) with 6 mobile UAV relays, one fixed Source, and one fixed Ground Station. It compares two routing systems on the same traffic, mobility and radio-connectivity snapshot:

- **LLM+RL**: LLM-guided reward shaping + tabular Q-learning + utility-based next-hop selection.
- **Q-FANET**: pure Q-learning baseline without LLM reasoning.

## Current network model

The corrected default scenario uses:

- horizontal environment: **500 m × 500 m**
- hard radio communication range: **180 m maximum (3-D Euclidean distance)**
- simulation tick: **250 ms**
- 6 Gauss-Markov mobile UAV relays
- Source at `(250, 475, 150)`
- Ground Station at `(250, 25, 0)`
- packet deadline choices: **1000, 1500, 2000, 2500, 3000 ms**
- unlimited queues; packets can be lost by deadline expiry

A UAV-UAV or UAV-GS link can exist only if the pair is physically within 180 m. For pairs inside that range, the existing Poisson/exponential link-churn model is then applied to represent transient radio availability/fading.

The UAVs start in a lightly jittered relay corridor between Source and GS. This avoids experiments being dominated by a permanently disconnected random initial topology; Gauss-Markov mobility then changes the topology during the simulation.

## Corrected transition / LLM reasoning sequence

The LLM+RL cycle is now explicitly:

```text
Observation S_t
    -> LLM reasoning / shaping
    -> RL action A_t
    -> Environment executes action
    -> Environment computes reward R_t
    -> all one-hop transfers are committed synchronously
    -> updated state S_(t+1)
    -> Q-learning update
    -> next LLM reasoning receives BOTH R_t and the updated-state feedback
```

The previous reward is included explicitly in the next LLM prompt under `PREVIOUS TRANSITION FEEDBACK`, together with the previous action and the resulting updated-state label.

## Deadline model

Packets are represented as independent `PacketBatch` objects rather than one aggregate queue deadline. Each batch stores:

- batch ID
- number of packets
- creation time
- initial deadline
- absolute deadline time
- hop count

This prevents a short deadline from incorrectly replacing the deadlines of all other packets in the same UAV queue.

Both LLM+RL and Q-FANET now use the same expiry rule: a packet that reaches its end-to-end deadline before successful delivery is dropped permanently and counted in `dropped_deadline`.

## Synchronous forwarding

A packet can move by **at most one hop per simulation tick**. Forwarded packets are first placed in an incoming buffer and are committed only after every UAV has made its decision for that tick. This removes the earlier iteration-order effect where a packet could cross several UAVs inside one 250 ms tick.

## Fair comparison

The two compared systems share:

- Source traffic
- Source packet deadlines
- UAV mobility
- physical positions
- the same active-link snapshot each tick

But they keep independent:

- packet queues
- battery states
- deadline losses
- Q-tables
- routing decisions

Q-FANET no longer reads the battery modified by LLM+RL actions.

## LTM

The long-term memory uses:

- LangSmith as persistent experience storage when configured
- SentenceTransformers embeddings
- FAISS local semantic vector retrieval

If FAISS/the embedding model or LangSmith is unavailable, the simulator falls back to local lexical retrieval instead of crashing.

## Project structure

```text
myagent/
├── main.py
├── neighbor_discovery.py
├── plot_comparison.py
├── multi_run_study.py
├── learning_curve_study.py
├── Dashboard.html
├── run_project.bat
├── requirements.txt
├── .env.example
├── .gitignore
├── q_tables.json
└── qfanet_q_tables.json
```

`comparison_results.json`, multi-run result files, and `charts/` are generated outputs.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own local credentials. Do not commit `.env`.

Typical variables are:

```env
GROQ_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=
LANGSMITH_PROJECT=
LTM_DATASET_NAME=FANETs
```

For the local Ollama backend:

```bash
ollama serve
ollama pull phi3
```

## Run the simulation

```bash
python main.py
```

The single-run output is written to `comparison_results.json`.

Important time-series fields include:

```text
time_ms
time_s
llm_pdr_over_time
qfanet_pdr_over_time
deadline_trace
packet_deadlines
llm_delivery_records
qfanet_delivery_records
deadline_drop_events
```

`packet_deadlines` contains each generated batch's creation time and assigned deadline. Delivery records include end-to-end latency and remaining deadline margin at delivery.

## Generate thesis figures

After running `main.py`:

```bash
python plot_comparison.py
```

The script produces, among other figures:

- `charts/pdr_vs_time.png` — **PDR versus simulated time in seconds**
- `charts/deadline_vs_time.png` — minimum remaining packet deadline versus time
- `charts/deadline_drops_vs_time.png` — cumulative deadline losses versus time
- `charts/cumulative_delivery.png`
- `charts/pdr_comparison.png`
- `charts/outcomes_breakdown.png`

## Multi-run studies

Independent variance study:

```bash
python multi_run_study.py 6 30 ollama
```

Sequential learning study with Q-table persistence:

```bash
python learning_curve_study.py 10 30
```

For large studies, a local Ollama backend avoids cloud token quotas.

## Security

`.env` is ignored by Git. The distributed project contains a blank `.env` / `.env.example`; add keys only on your own machine. If credentials were previously committed or shared, rotate them.
