"""
FANET Multi-UAV Relay Simulation: LLM Core (ReAct-inspired) + Reward Shaping + RL (Q-Learning)
============================================================================================

Architecture per UAV (same as the diagram):

    Capteurs (IMU: GPS/position, RSSI, Energie)  ----\
                                                        >--> LLM Core (ReAct-inspired) --bounded shaping--> Agent RL (Q-Learning, R+) --(1) action--> Env FANET
    File paquets (Deadlines, UNLIMITED queue)    ----/            ^        ^                                                              |
                                                                   |        |                                                             |
                                                                  STM      LTM  <---------------------------(3) save LTM------------------+
                                                               (Contexte) (LTM = LangSmith persistence + FAISS vector index)                        |
                                                                   ^--------------------------------(2) obs-------------------------------+

======================================================================
NETWORK MODEL AND ASSUMPTIONS  (for the thesis / report "Network model" section)
======================================================================

Topology
  - 1 fixed Source (emetteur fixe)      : generates data traffic, does not move.
  - 1 fixed Ground Station / GS         : the single destination (destinataire fixe), does not move.
  - N mobile UAV relay nodes            : N is selected at launch time and each UAV carries the LLM+RL decision loop.

IMU-based sensing (per UAV)
  Each UAV carries a simulated IMU composed of:
    - Accelerometer : linear acceleration (ax, ay, az), used to sense motion/vibration.
    - Gyroscope     : angular rates (roll/pitch/yaw), used for short-term orientation tracking.
    - Magnetometer  : heading reference used to correct long-term orientation drift (navigation).
  Fusing these three sensors, the IMU produces an on-board ESTIMATE of the UAV's own
  position, velocity and altitude (dead-reckoning with small drift + periodic correction) —
  this estimate is what feeds the LLM Core, distinct from the ground-truth physics state
  used only by the simulator itself.

Mobility model
  - All UAVs move under the SAME mobility model: the Gauss-Markov mobility model
    (temporally-correlated random walk: velocity/direction blend a memory term, a mean
    term, and a Gaussian random term, controlled by a "memory" parameter alpha).
  - The Source and the GS are stationary (fixed emitter / fixed destination).

Traffic & queueing
  - All UAVs are capable of being senders and relays (general capability of the code).
    In this configured scenario, only the fixed Source injects new traffic; the UAVs act
    as mobile multi-hop relays carrying that traffic toward the GS.
  - The GS is the single receiver (sink) of the network.
  - Packet queues are UNLIMITED: no packet is ever dropped due to queue capacity.
  - Every UAV has a FINITE DATA forwarding/service rate (packets/s). Under high
    offered load, packets wait longer in the unlimited queue and can miss their
    deadline. Therefore congestion reduces PDR through deadline expiry, not overflow.

Requires for Ollama mode: python-dotenv + local Ollama service. Groq/LangSmith are optional.
"""

# =====================================================================
# GUIDE DE LECTURE DU FICHIER
# =====================================================================
#
# Ce fichier contient deux systèmes de routage exécutés EN PARALLÈLE :
#
#   1) LLM + RL
#      - le LLM analyse l'observation courante,
#      - consulte la STM de la session,
#      - consulte la LTM historique (LangSmith + recherche sémantique),
#      - produit uniquement un shaping borné et des poids de routage,
#      - Q-learning choisit finalement FORWARD ou HOLD,
#      - RoutingUtility choisit le prochain saut.
#
#   2) Q-FANET / QMR-inspired
#      - aucun LLM,
#      - aucune STM/LTM,
#      - politique tabulaire Q-learning,
#      - mêmes conditions physiques, trafic, mobilité et batterie initiale.
#
# Les deux systèmes partagent :
#   - les positions physiques des UAV,
#   - la mobilité Gauss-Markov,
#   - le graphe de connectivité,
#   - les paquets générés par la Source,
#   - les conditions initiales de batterie.
#
# Mais ils gardent séparément :
#   - leurs files DATA,
#   - leurs batteries après le début de la simulation,
#   - leurs Q-tables,
#   - leurs décisions,
#   - leurs statistiques de livraison.
#
# FLUX PRINCIPAL DU SYSTÈME LLM+RL
# --------------------------------
#
#   Observation actuelle
#          |
#          +--> télémétrie / RSSI / batterie / deadline / voisins
#          |
#          +--> STM = mémoire récente de CETTE session pour CET UAV
#          |
#          +--> LTM = mémoire historique partagée
#                 |--> LangSmith
#                 |--> expériences récentes
#                 '--> expériences similaires via FAISS
#          |
#          v
#   LLM Core
#          |
#          |  raisonnement interne privé (CoT)
#          |
#          +--> bonus FORWARD / HOLD bornés
#          '--> poids w1 / w2 / w3
#          |
#          v
#   Q-learning choisit l'action
#          |
#          v
#   RoutingUtility choisit le prochain saut
#          |
#          v
#   NetworkEnv exécute l'action
#          |
#          v
#   Reward + nouvel état
#          |
#          +--> mise à jour Q-table
#          +--> mise à jour STM
#          '--> sauvegarde dans LTM
#
# IMPORTANT SUR LE PROMPT LLM
# ---------------------------
# Le prompt conserve volontairement la structure demandée :
#
#   GOAL
#   CONTEXT
#   CONDITIONS
#   TASKS
#   OUTPUT FORMAT
#
# La section TASKS contient :
#   - l'analyse du réseau,
#   - l'analyse deadline / énergie / RSSI,
#   - l'utilisation obligatoire de STM,
#   - l'utilisation obligatoire de LTM LangSmith,
#   - le raisonnement interne privé de type Chain-of-Thought,
#   - la production du shaping et des poids de routage.
#
# Le CoT complet n'est pas exporté : seule une courte "rationale" est retournée.
# =====================================================================

from dotenv import load_dotenv
import os
import re
import sys
import json
import random
import time
import math
import argparse
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
# ---------------------------------------------------------------------
# OPTIONAL LLM / MEMORY DEPENDENCIES
# ---------------------------------------------------------------------
# Ollama is called directly through its local HTTP API, so --backend ollama
# does not require langchain_core or langchain_ollama.
try:
    from langchain_groq import ChatGroq
    _GROQ_AVAILABLE = True
    _GROQ_IMPORT_ERROR = None
except Exception as _groq_exc:
    ChatGroq = None
    _GROQ_AVAILABLE = False
    _GROQ_IMPORT_ERROR = _groq_exc

try:
    from langsmith import Client
    try:
        from langsmith.schemas import DataType
    except Exception:
        DataType = None
    _LANGSMITH_AVAILABLE = True
    _LANGSMITH_IMPORT_ERROR = None
except Exception as _langsmith_exc:
    Client = None
    DataType = None
    _LANGSMITH_AVAILABLE = False
    _LANGSMITH_IMPORT_ERROR = _langsmith_exc

from urllib import request as urllib_request
from urllib import error as urllib_error
from types import SimpleNamespace

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _SEMANTIC_LTM_AVAILABLE = True
    _SEMANTIC_LTM_IMPORT_ERROR = None
except Exception as _semantic_exc:
    # FAISS / sentence-transformers are optional.  Catch any import-time
    # failure (including SyntaxError from a damaged site-packages install)
    # and continue with the lexical LTM fallback.
    faiss = None
    SentenceTransformer = None
    _SEMANTIC_LTM_AVAILABLE = False
    _SEMANTIC_LTM_IMPORT_ERROR = _semantic_exc
    print(
        f"[LTM] Semantic backend unavailable "
        f"({type(_semantic_exc).__name__}: {_semantic_exc}); "
        "using lexical fallback."
    )

# ---------------------------------------------------------------------
# DIRECT OLLAMA HTTP BACKEND
# ---------------------------------------------------------------------
# Minimal chat wrapper for Ollama. It uses only Python's standard library
# and exposes the same interface used later in this file:
#     response = self.llm.invoke(messages)
#     response.content
class DirectOllamaChat:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        num_predict: int = 160,
        keep_alive: str = "10m",
        timeout: float = 30.0,
        json_mode: bool = True,
    ):
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)
        self.keep_alive = keep_alive
        self.timeout = float(timeout)
        self.json_mode = bool(json_mode)

    @staticmethod
    def _normalize_messages(messages) -> list:
        normalized = []

        for item in messages or []:
            role = None
            content = None

            if isinstance(item, (tuple, list)) and len(item) >= 2:
                role = str(item[0])
                content = str(item[1])

            elif isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))

            else:
                content = str(getattr(item, "content", item))
                role = str(
                    getattr(item, "type", None)
                    or getattr(item, "role", None)
                    or "user"
                )

            role_map = {
                "human": "user",
                "user": "user",
                "system": "system",
                "ai": "assistant",
                "assistant": "assistant",
            }

            normalized.append({
                "role": role_map.get(role.lower(), "user"),
                "content": content,
            })

        return normalized

    def invoke(self, messages):
        payload = {
            "model": self.model,
            "messages": self._normalize_messages(messages),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        if self.json_mode:
            payload["format"] = "json"

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        req = urllib_request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(
                req,
                timeout=self.timeout,
            ) as response:
                raw = response.read()

        except urllib_error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)

            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {detail}"
            ) from exc

        except urllib_error.URLError as exc:
            raise RuntimeError(
                "Cannot reach Ollama at "
                f"{self.base_url}. Start Ollama and verify that model "
                f"'{self.model}' is installed. Original error: {exc}"
            ) from exc

        try:
            data = json.loads(
                raw.decode("utf-8")
            )
        except Exception as exc:
            raise RuntimeError(
                "Ollama returned a non-JSON response."
            ) from exc

        content = (
            data.get("message", {}).get("content")
            or data.get("response")
            or ""
        )

        if not isinstance(content, str):
            content = json.dumps(
                content,
                ensure_ascii=False,
            )

        return SimpleNamespace(
            content=content,
            raw=data,
        )


_OLLAMA_AVAILABLE = True

# P1 (environnement + connectivite Poisson), P2 (outils Get_Battery/Get_Distance/
# Get_Link_Quality) vivent dans neighbor_discovery.py -- ce fichier doit se trouver
# dans le meme dossier. gabriel_planarize() y reste disponible (documente pour P5)
# mais n'est plus appelee activement depuis que GPSR a ete remplace par QFANETRouter
# (Q-learning pur, pas de notion de graphe planarise/routage de perimetre).
from neighbor_discovery import (
    SharedLinkGraph,
    gabriel_planarize,
    get_battery,
    get_distance,
    get_link_quality,
)

load_dotenv()

# Keep explicit LangSmith LTM access, but disable automatic tracing of every LLM call.
# This prevents Ollama requests from being blocked by LangSmith trace uploads.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Force UTF-8 on stdout/stderr: on Windows the default console encoding (cp1252)
# cannot print characters an LLM may emit (e.g. non-breaking hyphen U+2011),
# which would otherwise crash print() calls, including inside error handlers.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# 0. Global constants
# ============================================================

ACTIONS = ["forward", "hold"]  # binaire (send/hold) -- reduit la dimension de la table Q
# Uses the shared "FANETs" LangSmith dataset (same one used by the other drone/orchestrator
# scripts), so all UAVs read from and write to the same accumulated long-term memory.
# NOTE: this dataset is a "chat"-type dataset (schema requires inputs={"messages":[...]},
# outputs={"message": {...}}), so LangSmithLTM below wraps/unwraps our plain telemetry
# dicts into that chat-message shape transparently -- see _wrap_as_chat / _unwrap_chat.
LTM_DATASET_NAME = os.getenv("LTM_DATASET_NAME") or "FANETs"

DEFAULT_NUM_UAVS = 6
AREA = (0.0, 500.0, 0.0, 500.0)          # 500 m x 500 m horizontal FANET area
GS_POSITION = (250.0, 25.0, 0.0)           # fixed ground station near the south edge
SOURCE_POSITION = (250.0, 475.0, 150.0)    # fixed emitter near the north edge
COMM_RANGE = 180.0                         # hard physical radio range: a link is impossible beyond 180 m
TICK_DURATION_MS = 250.0                   # one simulation tick = 250 ms
SIM_TICK_SECONDS = TICK_DURATION_MS / 1000.0

# Mean UAV speed used by the Gauss-Markov mobility model.
# Configurable with --speed-mps for PDR-vs-speed experiments.
DEFAULT_MEAN_SPEED_MPS = 12.0
DEADLINE_CHOICES_MS = (1000.0, 1500.0, 2000.0, 2500.0, 3000.0)
LARGE_DEADLINE = float("inf")              # empty-queue sentinel
MIN_OPERATIONAL_ENERGY = 1.0               # below this level a UAV cannot transmit
BATTERY_INITIAL_MIN_PCT = 50.0             # minimum random initial UAV battery
BATTERY_INITIAL_MAX_PCT = 100.0            # maximum random initial UAV battery
BATTERY_DRAIN_PER_TICK_PCT = 3.0           # deterministic battery cost per simulation tick

# QMR-inspired reward configuration.
# The paper formula supplied by the project uses delay and next-hop energy:
#   r_max                          if s_(t+1) is destination
#   r_min                          if s_t is a local minimum
#   omega*exp(-delay_ij) + (1-omega)*E_j   otherwise
#
# This implementation keeps that structure and adds RSSI explicitly, as required
# for this project, by blending the original delay-energy term with normalized RSSI:
#   R = beta*[omega*exp(-delay_s) + (1-omega)*E_j] + (1-beta)*RSSI_norm
# where beta=0.5 and omega=0.6 are configurable.
INITIAL_Q_VALUE = 0.5
QMR_R_MAX = 1.0
QMR_R_MIN = -1.0
QMR_OMEGA = 0.6
QMR_BETA = 0.5
RSSI_MIN_DBM = -100.0
RSSI_MAX_DBM = -40.0
LLM_SHAPING_MIN = -0.5
LLM_SHAPING_MAX = 0.5

# Traffic/control-plane configuration.
# ----------------------------------------------------------------
# DATA TRAFFIC LOAD
# ----------------------------------------------------------------
# ``DEFAULT_PACKET_RATE_PPS`` is the offered application load generated by
# the fixed Source, expressed directly in packets per simulated second.
#
# Examples:
#   10 packets/s  -> one packet every 100 ms
#   100 packets/s -> one packet every 10 ms
#
# ``DEFAULT_RELAY_CAPACITY_PPS`` is the abstract DATA forwarding/service
# capacity of EACH UAV. It is not a raw Wi-Fi PHY bit-rate. It models the
# finite number of DATA packets that one relay can process/forward per second
# after routing, MAC, queueing and control overhead.
#
# This finite service capacity is essential if we want a meaningful load test:
# at 10 pps the network is lightly loaded, while at 100 pps queues can build,
# deadlines can expire, and PDR can decrease naturally.
# ----------------------------------------------------------------
DEFAULT_PACKET_RATE_PPS = 10.0
DEFAULT_RELAY_CAPACITY_PPS = 40.0

# Kept only for backward-compatible CLI support.
DEFAULT_DATA_INTERVAL_MS = 1000.0 / DEFAULT_PACKET_RATE_PPS

DEFAULT_DATA_PACKET_SIZE_BYTES = 127   # parameter table supplied by the user
DEFAULT_HELLO_INTERVAL_MS = 100.0      # each UAV: one HELLO every 100 ms = 10 Hz
HELLO_EXPIRE_TIME_MS = 300.0           # neighbor information expiration (paper table)
DEFAULT_HELLO_PACKET_SIZE_BYTES = 32   # explicit simulation assumption; configurable from CLI

# Exploration schedule. Both compared systems use the same epsilon at each tick.
EPSILON_START = 0.10
EPSILON_MIN = 0.02
EPSILON_DECAY_PER_TICK = 0.95

# Q-FANET stabilization for short, deadline-constrained experiments.
# These do not give Q-FANET access to the LLM. They only make the tabular
# baseline respect the destination, current reachability and packet urgency.
QFANET_Q_TIE_TOLERANCE = 0.05
QFANET_HOP_PENALTY = 0.05
QFANET_URGENT_DEADLINE_MS = 2.0 * TICK_DURATION_MS
QFANET_CRITICAL_DEADLINE_MS = 1.0 * TICK_DURATION_MS

# ----------------------------------------------------------------
# LLM+RL deadline-aware performance guard.
#
# WHY:
# Q-FANET always forwards when it has a usable next hop, while LLM+RL has
# an additional HOLD action. In a deadline-constrained FANET, random or
# incorrectly rewarded HOLD decisions can destroy PDR even when the routes
# selected by LLM+RL are shorter. These constants keep HOLD meaningful but
# prevent avoidable deadline losses.
# ----------------------------------------------------------------
LLMRL_Q_MIN = -1.0
LLMRL_Q_MAX = 1.0

# Stronger deadline protection for the proposed controller. HOLD remains possible
# only when useful forward progress is unavailable.
LLMRL_HOLD_PENALTY = 0.40
LLMRL_URGENT_DEADLINE_MS = 2.0 * TICK_DURATION_MS
LLMRL_SOON_DEADLINE_MS = 4.0 * TICK_DURATION_MS
LLMRL_FORWARD_BONUS_FLOOR = 0.15
LLMRL_URGENT_FORWARD_BONUS_FLOOR = 0.45
LLMRL_URGENT_HOLD_BONUS_CEIL = -0.45

# In evaluation, random exploration among valid progress links is deliberately
# small. Training can still occur across repeated runs through the Q updates.
LLMRL_PROGRESS_EPSILON_CAP = 0.02

# ----------------------------------------------------------------
# NEIGHBOR-AWARE LLM + RL ARCHITECTURE
# ----------------------------------------------------------------
# The proposed controller no longer asks RL only "FORWARD or HOLD".
# RL directly evaluates the currently reachable next-hop candidates.
#
# Final action space for one UAV:
#     HOLD, UAV-1, UAV-2, ..., UAV-N, GS
#
# Only physically reachable actions are enabled at a given tick.
# The LLM supplies contextual scores for those candidates; it does NOT
# directly choose the final action.
# ----------------------------------------------------------------
HOLD_ACTION = "HOLD"

# Balanced fusion for next-hop selection. Q-learning remains important, but a
# partially trained/noisy Q-table must not dominate strong current topology/QoS
# evidence. All three terms are on approximately the same [-1, 1] scale.
LLMRL_Q_ACTION_WEIGHT = 0.35
LLMRL_LLM_NEIGHBOR_WEIGHT = 0.25
LLMRL_HEURISTIC_WEIGHT = 0.40

# Intermediate reward shaping for useful geographic progress. Regressive hops
# are penalized more strongly because they consume one complete 250 ms tick.
LLMRL_PROGRESS_REWARD_WEIGHT = 0.40
LLMRL_HOP_COST = 0.08
LLMRL_NON_PROGRESS_PENALTY = 0.30

# Quality-aware memory retrieval:
# semantic similarity remains dominant, while successful experiences are
# preferred over semantically similar deadline failures.
LTM_SIMILARITY_WEIGHT = 0.70
LTM_QUALITY_WEIGHT = 0.30

# Stop generating new traffic early enough that the last packet with the maximum
# deadline can still be routed before the experiment ends. ``steps`` remains the
# TOTAL number of simulation ticks; the final ticks are a drain/flush phase.
MAX_DEADLINE_MS = max(DEADLINE_CHOICES_MS)
DRAIN_TICKS = int(math.ceil(MAX_DEADLINE_MS / TICK_DURATION_MS))


# Génère la batterie initiale d'un UAV.
# La même valeur initiale est réutilisée dans LLM+RL et Q-FANET afin de
# garantir une comparaison équitable entre les deux approches.

def random_initial_battery() -> float:
    """Return a fresh random initial battery percentage for one UAV.

    A value is generated independently for every UAV at the beginning of a
    simulation run.  The same generated value is then given to the matching
    UAV in both LLM+RL and Q-FANET so that the comparison remains fair.
    """
    return round(
        random.uniform(BATTERY_INITIAL_MIN_PCT, BATTERY_INITIAL_MAX_PCT),
        2,
    )

# --- Neighbor model: HARD geometric gate first (<= COMM_RANGE), then in-range
#     Poisson/exponential radio churn/fading. ---
# Link churn rates are expressed per simulated SECOND.  Values preserve approximately
# the old per-tick behavior when one tick was implicitly treated as a whole time unit.
LAMBDA_RATE = 1.40   # activation rate / second for pairs that are physically within 180 m
MU_RATE = 0.60       # deactivation rate / second; mean active dwell ~= 1.67 s (~6.7 ticks)
MAX_DISTANCE = math.sqrt(
    (AREA[1] - AREA[0]) ** 2 + (AREA[3] - AREA[2]) ** 2 + 180.0 ** 2
)  # 3D upper bound used only to normalize geographic progress


# ============================================================
# QMR-inspired reward function (called by both routing systems)
# ============================================================

# Convertit un RSSI exprimé en dBm vers une valeur normalisée [0, 1].
# Cette normalisation permet d'intégrer la qualité radio dans une récompense
# composée sans mélanger directement des unités incompatibles.

def normalize_rssi(rssi_dbm: float) -> float:
    """Map RSSI in dBm to [0, 1]. Values <= -100 dBm -> 0, >= -40 dBm -> 1."""
    if RSSI_MAX_DBM <= RSSI_MIN_DBM:
        return 0.0
    value = (float(rssi_dbm) - RSSI_MIN_DBM) / (RSSI_MAX_DBM - RSSI_MIN_DBM)
    return max(0.0, min(value, 1.0))


# Calcule la récompense de base inspirée de QMR.
#
# Cas spéciaux :
#   - destination atteinte  -> r_max
#   - minimum local         -> r_min
#
# Sinon :
#   - délai,
#   - énergie du prochain saut,
#   - RSSI normalisé
# sont combinés pour produire la récompense de transition.
#
# Cette fonction est utilisée par LES DEUX systèmes afin que la comparaison
# conserve la même logique de récompense environnementale.

def calculate_qmr_reward(*, is_destination: bool, is_local_minimum: bool,
                         rssi_dbm: float, next_hop_energy_pct: float,
                         delay_ms: float, action: str = "forward"):
    """Return the QMR-inspired reward and its normalized components.

    The supplied QMR equation is preserved for the destination/local-minimum cases.
    In the normal case, the original delay-energy term is blended with RSSI because
    RSSI is an explicit requirement of this FANET implementation.

    delay is converted from milliseconds to seconds before exp(-delay), so the
    exponential remains numerically useful with the simulator's millisecond clock.
    For HOLD, RSSI contributes zero because no radio hop is actually used; delay and
    current energy still affect the reward.
    """
    delay_s = max(float(delay_ms), 0.0) / 1000.0
    delay_score = math.exp(-delay_s)
    energy_score = max(0.0, min(float(next_hop_energy_pct) / 100.0, 1.0))
    rssi_score = 0.0 if action == "hold" else normalize_rssi(rssi_dbm)

    if is_destination:
        reward = QMR_R_MAX
        case = "destination"
    elif is_local_minimum:
        reward = QMR_R_MIN
        case = "local_minimum"
    else:
        delay_energy = (
            QMR_OMEGA * delay_score
            + (1.0 - QMR_OMEGA) * energy_score
        )
        reward = (
            QMR_BETA * delay_energy
            + (1.0 - QMR_BETA) * rssi_score
        )
        case = "normal"

    return float(reward), {
        "case": case,
        "rssi_dbm": round(float(rssi_dbm), 2),
        "rssi_score": round(rssi_score, 4),
        "next_hop_energy_pct": round(float(next_hop_energy_pct), 2),
        "energy_score": round(energy_score, 4),
        "delay_ms": round(float(delay_ms), 2),
        "delay_s": round(delay_s, 4),
        "delay_score": round(delay_score, 4),
        "omega": QMR_OMEGA,
        "beta": QMR_BETA,
        "reward": round(float(reward), 4),
    }


# ============================================================
# Battery state model
# ============================================================

# État énergétique persistant d'un UAV.
# Le modèle est volontairement simple : batterie initiale aléatoire puis
# diminution déterministe à chaque tick de simulation.

@dataclass
class BatteryState:
    """Persistent battery state for one simulated UAV.

    By default a new battery starts at a random value in
    [BATTERY_INITIAL_MIN_PCT, BATTERY_INITIAL_MAX_PCT].  During a simulation
    run, explicit initial values are supplied so the LLM+RL UAV and the
    corresponding Q-FANET UAV start from exactly the same battery percentage.
    Every completed tick removes exactly 3 percentage points.
    """

    level_pct: float = field(default_factory=random_initial_battery)
    drain_per_tick_pct: float = BATTERY_DRAIN_PER_TICK_PCT
    history: list = field(default_factory=list)

    # Applique la consommation énergétique d'un tick et garde l'historique
    # avant/après pour l'analyse expérimentale.

    def consume_tick(self, step: int, time_ms: float) -> float:
        before = self.level_pct
        self.level_pct = max(0.0, self.level_pct - self.drain_per_tick_pct)
        self.history.append({
            "step": step,
            "time_ms": time_ms,
            "time_s": round(time_ms / 1000.0, 3),
            "before_pct": round(before, 2),
            "after_pct": round(self.level_pct, 2),
        })
        return self.level_pct

    @property
    def operational(self) -> bool:
        return self.level_pct >= MIN_OPERATIONAL_ENERGY


# ============================================================
# 1. Gauss-Markov mobility model  (same model, all UAVs)
# ============================================================

# Modèle de mobilité Gauss-Markov.
# Chaque UAV partage les mêmes paramètres statistiques, mais sa vitesse et
# sa direction évoluent indépendamment en fonction de son propre historique.

class GaussMarkovMobility:
    """Standard Gauss-Markov mobility model:
       v_t     = alpha*v_(t-1)     + (1-alpha)*v_mean     + sqrt(1-alpha^2)*N(0, sigma_v)
       theta_t = alpha*theta_(t-1) + (1-alpha)*theta_mean + sqrt(1-alpha^2)*N(0, sigma_theta)
    alpha in [0,1] is the "memory" of the model: alpha=1 -> pure Brownian motion continuity,
    alpha=0 -> fully random (memoryless) each step."""

    def __init__(self, x, y, z, area=AREA, alpha=0.75, mean_speed=DEFAULT_MEAN_SPEED_MPS,
                 mean_direction=None, speed_std=3.0, dir_std=0.35, alt_range=(80.0, 180.0)):
        self.position = [x, y, z]
        self.area = area
        self.alpha = alpha
        self.mean_speed = mean_speed
        self.mean_direction = mean_direction if mean_direction is not None else random.uniform(0, 2 * math.pi)
        self.speed = mean_speed
        self.direction = self.mean_direction
        self.speed_std = speed_std
        self.dir_std = dir_std
        self.alt_range = alt_range

    # Met à jour vitesse, direction et position avec mémoire Gauss-Markov.
    # ``dt`` correspond à la durée physique d'un tick (0.25 s par défaut).

    def update(self, dt=1.0):
        a = self.alpha
        self.speed = a * self.speed + (1 - a) * self.mean_speed + math.sqrt(1 - a ** 2) * random.gauss(0, self.speed_std)
        self.speed = max(0.5, self.speed)
        self.direction = a * self.direction + (1 - a) * self.mean_direction + math.sqrt(1 - a ** 2) * random.gauss(0, self.dir_std)

        x, y, z = self.position
        x += self.speed * math.cos(self.direction) * dt
        y += self.speed * math.sin(self.direction) * dt

        xmin, xmax, ymin, ymax = self.area
        if x < xmin or x > xmax:
            self.direction = math.pi - self.direction
            x = min(max(x, xmin), xmax)
        if y < ymin or y > ymax:
            self.direction = -self.direction
            y = min(max(y, ymin), ymax)

        z += random.uniform(-2.0, 2.0)
        z = min(max(z, self.alt_range[0]), self.alt_range[1])

        self.position = [x, y, z]
        return tuple(self.position), self.speed, self.direction


# ============================================================
# 2. IMU  (accelerometer + gyroscope + magnetometer -> on-board estimate)
# ============================================================

# Simulation simplifiée d'une IMU embarquée.
# Elle produit une estimation bruitée de la position, vitesse et altitude.
# Cette estimation est envoyée au LLM, tandis que l'état physique exact reste
# réservé au moteur de simulation.

class IMU:
    """Simulates the 3 IMU sensors and produces the UAV's own navigation ESTIMATE
    of position / velocity / altitude, separate from ground truth (dead-reckoning
    with small drift, periodically corrected -- like a real IMU+GPS/mag fusion)."""

    def __init__(self):
        self._drift = [0.0, 0.0, 0.0]

    # Produit les mesures simulées accéléromètre/gyroscope/magnétomètre et
    # l'estimation de navigation transmise au contrôleur.

    def sense(self, true_position, true_speed, true_direction):
        ax = true_speed * math.cos(true_direction) + random.gauss(0, 0.15)
        ay = true_speed * math.sin(true_direction) + random.gauss(0, 0.15)
        az = 9.81 + random.gauss(0, 0.05)  # gravity + noise

        gyro = (round(random.gauss(0, 1.0), 3),   # roll rate (deg/s)
                round(random.gauss(0, 1.0), 3),   # pitch rate (deg/s)
                round(random.gauss(0, 2.0), 3))   # yaw rate (deg/s)

        mag_heading = round((math.degrees(true_direction) + random.gauss(0, 3.0)) % 360, 1)

        # dead-reckoning drift, occasionally corrected (simulates GPS/mag fusion correction)
        self._drift = [d + random.gauss(0, 0.4) for d in self._drift]
        if random.random() < 0.2:
            self._drift = [d * 0.3 for d in self._drift]

        estimated_position = tuple(round(p + d, 2) for p, d in zip(true_position, self._drift))
        estimated_velocity = round(true_speed + random.gauss(0, 0.3), 2)
        estimated_altitude = round(true_position[2] + self._drift[2], 2)

        return {
            "accel": (round(ax, 3), round(ay, 3), round(az, 3)),
            "gyro": gyro,
            "mag_heading": mag_heading,
            "estimated_position": estimated_position,
            "estimated_velocity": estimated_velocity,
            "estimated_altitude": estimated_altitude,
        }


# ============================================================
# 3. Telemetry / PacketQueue
# ============================================================

# Structure regroupant l'observation courante d'un UAV.
# Elle sert d'interface propre entre la simulation physique et le contrôleur.

@dataclass
class Telemetry:
    uav_id: str
    energy: float
    speed: float
    true_position: tuple
    estimated_position: tuple
    estimated_velocity: float
    estimated_altitude: float
    direction_deg: float
    mag_heading: float
    accel: tuple
    gyro: tuple
    distance_to_gs: float
    rssi_best_neighbor: float
    connectivity: str   # "isolated" / "weak_link" / "strong_link"
    sens: str            # compass sense derived from direction


# Représentation d'un paquet DATA suivi individuellement.
# Chaque changement important est ajouté à ``trajectory`` afin de pouvoir
# reconstruire la route complète du paquet pour le dashboard et le mémoire.

@dataclass
class PacketBatch:
    """One deadline-aware DATA packet with explicit node/status tracking.

    ``current_node`` gives the packet's latest known location. ``status`` gives its
    latest lifecycle state. ``trajectory`` preserves every important state change so
    we can answer questions such as: "Where was DATA-000007 at step 12?"
    """
    batch_id: str                    # packet ID, e.g. DATA-000001
    count: int
    created_at_ms: float
    initial_deadline_ms: float
    deadline_at_ms: float
    size_bytes: int = DEFAULT_DATA_PACKET_SIZE_BYTES
    hops: int = 0
    last_hop_at_ms: float = None
    current_node: str = "SOURCE"
    status: str = "CREATED"
    trajectory: list = field(default_factory=list)

    @property
    def packet_id(self) -> str:
        return self.batch_id

    def remaining_ms(self, now_ms: float) -> float:
        return self.deadline_at_ms - now_ms

    # Ajoute un événement immuable dans la trajectoire du paquet.
    # Cette méthode est appelée à chaque génération, injection, forward, hold,
    # livraison ou expiration.

    def record_state(self, *, step, time_ms: float, node: str, status: str,
                     event: str, system: str = "shared", from_node: str = None):
        """Record an explicit packet state/location transition.

        The trajectory is append-only and copied when a packet moves to another
        node, so LLM+RL and Q-FANET maintain independent histories after injection.
        """
        self.current_node = str(node)
        self.status = str(status)
        entry = {
            "step": step,
            "time_ms": round(float(time_ms), 3),
            "time_s": round(float(time_ms) / 1000.0, 6),
            "node": str(node),
            "status": str(status),
            "event": str(event),
            "system": str(system),
            "hops": int(self.hops),
        }
        if from_node is not None:
            entry["from_node"] = str(from_node)
        self.trajectory.append(entry)
        return entry

    def state_at_step(self, step: int):
        """Return the latest known packet state at or before ``step``."""
        candidates = [
            event for event in self.trajectory
            if event.get("step") is not None and int(event["step"]) <= int(step)
        ]
        return candidates[-1] if candidates else None

    def tracking_record(self) -> dict:
        return {
            "packet_id": self.packet_id,
            "current_node": self.current_node,
            "status": self.status,
            "hops": self.hops,
            "created_at_ms": self.created_at_ms,
            "deadline_at_ms": self.deadline_at_ms,
            "size_bytes": self.size_bytes,
            "trajectory": [dict(event) for event in self.trajectory],
        }

    def forwarded_copy(self, forwarded_at_ms: float = None) -> "PacketBatch":
        return PacketBatch(
            batch_id=self.batch_id,
            count=self.count,
            created_at_ms=self.created_at_ms,
            initial_deadline_ms=self.initial_deadline_ms,
            deadline_at_ms=self.deadline_at_ms,
            size_bytes=self.size_bytes,
            hops=self.hops + 1,
            last_hop_at_ms=(
                float(forwarded_at_ms)
                if forwarded_at_ms is not None
                else self.last_hop_at_ms
            ),
            current_node=self.current_node,
            status=self.status,
            trajectory=[dict(event) for event in self.trajectory],
        )

    def clone(self) -> "PacketBatch":
        return PacketBatch(
            batch_id=self.batch_id,
            count=self.count,
            created_at_ms=self.created_at_ms,
            initial_deadline_ms=self.initial_deadline_ms,
            deadline_at_ms=self.deadline_at_ms,
            size_bytes=self.size_bytes,
            hops=self.hops,
            last_hop_at_ms=self.last_hop_at_ms,
            current_node=self.current_node,
            status=self.status,
            trajectory=[dict(event) for event in self.trajectory],
        )


# Structure d'un paquet HELLO du plan de contrôle.
# Les HELLO servent à maintenir une vision fraîche du voisinage radio.

@dataclass(frozen=True)
class HelloPacket:
    packet_id: str
    sender_id: str
    sequence: int
    created_at_ms: float
    expires_at_ms: float
    size_bytes: int = DEFAULT_HELLO_PACKET_SIZE_BYTES


# File DATA illimitée d'un UAV.
# Aucun paquet n'est perdu par overflow.
# Sous forte charge, la capacité de service finie peut créer une file d'attente;
# les pertes apparaissent alors naturellement par expiration des deadlines.

@dataclass
class PacketQueue:
    """Unlimited queue made of independent deadline-aware DATA packets."""
    batches: list = field(default_factory=list)
    current_time_ms: float = 0.0

    @property
    def pending_packets(self) -> int:
        return sum(batch.count for batch in self.batches)

    @property
    def pending_bytes(self) -> int:
        return sum(batch.count * batch.size_bytes for batch in self.batches)

    @property
    def min_deadline_ms(self) -> float:
        if not self.batches:
            return LARGE_DEADLINE
        return min(batch.remaining_ms(self.current_time_ms) for batch in self.batches)

    @property
    def batch_ids(self) -> set:
        return {batch.batch_id for batch in self.batches}

    def set_time(self, now_ms: float):
        self.current_time_ms = float(now_ms)

    def add_batch(self, batch: PacketBatch):
        self.batches.append(batch)

    def add_batches(self, batches):
        self.batches.extend(batches)

    def pop_all(self) -> list:
        batches = self.batches
        self.batches = []
        return batches

    def pop_for_transmission(self, max_packets: int) -> list:
        """Remove at most ``max_packets`` DATA packets using EDF scheduling.

        EDF = Earliest Deadline First.

        Why EDF?
        --------
        When the offered load becomes high (for example 100 packets/s), the
        relay queue can contain more packets than the UAV can forward during one
        250 ms tick. Selecting the most urgent packets first is a reasonable QoS
        policy for this deadline-constrained FANET.

        The configured Source creates one PacketBatch per DATA packet
        (``count=1``), therefore the normal path is exact packet-by-packet
        scheduling. The generic ``count`` field is still respected.
        """
        budget = max(0, int(max_packets))
        if budget <= 0 or not self.batches:
            return []

        # Earliest absolute deadline first; creation time and packet ID make
        # the result deterministic when deadlines are equal.
        ordered = sorted(
            self.batches,
            key=lambda b: (
                b.deadline_at_ms,
                b.created_at_ms,
                b.packet_id,
            ),
        )

        selected = []
        selected_ids = set()
        used = 0

        for batch in ordered:
            if used + batch.count > budget:
                continue

            selected.append(batch)
            selected_ids.add(id(batch))
            used += batch.count

            if used >= budget:
                break

        self.batches = [
            batch
            for batch in self.batches
            if id(batch) not in selected_ids
        ]

        return selected

    # Sépare les paquets expirés des paquets encore routables à l'instant donné.

    def expire_at(self, now_ms: float) -> list:
        self.set_time(now_ms)
        expired, alive = [], []
        for batch in self.batches:
            (expired if batch.deadline_at_ms <= now_ms else alive).append(batch)
        self.batches = alive
        return expired

    def mean_hop_wait_ms(self, now_ms: float) -> float:
        """Mean time DATA packets have waited at their current node."""
        if not self.batches:
            return 0.0
        weighted_sum = 0.0
        total = 0
        for batch in self.batches:
            entered_at = (
                batch.last_hop_at_ms
                if batch.last_hop_at_ms is not None
                else batch.created_at_ms
            )
            wait_ms = max(0.0, float(now_ms) - float(entered_at))
            weighted_sum += wait_ms * batch.count
            total += batch.count
        return weighted_sum / max(total, 1)

    def deadline_snapshot(self, now_ms: float) -> dict:
        self.set_time(now_ms)
        remaining = [max(0.0, b.remaining_ms(now_ms)) for b in self.batches]
        return {
            "pending_packets": self.pending_packets,
            "pending_bytes": self.pending_bytes,
            "pending_batches": len(self.batches),
            "min_remaining_ms": min(remaining) if remaining else None,
            "mean_remaining_ms": (sum(remaining) / len(remaining)) if remaining else None,
            "deadlines_ms": [round(v, 1) for v in remaining],
        }

    def as_memory_dict(self) -> dict:
        return {
            "pending_packets": self.pending_packets,
            "pending_bytes": self.pending_bytes,
            "min_deadline_ms": self.min_deadline_ms,
            "batch_ids": sorted(self.batch_ids),
            "batches": [asdict(b) for b in self.batches],
        }


# Convertit une direction en radians vers une direction cardinale lisible.

def compass_sense(direction_rad: float) -> str:
    deg = math.degrees(direction_rad) % 360
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((deg + 22.5) % 360) / 45)
    return dirs[idx]


# Approximation du RSSI à partir de la distance avec un modèle de path loss
# logarithmique simplifié.

def compute_rssi(distance_m: float) -> float:
    """Simple log-distance path loss model -> approximate RSSI in dBm."""
    d = max(distance_m, 1.0)
    tx_power = 20.0       # dBm
    path_loss_exponent = 2.5
    ref_loss_at_1m = 40.0  # dB
    path_loss = ref_loss_at_1m + 10 * path_loss_exponent * math.log10(d)
    return round(tx_power - path_loss, 1)


# ============================================================
# 4. Network: position registry + distance utilities.
#    UAV<->UAV / UAV<->GS connectivity is now decided by PeerLinkChurnModel
#    (stochastic, see below) instead of a geometric comm_range threshold.
#    Positions are still tracked here (from real Gauss-Markov mobility) because
#    they're still needed for: distance-to-GS (state/reward), RSSI once a link
#    is active, and the Source's single fixed link (kept geometric, see Source).
# ============================================================

# Registre physique du réseau.
# Il centralise les positions courantes des UAV, la position du GS et les
# fonctions de distance utilisées par le routage.

class Network:
    def __init__(self, gs_position=GS_POSITION, source_position=SOURCE_POSITION, comm_range=COMM_RANGE):
        self.gs_position = gs_position
        self.source_position = source_position
        self.comm_range = comm_range
        self.positions = {}       # uav_id -> (x,y,z)
        self.uav_lookup = {}      # uav_id -> UAVAgent

    def register(self, uav_agent):
        self.uav_lookup[uav_agent.uav_id] = uav_agent

    def update_positions(self):
        self.positions = {uid: agent.mobility.position for uid, agent in self.uav_lookup.items()}

    @staticmethod
    def distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def distance_to_gs(self, uav_id):
        return self.distance(self.positions[uav_id], self.gs_position)

    def nearest_uav_to(self, position):
        """Used only by the fixed Source to find its single entry point into the
        swarm -- kept geometric since it's one fixed link, not UAV<->UAV churn."""
        in_range = [(uid, self.distance(position, pos)) for uid, pos in self.positions.items()
                    if self.distance(position, pos) <= self.comm_range]
        if not in_range:
            return None
        best_id = min(in_range, key=lambda x: x[1])[0]
        return self.uav_lookup[best_id]


# ============================================================
# 4bis. Connectivite partagee (P1) : IMPORTEE depuis neighbor_discovery.py (voir
#       le haut de ce fichier), qui est le module livrable pour cette etape du
#       plan de simulation. SharedLinkGraph y implemente le meme processus de
#       naissance-mort de Poisson/exponentielle que documente la-bas, mais
#       applique par ARETE NON ORIENTEE (symetrique) plutot que par agent --
#       plus realiste (un lien radio est bidirectionnel) et necessaire pour que
#       les deux systemes compares partagent exactement la meme connectivite.
# ============================================================


# ============================================================
# 4ter. RoutingUtility -- U_i = w1*Proximite + w2*Stabilite + w3*Energie
#       Chooses which active link to use as next hop (LLM+RL system only --
#       QFANETRouter below uses its own Q-learning-based neighbor selection instead).
#       Weights are updated every tick from the LLM Core's CoT reasoning.
# ============================================================

# Classe utilisée UNIQUEMENT par le système LLM+RL pour classer les voisins.
#
# U_i = w1 * proximité_GS + w2 * stabilité + w3 * énergie
#
# Les poids w1/w2/w3 sont fournis dynamiquement par le LLM puis normalisés.

class RoutingUtility:
    """Dynamic next-hop utility used by the LLM+RL system.

    The LLM chooses the weights:
        w1 -> proximity to the Ground Station
        w2 -> link stability / radio reliability
        w3 -> next-hop energy

    IMPORTANT FIX:
    The previous stability term was identical for every link because every edge
    used the same churn parameter ``mu``. That meant changing w2 did not really
    change the ranking. Stability now combines:
        - stochastic link survival probability,
        - normalized RSSI of THIS specific link.

    Therefore all three LLM-produced weights now influence next-hop ranking.
    """

    def __init__(self, w1: float = 0.4, w2: float = 0.3, w3: float = 0.3):
        self.w1, self.w2, self.w3 = w1, w2, w3

    def update_weights(self, w1: float, w2: float, w3: float):
        # LLM outputs are constrained to non-negative [0, 1] weights before normalization.
        vals = [max(0.0, min(float(v), 1.0)) for v in (w1, w2, w3)]
        total = sum(vals)
        if total <= 0:
            vals = [0.4, 0.3, 0.3]
            total = 1.0
        self.w1, self.w2, self.w3 = (v / total for v in vals)

    @staticmethod
    def proximity_score(distance_to_gs: float, max_distance: float) -> float:
        return max(0.0, 1.0 - min(distance_to_gs / max_distance, 1.0))

    @staticmethod
    def stability_score(rssi_dbm: float, mu_rate: float = MU_RATE,
                        dt: float = SIM_TICK_SECONDS) -> float:
        """Link-specific stability score in [0, 1].

        ``exp(-mu*dt)`` models survival of an active edge during the next tick.
        RSSI differentiates links that otherwise share the same stochastic churn
        parameter.
        """
        churn_survival = math.exp(-mu_rate * dt)
        radio_quality = normalize_rssi(rssi_dbm)
        return 0.5 * churn_survival + 0.5 * radio_quality

    @staticmethod
    def energy_score(energy_pct: float) -> float:
        return max(0.0, min(energy_pct / 100.0, 1.0))

    def compute_utility(self, link: tuple, network: Network, max_distance: float,
                        mu_rate: float = MU_RATE) -> float:
        node_id, dist, rssi, is_gs, energy = link
        dest_distance = 0.0 if is_gs else network.distance_to_gs(node_id)

        prox = self.proximity_score(dest_distance, max_distance)
        stab = self.stability_score(rssi, mu_rate)
        energy_s = self.energy_score(energy)

        return (
            self.w1 * prox
            + self.w2 * stab
            + self.w3 * energy_s
        )

    def rank_links(self, links: list, network: Network, max_distance: float,
                   mu_rate: float = MU_RATE) -> list:
        """Return ``[(link, utility), ...]`` sorted best-first.

        Utility remains the primary criterion. Deterministic tie-breakers then
        prefer:
          1) direct Ground Station,
          2) smaller remaining distance to GS,
          3) stronger RSSI,
          4) higher energy.
        """
        scored = [
            (
                link,
                self.compute_utility(
                    link,
                    network,
                    max_distance,
                    mu_rate,
                ),
            )
            for link in links
        ]

        def sort_key(pair):
            link, utility = pair
            node_id, _dist, rssi, is_gs, energy = link
            dest_distance = (
                0.0
                if is_gs
                else network.distance_to_gs(node_id)
            )

            return (
                utility,
                1 if is_gs else 0,
                -dest_distance,
                rssi,
                energy,
            )

        scored.sort(
            key=sort_key,
            reverse=True,
        )
        return scored


# ============================================================
# 5. Fixed Source (emetteur fixe) and Ground Station (destinataire fixe)
# ============================================================

# Source DATA fixe.
# Elle génère les paquets à l'intervalle demandé et injecte une copie identique
# dans les deux systèmes comparés.

class Source:
    """Fixed DATA source with a configurable offered load in packets/s.

    The user configures the traffic directly with ``packet_rate_pps``:

        10 pps  -> 100 ms between DATA packets
        100 pps ->  10 ms between DATA packets

    Internally we keep the equivalent interval because the discrete-event
    generator works with exact millisecond timestamps.
    """

    def __init__(self, position=SOURCE_POSITION,
                 packet_rate_pps=DEFAULT_PACKET_RATE_PPS,
                 packet_size_bytes=DEFAULT_DATA_PACKET_SIZE_BYTES,
                 deadline_choices=DEADLINE_CHOICES_MS):
        self.position = position
        self.packet_rate_pps = float(packet_rate_pps)
        self.packet_size_bytes = int(packet_size_bytes)
        self.deadline_choices = tuple(float(v) for v in deadline_choices)

        if self.packet_rate_pps <= 0:
            raise ValueError("packet_rate_pps must be > 0")

        self.data_interval_ms = 1000.0 / self.packet_rate_pps
        if self.packet_size_bytes <= 0:
            raise ValueError("packet_size_bytes must be > 0")

        self.total_generated = 0
        self.total_generated_bytes = 0
        self.generation_log = []
        self.backlog_packets = []
        self.source_expired = 0
        self.source_expired_records = []
        self._next_packet_seq = 1

        # ------------------------------------------------------------
        # DATA GENERATION CLOCK
        # ------------------------------------------------------------
        # The first DATA packet is generated immediately at t = 0 ms.
        #
        # Example with --packet-rate-pps 10:
        #
        #   DATA-000001 -> t =    0 ms
        #   DATA-000002 -> t =  100 ms
        #   DATA-000003 -> t =  200 ms
        #   DATA-000004 -> t =  300 ms
        #   ...
        #
        # After every generation, Source.tick() increments this value by
        # ``self.data_interval_ms``.
        # ------------------------------------------------------------
        self.next_data_time_ms = 0.0

    # Crée un nouvel identifiant DATA unique et lui attribue une deadline absolue.

    def _new_data_packet(self, created_at_ms: float) -> PacketBatch:
        packet_id = f"DATA-{self._next_packet_seq:06d}"
        self._next_packet_seq += 1
        deadline = float(random.choice(self.deadline_choices))
        packet = PacketBatch(
            batch_id=packet_id,
            count=1,
            created_at_ms=float(created_at_ms),
            initial_deadline_ms=deadline,
            deadline_at_ms=float(created_at_ms) + deadline,
            size_bytes=self.packet_size_bytes,
            hops=0,
            last_hop_at_ms=float(created_at_ms),
            current_node="SOURCE",
            status="CREATED",
        )
        generation_step = max(1, int(math.ceil(float(created_at_ms) / TICK_DURATION_MS)))
        packet.record_state(
            step=generation_step,
            time_ms=created_at_ms,
            node="SOURCE",
            status="CREATED",
            event="DATA_GENERATED",
            system="shared_source",
        )
        self.total_generated += 1
        self.total_generated_bytes += self.packet_size_bytes
        self.generation_log.append({
            "packet_id": packet_id,
            "batch_id": packet_id,
            "packet_type": "DATA",
            "count": 1,
            "size_bytes": self.packet_size_bytes,
            "created_at_ms": float(created_at_ms),
            "created_at_s": round(float(created_at_ms) / 1000.0, 6),
            "deadline_ms": deadline,
            "deadline_at_ms": float(created_at_ms) + deadline,
            "injected_at_ms": None,
            "target_uav": None,
        })
        return packet

    # À chaque tick de routage, crée tous les DATA dont l'horodatage est arrivé,
    # puis tente de les injecter dans l'UAV d'entrée le plus proche.

    def tick(self, network: Network, qfanet_router: "QFANETRouter",
             simulation_time_ms: float, generation_cutoff_ms: float = None):
        """Generate every due DATA packet, then inject backlog into the nearest UAV.

        DATA generation starts immediately at t=0 ms.

        For example, with ``data_interval_ms=100`` the exact DATA timestamps are:

            0, 100, 200, 300, 400, ... ms

        ``simulation_time_ms`` is the simulation clock at the BEGINNING of the
        current 250 ms routing tick. Therefore:

          - Tick 1 starts at t=0 ms:
                DATA at 0 ms is generated and can be injected immediately.

          - Tick 2 starts at t=250 ms:
                DATA scheduled at 100 ms and 200 ms are generated with their exact
                creation timestamps and become available for routing at that tick.

        The DATA generation interval therefore remains independent from the routing
        tick duration.
        """
        now_ms = float(simulation_time_ms)
        generation_end_ms = (
            None
            if generation_cutoff_ms is None
            else float(generation_cutoff_ms)
        )

        generated_ids = []
        eps = 1e-9

        # A packet is due if its timestamp has already been reached by the
        # simulation clock. The global traffic end is EXCLUSIVE:
        #
        #   10 pps during 2.0 s -> timestamps 0,100,...,1900 ms = 20 packets
        #  100 pps during 2.0 s -> timestamps 0,10,...,1990 ms  = 200 packets
        #
        # This avoids an extra packet exactly at the first instant of the
        # drain phase.
        while (
            self.next_data_time_ms <= now_ms + eps
            and (
                generation_end_ms is None
                or self.next_data_time_ms < generation_end_ms - eps
            )
        ):
            packet = self._new_data_packet(
                self.next_data_time_ms
            )
            self.backlog_packets.append(packet)
            generated_ids.append(packet.packet_id)
            self.next_data_time_ms += self.data_interval_ms

        # Packets can expire while still waiting at the fixed Source for an entry UAV.
        alive_backlog = []
        source_step = max(1, int(math.floor(now_ms / TICK_DURATION_MS)) + 1)
        for packet in self.backlog_packets:
            if packet.deadline_at_ms <= now_ms:
                self.source_expired += packet.count
                packet.record_state(
                    step=source_step,
                    time_ms=now_ms,
                    node="SOURCE",
                    status="DROPPED_DEADLINE",
                    event="SOURCE_DEADLINE_EXPIRED",
                    system="shared_source",
                )
                self.source_expired_records.append(packet.tracking_record())
            else:
                # Keep a visible source-side state while waiting for an entry UAV.
                packet.record_state(
                    step=source_step,
                    time_ms=now_ms,
                    node="SOURCE",
                    status="WAITING_SOURCE",
                    event="WAITING_FOR_ENTRY_UAV",
                    system="shared_source",
                )
                alive_backlog.append(packet)
        self.backlog_packets = alive_backlog

        target = network.nearest_uav_to(self.position)
        injected_ids = []
        routing_step = max(1, int(math.floor(now_ms / TICK_DURATION_MS)) + 1)
        if target and self.backlog_packets:
            for packet in self.backlog_packets:
                llm_packet = packet.clone()
                llm_packet.record_state(
                    step=routing_step,
                    time_ms=now_ms,
                    node=target.uav_id,
                    status="QUEUED",
                    event="INJECTED_FROM_SOURCE",
                    system="llm_rl",
                    from_node="SOURCE",
                )
                target.packets.add_batch(llm_packet)
                qfanet_router.inject_batch(target.uav_id, packet, step=routing_step, time_ms=now_ms)
                injected_ids.append(packet.packet_id)
                for record in reversed(self.generation_log):
                    if record["packet_id"] == packet.packet_id:
                        record["injected_at_ms"] = now_ms
                        record["target_uav"] = target.uav_id
                        break
            self.backlog_packets = []

        if generated_ids:
            print(
                f"[Source t={now_ms/1000.0:.2f}s] generated {len(generated_ids)} DATA packet(s) "
                f"{generated_ids}; rate={self.packet_rate_pps:.2f} packet/s, "
                f"interval={self.data_interval_ms:.3f} ms, "
                f"size={self.packet_size_bytes} B"
            )
        if injected_ids:
            print(
                f"[Source t={now_ms/1000.0:.2f}s] injected {len(injected_ids)} DATA packet(s) "
                f"into {target.uav_id}: {injected_ids}"
            )
        elif self.backlog_packets:
            print(
                f"[Source t={now_ms/1000.0:.2f}s] no UAV within {COMM_RANGE:.0f} m -- "
                f"{len(self.backlog_packets)} DATA packet(s) waiting at Source"
            )


# Transforme la trajectoire détaillée d'un paquet en chaîne compacte :
#     UAV-2 -> UAV-5 -> UAV-13 -> GS
# Les états répétés sur le même UAV sont volontairement supprimés.

def trajectory_to_path(trajectory: list, include_source: bool = False) -> str:
    """Convert packet trajectory events to a compact route string.

    Repeated states at the same node (e.g. HELD/WAITING) are collapsed so the
    dashboard shows an actual route such as UAV-1 -> UAV-2 -> UAV-4 -> GS.
    """
    nodes = []
    for event in trajectory or []:
        node = event.get("node")
        if not node:
            continue
        if not include_source and node == "SOURCE":
            continue
        if not nodes or nodes[-1] != node:
            nodes.append(str(node))
    return " -> ".join(nodes) if nodes else "n/a"


# Destination finale du FANET.
# Elle réceptionne les paquets valides, calcule leur délai end-to-end et
# conserve l'historique complet de livraison.

class GroundStation:
    def __init__(self, position=GS_POSITION):
        self.position = position
        self.delivered_count = 0
        self.delivered_bytes = 0
        self.delivery_log = []
        self.delivery_records = []
        self.seen_batch_ids = set()

    # Termine la trajectoire d'un paquet, calcule sa latence end-to-end et
    # enregistre son chemin pour l'export/dashboard.

    def receive(self, uav_id: str, batches: list, step: int, simulation_time_ms: float,
                system: str = "llm_rl") -> int:
        """Receive non-expired DATA packets and log end-to-end delay/deadline."""
        delivered_now = 0
        delivered_bytes_now = 0
        new_ids = []
        for batch in batches:
            if batch.batch_id in self.seen_batch_ids:
                continue
            self.seen_batch_ids.add(batch.batch_id)
            batch.record_state(
                step=step,
                time_ms=simulation_time_ms,
                node="GS",
                status="DELIVERED",
                event="DELIVERED_TO_GS",
                system=system,
                from_node=uav_id,
            )
            delivered_now += batch.count
            delivered_bytes_now += batch.count * batch.size_bytes
            new_ids.append(batch.batch_id)
            latency_ms = simulation_time_ms - batch.created_at_ms
            route = trajectory_to_path(batch.trajectory, include_source=False)
            full_route = trajectory_to_path(batch.trajectory, include_source=True)
            self.delivery_records.append({
                "step": step,
                "time_ms": simulation_time_ms,
                "time_s": simulation_time_ms / 1000.0,
                "received_at_ms": simulation_time_ms,
                "received_at_s": simulation_time_ms / 1000.0,
                "uav_id": uav_id,
                "packet_id": batch.packet_id,
                "batch_id": batch.batch_id,
                "count": batch.count,
                "size_bytes": batch.size_bytes,
                "created_at_ms": batch.created_at_ms,
                "created_at_s": batch.created_at_ms / 1000.0,
                "initial_deadline_ms": batch.initial_deadline_ms,
                "deadline_at_ms": batch.deadline_at_ms,
                "latency_ms": latency_ms,
                "deadline_margin_ms": batch.deadline_at_ms - simulation_time_ms,
                "hops": batch.hops,
                "current_node": batch.current_node,
                "status": batch.status,
                "path": route,
                "full_path": full_route,
                "trajectory": [dict(event) for event in batch.trajectory],
            })

        if delivered_now > 0:
            self.delivered_count += delivered_now
            self.delivered_bytes += delivered_bytes_now
            self.delivery_log.append((step, uav_id, delivered_now))
            print(
                f"[GS t={simulation_time_ms/1000.0:.2f}s] received {delivered_now} DATA packet(s) "
                f"({delivered_bytes_now} B, IDs {new_ids}) from {uav_id}; total={self.delivered_count}"
            )
            for record in self.delivery_records[-len(new_ids):]:
                print(
                    f"  [GS history] {record['packet_id']} received_at={record['received_at_s']:.3f}s "
                    f"delay={record['latency_ms']:.1f}ms path={record['path']}"
                )
        return delivered_now


# ============================================================
# 5bis. QFANETRouter -- baseline concurrent : Q-learning pur (sans LLM), dans
#       l'esprit des protocoles Q-FANET / QMR de la litterature FANET (Q-routing
#       classique de Boyan & Littman, 1993). Tourne sur le MEME SharedLinkGraph
#       que le systeme LLM+RL pour une comparaison equitable et controlee.
#
#       SIMPLIFICATION (documentee, a mentionner dans votre rapport) : modele de
#       file AGREGEE, comme le reste du projet (un compteur de paquets en attente
#       + une deadline par noeud, comme PacketQueue), plutot qu'un etat par paquet
#       individuel. Une decision de transmission par noeud et par tick.
# ============================================================

# Baseline Q-FANET / QMR-inspired.
#
# Cette partie N'UTILISE PAS le LLM, la STM ou la LTM.
# Elle permet de comparer le système proposé à un routage Q-learning pur
# sous exactement les mêmes conditions de mobilité, trafic et connectivité.

class QFANETRouter:
    """Independent tabular Q-FANET/QMR-inspired baseline.

    The baseline shares only mobility, connectivity, traffic and initial battery
    conditions with LLM+RL. It has no LLM, STM or LTM access.

    Important routing safeguards:
      - GS is always selected when directly reachable.
      - Only physically reachable and operational actions are considered.
      - Geographic progress is used as an action mask when possible.
      - Equal/near-equal Q values are resolved with a deterministic link-quality
        tie-breaker instead of a random choice.
      - Urgent packets disable exploratory random hops.
      - TD bootstrapping uses only actions reachable from the next UAV in the
        CURRENT topology snapshot.
      - Intermediate hops receive a small cost and Q targets are clipped to the
        reward scale, preventing long paths from becoming more valuable than
        immediate delivery.
    """

    def __init__(self, network: Network, uav_ids: list, gs_id: str = "GS",
                 alpha: float = 0.3, gamma: float = 0.9, epsilon: float = EPSILON_START,
                 q_tables_init: dict = None, initial_batteries: dict = None,
                 relay_capacity_pps: float = DEFAULT_RELAY_CAPACITY_PPS):
        self.network = network
        self.gs_id = gs_id
        self.uav_ids = list(uav_ids)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random()

        # ------------------------------------------------------------
        # FINITE DATA SERVICE CAPACITY
        # ------------------------------------------------------------
        # Both Q-FANET and LLM+RL receive exactly the same per-UAV capacity.
        # A fractional residual keeps non-integer packet/s rates accurate
        # across 250 ms ticks without banking unused whole-packet capacity.
        # ------------------------------------------------------------
        self.relay_capacity_pps = float(relay_capacity_pps)
        if self.relay_capacity_pps <= 0:
            raise ValueError("relay_capacity_pps must be > 0")

        self.tx_capacity_residual = {
            uid: 0.0
            for uid in uav_ids
        }
        self.tx_budget_packets = {
            uid: 0
            for uid in uav_ids
        }

        self.queues = {uid: PacketQueue() for uid in uav_ids}

        # Q-FANET receives the SAME per-UAV initial battery values as LLM+RL.
        # The two systems then evolve independently.
        initial_batteries = initial_batteries or {}
        self.batteries = {
            uid: BatteryState(
                level_pct=(
                    float(initial_batteries[uid])
                    if uid in initial_batteries
                    else random_initial_battery()
                )
            )
            for uid in uav_ids
        }

        # Persistent table format is intentionally kept backward-compatible:
        # {uav_id: {peer_id: q_value}}.
        q_tables_init = q_tables_init or {}
        self.q = {}
        for uid in uav_ids:
            peers = [pid for pid in uav_ids if pid != uid] + [gs_id]
            saved = q_tables_init.get(uid, {})
            self.q[uid] = {
                pid: float(saved.get(pid, INITIAL_Q_VALUE))
                for pid in peers
            }

        self.delivered_count = 0
        self.dropped_deadline = 0
        self.dropped_no_path = 0
        self.delivery_log = []
        self.deadline_drop_log = []
        self.decision_log = []

    def inject_batch(self, uav_id: str, packet: PacketBatch, step=None, time_ms: float = None):
        clone = packet.clone()
        if time_ms is None:
            time_ms = packet.created_at_ms
        if step is None:
            step = max(1, int(math.floor(float(time_ms) / TICK_DURATION_MS)) + 1)

        clone.record_state(
            step=step,
            time_ms=time_ms,
            node=uav_id,
            status="QUEUED",
            event="INJECTED_FROM_SOURCE",
            system="qfanet",
            from_node="SOURCE",
        )
        self.queues[uav_id].add_batch(clone)

    def inject(self, uav_id: str, count: int, deadline: float, batch_id: str,
               created_at_ms: float, size_bytes: int = DEFAULT_DATA_PACKET_SIZE_BYTES):
        """Backward-compatible injection wrapper."""
        self.inject_batch(
            uav_id,
            PacketBatch(
                batch_id=batch_id,
                count=count,
                created_at_ms=created_at_ms,
                initial_deadline_ms=deadline,
                deadline_at_ms=created_at_ms + deadline,
                size_bytes=size_bytes,
                last_hop_at_ms=created_at_ms,
            ),
        )

    def all_positions(self) -> dict:
        pos = dict(self.network.positions)
        pos[self.gs_id] = self.network.gs_position
        return pos

    def _usable_neighbors(self, uav_id: str, active_neighbors: list,
                          require_progress: bool = True) -> list:
        """Return only currently reachable/operational actions.

        When at least one neighbor makes geographic progress toward the GS, only
        progress-making neighbors are returned. If none exists, all usable
        neighbors are retained so the protocol can escape temporary local minima.
        """
        usable = [
            pid for pid in active_neighbors
            if (
                pid == self.gs_id
                or (
                    pid in self.batteries
                    and self.batteries[pid].operational
                    and pid in self.network.positions
                )
            )
        ]

        if not usable:
            return []

        # Destination is terminal and always dominates another relay hop.
        if self.gs_id in usable:
            return [self.gs_id]

        if not require_progress:
            return usable

        current_dist = self.network.distance_to_gs(uav_id)
        progress = [
            pid for pid in usable
            if self.network.distance_to_gs(pid) < current_dist
        ]

        return progress if progress else usable

    def _is_local_minimum(self, uav_id: str, active_neighbors: list) -> bool:
        current_dist = self.network.distance_to_gs(uav_id)

        for pid in active_neighbors:
            if pid == self.gs_id:
                return False
            if pid not in self.network.positions:
                continue
            if pid not in self.batteries or not self.batteries[pid].operational:
                continue
            if self.network.distance_to_gs(pid) < current_dist:
                return False

        return True

    def _candidate_tiebreak_score(self, uav_id: str, peer_id: str,
                                  remaining_deadline_ms: float) -> float:
        """Topology/QoS score used ONLY to break near-equal Q values.

        Q-learning remains the main policy. This prevents a fresh 25-UAV table,
        whose unseen actions all start at Q=0.5, from choosing randomly among many
        geometrically very different neighbors.
        """
        if peer_id == self.gs_id:
            return 10.0

        current_pos = self.network.positions[uav_id]
        peer_pos = self.network.positions[peer_id]

        current_dist_gs = self.network.distance_to_gs(uav_id)
        next_dist_gs = self.network.distance_to_gs(peer_id)
        progress_m = max(0.0, current_dist_gs - next_dist_gs)
        progress_score = min(progress_m / COMM_RANGE, 1.0)

        link_dist = Network.distance(current_pos, peer_pos)
        rssi_score = normalize_rssi(compute_rssi(link_dist))
        energy_score = max(
            0.0,
            min(self.batteries[peer_id].level_pct / 100.0, 1.0)
        )

        if math.isfinite(remaining_deadline_ms):
            urgency = 1.0 - min(
                max(remaining_deadline_ms, 0.0) / MAX_DEADLINE_MS,
                1.0,
            )
        else:
            urgency = 0.0

        # As the deadline becomes critical, geographic progress dominates.
        w_progress = 0.55 + 0.25 * urgency
        w_rssi = 0.25 - 0.10 * urgency
        w_energy = 1.0 - w_progress - w_rssi

        return (
            w_progress * progress_score
            + w_rssi * rssi_score
            + w_energy * energy_score
        )

    # Sélection du prochain saut Q-FANET.
    # Le GS est prioritaire s'il est directement joignable.
    # En cas de Q-values proches, un tie-breaker réseau évite un choix arbitraire.

    def _choose_peer(self, uav_id: str, active_neighbors: list,
                     remaining_deadline_ms: float = LARGE_DEADLINE):
        usable = self._usable_neighbors(
            uav_id,
            active_neighbors,
            require_progress=True,
        )

        if not usable:
            return None

        # Never take an unnecessary relay hop when GS is already reachable.
        if usable == [self.gs_id]:
            return self.gs_id

        # Random exploration is useful while learning, but an urgent packet should
        # not be sacrificed to an arbitrary exploratory hop.
        effective_epsilon = self.epsilon
        if (
            math.isfinite(remaining_deadline_ms)
            and remaining_deadline_ms <= QFANET_URGENT_DEADLINE_MS
        ):
            effective_epsilon = 0.0

        if self.rng.random() < effective_epsilon:
            return self.rng.choice(usable)

        table = self.q[uav_id]
        best_q = max(table.get(pid, INITIAL_Q_VALUE) for pid in usable)

        # Include near ties. Fresh/unseen actions often share exactly Q=0.5, while
        # tiny numeric differences should not force a geometrically poor route.
        near_best = [
            pid for pid in usable
            if (
                best_q - table.get(pid, INITIAL_Q_VALUE)
                <= QFANET_Q_TIE_TOLERANCE
            )
        ]

        return max(
            near_best,
            key=lambda pid: self._candidate_tiebreak_score(
                uav_id,
                pid,
                remaining_deadline_ms,
            ),
        )

    def _reward(self, uav_id: str, peer_id: str, positions: dict,
                active_neighbors: list, delay_ms: float):
        link_dist = (
            Network.distance(positions[uav_id], self.network.gs_position)
            if peer_id == self.gs_id
            else Network.distance(positions[uav_id], positions[peer_id])
        )

        rssi = compute_rssi(link_dist)
        next_energy = (
            100.0
            if peer_id == self.gs_id
            else self.batteries[peer_id].level_pct
        )
        local_minimum = self._is_local_minimum(
            uav_id,
            active_neighbors,
        )

        return calculate_qmr_reward(
            is_destination=(peer_id == self.gs_id),
            is_local_minimum=local_minimum,
            rssi_dbm=rssi,
            next_hop_energy_pct=next_energy,
            delay_ms=delay_ms,
            action="forward",
        )

    # Calcule le futur Q seulement sur les actions réellement joignables dans
    # la topologie COURANTE, et non sur tous les voisins théoriques de la table.

    def _reachable_next_best(self, node_id: str, active_adj: dict) -> float:
        """Best Q value among actions that are feasible from node_id NOW."""
        if node_id == self.gs_id or node_id not in self.q:
            return 0.0

        candidates = self._usable_neighbors(
            node_id,
            active_adj.get(node_id, []),
            require_progress=True,
        )

        if not candidates:
            return 0.0

        return max(
            self.q[node_id].get(pid, INITIAL_Q_VALUE)
            for pid in candidates
        )

    def _record_expired(self, uav_id: str, expired: list, step: int, time_ms: float):
        if not expired:
            return

        count = sum(b.count for b in expired)
        self.dropped_deadline += count

        for b in expired:
            b.record_state(
                step=step,
                time_ms=time_ms,
                node=uav_id,
                status="DROPPED_DEADLINE",
                event="DEADLINE_EXPIRED",
                system="qfanet",
            )

            self.deadline_drop_log.append({
                "system": "qfanet",
                "step": step,
                "time_ms": time_ms,
                "time_s": time_ms / 1000.0,
                "uav_id": uav_id,
                "batch_id": b.batch_id,
                "packet_id": b.packet_id,
                "count": b.count,
                "initial_deadline_ms": b.initial_deadline_ms,
                "created_at_ms": b.created_at_ms,
                "hops": b.hops,
                "current_node": b.current_node,
                "status": b.status,
                "trajectory": [dict(event) for event in b.trajectory],
            })

    def _refresh_tx_budgets(self):
        """Compute the DATA forwarding budget for this 250 ms tick.

        Example with 40 packet/s relay capacity:
            40 * 0.25 = 10 DATA packets maximum per UAV per tick.
        """
        per_tick = (
            self.relay_capacity_pps
            * SIM_TICK_SECONDS
        )

        for uid in self.uav_ids:
            exact_budget = (
                per_tick
                + self.tx_capacity_residual[uid]
            )

            whole_packets = int(
                math.floor(exact_budget + 1e-12)
            )

            self.tx_capacity_residual[uid] = (
                exact_budget
                - whole_packets
            )

            self.tx_budget_packets[uid] = (
                whole_packets
            )

    # Exécute un tick complet de la baseline Q-FANET :
    # sélection du prochain saut, reward, TD update et transfert staged.

    def tick(self, step: int, active_adj: dict,
             ground_station_qfanet: GroundStation,
             simulation_time_ms: float):
        """Execute one synchronous Q-FANET tick (max one hop per packet batch)."""
        start_ms = simulation_time_ms
        end_ms = start_ms + TICK_DURATION_MS
        positions = self.all_positions()
        incoming = {uid: [] for uid in self.uav_ids}

        # Every UAV receives a fresh forwarding service budget for this tick.
        self._refresh_tx_budgets()

        for q in self.queues.values():
            q.set_time(start_ms)

        for uav_id in self.uav_ids:
            q = self.queues[uav_id]

            if q.pending_packets == 0:
                continue

            # A depleted UAV cannot transmit; packets remain queued until expiry.
            if not self.batteries[uav_id].operational:
                expired = q.expire_at(end_ms)
                self._record_expired(
                    uav_id,
                    expired,
                    step,
                    end_ms,
                )

                for b in q.batches:
                    b.record_state(
                        step=step,
                        time_ms=end_ms,
                        node=uav_id,
                        status="QUEUED",
                        event="HELD_NO_ENERGY",
                        system="qfanet",
                    )
                continue

            remaining_deadline_ms = q.min_deadline_ms

            chosen = self._choose_peer(
                uav_id,
                active_adj.get(uav_id, []),
                remaining_deadline_ms=remaining_deadline_ms,
            )

            if chosen is None:
                self.dropped_no_path += q.pending_packets
                expired = q.expire_at(end_ms)
                self._record_expired(
                    uav_id,
                    expired,
                    step,
                    end_ms,
                )

                for b in q.batches:
                    b.record_state(
                        step=step,
                        time_ms=end_ms,
                        node=uav_id,
                        status="QUEUED",
                        event="HELD_NO_PATH",
                        system="qfanet",
                    )
                continue

            estimated_delay_ms = (
                q.mean_hop_wait_ms(start_ms)
                + TICK_DURATION_MS
            )

            # Save the deadline before service for reward/urgency analysis.
            min_deadline_before_hop = q.min_deadline_ms

            # --------------------------------------------------------
            # CAPACITY-LIMITED SERVICE
            # --------------------------------------------------------
            # Only a finite number of DATA packets may leave this UAV during
            # the current tick. EDF serves the most urgent packets first.
            # --------------------------------------------------------
            tx_budget = self.tx_budget_packets.get(
                uav_id,
                0,
            )

            batches = q.pop_for_transmission(
                tx_budget
            )

            expired = [
                b for b in batches
                if b.deadline_at_ms <= end_ms
            ]
            alive = [
                b for b in batches
                if b.deadline_at_ms > end_ms
            ]

            self._record_expired(
                uav_id,
                expired,
                step,
                end_ms,
            )

            # Packets not selected because of finite capacity remain queued.
            # They can naturally expire while waiting, which is exactly the
            # congestion effect we want to measure at high offered load.
            capacity_expired = q.expire_at(
                end_ms
            )
            self._record_expired(
                uav_id,
                capacity_expired,
                step,
                end_ms,
            )

            for waiting_batch in q.batches:
                waiting_batch.record_state(
                    step=step,
                    time_ms=end_ms,
                    node=uav_id,
                    status="QUEUED",
                    event="WAITING_RELAY_CAPACITY",
                    system="qfanet",
                )

            reward, reward_components = self._reward(
                uav_id,
                chosen,
                positions,
                active_adj.get(uav_id, []),
                estimated_delay_ms,
            )

            # Terminal delivery keeps QMR r_max. Intermediate hops pay a small cost
            # so a long chain of positive rewards cannot dominate direct delivery.
            if chosen != self.gs_id:
                reward -= QFANET_HOP_PENALTY
                reward_components["hop_penalty"] = QFANET_HOP_PENALTY

                if math.isfinite(min_deadline_before_hop):
                    post_hop_remaining = (
                        min_deadline_before_hop
                        - TICK_DURATION_MS
                    )

                    if post_hop_remaining <= 0:
                        reward = QMR_R_MIN
                        reward_components["case"] = "deadline_critical"
                    elif post_hop_remaining <= QFANET_CRITICAL_DEADLINE_MS:
                        reward -= 0.20
                        reward_components["deadline_pressure_penalty"] = 0.20
                    elif post_hop_remaining <= QFANET_URGENT_DEADLINE_MS:
                        reward -= 0.10
                        reward_components["deadline_pressure_penalty"] = 0.10

            if expired:
                reward = min(reward, QMR_R_MIN)
                reward_components["case"] = "deadline_expired"

            reward = max(
                QMR_R_MIN,
                min(QMR_R_MAX, float(reward)),
            )
            reward_components["reward"] = round(reward, 4)

            # CRITICAL FIX: bootstrap only from actions actually reachable from
            # the selected next UAV in this same topology snapshot.
            next_best = (
                0.0
                if chosen == self.gs_id
                else self._reachable_next_best(
                    chosen,
                    active_adj,
                )
            )

            old_value = self.q[uav_id][chosen]
            td_target = reward + self.gamma * next_best

            # QMR reward scale is [-1, 1]. Clipping prevents repeated positive
            # intermediate rewards from making a long path more valuable than
            # immediate GS delivery.
            td_target = max(
                QMR_R_MIN,
                min(QMR_R_MAX, td_target),
            )

            new_value = (
                old_value
                + self.alpha * (td_target - old_value)
            )
            self.q[uav_id][chosen] = new_value

            self.decision_log.append({
                "step": step,
                "time_ms": start_ms,
                "uav_id": uav_id,
                "chosen": chosen,
                "remaining_deadline_ms": (
                    round(min_deadline_before_hop, 3)
                    if math.isfinite(min_deadline_before_hop)
                    else None
                ),
                "reward": round(reward, 4),
                "next_best_reachable": round(next_best, 4),
                "old_q": round(old_value, 4),
                "new_q": round(new_value, 4),
                "relay_capacity_pps": self.relay_capacity_pps,
                "tx_budget_packets": tx_budget,
                "queue_after_service": q.pending_packets,
            })

            if alive:
                forwarded = [
                    b.forwarded_copy(end_ms)
                    for b in alive
                ]

                if chosen == self.gs_id:
                    delivered_now = (
                        ground_station_qfanet.receive(
                            uav_id,
                            forwarded,
                            step,
                            end_ms,
                            system="qfanet",
                        )
                    )

                    self.delivered_count += delivered_now

                    if delivered_now > 0:
                        self.delivery_log.append(
                            (step, uav_id, delivered_now)
                        )

                else:
                    for b in forwarded:
                        b.record_state(
                            step=step,
                            time_ms=end_ms,
                            node=chosen,
                            status="QUEUED",
                            event="FORWARDED",
                            system="qfanet",
                            from_node=uav_id,
                        )

                    incoming[chosen].extend(
                        forwarded
                    )

        # Commit all transmissions only after every node has acted:
        # no same-tick multi-hop.
        for uid, batches in incoming.items():
            if batches:
                self.queues[uid].add_batches(
                    batches
                )

            self.queues[uid].set_time(
                end_ms
            )

        # Identical battery model to the LLM+RL side.
        for battery in self.batteries.values():
            battery.consume_tick(
                step,
                end_ms,
            )

    def deadline_snapshot(self, now_ms: float) -> dict:
        snaps = [
            q.deadline_snapshot(now_ms)
            for q in self.queues.values()
        ]

        mins = [
            s["min_remaining_ms"]
            for s in snaps
            if s["min_remaining_ms"] is not None
        ]

        means = [
            s["mean_remaining_ms"]
            for s in snaps
            if s["mean_remaining_ms"] is not None
        ]

        return {
            "pending_packets": sum(
                s["pending_packets"]
                for s in snaps
            ),
            "min_remaining_ms": (
                min(mins)
                if mins
                else None
            ),
            "mean_remaining_ms": (
                sum(means) / len(means)
                if means
                else None
            ),
            "dropped_deadline": self.dropped_deadline,
        }


# 6. STM (Contexte) — short-term rolling memory, per UAV
# ============================================================

# Short-Term Memory.
#
# Une STM est propre à :
#   - un UAV,
#   - une session de simulation.
#
# Elle mémorise seulement les transitions récentes du run courant et est
# injectée dans le prompt LLM comme contexte court terme.

class STM:
    """Short-Term Memory (STM) for ONE UAV in the CURRENT simulation session.

    ROLE IN THE LLM PIPELINE
    ------------------------
    The STM is the local, recent memory of an individual UAV.  It stores only the
    most recent transitions of the current simulation run (rolling deque).

    The LLM uses this memory to answer questions such as:
      - What action did this UAV take recently?
      - What reward followed that action?
      - Did the next state improve?
      - Was the packet deadline becoming more urgent?
      - Was energy decreasing?
      - Did FORWARD or HOLD work well in the last few steps?

    IMPORTANT:
      * STM is session-specific.
      * STM is UAV-specific.
      * STM is NOT LangSmith.
      * Historical experiences from earlier runs belong to LTM.

    The explicit ``session_id`` lets the prompt distinguish current-session
    information from long-term historical experience.
    """

    def __init__(self, maxlen=5, session_id: str = "unknown-session", owner_id: str = None):
        self.buffer = deque(maxlen=maxlen)
        self.session_id = session_id
        self.owner_id = owner_id

    # Ajoute une transition récente à la mémoire de session de l'UAV.
    # La deque limite automatiquement la quantité d'historique conservé.

    def add(self, entry: dict):
        item = dict(entry)
        item.setdefault("session_id", self.session_id)
        if self.owner_id is not None:
            item.setdefault("uav_id", self.owner_id)
        self.buffer.append(item)

    def update_last(self, **fields):
        """Complete the most recent transition once S_(t+1) becomes available."""
        if self.buffer:
            self.buffer[-1].update(fields)

    # Convertit la STM en texte structuré directement insérable dans le prompt LLM.

    def context_text(self) -> str:
        if not self.buffer:
            return f"Session {self.session_id}: no recent transition yet."

        lines = [f"Session ID: {self.session_id}"]
        for e in self.buffer:
            step = e.get("step", "?")
            time_ms = e.get("time_ms", 0.0)
            state = e.get("state", "n/a")
            next_state = e.get("next_state", "pending")
            action = e.get("action", "n/a")
            reward = float(e.get("reward", 0.0))
            raw_reward = float(e.get("raw_reward", 0.0))
            energy = float(e.get("energy", 0.0))
            pending = e.get("pending_packets", "?")
            deadline = e.get("min_deadline_ms", "n/a")
            lines.append(
                f"- step={step} t={float(time_ms)/1000.0:.3f}s "
                f"state={state} action={action} raw_R={raw_reward:.3f} "
                f"final_R={reward:.3f} next_state={next_state} "
                f"energy={energy:.1f}% pending={pending} min_deadline_ms={deadline}"
            )
        return "\n".join(lines)


# ============================================================
# 7. LTM (Vector DB) = LangSmith Dataset "FANETs"
# ============================================================

# Vectorisation lexicale de secours utilisée lorsque FAISS / embeddings
# ne sont pas disponibles.

def _vectorize(text: str) -> dict:
    words = re.findall(r"[a-zA-Z_]+", text.lower())
    vec = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    return vec


# Similarité cosinus entre deux vecteurs lexicaux de secours.

def _cosine(v1: dict, v2: dict) -> float:
    common = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in common)
    n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    return 0.0 if n1 == 0 or n2 == 0 else dot / (n1 * n2)


# Convertit une expérience structurée en texte indexable dans la mémoire.

def _to_text(inputs: dict, outputs: dict) -> str:
    in_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
    out_str = ", ".join(f"{k}={v}" for k, v in outputs.items())
    return f"[{in_str}] -> [{out_str}]"


# Adapte une expérience FANET au schéma ``chat`` exigé par le dataset
# LangSmith FANETs.

def _wrap_as_chat(inputs: dict, outputs: dict):
    """Wrap plain telemetry/decision dicts into the {"messages":[...]}/{"message":{...}}
    shape required by a chat-type LangSmith dataset (which is what the shared "FANETs"
    dataset is typed as). The real data travels as a JSON string inside the message
    content, so nothing is lost -- it's just repackaged to satisfy the schema."""
    wrapped_inputs = {"messages": [{"role": "user", "content": json.dumps(inputs, default=str)}]}
    wrapped_outputs = {"message": {"role": "assistant", "content": json.dumps(outputs, default=str)}}
    return wrapped_inputs, wrapped_outputs


# Opération inverse : reconstruit les dictionnaires FANET originaux depuis
# un exemple LangSmith stocké au format chat.

def _unwrap_chat(ex_inputs: dict, ex_outputs: dict):
    """Inverse of _wrap_as_chat -- recovers the original plain dicts from a chat-shaped
    example. Falls back to the raw dict as-is if it isn't chat-wrapped (e.g. a dataset
    that turns out to be plain kv), so retrieval never breaks either way."""
    inputs = ex_inputs or {}
    outputs = ex_outputs or {}
    try:
        content = inputs["messages"][0]["content"]
        inputs = json.loads(content)
    except Exception:
        pass
    try:
        content = outputs["message"]["content"]
        outputs = json.loads(content)
    except Exception:
        pass
    return inputs, outputs


# Long-Term Memory partagée.
#
# Rôle :
#   1) persister les expériences historiques dans LangSmith ;
#   2) maintenir une copie locale des expériences ;
#   3) construire un index FAISS ;
#   4) fournir au LLM :
#        - les expériences les plus récentes,
#        - les expériences historiquement similaires à la situation courante.
#
# Si LangSmith est indisponible, la simulation continue avec la mémoire locale.

class LangSmithLTM:
    """Shared Long-Term Memory (LTM): LangSmith + local semantic retrieval.

    ROLE IN THE LLM PIPELINE
    ------------------------
    This object represents the historical memory shared by the FANET swarm.

    Primary persistence:
      - LangSmith dataset ``FANETs`` stores routing experiences across runs.

    Local semantic layer:
      - recent experiences are mirrored in ``local_cache``;
      - SentenceTransformers creates embeddings;
      - FAISS retrieves semantically similar historical situations.

    The LLM receives TWO LTM views:
      1) the most recent historical experiences;
      2) semantically similar historical experiences.

    Therefore the memory architecture is:

        CURRENT RUN                       HISTORICAL RUNS
        -----------                       ---------------
        UAV-specific STM                  Shared LTM
        current session                   LangSmith dataset
             |                                  |
             +--------------+-------------------+
                            |
                            v
                         LLM Core

    If LangSmith is temporarily unavailable, the simulation continues with the
    local cache/FAISS fallback rather than stopping.  This fallback does not
    change the conceptual role of LTM in the architecture.
    """

    _shared_embedder = None
    _embedder_failed = False

    def __init__(self, dataset_name: str = LTM_DATASET_NAME, cache_size: int = 150,
                 refresh_every: int = 6, embedding_model: str = "all-MiniLM-L6-v2"):
        self.dataset_name = dataset_name
        self.cache_size = cache_size
        self.refresh_every = refresh_every
        self.embedding_model = embedding_model
        self._ticks_since_refresh = 0
        self.local_cache = deque(maxlen=cache_size)
        self.faiss_index = None
        self._semantic_texts = []

        if not _LANGSMITH_AVAILABLE:
            self.client = None
            self.dataset_id = None
            print(
                f"[LTM:{self.dataset_name}] LangSmith package unavailable "
                f"({_LANGSMITH_IMPORT_ERROR}); using local memory only."
            )
        else:
            try:
                # LangSmith is optional for execution. Fail fast and use the local
                # FAISS/cache fallback when the remote endpoint is unavailable.
                try:
                    self.client = Client(
                        timeout_ms=(1500, 3000),
                        auto_batch_tracing=False,
                    )
                except TypeError:
                    # Compatibility with older langsmith versions.
                    self.client = Client()

                self.dataset_id = self._get_or_create_dataset()
            except Exception as e:
                self.client = None
                self.dataset_id = None
                print(
                    f"[LTM:{self.dataset_name}] LangSmith unavailable "
                    f"({type(e).__name__}: {e}); using local memory only."
                )

        self._refresh_from_langsmith()

    def _get_or_create_dataset(self):
        try:
            ds = self.client.read_dataset(dataset_name=self.dataset_name)
        except Exception:
            create_kwargs = {
                "dataset_name": self.dataset_name,
                "description": (
                    "Shared LTM for FANET UAV agents: "
                    "network state -> routing decision + outcome"
                ),
            }

            if DataType is not None:
                create_kwargs["data_type"] = DataType.chat

            ds = self.client.create_dataset(
                **create_kwargs
            )
        return ds.id

    @classmethod
    def _get_embedder(cls, model_name: str):
        if not _SEMANTIC_LTM_AVAILABLE or cls._embedder_failed:
            return None
        if cls._shared_embedder is None:
            try:
                cls._shared_embedder = SentenceTransformer(model_name)
            except Exception as e:
                cls._embedder_failed = True
                print(f"[LTM] semantic embedder unavailable ({e}); using lexical fallback.")
                return None
        return cls._shared_embedder

    def _rebuild_faiss(self):
        self.faiss_index = None
        self._semantic_texts = [m["text"] for m in self.local_cache]
        if not self._semantic_texts:
            return
        embedder = self._get_embedder(self.embedding_model)
        if embedder is None or faiss is None:
            return
        try:
            embeddings = embedder.encode(
                self._semantic_texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index.add(embeddings)
        except Exception as e:
            print(f"[LTM:{self.dataset_name}] FAISS rebuild failed ({e}); using lexical fallback.")
            self.faiss_index = None

    @staticmethod
    def _experience_quality(outputs: dict) -> float:
        """Return an outcome-quality score in [-1, 1].

        High-quality memories are successful low-failure routing outcomes.
        Deadline failures are deliberately down-ranked during retrieval.
        """
        outputs = outputs or {}

        reward = float(
            outputs.get(
                "shaped_reward",
                outputs.get("raw_reward", 0.0),
            )
            or 0.0
        )

        reward = max(-1.0, min(1.0, reward))

        delivered_now = int(outputs.get("delivered_now", 0) or 0)
        deadline_drops = int(outputs.get("deadline_drops", 0) or 0)

        quality = reward

        if delivered_now > 0:
            quality += 0.35

        if deadline_drops > 0:
            quality -= 0.60

        return max(-1.0, min(1.0, quality))

    # Sauvegarde l'expérience dans LangSmith si possible ET dans le cache local.
    # Après une panne LangSmith, le code continue localement sans interrompre le run.

    def save_generic(self, inputs: dict, outputs: dict):
        wrapped_inputs, wrapped_outputs = _wrap_as_chat(inputs, outputs)
        remote_saved = False

        if self.client is not None and self.dataset_id is not None:
            try:
                self.client.create_example(
                    inputs=wrapped_inputs,
                    outputs=wrapped_outputs,
                    dataset_id=self.dataset_id,
                )
                remote_saved = True
            except Exception as e:
                print(
                    f"[LTM:{self.dataset_name}] LangSmith unavailable during save "
                    f"({type(e).__name__}). Switching to local LTM for this run."
                )
                # Stop retrying the unreachable remote service for the rest of this run.
                self.client = None
                self.dataset_id = None

        # Local LTM is always updated.
        text = _to_text(inputs, outputs)
        self.local_cache.append({
            "text": text,
            "vec": _vectorize(text),
            "created_at": time.time(),
            "source": "langsmith+local" if remote_saved else "local",
            "quality": self._experience_quality(outputs),
        })
        self._rebuild_faiss()

    def save(self, telemetry: Telemetry, packets: PacketQueue, action: str, rationale: str,
             raw_reward: float, shaped_reward: float, step: int, session_id: str = None,
             chosen_neighbor: str = None, delivered_now: int = 0,
             deadline_drops: int = 0, reward_components: dict = None):
        inputs = {
            **asdict(telemetry),
            **packets.as_memory_dict(),
            "step": step,
            "session_id": session_id,
        }
        outputs = {
            "action": action,
            "chosen_neighbor": chosen_neighbor,
            "rationale": rationale,
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward,
            "delivered_now": int(delivered_now or 0),
            "deadline_drops": int(deadline_drops or 0),
            "success": bool((delivered_now or 0) > 0 and (deadline_drops or 0) == 0),
            "reward_components": reward_components or {},
        }
        self.save_generic(inputs, outputs)

    # Recharge périodiquement les expériences historiques du dataset LangSmith
    # puis reconstruit l'index local de recherche sémantique.

    def _refresh_from_langsmith(self):
        if self.client is None or self.dataset_id is None:
            return
        try:
            examples = list(self.client.list_examples(dataset_id=self.dataset_id, limit=self.cache_size))
        except Exception as e:
            print(
                f"[LTM:{self.dataset_name}] LangSmith refresh unavailable "
                f"({type(e).__name__}). Using local LTM only."
            )
            self.client = None
            self.dataset_id = None
            return
        # Keep cache ordered oldest -> newest so the right side always contains
        # the most recent LangSmith experiences.
        examples.sort(key=lambda e: getattr(e, "created_at", 0) or 0)
        self.local_cache.clear()
        for ex in examples:
            inputs, outputs = _unwrap_chat(ex.inputs or {}, ex.outputs or {})
            text = _to_text(inputs, outputs)
            created_at = getattr(ex, "created_at", None)
            self.local_cache.append({
                "text": text,
                "vec": _vectorize(text),
                "created_at": str(created_at) if created_at is not None else None,
                "source": "langsmith",
                "quality": self._experience_quality(outputs),
            })
        self._ticks_since_refresh = 0
        self._rebuild_faiss()

    # Retourne les dernières expériences chronologiques de la LTM.

    def recent_experiences(self, k: int = 3) -> list:
        """Return the latest LTM experiences, newest first.

        When LangSmith is available the cache is periodically refreshed from the remote
        dataset.  Newly saved experiences are also inserted locally immediately, so the
        prompt always has recent context even between remote refreshes.
        """
        if not self.local_cache:
            return []
        recent = list(self.local_cache)[-max(1, int(k)):]
        recent.reverse()
        return [
            {
                "text": item.get("text", ""),
                "created_at": item.get("created_at"),
                "source": item.get("source", "unknown"),
                "quality": round(float(item.get("quality", 0.0)), 3),
            }
            for item in recent
        ]

    # Recherche les expériences historiquement les plus similaires à l'état courant.
    # FAISS est utilisé si disponible ; sinon une similarité lexicale sert de fallback.

    def retrieve(self, query: str, k: int = 3) -> list:
        """Retrieve memories by semantic similarity + outcome quality.

        Similarity answers: "is this experience relevant?"
        Quality answers:    "did this experience work?"

        This prevents the LLM from blindly receiving deadline failures merely
        because they are semantically similar to the current state.
        """
        self._ticks_since_refresh += 1

        if self._ticks_since_refresh >= self.refresh_every:
            self._refresh_from_langsmith()

        if not self.local_cache:
            return []

        cache = list(self.local_cache)
        candidate_count = min(
            len(cache),
            max(int(k) * 4, int(k)),
        )

        embedder = self._get_embedder(self.embedding_model)

        if self.faiss_index is not None and embedder is not None:
            try:
                q = embedder.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype("float32")

                scores, indices = self.faiss_index.search(
                    q,
                    candidate_count,
                )

                reranked = []

                for similarity, index in zip(scores[0], indices[0]):
                    if index < 0:
                        continue

                    item = cache[index]

                    # Inner-product cosine similarity for normalized embeddings
                    # is mapped approximately to [0, 1].
                    similarity01 = max(
                        0.0,
                        min(1.0, (float(similarity) + 1.0) / 2.0),
                    )

                    quality01 = (
                        max(-1.0, min(1.0, float(item.get("quality", 0.0))))
                        + 1.0
                    ) / 2.0

                    combined = (
                        LTM_SIMILARITY_WEIGHT * similarity01
                        + LTM_QUALITY_WEIGHT * quality01
                    )

                    reranked.append(
                        (combined, item.get("text", ""))
                    )

                reranked.sort(
                    key=lambda pair: pair[0],
                    reverse=True,
                )

                return [
                    text
                    for score, text in reranked[:max(1, int(k))]
                    if text
                ]

            except Exception as e:
                print(
                    f"[LTM:{self.dataset_name}] semantic retrieval failed "
                    f"({e}); using lexical fallback."
                )

        qvec = _vectorize(query)
        scored = []

        for item in cache:
            similarity = _cosine(
                qvec,
                item["vec"],
            )

            if similarity <= 0:
                continue

            quality01 = (
                max(-1.0, min(1.0, float(item.get("quality", 0.0))))
                + 1.0
            ) / 2.0

            combined = (
                LTM_SIMILARITY_WEIGHT * similarity
                + LTM_QUALITY_WEIGHT * quality01
            )

            scored.append(
                (combined, item["text"])
            )

        scored.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        return [
            text
            for score, text in scored[:max(1, int(k))]
        ]


# ============================================================
# 8. Agent RL (tabular Q-learning with shaped reward)
# ============================================================

Q_TABLE_FILENAME = "q_tables_neighbor_llmrl_v2.json"
QFANET_Q_TABLE_FILENAME = "qfanet_q_tables_qmr_v1.json"  # table Q separee pour le baseline Q-FANET (pas de LLM)


# Retourne le dossier contenant main.py afin de sauvegarder les artefacts
# (Q-tables, JSON, figures) au même emplacement.

def _base_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def save_json_utf8_atomic(data: dict, output_path: str) -> str:
    """Write JSON as UTF-8 using atomic replacement.

    The dashboard reads UTF-8 JSON. Always use --output-json and never use
    shell redirection such as:

        python main.py ... > result.json

    because that redirects terminal output instead of writing a real JSON file.
    """

    path = os.path.abspath(output_path)
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)

    temp_path = path + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    os.replace(
        temp_path,
        path,
    )

    return path


# Charge les Q-tables persistantes d'une exécution précédente.

def load_q_tables(filename: str = Q_TABLE_FILENAME) -> dict:
    """Loads {node_id: {...}} from disk -- reused for both the LLM+RL system's
    Q-tables (q_tables.json, {state: {action: value}} per UAV) and the Q-FANET
    baseline's Q-tables (qfanet_q_tables.json, {peer_id: value} per UAV), via
    the filename parameter, so learned policy carries over between runs for
    BOTH systems being compared."""
    path = os.path.join(_base_dir(), filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_states = sum(len(v) for v in data.values())
        print(f"[Q-table] {total_states} etat(s) appris charges depuis {path}")
        return data
    except Exception as e:
        print(f"[Q-table] Echec du chargement ({e}) -- demarrage avec des tables vides.")
        return {}


# Sauvegarde les Q-tables afin que l'apprentissage puisse continuer entre runs.

def save_q_tables(q_tables: dict, filename: str = Q_TABLE_FILENAME):
    path = os.path.join(_base_dir(), filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(q_tables, f, indent=2, ensure_ascii=False)
        total_states = sum(len(v) for v in q_tables.values())
        print(f"[Q-table] {total_states} etat(s) appris sauvegardes dans {path}")
    except Exception as e:
        print(f"[Q-table] Echec de la sauvegarde: {e}")


# Vérifie qu'une ancienne Q-table Q-FANET correspond exactement au nombre
# actuel d'UAV et à son espace d'actions.
# Une table d'une topologie différente est rejetée pour éviter un biais.

def qfanet_table_matches_topology(saved_tables: dict, uav_ids: list, gs_id: str = "GS") -> bool:
    """Return True only when a persisted Q-FANET table matches this exact fleet.

    Q-FANET actions are neighbor IDs, so a table learned with 6 UAVs must not be
    silently reused for a 25-UAV experiment. Reusing it biases old peers while all
    newly introduced peers start at INITIAL_Q_VALUE.
    """
    if not saved_tables:
        return True

    expected_uavs = set(uav_ids)
    if set(saved_tables.keys()) != expected_uavs:
        return False

    for uid in uav_ids:
        table = saved_tables.get(uid)
        if not isinstance(table, dict):
            return False

        expected_peers = (expected_uavs - {uid}) | {gs_id}
        if set(table.keys()) != expected_peers:
            return False

    return True


# Agent Q-learning tabulaire du système LLM+RL.
#
# Le LLM ne remplace pas cet agent :
# il fournit seulement un petit shaping qui influence temporairement
# l'évaluation FORWARD/HOLD.

class QLearningAgent:
    def __init__(self, actions=ACTIONS, alpha=0.3, gamma=0.9, epsilon=EPSILON_START, initial_q: dict = None):
        self.q = initial_q if initial_q is not None else {}  # loaded from q_tables.json when available
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # Transforme l'observation continue en état discret compact pour la Q-table :
    # énergie + connectivité + deadline + distance GS + charge de queue.

    @staticmethod
    def discretize_state(t: Telemetry, p: PacketQueue) -> str:
        """Compact multi-objective state used by tabular Q-learning.

        The previous state only contained energy/connectivity/deadline.  Adding
        distance-to-GS and queue load lets the same forward/hold action have
        different learned values in meaningfully different FANET situations.
        """
        e_bucket = "low" if t.energy < 30 else ("mid" if t.energy < 70 else "high")

        if p.pending_packets == 0:
            d_bucket = "empty"
        elif p.min_deadline_ms <= 2 * TICK_DURATION_MS:
            d_bucket = "urgent"
        elif p.min_deadline_ms <= 4 * TICK_DURATION_MS:
            d_bucket = "soon"
        else:
            d_bucket = "relaxed"

        if t.distance_to_gs <= COMM_RANGE:
            g_bucket = "near_gs"
        elif t.distance_to_gs <= 2 * COMM_RANGE:
            g_bucket = "mid_gs"
        else:
            g_bucket = "far_gs"

        if p.pending_packets == 0:
            q_bucket = "empty_q"
        elif p.pending_packets <= 3:
            q_bucket = "light_q"
        else:
            q_bucket = "heavy_q"

        return f"{e_bucket}-{t.connectivity}-{d_bucket}-{g_bucket}-{q_bucket}"

    def _ensure_state(self, state: str):
        if state not in self.q:
            self.q[state] = {a: INITIAL_Q_VALUE for a in self.actions}

    # Politique epsilon-greedy.
    # Le bonus LLM est ajouté uniquement au moment du choix pour guider l'action.

    def choose_action(
        self,
        state: str,
        shaping_bonus: dict,
        allowed_actions: list = None,
        epsilon_override: float = None,
    ) -> str:
        """Choose the final RL action with optional safety action masking.

        ``allowed_actions`` is NOT an LLM decision. It represents hard routing
        feasibility/deadline constraints supplied by the UAV controller.

        Examples:
          - no usable link       -> HOLD only
          - GS directly reachable -> FORWARD only
          - critical deadline + progress link -> FORWARD only
        """
        self._ensure_state(state)

        allowed = (
            [a for a in self.actions if a in allowed_actions]
            if allowed_actions is not None
            else list(self.actions)
        )

        if not allowed:
            allowed = list(self.actions)

        epsilon = (
            self.epsilon
            if epsilon_override is None
            else max(0.0, min(float(epsilon_override), 1.0))
        )

        if random.random() < epsilon:
            return random.choice(allowed)

        shaped_q = {
            action: (
                self.q[state][action]
                + shaping_bonus.get(action, 0.0)
            )
            for action in allowed
        }

        return max(
            shaped_q,
            key=shaped_q.get,
        )

    # Mise à jour TD classique :
    # Q(s,a) <- Q(s,a) + alpha * [r + gamma*max Q(s',.) - Q(s,a)]
    #
    # The Q values are bounded to the same [-1, 1] scale as the environment
    # reward. Without this guard, old cumulative Q values can become so large
    # that the bounded LLM shaping signal [-0.5, +0.5] becomes irrelevant.

    def update(self, state, action, shaped_reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)

        best_next = max(
            self.q[next_state].values()
        )

        td_target = (
            float(shaped_reward)
            + self.gamma * best_next
        )

        td_target = max(
            LLMRL_Q_MIN,
            min(LLMRL_Q_MAX, td_target),
        )

        td_error = (
            td_target
            - self.q[state][action]
        )

        updated = (
            self.q[state][action]
            + self.alpha * td_error
        )

        self.q[state][action] = max(
            LLMRL_Q_MIN,
            min(LLMRL_Q_MAX, updated),
        )


# ============================================================
# 8B. Neighbor-aware Q-learning for the proposed LLM+RL model
# ============================================================

class NeighborQLearningAgent:
    """State-conditioned Q-learning where actions are actual next hops.

    Unlike the legacy binary FORWARD/HOLD controller, this agent directly
    evaluates currently reachable neighbors:

        HOLD, UAV-x, UAV-y, ..., GS

    The LLM contributes contextual candidate scores. Q-learning remains the
    final action selector.
    """

    def __init__(
        self,
        action_space: list,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = EPSILON_START,
        initial_q: dict = None,
    ):
        self.actions = list(dict.fromkeys(action_space))
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        loaded = initial_q if isinstance(initial_q, dict) else {}

        # Old binary tables are incompatible with this next-hop action space.
        incompatible = any(
            isinstance(values, dict)
            and (
                "forward" in values
                or "hold" in values
            )
            for values in loaded.values()
        )

        if incompatible:
            print(
                "[LLM+RL] Legacy FORWARD/HOLD Q-table detected -> "
                "starting fresh neighbor-aware Q-table."
            )
            loaded = {}

        self.q = loaded

    @staticmethod
    def discretize_state(
        telemetry: Telemetry,
        packets: PacketQueue,
        links: list,
        network: Network,
        uav_id: str,
    ) -> str:
        """Create a compact state that also represents local topology."""

        energy_bucket = (
            "low"
            if telemetry.energy < 30
            else (
                "mid"
                if telemetry.energy < 70
                else "high"
            )
        )

        if packets.pending_packets == 0:
            deadline_bucket = "empty"
        elif packets.min_deadline_ms <= LLMRL_URGENT_DEADLINE_MS:
            deadline_bucket = "urgent"
        elif packets.min_deadline_ms <= LLMRL_SOON_DEADLINE_MS:
            deadline_bucket = "soon"
        else:
            deadline_bucket = "relaxed"

        if telemetry.distance_to_gs <= COMM_RANGE:
            gs_bucket = "near_gs"
        elif telemetry.distance_to_gs <= 2.0 * COMM_RANGE:
            gs_bucket = "mid_gs"
        else:
            gs_bucket = "far_gs"

        if packets.pending_packets == 0:
            queue_bucket = "empty_q"
        elif packets.pending_packets <= 3:
            queue_bucket = "light_q"
        elif packets.pending_packets <= 10:
            queue_bucket = "mid_q"
        else:
            queue_bucket = "heavy_q"

        if not links:
            neighbor_bucket = "n0"
        elif len(links) <= 2:
            neighbor_bucket = "n1_2"
        elif len(links) <= 5:
            neighbor_bucket = "n3_5"
        else:
            neighbor_bucket = "n6plus"

        current_distance = network.distance_to_gs(uav_id)

        progress_values = []

        for node_id, _dist, _rssi, is_gs, _energy in links:
            next_distance = (
                0.0
                if is_gs
                else network.distance_to_gs(node_id)
            )

            progress_values.append(
                current_distance - next_distance
            )

        best_progress = max(progress_values, default=-COMM_RANGE)

        if best_progress >= 0.50 * COMM_RANGE:
            progress_bucket = "big_progress"
        elif best_progress > 0:
            progress_bucket = "some_progress"
        else:
            progress_bucket = "no_progress"

        return (
            f"{energy_bucket}-"
            f"{telemetry.connectivity}-"
            f"{deadline_bucket}-"
            f"{gs_bucket}-"
            f"{queue_bucket}-"
            f"{neighbor_bucket}-"
            f"{progress_bucket}"
        )

    def _ensure_state(self, state: str):
        if state not in self.q or not isinstance(self.q[state], dict):
            self.q[state] = {}

        # Keep only valid actions for this fleet and add missing ones.
        self.q[state] = {
            action: float(value)
            for action, value in self.q[state].items()
            if action in self.actions
        }

        for action in self.actions:
            self.q[state].setdefault(
                action,
                INITIAL_Q_VALUE,
            )

    def choose_next_hop(
        self,
        state: str,
        allowed_actions: list,
        llm_scores: dict,
        heuristic_scores: dict,
        epsilon_override: float = None,
    ) -> tuple:
        """Return (selected_action, diagnostic_scores).

        FinalScore(a) =
            Q(s,a)
            + λ_llm * centered_LLM_score(a)
            + λ_prior * centered_topology_QoS_score(a)

        The LLM therefore guides Q-learning but never directly controls the UAV.
        """

        self._ensure_state(state)

        allowed = [
            action
            for action in allowed_actions
            if action in self.actions
        ]

        if not allowed:
            allowed = [HOLD_ACTION]

        epsilon = (
            self.epsilon
            if epsilon_override is None
            else max(0.0, min(1.0, float(epsilon_override)))
        )

        diagnostic = {}

        for action in allowed:
            q_value = self.q[state].get(
                action,
                INITIAL_Q_VALUE,
            )

            llm01 = max(
                0.0,
                min(
                    1.0,
                    float(llm_scores.get(action, 0.5)),
                ),
            )

            prior01 = max(
                0.0,
                min(
                    1.0,
                    float(heuristic_scores.get(action, 0.5)),
                ),
            )

            llm_centered = 2.0 * llm01 - 1.0
            prior_centered = 2.0 * prior01 - 1.0

            total = (
                LLMRL_Q_ACTION_WEIGHT * q_value
                + LLMRL_LLM_NEIGHBOR_WEIGHT * llm_centered
                + LLMRL_HEURISTIC_WEIGHT * prior_centered
            )

            diagnostic[action] = {
                "q": round(q_value, 4),
                "llm_score": round(llm01, 4),
                "heuristic_score": round(prior01, 4),
                "final_score": round(total, 4),
            }

        if random.random() < epsilon:
            selected = random.choice(allowed)
        else:
            selected = max(
                allowed,
                key=lambda action: diagnostic[action]["final_score"],
            )

        return selected, diagnostic

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_allowed_actions: list = None,
        terminal: bool = False,
    ):
        self._ensure_state(state)
        self._ensure_state(next_state)

        if action not in self.actions:
            return

        if terminal:
            best_next = 0.0
        else:
            next_allowed = [
                candidate
                for candidate in (next_allowed_actions or self.actions)
                if candidate in self.actions
            ]

            if not next_allowed:
                next_allowed = [HOLD_ACTION]

            best_next = max(
                self.q[next_state][candidate]
                for candidate in next_allowed
            )

        td_target = (
            float(reward)
            + self.gamma * best_next
        )

        td_target = max(
            LLMRL_Q_MIN,
            min(LLMRL_Q_MAX, td_target),
        )

        old_value = self.q[state][action]

        updated = (
            old_value
            + self.alpha * (td_target - old_value)
        )

        self.q[state][action] = max(
            LLMRL_Q_MIN,
            min(LLMRL_Q_MAX, updated),
        )


# ============================================================
# 9. LLM Core (CoT + ReAct) -> reward-shaping bonuses, grounded in real topology
# ============================================================

# Coeur de guidance LLM.
#
# Entrées :
#   - observation courante,
#   - feedback RL précédent,
#   - STM de la session actuelle,
#   - LTM LangSmith / FAISS.
#
# Sorties :
#   - bonus bornés FORWARD/HOLD,
#   - poids w1/w2/w3,
#   - courte rationale.
#
# Le prompt garde exactement :
#   GOAL / CONTEXT / CONDITIONS / TASKS / OUTPUT FORMAT
# et demande un raisonnement interne privé de type CoT.

class LLMCore:
    """LLM guidance layer used before the Q-learning routing action.

    ============================================================
    WHY THIS CLASS EXISTS
    ============================================================
    The LLM does NOT directly route a DATA packet.

    Instead, it reads:
      - the current FANET observation,
      - the previous RL feedback,
      - STM from the CURRENT simulation session,
      - LTM from LangSmith / semantic retrieval,

    then performs PRIVATE internal Chain-of-Thought (CoT) reasoning.

    From that reasoning it returns only:
      1) bounded shaping bonuses for FORWARD and HOLD;
      2) routing-utility weights w1, w2 and w3;
      3) a short rationale.

    The complete internal CoT is intentionally NOT returned or stored in the
    JSON response.  Only the compact rationale is exposed.

    ============================================================
    CONTROL RESPONSIBILITIES
    ============================================================
    LLM:
        analyzes observations + memory and produces guidance.

    Q-learning:
        chooses the final action: FORWARD or HOLD.

    RoutingUtility:
        chooses the final next hop among reachable neighbors.

    NetworkEnv:
        executes the action and calculates the base QMR-inspired reward.

    ============================================================
    PROMPT STRUCTURE
    ============================================================
    The prompt keeps the requested five-part structure:

        GOAL
        CONTEXT
        CONDITIONS
        TASKS
        OUTPUT FORMAT

    The CoT reasoning requirement is contained INSIDE the TASKS section.
    The memory requirement is also explicit inside TASKS:
        - STM = current-session memory
        - LTM = LangSmith historical memory
    """

    # Initialise le backend LLM et construit le System Prompt permanent.
    # Le prompt fixe le rôle, les contraintes, les tâches mémoire et le format JSON.

    def __init__(
        self,
        uav_id: str,
        backend: str = "groq",
        groq_model: str = "openai/gpt-oss-120b",
        ollama_model: str = None,
    ):
        # ------------------------------------------------------------
        # Identity of the UAV receiving LLM guidance.
        # ------------------------------------------------------------
        self.uav_id = uav_id

        # ------------------------------------------------------------
        # Backend selected for this UAV.
        # In the current project Ollama is normally used locally.
        # ------------------------------------------------------------
        self.backend = backend

        # ------------------------------------------------------------
        # Ollama parameters can be changed from the .env file:
        #
        #   OLLAMA_MODEL=phi3
        #   OLLAMA_BASE_URL=http://localhost:11434
        #
        # This avoids hard-coding the model in several places.
        # ------------------------------------------------------------
        ollama_model = (
            ollama_model
            or os.getenv("OLLAMA_MODEL", "phi3")
        )

        ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        # ------------------------------------------------------------
        # Initialize the selected LLM backend.
        # ------------------------------------------------------------
        if backend == "ollama":
            # Direct local Ollama API.
            # No langchain_core/langchain_ollama dependency is required.
            self.llm = DirectOllamaChat(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0.2,
                num_predict=160,
                keep_alive="10m",
                timeout=30.0,
                json_mode=True,
            )

        elif backend == "groq":
            if not _GROQ_AVAILABLE or ChatGroq is None:
                raise RuntimeError(
                    "Backend 'groq' was requested, but langchain-groq "
                    "could not be imported. "
                    f"Import error: {_GROQ_IMPORT_ERROR}"
                )

            self.llm = ChatGroq(
                model=groq_model,
                temperature=0.2,
            )

        else:
            raise ValueError(
                f"Unsupported LLM backend: {backend!r}. "
                "Use 'ollama' or 'groq'."
            )

        # ============================================================
        # SYSTEM PROMPT
        # ============================================================
        #
        # This is the permanent instruction given to the LLM.
        #
        # IMPORTANT:
        # We KEEP the requested prompt organization:
        #
        #   GOAL
        #   CONTEXT
        #   CONDITIONS
        #   TASKS
        #   OUTPUT FORMAT
        #
        # The prompt explicitly asks the LLM to use PRIVATE internal
        # Chain-of-Thought reasoning.  It does not ask the model to print
        # the full reasoning.  Only a short rationale is returned.
        #
        # Memory is an explicit reasoning task:
        #   STM -> current session / recent transitions
        #   LTM -> LangSmith historical experiences
        # ============================================================

        self.system_prompt = f"""
==============================
GOAL
==============================
You are the LLM guidance component for UAV {uav_id} in a multi-hop FANET.

Assist a Q-learning routing controller in selecting the most promising
currently reachable next hop for deadline-constrained DATA packets.

The final action is selected by Q-learning.
Your role is to provide bounded guidance scores for the currently feasible
neighbor actions and for HOLD.

Primary objectives, in order of importance:

1. Deliver packets before their deadline.
2. Make useful progress toward the Ground Station.
3. Avoid unstable or poor-quality links.
4. Avoid congested next hops.
5. Preserve sufficient UAV energy.
6. Use successful STM/LTM experience when relevant.


==============================
CONTEXT
==============================
For every candidate next hop you may receive:

- candidate ID,
- whether the candidate is the Ground Station,
- link distance,
- candidate distance to Ground Station,
- geographic progress toward Ground Station,
- RSSI / radio quality,
- link stability,
- candidate battery / energy score,
- packet deadline remaining,
- deadline feasibility,
- queue/congestion information.

You may also receive:

- previous routing transition and reward,
- recent Short-Term Memory from this simulation,
- recent Long-Term Memory,
- semantically similar historical experiences.

STM represents recent experience in the current topology.

LTM represents historical experience and may be outdated if topology,
traffic or mobility has changed.

The environment reward remains authoritative.
Do not redefine it.


==============================
CONDITIONS
==============================
1. Score ONLY the candidate actions explicitly provided.

2. Never invent UAVs, links, measurements, packets or memories.

3. The Ground Station should receive the highest preference when directly
   reachable and deadline-feasible.

4. Under critical deadline pressure, prioritize actions that provide strong
   positive progress and sufficient link reliability.

5. Penalize candidates that:
   - move significantly away from the Ground Station,
   - have poor RSSI,
   - have poor stability,
   - are heavily congested,
   - have critically low energy.

6. HOLD should normally receive a low score when at least one safe,
   progress-making next hop exists.

7. HOLD may receive a high score when:
   - no usable link exists,
   - all candidates are strongly regressive,
   - all candidate links are temporarily unreliable,
   - forwarding would likely cause immediate failure.

8. Use STM before older LTM when current topology has changed.

9. Historical negative-reward experiences are failure evidence.
   Do not imitate them blindly.

10. Prefer historical experiences associated with successful delivery,
    positive deadline margin and good reward.

11. Do not output hidden reasoning.
    Return only a concise rationale.


==============================
TASKS
==============================
Evaluate each currently feasible candidate according to:

A. DEADLINE FEASIBILITY
Can this action realistically help deliver the packet before expiration?

B. GEOGRAPHIC PROGRESS
How much does this candidate reduce distance to the Ground Station?

C. LINK QUALITY
Consider RSSI and radio quality.

D. LINK STABILITY
Prefer links likely to remain available under UAV mobility.

E. CONGESTION
Avoid candidates whose queue state makes deadline failure likely.

F. ENERGY
Avoid critically depleted relays when comparable alternatives exist.

G. MEMORY
Use recent STM and high-quality relevant LTM to adjust confidence.

H. PREVIOUS FEEDBACK
Use previous rewards and deadline outcomes to avoid repeating recently
unsuccessful choices.

Assign every candidate a score between 0.0 and 1.0.

A score near 1.0 means strongly recommended.
A score near 0.0 means strongly discouraged.

The scores are guidance only.
Q-learning remains the final controller.


==============================
OUTPUT FORMAT
==============================
Return ONLY valid JSON.

Use exactly:

{{
  "rationale": "short explanation of the dominant routing factors",
  "neighbor_scores": {{
    "candidate-id": 0.0
  }},
  "hold_score": 0.0,
  "weights": {{
    "progress": 0.40,
    "stability": 0.25,
    "energy": 0.10,
    "deadline": 0.25
  }}
}}
""".strip()

    # Construit le Human Prompt dynamique pour UN UAV et UN tick.
    #
    # Cette méthode rassemble explicitement :
    #   - observation courante,
    #   - feedback précédent,
    #   - STM de la session,
    #   - LTM récente,
    #   - LTM sémantique,
    # puis appelle le LLM.

    def shape(
        self,
        telemetry: Telemetry,
        packets: PacketQueue,
        stm: STM,
        ltm: LangSmithLTM,
        neighbors_summary: str,
        previous_feedback: dict = None,
        simulation_time_ms: float = 0.0,
    ) -> dict:
        """Create the dynamic prompt for ONE UAV at ONE simulation tick.

        ============================================================
        DATA FLOW
        ============================================================

        Current telemetry ------------------------+
        Current packet queue ---------------------+
        Previous RL feedback ---------------------+
        STM (current session) --------------------+--> Human prompt
        LTM recent historical experiences --------+
        LTM semantic similar experiences ---------+
                                                   |
                                                   v
                                                 LLM
                                                   |
                                                   v
                              bonuses + routing weights + rationale

        The model then performs private internal CoT as requested by the
        SystemMessage.  The full internal reasoning is not returned.
        """

        # No previous transition exists on the first decision.
        previous_feedback = (
            previous_feedback
            or {}
        )

        # Previous shaped reward is useful in the semantic-memory query
        # because it helps retrieve experiences with similar outcomes.
        last_reward = previous_feedback.get(
            "shaped_reward"
        )

        # ============================================================
        # 1. BUILD THE SEMANTIC-LTM SEARCH QUERY
        # ============================================================
        # The query summarizes the current FANET situation.
        #
        # LangSmithLTM.retrieve() uses this text to find experiences that
        # are semantically similar to the current routing state.
        # ============================================================

        query = (
            f"energy={telemetry.energy} "
            f"connectivity={telemetry.connectivity} "
            f"distance_to_gs={telemetry.distance_to_gs} "
            f"deadline={packets.min_deadline_ms} "
            f"rssi={telemetry.rssi_best_neighbor} "
            f"pending_packets={packets.pending_packets} "
            f"previous_reward={last_reward}"
        )

        # ============================================================
        # 2. PREVIOUS RL FEEDBACK
        # ============================================================
        # This closes the RL feedback loop.
        #
        # The LLM sees what happened after its previous guidance:
        #     S_t -> A_t -> R_t -> S_(t+1)
        #
        # It can therefore adapt the next shaping suggestion.
        # ============================================================

        if previous_feedback:
            components = (
                previous_feedback.get(
                    "reward_components",
                    {},
                )
                or {}
            )

            feedback_text = (
                f"Previous state: "
                f"{previous_feedback.get('state', 'n/a')}\n"

                f"Previous action: "
                f"{previous_feedback.get('action', 'n/a')}\n"

                f"Base environment reward R_t: "
                f"{previous_feedback.get('raw_reward', 0.0):.3f}\n"

                f"LLM shaping used: "
                f"{previous_feedback.get('llm_bonus', 0.0):.3f}\n"

                f"Final reward used by Q-learning: "
                f"{previous_feedback.get('shaped_reward', 0.0):.3f}\n"

                f"Updated state S_(t+1): "
                f"{previous_feedback.get('next_state', 'n/a')}\n"

                f"Deadline-dropped packets: "
                f"{previous_feedback.get('deadline_drops', 0)}\n"

                f"Base-reward components: "
                f"{json.dumps(components, ensure_ascii=False)}"
            )

        else:
            feedback_text = (
                "No previous transition yet "
                "(first reasoning cycle)."
            )

        # ============================================================
        # 3. STM: CURRENT-SESSION MEMORY
        # ============================================================
        #
        # stm.context_text() returns the most recent transitions of this
        # UAV in THIS simulation session.
        #
        # This is the short-term context used by the LLM.
        # ============================================================

        stm_text = stm.context_text()

        # ============================================================
        # 4. LTM VIEW A: MOST RECENT HISTORICAL EXPERIENCES
        # ============================================================
        #
        # The LTM object is shared by the swarm.
        #
        # When LangSmith is available, historical examples are loaded
        # from the LangSmith dataset "FANETs". Newly created experiences
        # are also mirrored locally.
        #
        # We send the three latest experiences to the LLM.
        # ============================================================

        recent_ltm = ltm.recent_experiences(
            k=3
        )

        if recent_ltm:
            recent_ltm_text = "\n".join(
                (
                    f"- [{item.get('source', 'ltm')} | "
                    f"{item.get('created_at', 'time n/a')}] "
                    f"{item.get('text', '')}"
                )
                for item in recent_ltm
            )
        else:
            recent_ltm_text = (
                "No previous long-term experience "
                "available yet."
            )

        # ============================================================
        # 5. LTM VIEW B: SEMANTICALLY SIMILAR EXPERIENCES
        # ============================================================
        #
        # retrieve(query) compares the current FANET state with the LTM:
        #
        #   LangSmith historical examples
        #             |
        #             v
        #      local embeddings / FAISS
        #             |
        #             v
        #   semantically similar experiences
        #
        # This gives the LLM historical examples that are relevant to
        # the CURRENT routing situation, not only the newest records.
        # ============================================================

        retrieved = ltm.retrieve(
            query
        )

        if retrieved:
            ltm_text = "\n".join(
                f"- {item}"
                for item in retrieved
            )
        else:
            ltm_text = (
                "No semantically similar "
                "past memory yet."
            )

        # ============================================================
        # 6. FORMAT THE CURRENT DEADLINE
        # ============================================================

        min_deadline_text = (
            "none (queue empty)"
            if not math.isfinite(
                packets.min_deadline_ms
            )
            else (
                f"{packets.min_deadline_ms:.1f} ms"
            )
        )

        # ============================================================
        # 7. BUILD THE DYNAMIC HUMAN PROMPT
        # ============================================================
        #
        # IMPORTANT:
        # The prompt structure remains EXACTLY:
        #
        #       GOAL
        #       CONTEXT
        #       CONDITIONS
        #       TASKS
        #       OUTPUT FORMAT
        #
        # CONTEXT contains the actual runtime data.
        #
        # TASKS explicitly require:
        #   - private CoT reasoning,
        #   - STM consultation,
        #   - LangSmith LTM consultation,
        #   - feedback analysis.
        # ============================================================

        human_prompt = f"""
==============================
GOAL
==============================
Provide routing guidance for UAV {self.uav_id} at the CURRENT simulation tick.

Help the RL controller improve DATA packet delivery before deadline while
balancing:

- progress toward the Ground Station,
- deadline urgency,
- link reliability,
- energy,
- recent STM experience,
- historical LangSmith LTM experience.

Do not make the final routing decision.


==============================
CONTEXT
==============================

[SIMULATION]
UAV ID: {self.uav_id}
Simulation time: {simulation_time_ms / 1000.0:.3f} s
Simulation time ms: {simulation_time_ms:.0f} ms

[NAVIGATION AND NETWORK]
Estimated position: {telemetry.estimated_position}
Estimated velocity: {telemetry.estimated_velocity} m/s
Estimated altitude: {telemetry.estimated_altitude} m
Battery: {telemetry.energy}%
Distance to Ground Station: {telemetry.distance_to_gs:.1f} m
Connectivity state: {telemetry.connectivity}
Best neighbor RSSI: {telemetry.rssi_best_neighbor} dBm

Reachable active neighbors within {COMM_RANGE:.0f} m:
{neighbors_summary}

[TRAFFIC AND QoS]
Pending DATA packets: {packets.pending_packets}
Pending packet batches: {len(packets.batches)}
Minimum remaining deadline: {min_deadline_text}

[PREVIOUS TRANSITION FEEDBACK]
{feedback_text}

[STM - CURRENT SESSION]
STM session ID: {stm.session_id}
STM owner UAV: {stm.owner_id}
{stm_text}

[LTM - LANGSMITH RECENT HISTORICAL EXPERIENCES]
LangSmith dataset: {LTM_DATASET_NAME}
{recent_ltm_text}

[LTM - LANGSMITH / SEMANTICALLY SIMILAR EXPERIENCES]
LangSmith dataset: {LTM_DATASET_NAME}
{ltm_text}


==============================
CONDITIONS
==============================
- Use only the neighbors listed in the current context.
- Do not invent network information.
- Do not invent STM or LTM experiences.
- Do not select the final next hop.
- Do not select the final FORWARD/HOLD action.
- Q-learning remains the final action selector.
- RoutingUtility remains the final next-hop selector.
- Do not modify the QMR-inspired base reward.
- Keep both shaping bonuses inside:
  [{LLM_SHAPING_MIN}, {LLM_SHAPING_MAX}]
- Keep w1, w2 and w3 non-negative.
- Make w1 + w2 + w3 approximately equal to 1.
- You MUST use STM if it contains current-session history.
- You MUST use LTM if historical experiences are available.
- Treat STM as session-specific recent memory.
- Treat LangSmith LTM as historical routing memory.
- Use previous reward/state feedback when available.
- Perform private internal Chain-of-Thought reasoning.
- Do not output the full Chain-of-Thought.
- Output only the requested compact JSON guidance.


==============================
TASKS
==============================
Perform the following PRIVATE Chain-of-Thought reasoning sequence internally.

1. OBSERVE
   Analyze:
   - battery,
   - packet queue,
   - deadline,
   - distance to GS,
   - connectivity,
   - neighbors,
   - RSSI.

2. DEADLINE
   Determine whether the deadline is:
   - comfortable,
   - short,
   - critical.

3. ROUTING PROGRESS
   Evaluate whether the reachable topology can move the packet toward the GS.

4. LINK QUALITY
   Evaluate RSSI and current link reliability.

5. ENERGY
   Evaluate current and neighbor energy conditions.

6. MEMORY
   This task is mandatory.

   6A. STM / CURRENT SESSION
       Read the STM provided above.
       Use recent state -> action -> reward -> next-state transitions.
       Determine what worked or failed recently for this UAV in THIS run.

   6B. LTM / LANGSMITH HISTORY
       Read the recent LangSmith historical experiences provided above.
       Identify useful previously observed routing outcomes.

   6C. SEMANTIC LTM
       Read semantically similar LangSmith/LTM experiences.
       Compare them with the current observation.

   6D. MEMORY SYNTHESIS
       Combine STM and LTM evidence.
       Give current-session STM appropriate importance when the current
       topology differs from historical situations.

7. PREVIOUS FEEDBACK
   Analyze the previous:
   - state,
   - action,
   - base reward,
   - shaping bonus,
   - final reward,
   - deadline drops,
   - updated state.

8. PRIVATE CHAIN-OF-THOUGHT SYNTHESIS
   Privately reason step by step using:
   - current observation,
   - deadline,
   - routing progress,
   - link quality,
   - energy,
   - previous feedback,
   - STM,
   - LangSmith LTM.

   Compare FORWARD and HOLD internally.

   Do NOT expose the complete reasoning.

9. ACTION SHAPING
   Produce a SMALL bounded preference for:
   - forward
   - hold

   Routing-performance rules:
   - Direct GS reachable -> strongly prefer FORWARD.
   - Critical deadline + useful progress link -> strongly prefer FORWARD.
   - Short deadline + reliable progress link -> normally prefer FORWARD.
   - Favor HOLD mainly when no useful progress link exists or forwarding is
     currently unsafe/poor.
   - Use STM reward outcomes to avoid repeating a recent bad action.
   - Use LTM reward outcomes as historical evidence, not as a command.
   - Do not imitate negative-reward LTM actions blindly.

10. ROUTING WEIGHTS
    Produce:
    - w1 = proximity to GS
    - w2 = link stability
    - w3 = next-hop energy

11. SHORT RATIONALE
    Summarize only the principal factors that influenced the guidance.

    Mention relevant memory evidence when useful, for example:
    - "STM shows the previous forward action improved the state"
    - "LTM contains similar successful forwarding situations"

    Do NOT expose the full private Chain-of-Thought.


==============================
OUTPUT FORMAT
==============================
Return ONLY valid JSON.

Do not return Markdown.
Do not return code fences.
Do not return text outside the JSON object.

Use exactly this schema:

{{
  "rationale": "short summary of the main routing and memory factors",
  "bonuses": {{
    "forward": 0.0,
    "hold": 0.0
  }},
  "weights": {{
    "w1": 0.4,
    "w2": 0.3,
    "w3": 0.3
  }}
}}
""".strip()

        # ============================================================
        # 8. SEND SYSTEM + HUMAN PROMPTS TO THE LLM
        # ============================================================
        #
        # SystemMessage:
        #   permanent role, rules, CoT policy, memory interpretation.
        #
        # HumanMessage:
        #   live state for the current UAV/tick.
        # ============================================================

        messages = [
            ("system", self.system_prompt),
            ("human", human_prompt),
        ]

        # ============================================================
        # 9. LLM INVOCATION WITH SAFE FALLBACK
        # ============================================================
        #
        # If Ollama/Groq fails, the whole FANET simulation should NOT
        # crash. We return neutral shaping and default routing weights.
        # ============================================================

        try:
            response = self.llm.invoke(
                messages
            )

        except Exception as e:
            print(
                f"[LLMCore:{self.uav_id}:{self.backend}] "
                f"LLM call failed "
                f"({type(e).__name__}: {e}) "
                f"-- using neutral shaping for this tick."
            )

            return {
                "rationale": (
                    f"llm_call_failed: "
                    f"{type(e).__name__}"
                ),
                "bonuses": {
                    action: 0.0
                    for action in ACTIONS
                },
                "weights": {
                    "w1": 0.4,
                    "w2": 0.3,
                    "w3": 0.3,
                },
            }

        # Parse, validate and clamp the JSON returned by the model.
        return self._parse_json(
            response.content
        )

    # Valide la réponse LLM et borne les valeurs pour empêcher une sortie du modèle
    # de casser le contrôleur ou de dépasser les limites de shaping autorisées.

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Validate the compact JSON guidance returned by the LLM.

        The complete private CoT is never expected in this JSON.

        Expected fields:
            rationale
            bonuses.forward
            bonuses.hold
            weights.w1
            weights.w2
            weights.w3

        This parser:
          1) removes accidental Markdown fences;
          2) extracts/parses the JSON object;
          3) clamps bonuses to the allowed shaping interval;
          4) clamps each routing weight to [0, 1];
          5) returns neutral guidance if parsing fails.
        """

        try:
            # Some LLMs may accidentally surround JSON with ```json fences.
            cleaned = re.sub(
                r"```(?:json)?\s*|\s*```",
                "",
                text,
            ).strip()

            # Extract the first JSON object if additional whitespace/text
            # accidentally appears around it.
            match = re.search(
                r"\{.*\}",
                cleaned,
                re.DOTALL,
            )

            data = (
                json.loads(match.group(0))
                if match
                else json.loads(cleaned)
            )

            # Generic numeric safety helper.
            def clamp(value, lo, hi):
                return max(
                    lo,
                    min(float(value), hi),
                )

            # --------------------------------------------------------
            # Validate / clamp LLM shaping bonuses.
            # --------------------------------------------------------
            raw_bonuses = data.get(
                "bonuses",
                {},
            )

            bonuses = {
                action: clamp(
                    raw_bonuses.get(
                        action,
                        0.0,
                    ),
                    LLM_SHAPING_MIN,
                    LLM_SHAPING_MAX,
                )
                for action in ACTIONS
            }

            # --------------------------------------------------------
            # Validate / clamp dynamic RoutingUtility weights.
            #
            # RoutingUtility.update_weights() performs the final
            # normalization so their exact sum becomes 1.
            # --------------------------------------------------------
            raw_weights = data.get(
                "weights",
                {},
            )

            weights = {
                "w1": clamp(
                    raw_weights.get(
                        "w1",
                        0.4,
                    ),
                    0.0,
                    1.0,
                ),
                "w2": clamp(
                    raw_weights.get(
                        "w2",
                        0.3,
                    ),
                    0.0,
                    1.0,
                ),
                "w3": clamp(
                    raw_weights.get(
                        "w3",
                        0.3,
                    ),
                    0.0,
                    1.0,
                ),
            }

            # Only a short explanatory rationale is stored.
            # The model's complete internal CoT is not part of the API output.
            return {
                "rationale": str(
                    data.get(
                        "rationale",
                        "",
                    )
                )[:500],
                "bonuses": bonuses,
                "weights": weights,
            }

        except Exception:
            # Invalid LLM output should never terminate the simulation.
            return {
                "rationale": (
                    "parse_error, using neutral shaping"
                ),
                "bonuses": {
                    action: 0.0
                    for action in ACTIONS
                },
                "weights": {
                    "w1": 0.4,
                    "w2": 0.3,
                    "w3": 0.3,
                },
            }


# ============================================================
# 9B. LLM guidance for reachable next-hop candidates
# ============================================================

class NeighborGuidanceCore(LLMCore):
    """LLM guidance that scores neighbors instead of FORWARD/HOLD.

    Q-learning still makes the final decision. The model only provides:
      - one score in [0,1] for each reachable candidate,
      - a HOLD score in [0,1],
      - contextual routing weights,
      - a short rationale.

    The full private reasoning is not returned.
    """

    def __init__(
        self,
        uav_id: str,
        backend: str = "groq",
        groq_model: str = "openai/gpt-oss-120b",
        ollama_model: str = None,
    ):
        super().__init__(
            uav_id=uav_id,
            backend=backend,
            groq_model=groq_model,
            ollama_model=ollama_model,
        )

        self.system_prompt = f"""
==============================
GOAL
==============================
You are the contextual routing-guidance LLM for UAV {uav_id} in a FANET.

Help a neighbor-aware Q-learning controller select a reliable next hop that
maximizes deadline-constrained packet delivery.

You provide SCORES and contextual guidance only.
Q-learning makes the final routing decision.


==============================
CONTEXT
==============================
The controller observes the current UAV, packet deadlines, reachable neighbors,
link quality, energy, Short-Term Memory (STM), and historical Long-Term Memory
(LTM) backed by the LangSmith dataset '{LTM_DATASET_NAME}'.

Control loop:

OBSERVE
-> READ STM + LTM
-> PRIVATE INTERNAL REASONING
-> LLM CANDIDATE SCORES
-> Q-LEARNING NEXT-HOP SELECTION
-> FANET ENVIRONMENT
-> REWARD
-> STM/LTM UPDATE
-> NEXT STATE


==============================
CONDITIONS
==============================
1. Do not invent neighbors.
2. Score ONLY candidate IDs explicitly provided in the current prompt.
3. A score is guidance, not the final decision.
4. GS should receive a very high score when directly reachable.
5. Prefer candidates that provide strong progress toward GS.
6. Prefer reliable RSSI/stability when progress is comparable.
7. Avoid low-energy relays when a similarly good alternative exists.
8. Under critical deadlines, prioritize fast geographic progress and reliable delivery.
9. HOLD must be near zero when at least one useful progress-making link exists.
10. HOLD can be higher only when no progress-making link exists or all usable links are unavailable.
10B. Prefer a less congested progress-making relay when progress and radio quality are comparable.
11. Read STM to learn from the current run.
12. Read LTM as historical evidence.
13. Negative-reward/deadline-drop memories are failure examples, not actions to imitate.
14. When current STM conflicts with old LTM because topology changed, prioritize STM.
15. Perform private internal step-by-step reasoning.
16. Do not output the complete private reasoning.


==============================
TASKS
==============================
1. Analyze deadline urgency.
2. Compare every provided candidate:
   - progress toward GS,
   - RSSI,
   - stability,
   - candidate energy,
   - whether the candidate is GS.
3. Read recent STM transitions.
4. Read recent and semantically similar LTM experiences.
5. Use outcome quality: successful positive-reward experiences are stronger evidence
   than deadline failures.
6. Score each candidate from 0.0 to 1.0.
7. Score HOLD from 0.0 to 1.0.
8. Produce routing weights:
   - progress
   - stability
   - energy
   - deadline
   - congestion
9. Return a short rationale only.


==============================
OUTPUT FORMAT
==============================
Return ONLY valid JSON.

Use exactly this structure:

{{
  "rationale": "short routing explanation",
  "neighbor_scores": {{
    "CANDIDATE_ID": 0.0
  }},
  "hold_score": 0.0,
  "weights": {{
    "progress": 0.40,
    "stability": 0.20,
    "energy": 0.10,
    "deadline": 0.15,
    "congestion": 0.15
  }}
}}
""".strip()

    def guide_neighbors(
        self,
        telemetry: Telemetry,
        packets: PacketQueue,
        stm: STM,
        ltm: LangSmithLTM,
        candidate_context: list,
        previous_feedback: dict = None,
        simulation_time_ms: float = 0.0,
    ) -> dict:
        previous_feedback = previous_feedback or {}

        candidate_ids = [
            str(item["id"])
            for item in candidate_context
        ]

        query = (
            f"energy={telemetry.energy} "
            f"connectivity={telemetry.connectivity} "
            f"distance_to_gs={telemetry.distance_to_gs} "
            f"deadline={packets.min_deadline_ms} "
            f"pending={packets.pending_packets} "
            f"candidates={candidate_ids} "
            f"previous_reward={previous_feedback.get('shaped_reward')}"
        )

        recent = ltm.recent_experiences(k=3)
        recent_text = (
            "\n".join(
                f"- quality={item.get('quality', 0.0)} "
                f"[{item.get('source', 'ltm')}] {item.get('text', '')}"
                for item in recent
            )
            if recent
            else "No recent LTM experience."
        )

        similar = ltm.retrieve(query, k=3)
        similar_text = (
            "\n".join(
                f"- {item}"
                for item in similar
            )
            if similar
            else "No similar historical experience."
        )

        previous_text = (
            json.dumps(
                previous_feedback,
                ensure_ascii=False,
                default=str,
            )
            if previous_feedback
            else "No previous transition."
        )

        human_prompt = f"""
==============================
GOAL
==============================
Score the CURRENT reachable next-hop candidates for UAV {self.uav_id}.


==============================
CONTEXT
==============================
Simulation time: {simulation_time_ms / 1000.0:.3f} s
Battery: {telemetry.energy}%
Distance to GS: {telemetry.distance_to_gs:.1f} m
Connectivity: {telemetry.connectivity}
Best RSSI: {telemetry.rssi_best_neighbor} dBm
Pending packets: {packets.pending_packets}
Minimum remaining deadline: {packets.min_deadline_ms}

[CANDIDATES]
{json.dumps(candidate_context, indent=2, ensure_ascii=False)}

[PREVIOUS FEEDBACK]
{previous_text}

[STM - CURRENT SESSION]
Session: {stm.session_id}
Owner: {stm.owner_id}
{stm.context_text()}

[LTM - RECENT HISTORICAL EXPERIENCES]
{recent_text}

[LTM - QUALITY-AWARE SIMILAR EXPERIENCES]
{similar_text}


==============================
CONDITIONS
==============================
- Score ONLY these candidate IDs: {candidate_ids}
- Candidate score range: 0.0 to 1.0.
- HOLD score range: 0.0 to 1.0.
- Direct GS should be strongly preferred.
- Critical deadline increases the value of immediate progress.
- When two candidates make similar progress, prefer the one with better stability and lower queue pressure.
- If a progress-making candidate exists, HOLD should normally be <= 0.05.
- Give clearly different scores when candidates are materially different; avoid returning all candidates near 0.5.
- Negative/deadline-failure memory is evidence of what to avoid.
- Do not output hidden reasoning.


==============================
TASKS
==============================
1. Compare all candidates.
2. Use deadline urgency.
3. Use progress toward GS.
4. Use RSSI/stability.
5. Use relay energy.
6. Use neighbor queue pressure / congestion.
7. Use STM.
7. Use recent LTM.
8. Use quality-aware similar LTM.
9. Produce candidate scores and HOLD score.
10. Produce progress/stability/energy/deadline weights.
11. Return a short rationale.


==============================
OUTPUT FORMAT
==============================
Return ONLY JSON:

{{
  "rationale": "short explanation",
  "neighbor_scores": {{
    "candidate-id": 0.0
  }},
  "hold_score": 0.0,
  "weights": {{
    "progress": 0.40,
    "stability": 0.20,
    "energy": 0.10,
    "deadline": 0.15,
    "congestion": 0.15
  }}
}}
""".strip()

        messages = [
            ("system", self.system_prompt),
            ("human", human_prompt),
        ]

        try:
            response = self.llm.invoke(messages)

            return self._parse_neighbor_json(
                response.content,
                candidate_ids,
            )

        except Exception as e:
            print(
                f"[NeighborGuidance:{self.uav_id}:{self.backend}] "
                f"LLM failed ({type(e).__name__}: {e}) -> neutral guidance."
            )

            return self._neutral_neighbor_guidance(
                candidate_ids,
                rationale=f"llm_call_failed:{type(e).__name__}",
            )

    @staticmethod
    def _neutral_neighbor_guidance(
        candidate_ids: list,
        rationale: str = "neutral_guidance",
    ) -> dict:
        return {
            "rationale": rationale,
            "neighbor_scores": {
                candidate_id: 0.5
                for candidate_id in candidate_ids
            },
            "hold_score": 0.5,
            "weights": {
                "progress": 0.40,
                "stability": 0.20,
                "energy": 0.10,
                "deadline": 0.15,
                "congestion": 0.15,
            },
        }

    @classmethod
    def _parse_neighbor_json(
        cls,
        text: str,
        candidate_ids: list,
    ) -> dict:
        try:
            cleaned = re.sub(
                r"```(?:json)?\s*|\s*```",
                "",
                text,
            ).strip()

            match = re.search(
                r"\{.*\}",
                cleaned,
                re.DOTALL,
            )

            data = json.loads(
                match.group(0)
                if match
                else cleaned
            )

            def clamp(value, lo=0.0, hi=1.0):
                try:
                    value = float(value)
                except Exception:
                    value = 0.5

                return max(
                    lo,
                    min(hi, value),
                )

            raw_scores = data.get(
                "neighbor_scores",
                {},
            )

            scores = {
                candidate_id: clamp(
                    raw_scores.get(candidate_id, 0.5)
                )
                for candidate_id in candidate_ids
            }

            raw_weights = data.get(
                "weights",
                {},
            )

            weights = {
                "progress": clamp(raw_weights.get("progress", 0.40)),
                "stability": clamp(raw_weights.get("stability", 0.20)),
                "energy": clamp(raw_weights.get("energy", 0.10)),
                "deadline": clamp(raw_weights.get("deadline", 0.15)),
                "congestion": clamp(raw_weights.get("congestion", 0.15)),
            }

            total = sum(weights.values())

            if total <= 0:
                weights = {
                    "progress": 0.40,
                    "stability": 0.20,
                    "energy": 0.10,
                    "deadline": 0.15,
                    "congestion": 0.15,
                }
            else:
                weights = {
                    key: value / total
                    for key, value in weights.items()
                }

            return {
                "rationale": str(
                    data.get("rationale", "")
                )[:500],
                "neighbor_scores": scores,
                "hold_score": clamp(
                    data.get("hold_score", 0.5)
                ),
                "weights": weights,
            }

        except Exception:
            return cls._neutral_neighbor_guidance(
                candidate_ids,
                rationale="parse_error",
            )


# ============================================================
# 10. NetworkEnv — executes the action against real topology -> reward
# ============================================================

# Environnement d'exécution du système LLM+RL.
# Il applique réellement FORWARD/HOLD, déplace les paquets et calcule la
# récompense de base utilisée ensuite par Q-learning.

class NetworkEnv:
    """FANET environment using the QMR-inspired RSSI/Energy/Delay reward."""

    @staticmethod
    def _is_local_minimum(network: Network, uav_id: str, links: list) -> bool:
        current_dist = network.distance_to_gs(uav_id)
        for node_id, _dist, _rssi, is_gs, _energy in links:
            if is_gs:
                return False
            if network.distance_to_gs(node_id) < current_dist:
                return False
        return True

    # Applique l'action choisie par Q-learning.
    # Cette fonction est la frontière entre "décision" et "effet réel" dans le FANET.

    def step(self, uav, network: Network, ground_station: GroundStation, action: str, step: int,
             links: list, best_hop, incoming_buffers: dict, simulation_time_ms: float):
        """Execute one action and CALL calculate_qmr_reward() for the RL transition."""
        start_ms = simulation_time_ms
        end_ms = start_ms + TICK_DURATION_MS
        uav.packets.set_time(start_ms)

        pending_before = uav.packets.pending_packets
        if pending_before == 0:
            return 0.0, 0, {
                "case": "empty_queue", "rssi_dbm": -120.0, "rssi_score": 0.0,
                "next_hop_energy_pct": round(uav.energy, 2), "energy_score": round(uav.energy / 100.0, 4),
                "delay_ms": 0.0, "delay_s": 0.0, "delay_score": 1.0,
                "omega": QMR_OMEGA, "beta": QMR_BETA, "reward": 0.0,
            }

        estimated_delay_ms = uav.packets.mean_hop_wait_ms(start_ms) + TICK_DURATION_MS
        local_minimum = self._is_local_minimum(network, uav.uav_id, links)
        deadline_drops = 0
        dropped_packets = 0
        delivered_now = 0

        can_forward = (
            action == "forward"
            and best_hop is not None
            and uav.energy >= MIN_OPERATIONAL_ENERGY
        )

        if can_forward:
            node_id, dist, rssi, is_gs, next_hop_energy = best_hop
            base_reward, components = calculate_qmr_reward(
                is_destination=is_gs,
                is_local_minimum=local_minimum,
                rssi_dbm=rssi,
                next_hop_energy_pct=next_hop_energy,
                delay_ms=estimated_delay_ms,
                action="forward",
            )

            # --------------------------------------------------------
            # PROGRESS-AWARE REWARD FOR THE PROPOSED MODEL
            # --------------------------------------------------------
            # Successful delivery remains the terminal +1 reward.
            # Intermediate relays receive only a small progress reward and
            # a hop cost, so the agent does not learn unnecessarily long paths.
            # --------------------------------------------------------
            if not is_gs:
                current_distance = network.distance_to_gs(
                    uav.uav_id
                )
                next_distance = network.distance_to_gs(
                    node_id
                )

                progress_m = (
                    current_distance
                    - next_distance
                )

                progress_norm = max(
                    -1.0,
                    min(
                        1.0,
                        progress_m / COMM_RANGE,
                    ),
                )

                base_reward += (
                    LLMRL_PROGRESS_REWARD_WEIGHT
                    * progress_norm
                )

                base_reward -= LLMRL_HOP_COST

                if progress_m <= 0:
                    base_reward -= (
                        LLMRL_NON_PROGRESS_PENALTY
                    )

                base_reward = max(
                    QMR_R_MIN,
                    min(QMR_R_MAX, base_reward),
                )

                components["progress_m"] = round(
                    progress_m,
                    3,
                )
                components["progress_norm"] = round(
                    progress_norm,
                    4,
                )
                components["hop_cost"] = (
                    LLMRL_HOP_COST
                )
                components["reward"] = round(
                    base_reward,
                    4,
                )

            # --------------------------------------------------------
            # CAPACITY-LIMITED EDF SERVICE
            # --------------------------------------------------------
            # At low offered load the complete queue will normally fit inside
            # the tick budget. At high load, excess packets stay queued and
            # accumulate queueing delay; they may later miss their deadline.
            # --------------------------------------------------------
            tx_budget = max(
                0,
                int(uav.tx_budget_packets),
            )

            batches = uav.packets.pop_for_transmission(
                tx_budget
            )

            expired = [
                b for b in batches
                if b.deadline_at_ms <= end_ms
            ]
            alive = [
                b for b in batches
                if b.deadline_at_ms > end_ms
            ]

            if expired:
                dropped_packets = uav.record_deadline_drops(
                    expired,
                    step,
                    end_ms,
                )
                deadline_drops += dropped_packets

                # A deadline miss is a failed transition for learning.
                base_reward = min(
                    base_reward,
                    QMR_R_MIN,
                )
                components["case"] = "deadline_expired"
                components["reward"] = round(
                    base_reward,
                    4,
                )

            # Packets that could not be served this tick remain in the local
            # queue. Expire any whose deadline is reached by t+1.
            capacity_expired = uav.packets.expire_at(
                end_ms
            )

            if capacity_expired:
                dropped_capacity = uav.record_deadline_drops(
                    capacity_expired,
                    step,
                    end_ms,
                )
                deadline_drops += dropped_capacity
                dropped_packets += dropped_capacity

                base_reward = min(
                    base_reward,
                    QMR_R_MIN,
                )
                components["case"] = (
                    "deadline_expired_while_waiting_capacity"
                )
                components["reward"] = round(
                    base_reward,
                    4,
                )

            for waiting_batch in uav.packets.batches:
                waiting_batch.record_state(
                    step=step,
                    time_ms=end_ms,
                    node=uav.uav_id,
                    status="QUEUED",
                    event="WAITING_RELAY_CAPACITY",
                    system="llm_rl",
                )

            components["relay_capacity_pps"] = (
                uav.relay_capacity_pps
            )
            components["tx_budget_packets"] = (
                tx_budget
            )
            components["queue_after_service"] = (
                uav.packets.pending_packets
            )

            if alive:
                forwarded = [b.forwarded_copy(end_ms) for b in alive]
                if is_gs:
                    delivered_now = ground_station.receive(
                        uav.uav_id, forwarded, step, end_ms, system="llm_rl"
                    )
                else:
                    for b in forwarded:
                        b.record_state(
                            step=step,
                            time_ms=end_ms,
                            node=node_id,
                            status="QUEUED",
                            event="FORWARDED",
                            system="llm_rl",
                            from_node=uav.uav_id,
                        )
                    incoming_buffers[node_id].extend(forwarded)

        else:
            # HOLD uses no radio hop, so RSSI contribution is zero. Delay and the
            # sender's remaining energy still change the QMR reward. If the node has
            # no closer neighbor, the QMR local-minimum r_min case applies.
            ref_rssi = max((link[2] for link in links), default=RSSI_MIN_DBM)
            base_reward, components = calculate_qmr_reward(
                is_destination=False,
                is_local_minimum=local_minimum or (action == "forward" and best_hop is None),
                rssi_dbm=ref_rssi,
                next_hop_energy_pct=uav.energy,
                delay_ms=estimated_delay_ms,
                action="hold",
            )

            expired = uav.packets.expire_at(end_ms)
            if expired:
                dropped_packets = uav.record_deadline_drops(expired, step, end_ms)
                deadline_drops += dropped_packets
                base_reward = QMR_R_MIN
                components["case"] = "deadline_expired"
                components["reward"] = round(base_reward, 4)

            # --------------------------------------------------------
            # HOLD PENALTY WHEN USEFUL PROGRESS WAS AVAILABLE
            # --------------------------------------------------------
            # The QMR forwarding equation was not designed to make waiting a
            # profitable routing action. Without an explicit HOLD cost, HOLD can
            # receive a positive reward from delay/energy terms and be learned
            # too often, which is disastrous under hard packet deadlines.
            #
            # We penalize HOLD only when a usable progress link existed.
            # If the UAV is isolated/local-minimum, HOLD remains legitimate.
            # --------------------------------------------------------
            current_dist = network.distance_to_gs(uav.uav_id)
            progress_available = any(
                is_gs
                or (
                    node_id in network.positions
                    and network.distance_to_gs(node_id) < current_dist
                )
                for node_id, _dist, _rssi, is_gs, _energy in links
            )

            if action == "hold" and progress_available:
                deadline_now = uav.packets.min_deadline_ms

                urgency_multiplier = 1.0
                if (
                    math.isfinite(deadline_now)
                    and deadline_now <= LLMRL_URGENT_DEADLINE_MS
                ):
                    urgency_multiplier = 2.0
                elif (
                    math.isfinite(deadline_now)
                    and deadline_now <= LLMRL_SOON_DEADLINE_MS
                ):
                    urgency_multiplier = 1.5

                hold_penalty = (
                    LLMRL_HOLD_PENALTY
                    * urgency_multiplier
                )

                base_reward -= hold_penalty
                base_reward = max(
                    QMR_R_MIN,
                    min(QMR_R_MAX, base_reward),
                )

                components["hold_penalty"] = round(
                    hold_penalty,
                    4,
                )
                components["reward"] = round(
                    base_reward,
                    4,
                )

            # Explicitly keep the packet at the current UAV for this step.
            # This makes "where was packet X at step Y?" directly traceable.
            for b in uav.packets.batches:
                b.record_state(
                    step=step,
                    time_ms=end_ms,
                    node=uav.uav_id,
                    status="QUEUED",
                    event="HELD",
                    system="llm_rl",
                )

        components["action"] = action
        components["pending_before"] = pending_before
        components["delivered_now"] = delivered_now
        components["dropped_packets"] = dropped_packets
        components["base_reward"] = round(base_reward, 4)
        uav.packets.set_time(end_ms)
        return base_reward, deadline_drops, components

# ============================================================
# 11. UAVAgent — mobility + IMU + LLM Core + Agent RL + STM + LTM
# ============================================================

# Agent UAV complet du système proposé.
#
# Il agrège :
#   mobilité + IMU + queue DATA + batterie + STM + LTM + LLM + Q-learning.
#
# Sa méthode ``tick`` réalise une itération complète Observe -> Reason -> Act.

class UAVAgent:
    def __init__(self, uav_id: str, network: Network, backend: str = "groq",
                 q_table_init: dict = None, initial_position: tuple = None,
                 initial_battery_pct: float = None,
                 hello_interval_ms: float = DEFAULT_HELLO_INTERVAL_MS,
                 hello_packet_size_bytes: int = DEFAULT_HELLO_PACKET_SIZE_BYTES,
                 session_id: str = "unknown-session",
                 shared_ltm=None,
                 relay_capacity_pps: float = DEFAULT_RELAY_CAPACITY_PPS,
                 mean_speed_mps: float = DEFAULT_MEAN_SPEED_MPS,
                 all_uav_ids: list = None):
        self.uav_id = uav_id
        self.network = network
        xmin, xmax, ymin, ymax = AREA
        if initial_position is None:
            initial_position = (
                random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(80, 180)
            )
        self.mobility = GaussMarkovMobility(
            x=initial_position[0],
            y=initial_position[1],
            z=initial_position[2],
            area=AREA,
            mean_speed=float(mean_speed_mps),
        )
        self.imu = IMU()
        self.session_id = session_id

        # STM remains private to this UAV and this simulation session.
        self.stm = STM(session_id=session_id, owner_id=uav_id)

        # LTM is shared by the entire swarm. With --uavs 25 this creates only
        # one LangSmith client instead of 25 independent clients.
        self.ltm = (
            shared_ltm
            if shared_ltm is not None
            else LangSmithLTM(dataset_name=LTM_DATASET_NAME)
        )

        # ------------------------------------------------------------
        # NEIGHBOR-AWARE RL ACTION SPACE
        # ------------------------------------------------------------
        # RL directly selects the next-hop candidate instead of making only a
        # binary FORWARD/HOLD decision.
        # ------------------------------------------------------------
        all_uav_ids = list(all_uav_ids or [uav_id])

        next_hop_actions = [
            HOLD_ACTION,
            *[
                peer_id
                for peer_id in all_uav_ids
                if peer_id != uav_id
            ],
            "GS",
        ]

        self.rl = NeighborQLearningAgent(
            action_space=next_hop_actions,
            initial_q=q_table_init,
        )

        self.llm_core = NeighborGuidanceCore(
            uav_id,
            backend=backend,
        )

        self.env = NetworkEnv()
        self.routing_utility = RoutingUtility()

        if initial_battery_pct is None:
            initial_battery_pct = random_initial_battery()
        initial_battery_pct = max(0.0, min(100.0, float(initial_battery_pct)))
        self.battery = BatteryState(level_pct=initial_battery_pct)
        self.packets = PacketQueue()
        self.dropped_deadline = 0
        self.deadline_drop_log = []
        self.last_feedback = {}
        self.pending_transition = None
        self.reward_log = []

        # ------------------------------------------------------------
        # FINITE DATA FORWARDING CAPACITY
        # ------------------------------------------------------------
        # Same capacity model as Q-FANET. The credit is recomputed every tick
        # from packets/s * tick duration. Only the fractional remainder carries
        # over; unused whole-packet capacity is not banked.
        # ------------------------------------------------------------
        self.relay_capacity_pps = float(
            relay_capacity_pps
        )
        if self.relay_capacity_pps <= 0:
            raise ValueError(
                "relay_capacity_pps must be > 0"
            )

        self.tx_capacity_residual = 0.0
        self.tx_budget_packets = 0

        # HELLO control-plane state. Every mobile UAV broadcasts HELLO packets
        # periodically; Source does not generate HELLO packets.
        self.hello_interval_ms = float(hello_interval_ms)
        self.hello_packet_size_bytes = int(hello_packet_size_bytes)
        if self.hello_interval_ms <= 0:
            raise ValueError("hello_interval_ms must be > 0")
        if self.hello_packet_size_bytes <= 0:
            raise ValueError("hello_packet_size_bytes must be > 0")
        self.hello_sequence = 0
        self.next_hello_time_ms = self.hello_interval_ms
        self.hello_tx_count = 0
        self.hello_tx_bytes = 0
        self.hello_log = []
        self.last_hello_rx_ms = {}

        network.register(self)

    @property
    def energy(self) -> float:
        """Compatibility view used by telemetry/routing helpers."""
        return self.battery.level_pct

    def refresh_tx_budget(self) -> int:
        """Refresh this UAV's DATA forwarding budget for one routing tick."""
        exact_budget = (
            self.relay_capacity_pps
            * SIM_TICK_SECONDS
            + self.tx_capacity_residual
        )

        whole_packets = int(
            math.floor(exact_budget + 1e-12)
        )

        self.tx_capacity_residual = (
            exact_budget
            - whole_packets
        )

        self.tx_budget_packets = whole_packets
        return whole_packets

    def update_battery(self, step: int, time_ms: float) -> float:
        """Drain exactly 3 percentage points once per completed tick."""
        return self.battery.consume_tick(step, time_ms)

    # Génère tous les HELLO dus dans l'intervalle du tick et met à jour la fraîcheur
    # de voisinage des UAV destinataires.

    def generate_hello_packets(self, start_ms: float, end_ms: float,
                               active_peer_ids: list) -> list:
        """Generate all HELLO events due in (start_ms, end_ms].

        Because HELLO_INTERVAL_MS=100 ms can be smaller than the 250 ms simulation
        tick, a UAV can generate two or three HELLO packets inside one routing tick.
        """
        generated = []
        eps = 1e-9
        while self.next_hello_time_ms <= end_ms + eps:
            tx_time = self.next_hello_time_ms
            if tx_time > start_ms + eps:
                self.hello_sequence += 1
                packet_id = f"HELLO-{self.uav_id}-{self.hello_sequence:06d}"
                hello = HelloPacket(
                    packet_id=packet_id,
                    sender_id=self.uav_id,
                    sequence=self.hello_sequence,
                    created_at_ms=tx_time,
                    expires_at_ms=tx_time + HELLO_EXPIRE_TIME_MS,
                    size_bytes=self.hello_packet_size_bytes,
                )
                recipients = [
                    pid for pid in active_peer_ids
                    if pid != "GS" and pid in self.network.uav_lookup
                ]
                for pid in recipients:
                    self.network.uav_lookup[pid].last_hello_rx_ms[self.uav_id] = tx_time

                self.hello_tx_count += 1
                self.hello_tx_bytes += hello.size_bytes
                record = asdict(hello)
                record["recipients"] = recipients
                self.hello_log.append(record)
                generated.append(hello)
            self.next_hello_time_ms += self.hello_interval_ms
        return generated

    # Filtre les voisins dont le dernier HELLO est encore valide.

    def fresh_neighbors(self, active_peer_ids: list, now_ms: float) -> list:
        """Return active peers whose latest HELLO has not expired.

        The GS is kept directly from the radio graph because this model only requires
        the mobile UAVs to generate HELLO control packets.
        """
        fresh = []
        for pid in active_peer_ids:
            if pid == "GS":
                fresh.append(pid)
                continue
            last_rx = self.last_hello_rx_ms.get(pid)
            if last_rx is not None and (now_ms - last_rx) <= HELLO_EXPIRE_TIME_MS:
                fresh.append(pid)
        return fresh

    # Construit les liens réellement utilisables :
    # lien actif + distance <= portée + énergie suffisante.

    def current_links(self, active_peer_ids: list) -> list:
        """Return only physically valid active links (distance <= COMM_RANGE)."""
        my_pos = self.network.positions[self.uav_id]
        links = []
        for pid in active_peer_ids:
            if pid == "GS":
                d = self.network.distance_to_gs(self.uav_id)
                if d <= COMM_RANGE:
                    links.append(("GS", d, compute_rssi(d), True, 100.0))
            else:
                other = self.network.uav_lookup.get(pid)
                if other is None or other.energy < MIN_OPERATIONAL_ENERGY:
                    continue
                d = get_distance(my_pos, self.network.positions[pid])
                if d <= COMM_RANGE:
                    links.append((pid, d, compute_rssi(d), False, get_battery(other)))
        return links

    # Fusionne état physique, IMU, RSSI, énergie et distance GS en une Telemetry.

    def _build_telemetry(self, links: list) -> Telemetry:
        true_position = tuple(self.mobility.position)
        speed = self.mobility.speed
        direction = self.mobility.direction
        imu_reading = self.imu.sense(true_position, speed, direction)

        best_rssi = max((l[2] for l in links), default=-120.0)
        if not links:
            connectivity = "isolated"
        elif best_rssi < -75:
            connectivity = "weak_link"
        else:
            connectivity = "strong_link"

        return Telemetry(
            uav_id=self.uav_id,
            energy=round(self.energy, 1),
            speed=round(speed, 2),
            true_position=tuple(round(c, 1) for c in true_position),
            estimated_position=imu_reading["estimated_position"],
            estimated_velocity=imu_reading["estimated_velocity"],
            estimated_altitude=imu_reading["estimated_altitude"],
            direction_deg=round(math.degrees(direction) % 360, 1),
            mag_heading=imu_reading["mag_heading"],
            accel=imu_reading["accel"],
            gyro=imu_reading["gyro"],
            distance_to_gs=round(self.network.distance_to_gs(self.uav_id), 1),
            rssi_best_neighbor=best_rssi,
            connectivity=connectivity,
            sens=compass_sense(direction),
        )

    def _build_candidate_context(
        self,
        links: list,
        packets: PacketQueue,
    ) -> list:
        """Build normalized per-neighbor features for LLM and RL guidance."""

        current_distance = (
            self.network.distance_to_gs(
                self.uav_id
            )
        )

        if (
            math.isfinite(
                packets.min_deadline_ms
            )
        ):
            urgency = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - packets.min_deadline_ms
                    / max(DEADLINE_CHOICES_MS),
                ),
            )
        else:
            urgency = 0.0

        context = []

        for (
            node_id,
            link_distance,
            rssi,
            is_gs,
            energy,
        ) in links:

            next_distance = (
                0.0
                if is_gs
                else self.network.distance_to_gs(
                    node_id
                )
            )

            progress_m = (
                current_distance
                - next_distance
            )

            progress_score = max(
                0.0,
                min(
                    1.0,
                    progress_m / COMM_RANGE,
                ),
            )

            radio_score = normalize_rssi(
                rssi
            )

            stability_score = (
                self.routing_utility.stability_score(
                    rssi
                )
            )

            energy_score = max(
                0.0,
                min(
                    1.0,
                    energy / 100.0,
                ),
            )

            # Candidate-side congestion is an important advantage of the
            # proposed model at medium/high load. Q-FANET's tie-breaker does not
            # use the neighbor DATA queue, while LLM+RL can avoid overloaded
            # relays when another progress-making path is available.
            if is_gs:
                neighbor_queue_packets = 0
                neighbor_service_per_tick = 1.0
                congestion_score = 1.0
            else:
                neighbor_agent = self.network.uav_lookup.get(node_id)
                if neighbor_agent is not None:
                    neighbor_queue_packets = int(
                        neighbor_agent.packets.pending_packets
                    )
                    neighbor_service_per_tick = max(
                        1.0,
                        float(neighbor_agent.relay_capacity_pps)
                        * SIM_TICK_SECONDS,
                    )
                else:
                    neighbor_queue_packets = 0
                    neighbor_service_per_tick = max(
                        1.0,
                        self.relay_capacity_pps * SIM_TICK_SECONDS,
                    )

                queue_pressure = (
                    neighbor_queue_packets
                    / neighbor_service_per_tick
                )
                congestion_score = 1.0 / (1.0 + queue_pressure)

            # Under urgency, progress matters more.
            deadline_score = (
                1.0
                if is_gs
                else (
                    urgency * progress_score
                    + (1.0 - urgency)
                    * (
                        0.5
                        + 0.5 * progress_score
                    )
                )
            )

            context.append({
                "id": node_id,
                "is_gs": bool(is_gs),
                "link_distance_m": round(link_distance, 2),
                "distance_to_gs_m": round(next_distance, 2),
                "progress_m": round(progress_m, 2),
                "progress_score": round(progress_score, 4),
                "rssi_dbm": round(rssi, 2),
                "radio_score": round(radio_score, 4),
                "stability_score": round(stability_score, 4),
                "energy_pct": round(energy, 2),
                "energy_score": round(energy_score, 4),
                "neighbor_queue_packets": neighbor_queue_packets,
                "neighbor_service_packets_per_tick": round(
                    neighbor_service_per_tick,
                    3,
                ),
                "congestion_score": round(congestion_score, 4),
                "deadline_score": round(deadline_score, 4),
            })

        return context

    @staticmethod
    def _candidate_heuristic_scores(
        candidate_context: list,
        weights: dict,
        progress_available: bool,
    ) -> dict:
        """Topology/QoS prior used in addition to learned Q and LLM scores."""

        defaults = {
            "progress": 0.40,
            "stability": 0.20,
            "energy": 0.10,
            "deadline": 0.15,
            "congestion": 0.15,
        }

        raw = {
            key: max(
                0.0,
                float(
                    weights.get(
                        key,
                        default,
                    )
                ),
            )
            for key, default in defaults.items()
        }

        total = sum(raw.values())

        if total <= 0:
            raw = defaults
            total = 1.0

        normalized = {
            key: value / total
            for key, value in raw.items()
        }

        scores = {}

        for item in candidate_context:
            score = (
                normalized["progress"]
                * item["progress_score"]
                + normalized["stability"]
                * item["stability_score"]
                + normalized["energy"]
                * item["energy_score"]
                + normalized["deadline"]
                * item["deadline_score"]
                + normalized["congestion"]
                * item.get("congestion_score", 1.0)
            )

            if item["is_gs"]:
                score = 1.0

            scores[item["id"]] = max(
                0.0,
                min(1.0, score),
            )

        # HOLD should never compete with a safe progress-making action.
        # It remains available as a recovery action only when progress is absent.
        scores[HOLD_ACTION] = (
            0.02
            if progress_available
            else 0.45
        )

        return scores

    def update_mobility(self, dt=1.0):
        self.mobility.update(dt)

    def record_deadline_drops(self, expired: list, step: int, time_ms: float) -> int:
        count = sum(b.count for b in expired)
        self.dropped_deadline += count
        for b in expired:
            b.record_state(
                step=step,
                time_ms=time_ms,
                node=self.uav_id,
                status="DROPPED_DEADLINE",
                event="DEADLINE_EXPIRED",
                system="llm_rl",
            )
            self.deadline_drop_log.append({
                "system": "llm_rl", "step": step, "time_ms": time_ms, "time_s": time_ms / 1000.0,
                "uav_id": self.uav_id, "batch_id": b.batch_id, "packet_id": b.packet_id, "count": b.count,
                "initial_deadline_ms": b.initial_deadline_ms, "created_at_ms": b.created_at_ms,
                "hops": b.hops, "current_node": b.current_node, "status": b.status,
                "trajectory": [dict(event) for event in b.trajectory],
            })
        return count

    # Boucle décisionnelle principale d'un UAV LLM+RL.
    #
    # Ordre logique :
    #   OBSERVE
    #   -> récupérer STM/LTM
    #   -> guidance LLM
    #   -> action Q-learning
    #   -> classement du next hop
    #   -> environnement / reward
    #   -> mémorisation de la transition
    #
    # Si la queue DATA est vide, l'appel LLM est volontairement évité.

    def tick(
        self,
        step: int,
        ground_station: GroundStation,
        active_peer_ids: list,
        incoming_buffers: dict,
        simulation_time_ms: float,
    ):
        """Neighbor-aware LLM + Q-learning routing decision.

        Control flow:

            OBSERVE
              -> candidate features
              -> STM + quality-aware LTM
              -> LLM candidate scores
              -> Q-learning chooses actual next hop
              -> environment executes FORWARD/HOLD
              -> reward
              -> STM/LTM update
        """

        self.packets.set_time(
            simulation_time_ms
        )

        self.refresh_tx_budget()

        links = self.current_links(
            active_peer_ids
        )

        telemetry = self._build_telemetry(
            links
        )

        neighbors_summary = ", ".join(
            (
                f"{nid}(d={d:.0f}m,"
                f"rssi={r:.1f}dBm,"
                f"E={e:.0f}%"
                f"{',GS' if is_gs else ''})"
            )
            for nid, d, r, is_gs, e in links
        ) or f"none within {COMM_RANGE:.0f} m"

        if self.packets.pending_packets == 0:
            self.pending_transition = None

            print(
                f"\n[{self.uav_id}] tick {step} "
                f"t={simulation_time_ms/1000.0:.2f}s "
                f"queue empty -> skip LLM/RL"
            )

            return

        candidate_context = (
            self._build_candidate_context(
                links,
                self.packets,
            )
        )

        state = (
            self.rl.discretize_state(
                telemetry,
                self.packets,
                links,
                self.network,
                self.uav_id,
            )
        )

        current_distance = (
            self.network.distance_to_gs(
                self.uav_id
            )
        )

        progress_links = [
            link
            for link in links
            if (
                link[3]
                or self.network.distance_to_gs(
                    link[0]
                ) < current_distance
            )
        ]

        progress_available = bool(
            progress_links
        )

        direct_gs = next(
            (
                link
                for link in links
                if link[3]
            ),
            None,
        )

        min_deadline = (
            self.packets.min_deadline_ms
        )

        # --------------------------------------------------------
        # LLM GUIDANCE
        # --------------------------------------------------------
        # Obvious terminal/isolated cases do not require an LLM call.
        # --------------------------------------------------------
        if direct_gs is not None:
            guidance = {
                "rationale": (
                    "Ground Station is directly reachable; "
                    "terminal delivery is preferred."
                ),
                "neighbor_scores": {
                    item["id"]: (
                        1.0
                        if item["id"] == "GS"
                        else 0.2
                    )
                    for item in candidate_context
                },
                "hold_score": 0.0,
                "weights": {
                    "progress": 0.55,
                    "stability": 0.15,
                    "energy": 0.05,
                    "deadline": 0.15,
                    "congestion": 0.10,
                },
            }

        elif links:
            guidance = (
                self.llm_core.guide_neighbors(
                    telemetry=telemetry,
                    packets=self.packets,
                    stm=self.stm,
                    ltm=self.ltm,
                    candidate_context=candidate_context,
                    previous_feedback=self.last_feedback,
                    simulation_time_ms=simulation_time_ms,
                )
            )

        else:
            guidance = {
                "rationale": (
                    "No physically reachable next hop; HOLD is required."
                ),
                "neighbor_scores": {},
                "hold_score": 1.0,
                "weights": {
                    "progress": 0.40,
                    "stability": 0.20,
                    "energy": 0.10,
                    "deadline": 0.15,
                    "congestion": 0.15,
                },
            }

        llm_scores = {
            **guidance.get(
                "neighbor_scores",
                {},
            ),
            HOLD_ACTION: float(
                guidance.get(
                    "hold_score",
                    0.5,
                )
            ),
        }

        heuristic_scores = (
            self._candidate_heuristic_scores(
                candidate_context,
                guidance.get(
                    "weights",
                    {},
                ),
                progress_available,
            )
        )

        # --------------------------------------------------------
        # HARD ACTION FEASIBILITY / DEADLINE MASK
        # --------------------------------------------------------
        if self.energy < MIN_OPERATIONAL_ENERGY:
            allowed_actions = [
                HOLD_ACTION
            ]
            decision_guard = (
                "battery_depleted"
            )

        elif not links:
            allowed_actions = [
                HOLD_ACTION
            ]
            decision_guard = (
                "no_usable_link"
            )

        elif direct_gs is not None:
            allowed_actions = [
                "GS"
            ]
            decision_guard = (
                "direct_gs"
            )

        elif progress_available:
            # Performance/safety mask: when at least one reachable relay moves
            # the packet closer to GS, do not spend a full tick on HOLD or on a
            # regressive hop. The LLM+RL policy still chooses WHICH progress
            # relay is best using Q + LLM + QoS/congestion guidance.
            allowed_actions = [
                link[0]
                for link in progress_links
            ]

            decision_guard = (
                "progress_only"
            )

        else:
            # No progress link exists: allow recovery through a lateral/regressive
            # relay or HOLD until mobility changes the local topology.
            allowed_actions = [
                link[0]
                for link in links
            ] + [
                HOLD_ACTION
            ]

            decision_guard = (
                "local_minimum_recovery"
            )

        # Keep evaluation exploration small whenever useful progress exists.
        # This is especially important with 25/50 UAVs, where random exploration
        # over many reachable relays can dominate the PDR loss.
        if len(allowed_actions) == 1:
            epsilon_override = 0.0
        elif progress_available:
            epsilon_override = min(
                self.rl.epsilon,
                LLMRL_PROGRESS_EPSILON_CAP,
            )
        elif (
            math.isfinite(min_deadline)
            and min_deadline <= LLMRL_SOON_DEADLINE_MS
        ):
            epsilon_override = min(
                self.rl.epsilon,
                0.03,
            )
        else:
            epsilon_override = None

        selected_action, score_breakdown = (
            self.rl.choose_next_hop(
                state=state,
                allowed_actions=allowed_actions,
                llm_scores=llm_scores,
                heuristic_scores=heuristic_scores,
                epsilon_override=epsilon_override,
            )
        )

        if selected_action == HOLD_ACTION:
            env_action = "hold"
            best_hop = None
            chosen_neighbor = None

        else:
            env_action = "forward"
            chosen_neighbor = selected_action

            best_hop = next(
                (
                    link
                    for link in links
                    if link[0] == selected_action
                ),
                None,
            )

            # Safety fallback if the topology changed unexpectedly between
            # candidate construction and execution.
            if best_hop is None:
                env_action = "hold"
                selected_action = HOLD_ACTION
                chosen_neighbor = None
                decision_guard = (
                    "selected_link_unavailable"
                )

        raw_reward, deadline_drops, reward_components = (
            self.env.step(
                self,
                self.network,
                ground_station,
                env_action,
                step,
                links,
                best_hop,
                incoming_buffers,
                simulation_time_ms,
            )
        )

        # The LLM influences action selection, not the environment reward.
        # This keeps the learned value grounded in actual routing outcome.
        learning_reward = raw_reward
        llm_bonus = 0.0
        shaped_reward = learning_reward

        delivered_now = int(
            reward_components.get(
                "delivered_now",
                0,
            )
            or 0
        )

        memory_action = (
            HOLD_ACTION
            if chosen_neighbor is None
            else f"FORWARD_TO_{chosen_neighbor}"
        )

        self.stm.add({
            "step": step,
            "state": state,
            "action": memory_action,
            "chosen_neighbor": chosen_neighbor,
            "reward": shaped_reward,
            "raw_reward": raw_reward,
            "delivered_now": delivered_now,
            "deadline_drops": deadline_drops,
            "energy": self.energy,
            "time_ms": simulation_time_ms,
            "pending_packets": self.packets.pending_packets,
            "min_deadline_ms": (
                round(
                    self.packets.min_deadline_ms,
                    3,
                )
                if math.isfinite(
                    self.packets.min_deadline_ms
                )
                else "inf"
            ),
            "distance_to_gs_m": telemetry.distance_to_gs,
            "best_rssi_dbm": telemetry.rssi_best_neighbor,
        })

        self.ltm.save(
            telemetry=telemetry,
            packets=self.packets,
            action=memory_action,
            chosen_neighbor=chosen_neighbor,
            rationale=guidance.get(
                "rationale",
                "",
            ),
            raw_reward=raw_reward,
            shaped_reward=shaped_reward,
            step=step,
            session_id=self.session_id,
            delivered_now=delivered_now,
            deadline_drops=deadline_drops,
            reward_components=reward_components,
        )

        self.pending_transition = {
            "state": state,
            "action": selected_action,
            "memory_action": memory_action,
            "chosen_neighbor": chosen_neighbor,
            "env_action": env_action,
            "raw_reward": raw_reward,
            "llm_bonus": llm_bonus,
            "shaped_reward": shaped_reward,
            "deadline_drops": deadline_drops,
            "delivered_now": delivered_now,
            "reward_components": reward_components,
            "decision_guard": decision_guard,
            "allowed_actions": list(
                allowed_actions
            ),
            "score_breakdown": score_breakdown,
            "llm_neighbor_scores": dict(
                llm_scores
            ),
            "heuristic_scores": dict(
                heuristic_scores
            ),
            "rationale": guidance.get(
                "rationale",
                "",
            ),
        }

        self.reward_log.append({
            "step": step,
            "time_ms": simulation_time_ms,
            "time_s": round(
                simulation_time_ms / 1000.0,
                3,
            ),
            "state": state,
            "action": selected_action,
            "env_action": env_action,
            "chosen_neighbor": chosen_neighbor,
            "base_reward": round(
                raw_reward,
                4,
            ),
            "final_reward": round(
                shaped_reward,
                4,
            ),
            "components": reward_components,
            "decision_guard": decision_guard,
            "allowed_actions": list(
                allowed_actions
            ),
            "score_breakdown": score_breakdown,
            "llm_neighbor_scores": dict(
                llm_scores
            ),
            "heuristic_scores": dict(
                heuristic_scores
            ),
            "rationale": guidance.get(
                "rationale",
                "",
            ),
        })

        print(
            f"\n[{self.uav_id}] tick {step} "
            f"t={simulation_time_ms/1000.0:.2f}s "
            f"(LLM={self.llm_core.backend})"
        )

        print(
            f"  pos={telemetry.true_position} "
            f"dist_to_GS={telemetry.distance_to_gs:.0f}m "
            f"connectivity={telemetry.connectivity} "
            f"energy={telemetry.energy:.1f}%"
        )

        print(
            f"  links: {neighbors_summary}"
        )

        print(
            f"  state={state} "
            f"selected={selected_action} "
            f"env_action={env_action} "
            f"reward={raw_reward:.3f} "
            f"drops={deadline_drops}"
        )

        print(
            f"  guard={decision_guard} "
            f"allowed={allowed_actions}"
        )

        print(
            f"  scores={score_breakdown}"
        )

        print(
            f"  queue_pending={self.packets.pending_packets} "
            f"min_deadline={self.packets.min_deadline_ms}"
        )

    def finalize_transition(
        self,
        active_peer_ids: list,
        end_time_ms: float,
    ):
        """Update Q(S_t, next-hop) using the real synchronized S_(t+1)."""

        self.packets.set_time(
            end_time_ms
        )

        if not self.pending_transition:
            return

        links = self.current_links(
            active_peer_ids
        )

        next_telemetry = (
            self._build_telemetry(
                links
            )
        )

        next_state = (
            self.rl.discretize_state(
                next_telemetry,
                self.packets,
                links,
                self.network,
                self.uav_id,
            )
        )

        next_current_distance = (
            self.network.distance_to_gs(
                self.uav_id
            )
        )

        next_progress_links = [
            link
            for link in links
            if (
                link[3]
                or self.network.distance_to_gs(
                    link[0]
                ) < next_current_distance
            )
        ]

        direct_gs = any(
            link[3]
            for link in links
        )

        next_deadline = (
            self.packets.min_deadline_ms
        )

        if self.packets.pending_packets == 0:
            next_allowed = [
                HOLD_ACTION
            ]

        elif not links:
            next_allowed = [
                HOLD_ACTION
            ]

        elif direct_gs:
            next_allowed = [
                "GS"
            ]

        elif next_progress_links:
            next_allowed = [
                link[0]
                for link in next_progress_links
            ]

        else:
            next_allowed = [
                link[0]
                for link in links
            ] + [
                HOLD_ACTION
            ]

        tr = self.pending_transition

        terminal = (
            self.packets.pending_packets == 0
            and (
                tr.get(
                    "delivered_now",
                    0,
                ) > 0
                or tr.get(
                    "deadline_drops",
                    0,
                ) > 0
            )
        )

        self.rl.update(
            state=tr["state"],
            action=tr["action"],
            reward=tr["shaped_reward"],
            next_state=next_state,
            next_allowed_actions=next_allowed,
            terminal=terminal,
        )

        self.last_feedback = {
            **tr,
            "next_state": next_state,
            "time_ms": end_time_ms,
        }

        self.stm.update_last(
            next_state=next_state,
            time_end_ms=end_time_ms,
        )

        self.pending_transition = None

    def deadline_snapshot(self, now_ms: float) -> dict:
        snap = self.packets.deadline_snapshot(now_ms)
        snap["dropped_deadline"] = self.dropped_deadline
        return snap


# ============================================================
# 12. Run: 1 fixed Source -> N mobile UAV relays -> 1 fixed GS
# ============================================================
# Default scenario: 500 m x 500 m area, 180 m hard radio range, one hop per 250 ms tick.
# UAVs start in a lightly jittered relay corridor so the experiment is not dominated by a
# permanently disconnected initial topology; Gauss-Markov mobility and in-range link churn
# then make connectivity dynamic during the run.
#
# UAV backends alternate between Groq (cloud) and Ollama/Phi-3 (local) unless forced.
# backend, so a single run gives a side-by-side comparison of the two. Both backends'
# calls are traced to LangSmith automatically (LANGSMITH_TRACING=true in your .env),
# so you can also compare them directly in the LangSmith UI afterward.

# Génère une disposition initiale en corridor/maillage entre Source et GS.
# L'objectif est d'éviter qu'un placement aléatoire rende le réseau totalement
# déconnecté dès le début et masque les performances du routage.

def make_initial_relay_positions(num_uavs: int) -> list:
    """Create a connected-but-nontrivial relay corridor between Source and GS.

    Pure uniform random placement can easily leave a sparse 180 m network permanently
    disconnected from the GS, making PDR=0 regardless of routing quality.  This stratified
    initialization keeps neighboring relays within radio range at t=0 while Gauss-Markov
    mobility and link churn subsequently change the topology.
    """
    sx, sy, sz = SOURCE_POSITION
    gx, gy, _ = GS_POSITION
    positions = []
    for i in range(num_uavs):
        frac = (i + 1) / (num_uavs + 1)
        center_x = sx + frac * (gx - sx)
        center_y = sy + frac * (gy - sy)
        # Alternating lateral offsets create a mesh-like zig-zag rather than a straight line.
        lateral = (45.0 if i % 2 == 0 else -45.0) if num_uavs >= 4 else 25.0 * (-1 if i % 2 else 1)
        x = min(max(center_x + lateral + random.uniform(-10.0, 10.0), AREA[0]), AREA[1])
        y = min(max(center_y + random.uniform(-8.0, 8.0), AREA[2]), AREA[3])
        z = random.uniform(90.0, 130.0)
        positions.append((x, y, z))
    return positions


# Normalise une entrée de log afin qu'elle puisse être utilisée dans l'export
# de suivi détaillé des paquets.

def _packet_record_from_log(entry: dict) -> dict:
    """Normalize a delivered/dropped log entry into the tracking export format."""
    return {
        "packet_id": entry.get("packet_id", entry.get("batch_id")),
        "current_node": entry.get("current_node", entry.get("uav_id", "UNKNOWN")),
        "status": entry.get("status", "UNKNOWN"),
        "hops": entry.get("hops", 0),
        "created_at_ms": entry.get("created_at_ms"),
        "deadline_at_ms": entry.get("deadline_at_ms"),
        "size_bytes": entry.get("size_bytes"),
        "trajectory": [dict(event) for event in entry.get("trajectory", [])],
    }


# Reconstruit l'état final et la trajectoire complète de chaque paquet DATA
# pour un système donné.

def _build_packet_tracking(source: Source, uavs: list, ground_station: GroundStation,
                           qfanet_router: QFANETRouter, system: str) -> dict:
    """Collect the latest complete record of every DATA packet for one system."""
    records = {}

    # Packets that never entered either routing system still belong to the experiment.
    for rec in source.source_expired_records:
        records[rec["packet_id"]] = dict(rec)
    for packet in source.backlog_packets:
        records[packet.packet_id] = packet.tracking_record()

    if system == "llm_rl":
        for uav in uavs:
            for packet in uav.packets.batches:
                records[packet.packet_id] = packet.tracking_record()
            for entry in uav.deadline_drop_log:
                records[entry.get("packet_id", entry.get("batch_id"))] = _packet_record_from_log(entry)
    elif system == "qfanet":
        for queue in qfanet_router.queues.values():
            for packet in queue.batches:
                records[packet.packet_id] = packet.tracking_record()
        for entry in qfanet_router.deadline_drop_log:
            records[entry.get("packet_id", entry.get("batch_id"))] = _packet_record_from_log(entry)
    else:
        raise ValueError(f"Unknown tracking system: {system}")

    for entry in ground_station.delivery_records:
        records[entry.get("packet_id", entry.get("batch_id"))] = _packet_record_from_log(entry)

    return {packet_id: records[packet_id] for packet_id in sorted(records)}


# Construit une vue temporelle :
#     tick -> packet_id -> position / état / événement
# utile pour le dashboard et l'analyse expérimentale.

def _build_packet_state_by_step(packet_records: dict, total_steps: int) -> dict:
    """Build a direct lookup: step -> packet_id -> latest node/status at that step."""
    timeline = {}
    for step in range(1, total_steps + 1):
        step_state = {}
        for packet_id, record in packet_records.items():
            events = [
                event for event in record.get("trajectory", [])
                if event.get("step") is not None and int(event["step"]) <= step
            ]
            if not events:
                continue
            event = events[-1]
            step_state[packet_id] = {
                "node": event.get("node"),
                "status": event.get("status"),
                "event": event.get("event"),
                "hops": event.get("hops", 0),
                "time_ms": event.get("time_ms"),
            }
        timeline[str(step)] = step_state
    return timeline


# Fonction principale de simulation.
#
# Elle orchestre synchroniquement :
#   1) mobilité,
#   2) connectivité,
#   3) HELLO,
#   4) génération DATA,
#   5) décisions LLM+RL,
#   6) décisions Q-FANET,
#   7) mises à jour Q-learning,
#   8) batterie,
#   9) statistiques,
#  10) export JSON.
#
# Les transmissions sont "staged" : un paquet ne peut effectuer qu'un seul
# saut par tick, ce qui évite un multi-hop instantané irréaliste.

def run_simulation(steps=20, num_uavs=DEFAULT_NUM_UAVS, verbose=True, persist_q_table=True, export_json=True,
                     progress_callback=None, force_backend=None,
                     packet_rate_pps=DEFAULT_PACKET_RATE_PPS,
                     relay_capacity_pps=DEFAULT_RELAY_CAPACITY_PPS,
                     mean_speed_mps=DEFAULT_MEAN_SPEED_MPS,
                     packet_size_bytes=DEFAULT_DATA_PACKET_SIZE_BYTES,
                     hello_interval_ms=DEFAULT_HELLO_INTERVAL_MS,
                     hello_packet_size_bytes=DEFAULT_HELLO_PACKET_SIZE_BYTES,
                     output_json="comparison_results.json"):
    """Run a synchronized LLM+RL vs Q-FANET experiment.

    Both systems receive the same traffic and the same mobility/connectivity snapshot.  They keep
    independent queues, energy states and Q-tables.  One tick represents ``TICK_DURATION_MS`` and
    every packet batch has an absolute end-to-end deadline.
    """
    if not verbose:
        import builtins
        _real_print = builtins.print
        def _silent_print(*a, **kw):
            pass
        builtins.print = _silent_print

    try:
        if steps < 2:
            raise ValueError("steps must be >= 2")
        if num_uavs < 1:
            raise ValueError("num_uavs must be >= 1")

        # With the current Source/GS geometry and 180 m hard range, one relay
        # cannot geometrically bridge the full path. We allow the run for testing,
        # but warn because PDR may remain zero for a very small fleet.
        if num_uavs < 2:
            print(
                "[Warning] With COMM_RANGE=180 m, fewer than 2 UAVs may not "
                "provide an end-to-end Source->GS path."
            )

        # ``steps`` is the TOTAL experiment duration.  New packets are generated
        # only during the traffic phase.  The remaining ticks let in-flight packets
        # finish, preventing end-of-run packets from artificially depressing PDR.
        drain_ticks = min(DRAIN_TICKS, steps - 1)
        traffic_ticks = max(1, steps - drain_ticks)

        session_id = f"SIM-{int(time.time() * 1000)}"

        network = Network()
        ground_station = GroundStation()
        qfanet_ground_station = GroundStation()
        source = Source(
            packet_rate_pps=packet_rate_pps,
            packet_size_bytes=packet_size_bytes,
        )

        data_interval_ms = source.data_interval_ms

        # The first DATA packet is created at t=0 ms. The user controls the
        # offered load directly in packets per simulated second.
        print(
            f"[Traffic] Offered load = {packet_rate_pps:.2f} packet/s "
            f"(one DATA every {data_interval_ms:.3f} ms); "
            f"first DATA at t=0 ms."
        )

        saved_q_tables = load_q_tables() if persist_q_table else {}
        saved_qfanet_q_tables = load_q_tables(QFANET_Q_TABLE_FILENAME) if persist_q_table else {}

        if force_backend in ("groq", "ollama"):
            backend_cycle = [force_backend]
        else:
            backend_cycle = ["groq", "ollama"]
        initial_positions = make_initial_relay_positions(num_uavs)
        uav_ids = [f"UAV-{i+1}" for i in range(num_uavs)]

        # A Q-FANET table learned with another fleet size has a different action
        # space (neighbor IDs). Reusing it would unfairly bias old peers and leave
        # new peers at Q=0.5, so start a fresh baseline table for this topology.
        if not qfanet_table_matches_topology(saved_qfanet_q_tables, uav_ids):
            print(
                "[Q-FANET] Persisted Q-table topology does not match "
                f"the current {num_uavs}-UAV fleet -> starting a fresh Q-FANET table."
            )
            saved_qfanet_q_tables = {}

        # Generate the random batteries ONCE per run.  These exact values are
        # reused by the LLM+RL system and the Q-FANET baseline.
        initial_batteries = {
            uid: random_initial_battery()
            for uid in uav_ids
        }

        # Create ONE LTM for the complete swarm.
        print("[LTM] Initializing one shared long-term memory for the swarm...")
        shared_ltm = LangSmithLTM(dataset_name=LTM_DATASET_NAME)
        if shared_ltm.client is None:
            print("[LTM] Remote LangSmith unavailable -> local LTM fallback active.")
        else:
            print("[LTM] Shared LangSmith LTM connected.")

        uavs = [
            UAVAgent(
                uid,
                network,
                backend=backend_cycle[i % len(backend_cycle)],
                q_table_init=saved_q_tables.get(uid),
                initial_position=initial_positions[i],
                initial_battery_pct=initial_batteries[uid],
                hello_interval_ms=hello_interval_ms,
                hello_packet_size_bytes=hello_packet_size_bytes,
                session_id=session_id,
                shared_ltm=shared_ltm,
                relay_capacity_pps=relay_capacity_pps,
                mean_speed_mps=mean_speed_mps,
                all_uav_ids=uav_ids,
            )
            for i, uid in enumerate(uav_ids)
        ]
        network.update_positions()

        shared_graph = SharedLinkGraph(
            node_ids=uav_ids + ["GS"], lambda_rate=LAMBDA_RATE, mu_rate=MU_RATE,
            comm_range=COMM_RANGE,
        )
        qfanet_router = QFANETRouter(
            network,
            uav_ids=uav_ids,
            gs_id="GS",
            q_tables_init=saved_qfanet_q_tables,
            initial_batteries=initial_batteries,
            relay_capacity_pps=relay_capacity_pps,
        )

        print("===== Network initialized =====")
        print(f"Simulation session ID: {session_id}")
        print(f"Number of UAVs: {num_uavs}")
        if force_backend == "ollama":
            print(
                f"Ollama forced for all UAVs: model={os.getenv('OLLAMA_MODEL', 'phi3')} "
                f"server={os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}"
            )
        print(f"Area: {AREA[1]-AREA[0]:.0f} m x {AREA[3]-AREA[2]:.0f} m")
        print(f"Hard radio range: {COMM_RANGE:.0f} m (3-D distance)")
        print(f"Tick duration: {TICK_DURATION_MS:.0f} ms = {SIM_TICK_SECONDS:.2f} s")
        print(f"Packet deadlines: {list(DEADLINE_CHOICES_MS)} ms")
        print(
            f"DATA offered load: {packet_rate_pps:.2f} packet/s "
            f"(interval={data_interval_ms:.3f} ms), "
            f"size={packet_size_bytes} B"
        )
        print(
            f"Per-UAV DATA relay capacity: {relay_capacity_pps:.2f} packet/s "
            f"(~{relay_capacity_pps * SIM_TICK_SECONDS:.2f} packet/tick)"
        )
        if packet_rate_pps < relay_capacity_pps:
            print(
                "[Load regime] Source rate is below one relay's service capacity "
                "-> light/moderate offered load."
            )
        elif packet_rate_pps == relay_capacity_pps:
            print(
                "[Load regime] Source rate is approximately equal to one relay's "
                "service capacity -> saturation boundary."
            )
        else:
            print(
                "[Load regime] Source rate exceeds one relay's service capacity "
                "-> queueing/deadline pressure expected."
            )
        print(f"HELLO interval: {hello_interval_ms:.0f} ms ({1000.0/hello_interval_ms:.2f} Hz per UAV), size={hello_packet_size_bytes} B, expire={HELLO_EXPIRE_TIME_MS:.0f} ms")
        print(f"Initial Q value for unseen state/action or neighbor: {INITIAL_Q_VALUE}")
        print(f"QMR reward: omega={QMR_OMEGA}, beta={QMR_BETA}, r_max={QMR_R_MAX}, r_min={QMR_R_MIN}")
        print(
            f"Experiment phases: {traffic_ticks} traffic tick(s) + "
            f"{drain_ticks} drain tick(s) = {steps} total"
        )
        print(
            f"Exploration epsilon: {EPSILON_START:.2f} -> min {EPSILON_MIN:.2f}, "
            f"decay={EPSILON_DECAY_PER_TICK:.2f}/tick"
        )
        print(
            "LLM+RL architecture: neighbor-aware Q-learning + "
            "LLM candidate guidance + STM/LTM"
        )
        print(
            "LLM+RL performance guard: "
            f"urgent<={LLMRL_URGENT_DEADLINE_MS:.0f}ms, "
            f"hold_penalty={LLMRL_HOLD_PENALTY:.2f}, "
            f"Q-range=[{LLMRL_Q_MIN:.1f},{LLMRL_Q_MAX:.1f}], "
            "direct-GS priority enabled"
        )
        print(
            f"Battery model: random initial [{BATTERY_INITIAL_MIN_PCT:.0f}%, "
            f"{BATTERY_INITIAL_MAX_PCT:.0f}%], "
            f"-{BATTERY_DRAIN_PER_TICK_PCT:.0f}% per tick "
            f"({BATTERY_DRAIN_PER_TICK_PCT / SIM_TICK_SECONDS:.1f}% per simulated second)"
        )
        print("Initial UAV batteries (shared starting conditions):")
        for uid in uav_ids:
            print(f"  {uid}: {initial_batteries[uid]:.2f}%")
        print(f"Source: {SOURCE_POSITION} | GS: {GS_POSITION}")
        print(
            f"{num_uavs} mobile UAV relays, Gauss-Markov mobility "
            f"(mean speed={mean_speed_mps:.2f} m/s)"
        )
        print(
            f"In-range link churn: lambda={LAMBDA_RATE}/s, mu={MU_RATE}/s, "
            f"mean active dwell={1.0/MU_RATE:.2f}s"
        )
        print("Both systems share mobility/connectivity/traffic but keep independent energy/queues/Q-tables.")

        ticks_history = []
        time_ms_history, time_s_history = [], []
        llm_cum_history, qfanet_cum_history = [], []
        generated_cum_history = []
        llm_pdr_over_time, qfanet_pdr_over_time = [], []
        llm_deadline_min, qfanet_deadline_min = [], []
        llm_deadline_mean, qfanet_deadline_mean = [], []
        llm_pending_history, qfanet_pending_history = [], []
        llm_dropped_history, qfanet_dropped_history = [], []
        llm_battery_history = {uid: [] for uid in uav_ids}
        qfanet_battery_history = {uid: [] for uid in uav_ids}

        try:
            for step in range(1, steps + 1):
                start_time_ms = (step - 1) * TICK_DURATION_MS
                end_time_ms = step * TICK_DURATION_MS
                phase = "TRAFFIC" if step <= traffic_ticks else "DRAIN"
                print(
                    f"\n===== Tick {step}/{steps} [{phase}] | "
                    f"t={start_time_ms/1000.0:.2f}s -> {end_time_ms/1000.0:.2f}s ====="
                )

                # Identical decaying exploration schedule for both systems.
                current_epsilon = max(
                    EPSILON_MIN,
                    EPSILON_START * (EPSILON_DECAY_PER_TICK ** (step - 1)),
                )
                for uav in uavs:
                    uav.rl.epsilon = current_epsilon
                qfanet_router.epsilon = current_epsilon

                # Physical mobility is integrated using the actual 0.25 s tick duration.
                for uav in uavs:
                    uav.update_mobility(dt=SIM_TICK_SECONDS)
                network.update_positions()

                # Hard geometric gate (<=180 m) first, then stochastic in-range fading/churn.
                graph_positions = dict(network.positions)
                graph_positions["GS"] = network.gs_position
                active_edges = shared_graph.step(graph_positions, dt=SIM_TICK_SECONDS)
                active_adj = shared_graph.neighbors_by_node(active_edges)

                # HELLO control sub-phase. Each UAV generates HELLO packets every
                # hello_interval_ms (100 ms by default), independently of the 250 ms tick.
                for uav in uavs:
                    uav.generate_hello_packets(
                        start_time_ms, end_time_ms, active_adj.get(uav.uav_id, [])
                    )
                routing_adj = {
                    uav.uav_id: uav.fresh_neighbors(
                        active_adj.get(uav.uav_id, []), end_time_ms
                    )
                    for uav in uavs
                }

                # DATA generation is interval-based and starts at t=0 ms.
                #
                # Example with 100 ms DATA interval:
                #   0, 100, 200, 300, ... ms
                #
                # The Source is evaluated at the beginning of each routing tick and
                # creates every DATA event whose exact timestamp has elapsed.
                # Generation stops at traffic_end_ms.
                traffic_end_ms = traffic_ticks * TICK_DURATION_MS
                source.tick(
                    network, qfanet_router, start_time_ms,
                    generation_cutoff_ms=traffic_end_ms,
                )
                if step == traffic_ticks + 1:
                    print(
                        f"[Traffic] DATA generation stopped at t={traffic_end_ms/1000.0:.2f}s; "
                        f"draining in-flight packets for {drain_ticks} tick(s)."
                    )

                # LLM+RL uses staged incoming buffers so each DATA packet moves at most one hop this tick.
                llm_incoming = {uid: [] for uid in uav_ids}
                for uav in uavs:
                    uav.tick(
                        step, ground_station, routing_adj.get(uav.uav_id, []),
                        llm_incoming, start_time_ms,
                    )

                # Commit LLM+RL transmissions simultaneously at t+1.
                for uav in uavs:
                    incoming = llm_incoming.get(uav.uav_id, [])
                    if incoming:
                        uav.packets.add_batches(incoming)
                    uav.packets.set_time(end_time_ms)

                # Battery evolves synchronously: each UAV starts from its random initial
                # level and loses exactly 3 percentage points at the end of each completed tick.
                for uav in uavs:
                    uav.update_battery(step, end_time_ms)

                # Pure Q-FANET baseline on the exact same graph snapshot.
                qfanet_router.tick(
                    step, routing_adj, qfanet_ground_station, start_time_ms
                )

                # Now S_(t+1) is complete for LLM+RL: perform TD update and store reward feedback
                # that will be explicitly supplied to the next LLM reasoning cycle.
                for uav in uavs:
                    uav.finalize_transition(routing_adj.get(uav.uav_id, []), end_time_ms)

                # ----- Timer/PDR/deadline traces for thesis plots -----
                ticks_history.append(step)
                time_ms_history.append(end_time_ms)
                time_s_history.append(round(end_time_ms / 1000.0, 3))
                llm_cum_history.append(ground_station.delivered_count)
                qfanet_cum_history.append(qfanet_ground_station.delivered_count)
                generated_so_far = source.total_generated
                generated_cum_history.append(generated_so_far)
                llm_pdr_over_time.append(
                    round(ground_station.delivered_count / generated_so_far * 100, 2)
                    if generated_so_far else 0.0
                )
                qfanet_pdr_over_time.append(
                    round(qfanet_ground_station.delivered_count / generated_so_far * 100, 2)
                    if generated_so_far else 0.0
                )

                llm_snaps = [u.deadline_snapshot(end_time_ms) for u in uavs]
                llm_mins = [x["min_remaining_ms"] for x in llm_snaps if x["min_remaining_ms"] is not None]
                llm_means = [x["mean_remaining_ms"] for x in llm_snaps if x["mean_remaining_ms"] is not None]
                q_snap = qfanet_router.deadline_snapshot(end_time_ms)
                llm_deadline_min.append(round(min(llm_mins), 1) if llm_mins else None)
                llm_deadline_mean.append(
                    round(sum(llm_means) / len(llm_means), 1) if llm_means else None
                )
                qfanet_deadline_min.append(
                    round(q_snap["min_remaining_ms"], 1) if q_snap["min_remaining_ms"] is not None else None
                )
                qfanet_deadline_mean.append(
                    round(q_snap["mean_remaining_ms"], 1) if q_snap["mean_remaining_ms"] is not None else None
                )
                llm_pending_history.append(sum(x["pending_packets"] for x in llm_snaps))
                qfanet_pending_history.append(q_snap["pending_packets"])
                llm_dropped_history.append(sum(u.dropped_deadline for u in uavs))
                qfanet_dropped_history.append(qfanet_router.dropped_deadline)
                for u in uavs:
                    llm_battery_history[u.uav_id].append(round(u.energy, 2))
                for uid in uav_ids:
                    qfanet_battery_history[uid].append(
                        round(qfanet_router.batteries[uid].level_pct, 2)
                    )

                if progress_callback:
                    try:
                        progress_callback(step, steps)
                    except Exception:
                        pass
                if verbose:
                    time.sleep(0.3)
        finally:
            if persist_q_table:
                save_q_tables({uav.uav_id: uav.rl.q for uav in uavs})
                save_q_tables(qfanet_router.q, QFANET_Q_TABLE_FILENAME)

        total_generated = source.total_generated
        llm_dropped = sum(u.dropped_deadline for u in uavs)
        pdr_llm = (ground_station.delivered_count / total_generated * 100) if total_generated else 0.0
        pdr_qfanet = (qfanet_ground_station.delivered_count / total_generated * 100) if total_generated else 0.0

        def _mean_delivery_metric(records: list, key: str) -> float:
            values = [
                float(r[key])
                for r in records
                if r.get(key) is not None
            ]
            return (sum(values) / len(values)) if values else 0.0

        llm_mean_delay_ms = _mean_delivery_metric(
            ground_station.delivery_records,
            "latency_ms",
        )
        qfanet_mean_delay_ms = _mean_delivery_metric(
            qfanet_ground_station.delivery_records,
            "latency_ms",
        )
        llm_mean_hops = _mean_delivery_metric(
            ground_station.delivery_records,
            "hops",
        )
        qfanet_mean_hops = _mean_delivery_metric(
            qfanet_ground_station.delivery_records,
            "hops",
        )

        print("\n===== Final comparison =====")
        print(f"{'Metric':<32}{'LLM+RL':>15}{'Q-FANET':>15}")
        print(f"{'Generated DATA packets':<32}{total_generated:>15}{total_generated:>15}")
        print(f"{'Delivered DATA packets':<32}{ground_station.delivered_count:>15}{qfanet_ground_station.delivered_count:>15}")
        print(f"{'PDR (%)':<32}{pdr_llm:>14.1f}%{pdr_qfanet:>14.1f}%")
        print(f"{'Deadline drops':<32}{llm_dropped:>15}{qfanet_router.dropped_deadline:>15}")
        print(f"{'Mean E2E delay (ms)':<32}{llm_mean_delay_ms:>15.1f}{qfanet_mean_delay_ms:>15.1f}")
        print(f"{'Mean delivered hops':<32}{llm_mean_hops:>15.2f}{qfanet_mean_hops:>15.2f}")
        print("Deadline semantics, one-hop-per-tick timing, drain phase and physical link range are identical for both systems.")

        uav_backend = {uav.uav_id: uav.llm_core.backend for uav in uavs}
        backend_delivered = defaultdict(int)
        for _step, uav_id, count in ground_station.delivery_log:
            backend_delivered[uav_backend.get(uav_id, "unknown")] += count

        llm_drop_events = [event for u in uavs for event in u.deadline_drop_log]

        # Explicit per-packet location/state tracking for thesis analysis.
        llm_packet_tracking = _build_packet_tracking(
            source, uavs, ground_station, qfanet_router, system="llm_rl"
        )
        qfanet_packet_tracking = _build_packet_tracking(
            source, uavs, qfanet_ground_station, qfanet_router, system="qfanet"
        )
        llm_packet_state_by_step = _build_packet_state_by_step(llm_packet_tracking, steps)
        qfanet_packet_state_by_step = _build_packet_state_by_step(qfanet_packet_tracking, steps)

        results = {
            "session_id": session_id,
            "ticks": ticks_history,
            "time_ms": time_ms_history,
            "time_s": time_s_history,
            "llm_cumulative_delivered": llm_cum_history,
            "qfanet_cumulative_delivered": qfanet_cum_history,
            "generated_cumulative": generated_cum_history,
            "traffic_ticks": traffic_ticks,
            "drain_ticks": drain_ticks,
            "phase_by_tick": ["traffic" if t <= traffic_ticks else "drain" for t in ticks_history],
            "llm_pdr_over_time": llm_pdr_over_time,
            "qfanet_pdr_over_time": qfanet_pdr_over_time,
            "reward_trace": {u.uav_id: u.reward_log for u in uavs},
            "reward_design": {
                "formula": "QMR-inspired: destination=r_max; local_minimum=r_min; otherwise beta*(omega*exp(-delay_s)+(1-omega)*energy_norm)+(1-beta)*rssi_norm; final RL reward adds bounded LLM shaping",
                "initial_q_value": INITIAL_Q_VALUE,
                "r_max": QMR_R_MAX,
                "r_min": QMR_R_MIN,
                "omega": QMR_OMEGA,
                "beta": QMR_BETA,
                "rssi_normalization_dbm": [RSSI_MIN_DBM, RSSI_MAX_DBM],
                "llm_shaping_bounds": [LLM_SHAPING_MIN, LLM_SHAPING_MAX],
            },
            "traffic_design": {
                "packet_rate_pps": packet_rate_pps,
                "data_interval_ms": data_interval_ms,
                "data_frequency_pps": packet_rate_pps,
                "relay_capacity_pps": relay_capacity_pps,
                "offered_load_ratio": round(
                    packet_rate_pps / relay_capacity_pps,
                    6,
                ),
                "data_packet_size_bytes": packet_size_bytes,
                "hello_interval_ms": hello_interval_ms,
                "hello_frequency_hz_per_uav": round(1000.0 / hello_interval_ms, 6),
                "hello_packet_size_bytes": hello_packet_size_bytes,
                "hello_expire_time_ms": HELLO_EXPIRE_TIME_MS,
                "hello_generated_total": sum(u.hello_tx_count for u in uavs),
                "hello_generated_by_uav": {u.uav_id: u.hello_tx_count for u in uavs},
                "hello_logs": {u.uav_id: u.hello_log for u in uavs},
            },
            "battery_trace": {
                "initial_pct_by_uav": initial_batteries,
                "initial_min_pct": BATTERY_INITIAL_MIN_PCT,
                "initial_max_pct": BATTERY_INITIAL_MAX_PCT,
                "drain_per_tick_pct": BATTERY_DRAIN_PER_TICK_PCT,
                "llm_rl": llm_battery_history,
                "qfanet": qfanet_battery_history,
            },
            "deadline_trace": {
                "llm_min_remaining_ms": llm_deadline_min,
                "llm_mean_remaining_ms": llm_deadline_mean,
                "qfanet_min_remaining_ms": qfanet_deadline_min,
                "qfanet_mean_remaining_ms": qfanet_deadline_mean,
                "llm_pending_packets": llm_pending_history,
                "qfanet_pending_packets": qfanet_pending_history,
                "llm_dropped_cumulative": llm_dropped_history,
                "qfanet_dropped_cumulative": qfanet_dropped_history,
            },
            "packet_deadlines": source.generation_log,
            "packet_tracking": {
                "llm_rl": llm_packet_tracking,
                "qfanet": qfanet_packet_tracking,
                "state_by_step": {
                    "llm_rl": llm_packet_state_by_step,
                    "qfanet": qfanet_packet_state_by_step,
                },
                "status_values": [
                    "CREATED", "WAITING_SOURCE", "QUEUED", "DELIVERED", "DROPPED_DEADLINE"
                ],
                "event_values": [
                    "DATA_GENERATED", "WAITING_FOR_ENTRY_UAV", "INJECTED_FROM_SOURCE",
                    "FORWARDED", "HELD", "HELD_NO_PATH", "HELD_NO_ENERGY",
                    "DELIVERED_TO_GS", "DEADLINE_EXPIRED", "SOURCE_DEADLINE_EXPIRED"
                ],
            },
            "llm_delivery_records": ground_station.delivery_records,
            "qfanet_delivery_records": qfanet_ground_station.delivery_records,
            "qfanet_decision_log": qfanet_router.decision_log,
            "deadline_drop_events": {
                "llm": llm_drop_events,
                "qfanet": qfanet_router.deadline_drop_log,
            },
            "summary": {
                "total_generated": total_generated,
                "total_generated_bytes": source.total_generated_bytes,
                "source_expired": source.source_expired,
                "simulation_duration_ms": steps * TICK_DURATION_MS,
                "simulation_duration_s": round(steps * SIM_TICK_SECONDS, 3),
                "llm": {
                    "delivered": ground_station.delivered_count,
                    "delivered_bytes": ground_station.delivered_bytes,
                    "pdr_pct": round(pdr_llm, 2),
                    "dropped_deadline": llm_dropped,
                    "mean_e2e_delay_ms": round(llm_mean_delay_ms, 3),
                    "mean_delivered_hops": round(llm_mean_hops, 3),
                },
                "qfanet": {
                    "delivered": qfanet_ground_station.delivered_count,
                    "delivered_bytes": qfanet_ground_station.delivered_bytes,
                    "pdr_pct": round(pdr_qfanet, 2),
                    "dropped_deadline": qfanet_router.dropped_deadline,
                    "mean_e2e_delay_ms": round(qfanet_mean_delay_ms, 3),
                    "mean_delivered_hops": round(qfanet_mean_hops, 3),
                },
                "llm_backend_split": uav_backend,
                "llm_backend_delivered": dict(backend_delivered),
                "params": {
                    "num_uavs": num_uavs,
                    "steps": steps,
                    "mean_speed_mps": mean_speed_mps,
                    "traffic_ticks": traffic_ticks, "drain_ticks": drain_ticks,
                    "traffic_duration_s": round(traffic_ticks * SIM_TICK_SECONDS, 3),
                    "drain_duration_s": round(drain_ticks * SIM_TICK_SECONDS, 3),
                    "area_m": [AREA[1] - AREA[0], AREA[3] - AREA[2]],
                    "comm_range_m": COMM_RANGE,
                    "tick_duration_ms": TICK_DURATION_MS,
                    "battery_initial_min_pct": BATTERY_INITIAL_MIN_PCT,
                    "battery_initial_max_pct": BATTERY_INITIAL_MAX_PCT,
                    "battery_initial_pct_by_uav": initial_batteries,
                    "battery_drain_per_tick_pct": BATTERY_DRAIN_PER_TICK_PCT,
                    "deadline_choices_ms": list(DEADLINE_CHOICES_MS),
                    "packet_rate_pps": packet_rate_pps,
                    "data_interval_ms": data_interval_ms,
                    "relay_capacity_pps": relay_capacity_pps,
                    "offered_load_ratio": round(
                        packet_rate_pps / relay_capacity_pps,
                        6,
                    ),
                    "data_packet_size_bytes": packet_size_bytes,
                    "hello_interval_ms": hello_interval_ms,
                    "hello_packet_size_bytes": hello_packet_size_bytes,
                    "hello_expire_time_ms": HELLO_EXPIRE_TIME_MS,
                    "initial_q_value": INITIAL_Q_VALUE,
                    "qmr_omega": QMR_OMEGA,
                    "qmr_beta": QMR_BETA,
                    "epsilon_start": EPSILON_START,
                    "epsilon_min": EPSILON_MIN,
                    "epsilon_decay_per_tick": EPSILON_DECAY_PER_TICK,
                    "qfanet_q_tie_tolerance": QFANET_Q_TIE_TOLERANCE,
                    "qfanet_hop_penalty": QFANET_HOP_PENALTY,
                    "qfanet_urgent_deadline_ms": QFANET_URGENT_DEADLINE_MS,
                    "llmrl_q_min": LLMRL_Q_MIN,
                    "llmrl_q_max": LLMRL_Q_MAX,
                    "llmrl_hold_penalty": LLMRL_HOLD_PENALTY,
                    "llmrl_urgent_deadline_ms": LLMRL_URGENT_DEADLINE_MS,
                    "llmrl_soon_deadline_ms": LLMRL_SOON_DEADLINE_MS,
                    "llmrl_forward_bonus_floor": LLMRL_FORWARD_BONUS_FLOOR,
                    "llmrl_urgent_forward_bonus_floor": LLMRL_URGENT_FORWARD_BONUS_FLOOR,
                    "llmrl_architecture": "neighbor_aware_q_learning_with_llm_stm_ltm",
                    "llmrl_q_action_weight": LLMRL_Q_ACTION_WEIGHT,
                    "llmrl_llm_neighbor_weight": LLMRL_LLM_NEIGHBOR_WEIGHT,
                    "llmrl_heuristic_weight": LLMRL_HEURISTIC_WEIGHT,
                    "llmrl_progress_epsilon_cap": LLMRL_PROGRESS_EPSILON_CAP,
                    "llmrl_progress_reward_weight": LLMRL_PROGRESS_REWARD_WEIGHT,
                    "llmrl_hop_cost": LLMRL_HOP_COST,
                    "llmrl_non_progress_penalty": LLMRL_NON_PROGRESS_PENALTY,
                    "ltm_similarity_weight": LTM_SIMILARITY_WEIGHT,
                    "ltm_quality_weight": LTM_QUALITY_WEIGHT,
                    "lambda_rate_per_s": LAMBDA_RATE, "mu_rate_per_s": MU_RATE,
                },
            },
        }

        if export_json:
            requested_path = (
                output_json
                if os.path.isabs(output_json)
                else os.path.join(_base_dir(), output_json)
            )

            out_path = save_json_utf8_atomic(
                results,
                requested_path,
            )

            print(
                f"\n[Export] UTF-8 JSON saved atomically to: {out_path}"
            )
            print(
                "[Export] Validate with: "
                f'python -m json.tool "{out_path}"'
            )

        return results
    finally:
        if not verbose:
            builtins.print = _real_print


# Génère automatiquement la figure PDR-vs-time après la simulation.

def generate_pdr_vs_time_figure(json_path):
    """
    Generate charts/pdr_vs_time.png DIRECTLY from comparison_results.json.

    X = time_s
    Y = PDR (%)

    Curves:
        - llm_pdr_over_time
        - qfanet_pdr_over_time
    """

    import json
    from pathlib import Path
    import matplotlib.pyplot as plt

    base_dir = Path(__file__).resolve().parent

    json_path = Path(json_path)

    if not json_path.is_absolute():
        json_path = base_dir / json_path

    # ------------------------------------------------------------
    # READ THE SAME JSON USED BY THE DASHBOARD
    # ------------------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # ------------------------------------------------------------
    # EXACT SAME DATA AS DASHBOARD PDR vs TIME
    # ------------------------------------------------------------

    time_s = data.get(
        "time_s",
        [],
    )

    llm_pdr = data.get(
        "llm_pdr_over_time",
        [],
    )

    qfanet_pdr = data.get(
        "qfanet_pdr_over_time",
        [],
    )

    if not time_s:
        raise ValueError(
            "comparison_results.json: time_s is empty."
        )

    if not llm_pdr:
        raise ValueError(
            "comparison_results.json: "
            "llm_pdr_over_time is empty."
        )

    if not qfanet_pdr:
        raise ValueError(
            "comparison_results.json: "
            "qfanet_pdr_over_time is empty."
        )

    # ------------------------------------------------------------
    # VERIFY LENGTHS
    # ------------------------------------------------------------

    if not (
        len(time_s)
        == len(llm_pdr)
        == len(qfanet_pdr)
    ):
        raise ValueError(
            "PDR arrays do not have the same length: "
            f"time={len(time_s)}, "
            f"LLM={len(llm_pdr)}, "
            f"Q-FANET={len(qfanet_pdr)}"
        )

    # ------------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ------------------------------------------------------------

    charts_dir = (
        base_dir
        / "charts"
    )

    charts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        charts_dir
        / "pdr_vs_time.png"
    )

    # ------------------------------------------------------------
    # CREATE FIGURE
    # ------------------------------------------------------------

    plt.figure(
        figsize=(12, 7),
    )

    plt.plot(
        time_s,
        llm_pdr,
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="LLM + RL",
    )

    plt.plot(
        time_s,
        qfanet_pdr,
        marker="s",
        linestyle="--",
        linewidth=2.5,
        markersize=6,
        label="Q-FANET",
    )

    # ------------------------------------------------------------
    # SAME AXES AS DASHBOARD
    # ------------------------------------------------------------

    plt.xlabel(
        "Simulation Time (s)",
        fontsize=13,
    )

    plt.ylabel(
        "PDR (%)",
        fontsize=13,
    )

    plt.title(
        "PDR vs Simulation Time",
        fontsize=16,
    )

    plt.ylim(
        0,
        100,
    )

    plt.xlim(
        min(time_s),
        max(time_s),
    )

    plt.grid(
        True,
        alpha=0.30,
    )

    plt.legend(
        fontsize=12,
    )

    plt.tight_layout()

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------------
    # DEBUG / VERIFICATION
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("[PDR PNG] SOURCE = comparison_results.json")
    print("=" * 60)

    print(
        f"JSON: {json_path}"
    )

    print(
        f"Points: {len(time_s)}"
    )

    print(
        f"Final time: {time_s[-1]:.2f} s"
    )

    print(
        f"Final LLM+RL PDR: "
        f"{llm_pdr[-1]:.2f}%"
    )

    print(
        f"Final Q-FANET PDR: "
        f"{qfanet_pdr[-1]:.2f}%"
    )

    print(
        f"PNG: {output_path}"
    )

    print("=" * 60)

    return str(output_path)

# Déclare et valide les paramètres disponibles depuis CMD.
#
# Exemple :
# python main.py --uavs 25 --steps 20 --packet-rate-pps 10 ...

def parse_cli_args():
    """Parse launch parameters for one FANET simulation run.

    Examples:
        python main.py --uavs 6
        python main.py --uavs 10 --steps 40
        python main.py -n 8 -s 32 --backend groq
    """
    parser = argparse.ArgumentParser(
        description=(
            "FANET LLM+RL vs Q-FANET simulation. "
            "Choose the UAV fleet size at launch time."
        )
    )
    parser.add_argument(
        "-n", "--uavs",
        dest="num_uavs",
        type=int,
        required=True,
        help="number of mobile UAV relay nodes, e.g. --uavs 8",
    )
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default=20,
        help="total number of simulation ticks (default: 20; 1 tick = 0.25 s)",
    )
    parser.add_argument(
        "--backend",
        choices=["mixed", "groq", "ollama"],
        default="ollama",
        help="LLM backend assignment (default: ollama)",
    )
    traffic_group = parser.add_mutually_exclusive_group()

    traffic_group.add_argument(
        "--packet-rate-pps",
        type=float,
        default=None,
        help=(
            "DATA generation rate at Source in packets/s. "
            "Examples: --packet-rate-pps 10 or --packet-rate-pps 100. "
            f"Default: {DEFAULT_PACKET_RATE_PPS:g} packet/s"
        ),
    )

    traffic_group.add_argument(
        "--data-interval-ms",
        type=float,
        default=None,
        help=(
            "Backward-compatible alternative to --packet-rate-pps. "
            "Example: 100 ms = 10 packet/s."
        ),
    )

    parser.add_argument(
        "--relay-capacity-pps",
        type=float,
        default=DEFAULT_RELAY_CAPACITY_PPS,
        help=(
            "maximum DATA forwarding/service rate of each UAV in packets/s "
            f"(default: {DEFAULT_RELAY_CAPACITY_PPS:g}). "
            "This finite capacity creates realistic queueing under high load."
        ),
    )
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=DEFAULT_MEAN_SPEED_MPS,
        help=(
            "mean UAV speed for the Gauss-Markov mobility model in m/s "
            f"(default: {DEFAULT_MEAN_SPEED_MPS:g})"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="comparison_results.json",
        help=(
            "UTF-8 JSON output filename. Use this option instead of shell "
            "redirection (> file.json). Example: --output-json speed_25.json"
        ),
    )
    parser.add_argument(
        "--fresh-learning",
        action="store_true",
        help=(
            "ignore persisted Q-tables for this run. "
            "Useful for fair independent evaluation runs."
        ),
    )
    parser.add_argument(
        "--packet-size-bytes", type=int, default=DEFAULT_DATA_PACKET_SIZE_BYTES,
        help="DATA packet size in bytes (default: 127)",
    )
    parser.add_argument(
        "--hello-interval-ms", type=float, default=DEFAULT_HELLO_INTERVAL_MS,
        help="HELLO interval per UAV in ms (default: 100)",
    )
    parser.add_argument(
        "--hello-size-bytes", type=int, default=DEFAULT_HELLO_PACKET_SIZE_BYTES,
        help="HELLO packet size in bytes (default: 32, simulation assumption)",
    )
    args = parser.parse_args()

    if args.num_uavs < 1:
        parser.error("--uavs must be >= 1")
    if args.steps < 2:
        parser.error("--steps must be >= 2")

    # ------------------------------------------------------------
    # Normalize traffic input.
    # Primary interface: packets per second.
    # Backward-compatible interface: interval in milliseconds.
    # ------------------------------------------------------------
    if args.packet_rate_pps is None and args.data_interval_ms is None:
        args.packet_rate_pps = DEFAULT_PACKET_RATE_PPS
        args.data_interval_ms = (
            1000.0 / args.packet_rate_pps
        )

    elif args.packet_rate_pps is not None:
        if args.packet_rate_pps <= 0:
            parser.error(
                "--packet-rate-pps must be > 0"
            )

        args.data_interval_ms = (
            1000.0 / args.packet_rate_pps
        )

    else:
        if args.data_interval_ms <= 0:
            parser.error(
                "--data-interval-ms must be > 0"
            )

        args.packet_rate_pps = (
            1000.0 / args.data_interval_ms
        )

    if args.relay_capacity_pps <= 0:
        parser.error(
            "--relay-capacity-pps must be > 0"
        )

    if args.speed_mps < 0:
        parser.error(
            "--speed-mps must be >= 0"
        )

    if args.packet_size_bytes <= 0:
        parser.error("--packet-size-bytes must be > 0")
    if args.hello_interval_ms <= 0:
        parser.error("--hello-interval-ms must be > 0")
    if args.hello_size_bytes <= 0:
        parser.error("--hello-size-bytes must be > 0")

    return args


# =====================================================================
# POINT D'ENTRÉE CMD
# =====================================================================
# Cette section n'est exécutée que lorsque le fichier est lancé directement.
#
# Exemple demandé :
#
#   python main.py --uavs 25 --steps 20 --packet-rate-pps 10 \
#       --packet-size-bytes 128 --hello-interval-ms 100 \
#       --hello-size-bytes 32
#
# Le backend par défaut reste celui défini dans parse_cli_args().
# =====================================================================

if __name__ == "__main__":
    args = parse_cli_args()
    selected_backend = None if args.backend == "mixed" else args.backend

    print(
        f"[Launch] UAVs={args.num_uavs}, steps={args.steps}, "
        f"simulated_duration={args.steps * SIM_TICK_SECONDS:.2f} s, "
        f"backend={args.backend}, "
        f"DATA={args.packet_rate_pps:.2f}pps/"
        f"{args.packet_size_bytes}B, "
        f"relay_capacity={args.relay_capacity_pps:.2f}pps/UAV, "
        f"speed={args.speed_mps:.2f}m/s, "
        f"HELLO={args.hello_interval_ms:.0f}ms"
    )

  simulation_results = run_simulation(
    steps=args.steps,
    num_uavs=args.num_uavs,
    force_backend=selected_backend,
    packet_rate_pps=args.packet_rate_pps,
    relay_capacity_pps=args.relay_capacity_pps,
    mean_speed_mps=args.speed_mps,
    packet_size_bytes=args.packet_size_bytes,
    hello_interval_ms=args.hello_interval_ms,
    hello_packet_size_bytes=args.hello_size_bytes,
    output_json=args.output_json,
    persist_q_table=not args.fresh_learning,
)

# IMPORTANT:
# Generate the PNG from the exact JSON file used by Dashboard.html.
generate_pdr_vs_time_figure(
    args.output_json
)
