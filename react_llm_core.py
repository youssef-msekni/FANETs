"""
react_llm_core.py
======================================================================
ReAct LLM Core for the FANET LLM+RL router (matches the architecture diagram).

Role of the LLM (it NEVER picks the routing action):
    - it runs a ReAct loop:  Thought (step-by-step CoT) -> Action (read-only tool)
      -> Observation -> Thought -> ... -> finish
    - "finish" does not choose a next hop. It returns GUIDANCE only:
        * bounded REWARD-SHAPING bonuses (one per candidate + HOLD)
        * soft candidate scores / routing weights
        * a short rationale
    - the RL agent (Q-learning) takes the FINAL action.

The tools are read-only: they only read the context built by main.py
(telemetry, candidates, queues, STM, LTM). They cannot change the network.

This module is standalone (no import from main.py) to avoid circular imports.
"""

import json
import math
import os
import re

HOLD_ACTION = "HOLD"
MAX_REACT_STEPS = int(os.getenv("REACT_MAX_STEPS", "4"))

DEFAULT_WEIGHTS = {
    "progress": 0.40,
    "stability": 0.20,
    "energy": 0.10,
    "deadline": 0.15,
    "congestion": 0.15,
}


def _clamp(value, lo, hi, default=0.0):
    try:
        value = float(value)
    except Exception:
        value = default
    if not math.isfinite(value):
        value = default
    return max(lo, min(hi, value))


def _short(text, n):
    text = str(text)
    return text if len(text) <= n else text[: n - 3] + "..."


# ----------------------------------------------------------------------
# Read-only tools
# ----------------------------------------------------------------------
class _Toolbox:
    """Read-only observation tools exposed to the LLM."""

    DESCRIPTIONS = (
        '- get_battery  args {"id": "UAV-x" | "self"}      -> battery % of a candidate or of this UAV\n'
        '- get_distance args {"id": "UAV-x" | "GS" | "self"} -> link distance and distance-to-GS (m)\n'
        '- get_link_quality args {"id": "UAV-x" | "GS"}     -> RSSI, radio score, link stability\n'
        '- get_queue_load args {"id": "UAV-x" | "GS"}       -> neighbor queue size and congestion score\n'
        '- get_deadline_status args {}                       -> pending packets, min remaining deadline, urgency\n'
        '- read_stm args {"k": 3}                            -> last k transitions of THIS session\n'
        '- search_ltm args {"query": "text"}                 -> similar historical experiences (quality-ranked)\n'
        '- finish args {...guidance...}                      -> end the loop and return guidance'
    )

    def __init__(self, telemetry, packets, stm, ltm, candidates):
        self.t = telemetry
        self.packets = packets
        self.stm = stm
        self.ltm = ltm
        self.cand = {str(c["id"]): c for c in candidates}

    def names(self):
        return [
            "get_battery", "get_distance", "get_link_quality",
            "get_queue_load", "get_deadline_status", "read_stm", "search_ltm",
        ]

    def _candidate(self, args):
        cid = str(args.get("id", "")).strip()
        if cid not in self.cand:
            return None, {
                "error": f"unknown id '{cid}'",
                "valid_ids": list(self.cand.keys()),
            }
        return self.cand[cid], None

    def call(self, name, args):
        args = args if isinstance(args, dict) else {}
        try:
            if name == "get_battery":
                if str(args.get("id", "self")).lower() in ("self", ""):
                    return {"id": "self", "battery_pct": self.t.energy}
                c, err = self._candidate(args)
                return err or {"id": c["id"], "battery_pct": c["energy_pct"]}

            if name == "get_distance":
                if str(args.get("id", "self")).lower() in ("self", ""):
                    return {"id": "self", "distance_to_gs_m": self.t.distance_to_gs}
                c, err = self._candidate(args)
                return err or {
                    "id": c["id"],
                    "link_distance_m": c["link_distance_m"],
                    "candidate_distance_to_gs_m": c["distance_to_gs_m"],
                    "progress_toward_gs_m": c["progress_m"],
                }

            if name == "get_link_quality":
                c, err = self._candidate(args)
                return err or {
                    "id": c["id"],
                    "rssi_dbm": c["rssi_dbm"],
                    "radio_score": c["radio_score"],
                    "stability_score": c["stability_score"],
                }

            if name == "get_queue_load":
                c, err = self._candidate(args)
                return err or {
                    "id": c["id"],
                    "neighbor_queue_packets": c.get("neighbor_queue_packets", 0),
                    "congestion_score": c.get("congestion_score", 1.0),
                }

            if name == "get_deadline_status":
                dl = self.packets.min_deadline_ms
                if not math.isfinite(dl):
                    label = "none"
                elif dl <= 500:
                    label = "critical"
                elif dl <= 1000:
                    label = "urgent"
                else:
                    label = "comfortable"
                return {
                    "pending_packets": self.packets.pending_packets,
                    "min_remaining_deadline_ms": (
                        round(dl, 1) if math.isfinite(dl) else None
                    ),
                    "urgency": label,
                }

            if name == "read_stm":
                k = int(_clamp(args.get("k", 3), 1, 5, 3))
                items = list(self.stm.buffer)[-k:]
                return {
                    "transitions": [
                        {
                            "step": e.get("step"),
                            "action": e.get("action"),
                            "raw_reward": round(float(e.get("raw_reward", 0.0)), 3),
                            "final_reward": round(float(e.get("reward", 0.0)), 3),
                            "next_state": e.get("next_state", "pending"),
                            "deadline_drops": e.get("deadline_drops", 0),
                        }
                        for e in items
                    ]
                }

            if name == "search_ltm":
                query = str(args.get("query", "")) or (
                    f"energy={self.t.energy} dist_gs={self.t.distance_to_gs} "
                    f"pending={self.packets.pending_packets}"
                )
                found = self.ltm.retrieve(query, k=2)
                return {"experiences": [_short(x, 320) for x in found]}

        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        return {"error": f"unknown tool '{name}'", "available_tools": self.names() + ["finish"]}


# ----------------------------------------------------------------------
# ReAct guidance core
# ----------------------------------------------------------------------
class ReActGuidanceCore:
    """ReAct loop with step-by-step CoT. Returns guidance, never an action."""

    def __init__(self, uav_id, llm, backend="ollama",
                 shaping_min=-0.5, shaping_max=0.5,
                 max_steps=None, min_num_predict=450):
        self.uav_id = uav_id
        self.llm = llm
        self.backend = backend
        self.shaping_min = float(shaping_min)
        self.shaping_max = float(shaping_max)
        self.max_steps = int(max_steps or MAX_REACT_STEPS)

        # Thoughts need more tokens than the old 160-token JSON-only reply.
        if hasattr(self.llm, "num_predict"):
            try:
                self.llm.num_predict = max(int(self.llm.num_predict), min_num_predict)
            except Exception:
                pass

        self.system_prompt = f"""
You are the ReAct reasoning core for UAV {uav_id} in a deadline-constrained FANET.

YOU DO NOT CHOOSE THE ROUTING ACTION. A Q-learning agent takes the final action.
Your job: OBSERVE with tools, THINK step by step, then return GUIDANCE:
  - reward-shaping bonuses (one per candidate and one for HOLD),
  - soft candidate scores, routing weights, short rationale.

LOOP FORMAT. Every reply is ONE JSON object and nothing else:
{{"thought": "...", "action": "<tool name>", "args": {{...}}}}

THOUGHT RULES (chain-of-thought, step by step, numbered):
  1) Situation: what do I already know?
  2) Unknown: what is missing to judge the candidates?
  3) Plan: which single tool resolves it?
  After an Observation: 4) Update: what did I learn? 5) Next: another tool, or finish?
Keep each thought under ~120 words.

TOOLS (read-only, they cannot change the network):
{_Toolbox.DESCRIPTIONS}

RULES:
- You MUST call at least one observation tool before "finish".
- Use only candidate ids that were listed. Never invent data.
- Direct GS is the best candidate when listed. HOLD is bad when a progress-making candidate exists.
- STM = this session; LTM = history (may be outdated; failures are warnings, not examples).
- Shaping bonuses are small nudges in [{self.shaping_min}, {self.shaping_max}]:
  positive = the action looks good for on-time delivery, negative = it looks risky/wasteful.
- You have at most {self.max_steps} replies. Do not waste them.

"finish" ARGS FORMAT:
{{"rationale": "short why",
  "neighbor_scores": {{"UAV-x": 0.0}},
  "hold_score": 0.0,
  "shaping": {{"UAV-x": 0.0}},
  "hold_shaping": 0.0,
  "weights": {{"progress": 0.4, "stability": 0.2, "energy": 0.1, "deadline": 0.15, "congestion": 0.15}}}}
""".strip()

    # ------------------------------------------------------------------
    def _initial_prompt(self, telemetry, packets, candidates, previous_feedback, now_ms):
        dl = packets.min_deadline_ms
        dl_text = f"{dl:.0f} ms" if math.isfinite(dl) else "none"
        brief = [
            {
                "id": c["id"],
                "is_gs": c["is_gs"],
                "progress_m": c["progress_m"],
            }
            for c in candidates
        ]
        if previous_feedback:
            fb = {
                "action": previous_feedback.get("action"),
                "raw_reward": round(float(previous_feedback.get("raw_reward", 0.0)), 3),
                "llm_shaping_bonus": round(float(previous_feedback.get("llm_bonus", 0.0)), 3),
                "final_reward": round(float(previous_feedback.get("shaped_reward", 0.0)), 3),
                "deadline_drops": previous_feedback.get("deadline_drops", 0),
                "next_state": previous_feedback.get("next_state"),
            }
        else:
            fb = "none (first cycle)"

        return (
            f"Time: {now_ms / 1000.0:.3f} s | UAV: {self.uav_id}\n"
            f"Own battery: {telemetry.energy}% | distance to GS: {telemetry.distance_to_gs} m | "
            f"connectivity: {telemetry.connectivity}\n"
            f"Pending packets: {packets.pending_packets} | min remaining deadline: {dl_text}\n"
            f"Candidates (brief; details only through tools): {json.dumps(brief)}\n"
            f"Previous transition feedback: {json.dumps(fb)}\n\n"
            "Start with a numbered thought, then call ONE observation tool."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_step(text):
        try:
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", str(text)).strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            data = json.loads(match.group(0) if match else cleaned)
            thought = _short(data.get("thought", ""), 900)
            action = str(data.get("action", "")).strip().lower()
            args = data.get("args")
            if not isinstance(args, dict):
                args = {k: v for k, v in data.items() if k not in ("thought", "action", "args")}
            if not action:
                return None
            return thought, action, args
        except Exception:
            return None

    def _build_guidance(self, args, candidate_ids):
        raw_scores = args.get("neighbor_scores", {}) or {}
        raw_shape = args.get("shaping", {}) or {}
        raw_w = args.get("weights", {}) or {}

        scores = {cid: _clamp(raw_scores.get(cid, 0.5), 0.0, 1.0, 0.5) for cid in candidate_ids}
        shaping = {
            cid: _clamp(raw_shape.get(cid, 0.0), self.shaping_min, self.shaping_max, 0.0)
            for cid in candidate_ids
        }
        shaping[HOLD_ACTION] = _clamp(
            args.get("hold_shaping", 0.0), self.shaping_min, self.shaping_max, 0.0
        )

        weights = {k: _clamp(raw_w.get(k, d), 0.0, 1.0, d) for k, d in DEFAULT_WEIGHTS.items()}
        total = sum(weights.values())
        weights = (
            {k: v / total for k, v in weights.items()} if total > 0 else dict(DEFAULT_WEIGHTS)
        )

        return {
            "rationale": _short(args.get("rationale", ""), 500),
            "neighbor_scores": scores,
            "hold_score": _clamp(args.get("hold_score", 0.5), 0.0, 1.0, 0.5),
            "shaping": shaping,
            "weights": weights,
        }

    def _neutral(self, candidate_ids, rationale, trace):
        return {
            "rationale": rationale,
            "neighbor_scores": {cid: 0.5 for cid in candidate_ids},
            "hold_score": 0.5,
            "shaping": {**{cid: 0.0 for cid in candidate_ids}, HOLD_ACTION: 0.0},
            "weights": dict(DEFAULT_WEIGHTS),
            "react_trace": trace,
        }

    # ------------------------------------------------------------------
    def guide(self, telemetry, packets, stm, ltm, candidate_context,
              previous_feedback=None, simulation_time_ms=0.0):
        candidate_ids = [str(c["id"]) for c in candidate_context]
        toolbox = _Toolbox(telemetry, packets, stm, ltm, candidate_context)
        messages = [
            ("system", self.system_prompt),
            ("human", self._initial_prompt(
                telemetry, packets, candidate_context, previous_feedback, simulation_time_ms)),
        ]
        trace = []
        used_tool = False

        for step in range(1, self.max_steps + 1):
            if step == self.max_steps:
                messages.append((
                    "human",
                    'Last allowed reply: use action "finish" now with your guidance.'))

            try:
                reply = self.llm.invoke(messages).content
            except Exception as exc:
                print(f"[ReAct:{self.uav_id}:{self.backend}] LLM failed "
                      f"({type(exc).__name__}: {exc}) -> neutral guidance.")
                return self._neutral(candidate_ids, f"llm_call_failed:{type(exc).__name__}", trace)

            parsed = self._parse_step(reply)
            if parsed is None:
                trace.append({"step": step, "error": "invalid_json"})
                messages.append(("ai", _short(reply, 400)))
                messages.append(("human",
                                 'Invalid reply. Answer with ONE JSON object: '
                                 '{"thought": "...", "action": "...", "args": {...}}'))
                continue

            thought, action, args = parsed
            entry = {"step": step, "thought": thought, "action": action}
            trace.append(entry)
            messages.append(("ai", json.dumps(
                {"thought": thought, "action": action, "args": args}, ensure_ascii=False)))

            if action == "finish":
                if not used_tool and step < self.max_steps:
                    entry["rejected"] = "finish_before_observation"
                    messages.append(("human",
                                     "Rejected: you must observe with at least one tool "
                                     "before finishing. Call a tool now."))
                    continue
                guidance = self._build_guidance(args, candidate_ids)
                guidance["react_trace"] = trace
                return guidance

            observation = toolbox.call(action, args)
            used_tool = used_tool or ("error" not in observation)
            entry["args"] = args
            entry["observation"] = observation
            messages.append(("human", "Observation: " + _short(
                json.dumps(observation, ensure_ascii=False), 1200)))

        return self._neutral(candidate_ids, "react_no_finish_within_budget", trace)