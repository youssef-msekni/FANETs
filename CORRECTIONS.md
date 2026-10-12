# Corrections applied

This corrected version changes the simulator as follows.

1. Environment changed to 500 m × 500 m.
2. Hard communication range set to 180 m using 3-D Euclidean distance.
3. Physical range is checked before stochastic link activation/deactivation.
4. Packet deadlines are now per `PacketBatch` with absolute end-to-end expiry times.
5. Default packet deadlines are 1000/1500/2000/2500/3000 ms.
6. LLM+RL and Q-FANET now drop expired packets using the same deadline rule.
7. LLM+RL and Q-FANET maintain independent battery states.
8. Packets can traverse at most one hop per 250 ms tick; transmissions are committed synchronously.
9. A UAV below `MIN_OPERATIONAL_ENERGY` cannot transmit.
10. LLM action bonuses, routing weights and dynamic reward parameters are numerically clamped.
11. The reward from transition t is saved and explicitly inserted into the next LLM prompt together with the updated state.
12. Q-learning is updated only after the synchronized updated state S_(t+1) is available.
13. Simulation time is exported in milliseconds and seconds.
14. Deadline traces, generated packet deadlines, delivery latency, deadline margin and deadline-drop events are exported.
15. `plot_comparison.py` now generates `pdr_vs_time.png`, `deadline_vs_time.png`, and `deadline_drops_vs_time.png`.
16. LTM uses LangSmith persistence plus FAISS/SentenceTransformer semantic retrieval when available, with a lexical fallback.
17. Missing `langchain-groq` / `langchain-ollama` dependencies were added.
18. `.env` was sanitized and `.gitignore` added. Add your own keys locally.
19. Old Q-tables were reset because they were learned under the previous environment/reward/deadline semantics.
20. Initial UAV placement uses a lightly jittered source-to-GS relay corridor so a 6-UAV/180 m experiment is not trivially disconnected at startup.

## PDR versus time

Run:

```bash
python main.py
python plot_comparison.py
```

Then use:

```text
charts/pdr_vs_time.png
```

The underlying JSON arrays are:

```text
time_s
llm_pdr_over_time
qfanet_pdr_over_time
```

For deadline analysis, use `deadline_trace`, `packet_deadlines`, `llm_delivery_records`, `qfanet_delivery_records`, and `deadline_drop_events` in `comparison_results.json`.
