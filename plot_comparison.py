#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comparaison.py

Creates the three sensitivity studies used by Dashboard.html:

1. PDR vs DATA flow
2. PDR vs mean UAV speed
3. PDR vs number of UAVs

Every simulation is launched with --output-json, so the produced files are
real UTF-8 JSON files. No terminal redirection is used.

The script also creates:
    experiment_results.json

Dashboard.html loads that aggregate file automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent

MAIN_PATH = BASE_DIR / "main.py"

# Do not use the old "experiments" directory because its
# directory entry on the USB drive is corrupted.
EXPERIMENT_DIR = BASE_DIR
# Keep the aggregate JSON directly inside myagent so that
# Dashboard.html can load it automatically.
AGGREGATE_PATH = BASE_DIR / "experiment_results.json"

FLOW_VALUES = [10.0, 100.0, 1000.0]
SPEED_VALUES = [5.0, 12.0, 25.0]
NODE_VALUES = [10, 25, 50]

DEFAULT_STEPS = 20
DEFAULT_UAVS = 25
DEFAULT_PACKET_RATE_PPS = 10.0
DEFAULT_RELAY_CAPACITY_PPS = 40.0
DEFAULT_SPEED_MPS = 12.0
DEFAULT_PACKET_SIZE_BYTES = 128
DEFAULT_HELLO_INTERVAL_MS = 100.0
DEFAULT_HELLO_SIZE_BYTES = 32


def write_json_utf8_atomic(data: dict, path: Path) -> None:
    """Write UTF-8 JSON atomically."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open(
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
        tmp_path,
        path,
    )


def read_json_utf8(path: Path) -> dict:
    """Read and validate a UTF-8 JSON result."""
    raw = path.read_bytes()

    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path.name} is not UTF-8 JSON: {exc}. "
            "Regenerate it with --output-json."
        ) from exc

    text = text.strip()

    if not text:
        raise ValueError(
            f"{path.name} is empty."
        )

    if text[0] not in "[{":
        preview = text[:80].replace("\n", " ")
        raise ValueError(
            f"{path.name} is not JSON. First text: {preview!r}. "
            "Do not create JSON using > redirection."
        )

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path.name} contains invalid JSON: {exc}"
        ) from exc


def extract_run_metrics(
    result: dict,
    source_file: Path,
) -> dict:
    summary = result.get("summary", {})
    params = summary.get("params", {})

    llm = summary.get("llm", {})
    qfanet = summary.get("qfanet", {})

    generated = int(
        summary.get("total_generated", 0) or 0
    )

    def pdr(system: dict) -> float:
        if system.get("pdr_pct") is not None:
            return float(system["pdr_pct"])

        delivered = int(
            system.get("delivered", 0) or 0
        )

        if generated <= 0:
            return 0.0

        return (
            100.0
            * delivered
            / generated
        )

    try:
        relative_file = str(
            source_file.resolve().relative_to(
                BASE_DIR.resolve()
            )
        )
    except ValueError:
        relative_file = str(
            source_file.resolve()
        )

    return {
        "file": relative_file,
        "session_id": result.get("session_id"),
        "num_uavs": int(
            params.get("num_uavs", 0) or 0
        ),
        "packet_rate_pps": float(
            params.get(
                "packet_rate_pps",
                params.get("data_frequency_pps", 0.0),
            ) or 0.0
        ),
        "mean_speed_mps": float(
            params.get("mean_speed_mps", 0.0) or 0.0
        ),
        "relay_capacity_pps": float(
            params.get("relay_capacity_pps", 0.0) or 0.0
        ),
        "total_generated": generated,
        "llm_pdr_pct": pdr(llm),
        "qfanet_pdr_pct": pdr(qfanet),
        "llm_delivered": int(
            llm.get("delivered", 0) or 0
        ),
        "qfanet_delivered": int(
            qfanet.get("delivered", 0) or 0
        ),
        "llm_deadline_drops": int(
            llm.get("deadline_drops", 0) or 0
        ),
        "qfanet_deadline_drops": int(
            qfanet.get("deadline_drops", 0) or 0
        ),
        "llm_mean_e2e_delay_ms": llm.get(
            "mean_e2e_delay_ms"
        ),
        "qfanet_mean_e2e_delay_ms": qfanet.get(
            "mean_e2e_delay_ms"
        ),
        "llm_mean_delivered_hops": llm.get(
            "mean_delivered_hops"
        ),
        "qfanet_mean_delivered_hops": qfanet.get(
            "mean_delivered_hops"
        ),
    }


def mean_or_none(values):
    clean = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    ]

    if not clean:
        return None

    return statistics.fmean(clean)


def std_or_zero(values):
    clean = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    ]

    if len(clean) < 2:
        return 0.0

    return statistics.stdev(clean)


def aggregate_points(
    records: list[dict],
    x_key: str,
) -> list[dict]:
    grouped: dict[float, list[dict]] = {}

    for record in records:
        x = record.get(x_key)

        if x is None:
            continue

        x = float(x)

        grouped.setdefault(
            x,
            [],
        ).append(record)

    points = []

    for x in sorted(grouped):
        runs = grouped[x]

        llm_pdr = [
            item["llm_pdr_pct"]
            for item in runs
        ]

        q_pdr = [
            item["qfanet_pdr_pct"]
            for item in runs
        ]

        points.append({
            "x": x,
            "runs": len(runs),
            "llm_pdr_mean": mean_or_none(llm_pdr),
            "llm_pdr_std": std_or_zero(llm_pdr),
            "qfanet_pdr_mean": mean_or_none(q_pdr),
            "qfanet_pdr_std": std_or_zero(q_pdr),
            "source_files": [
                item["file"]
                for item in runs
            ],
        })

    return points


def run_main(
    *,
    output_path: Path,
    uavs: int,
    packet_rate_pps: float,
    speed_mps: float,
    args,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(MAIN_PATH),
        "--uavs",
        str(uavs),
        "--steps",
        str(args.steps),
        "--backend",
        args.backend,
        "--packet-rate-pps",
        str(packet_rate_pps),
        "--relay-capacity-pps",
        str(args.relay_capacity_pps),
        "--speed-mps",
        str(speed_mps),
        "--packet-size-bytes",
        str(args.packet_size_bytes),
        "--hello-interval-ms",
        str(args.hello_interval_ms),
        "--hello-size-bytes",
        str(args.hello_size_bytes),
        "--output-json",
        str(output_path),
    ]

    if args.fresh_learning:
        command.append(
            "--fresh-learning"
        )

    print()
    print("=" * 78)
    print("[comparaison.py] Running simulation:")
    print(
        " ".join(
            f'"{part}"'
            if " " in part
            else part
            for part in command
        )
    )
    print("=" * 78)

    subprocess.run(
        command,
        cwd=BASE_DIR,
        check=True,
    )

    read_json_utf8(
        output_path
    )

    print(
        f"[comparaison.py] Valid UTF-8 JSON: "
        f"{output_path}"
    )


def execute_study(
    study: str,
    runs: int,
    args,
) -> None:

    if not MAIN_PATH.exists():
        raise FileNotFoundError(
            f"main.py not found at {MAIN_PATH}"
        )

    if study == "flow":

        for value in FLOW_VALUES:

            for run_index in range(
                1,
                runs + 1
            ):

                output = (
                    BASE_DIR
                    / f"exp_flow_{value:g}_run_{run_index}.json"
                )

                run_main(
                    output_path=output,
                    uavs=args.base_uavs,
                    packet_rate_pps=value,
                    speed_mps=args.base_speed_mps,
                    args=args,
                )

    elif study == "speed":

        for value in SPEED_VALUES:

            for run_index in range(
                1,
                runs + 1
            ):

                output = (
                    BASE_DIR
                    / f"exp_speed_{value:g}_run_{run_index}.json"
                )

                run_main(
                    output_path=output,
                    uavs=args.base_uavs,
                    packet_rate_pps=args.base_packet_rate_pps,
                    speed_mps=value,
                    args=args,
                )

    elif study == "nodes":

        for value in NODE_VALUES:

            for run_index in range(
                1,
                runs + 1
            ):

                output = (
                    BASE_DIR
                    / f"exp_nodes_{value}_run_{run_index}.json"
                )

                run_main(
                    output_path=output,
                    uavs=value,
                    packet_rate_pps=args.base_packet_rate_pps,
                    speed_mps=args.base_speed_mps,
                    args=args,
                )

    else:

        raise ValueError(
            f"Unknown study: {study}"
        )


def collect_study_records(
    study: str,
) -> list[dict]:

    patterns = {
        "flow": "exp_flow_*_run_*.json",
        "speed": "exp_speed_*_run_*.json",
        "nodes": "exp_nodes_*_run_*.json",
    }

    pattern = patterns.get(
        study
    )

    if pattern is None:

        raise ValueError(
            f"Unknown study: {study}"
        )

    records = []

    for path in sorted(
        BASE_DIR.glob(pattern)
    ):

        try:

            result = read_json_utf8(
                path
            )

            records.append(
                extract_run_metrics(
                    result,
                    path,
                )
            )

        except Exception as exc:

            print(
                f"[plot_comparison.py] "
                f"Skipping invalid file "
                f"{path.name}: {exc}"
            )

    return records


def build_aggregate() -> dict:

    flow_runs = collect_study_records(
        "flow"
    )

    speed_runs = collect_study_records(
        "speed"
    )

    node_runs = collect_study_records(
        "nodes"
    )

    return {
        "schema_version": 1,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "description":
            "FANET sensitivity studies: LLM+RL vs Q-FANET",

        "studies": {

            "flow": {
                "x_key": "packet_rate_pps",
                "x_label": "DATA Flow (packets/s)",
                "y_label": "PDR (%)",

                "runs": flow_runs,

                "points": aggregate_points(
                    flow_runs,
                    "packet_rate_pps",
                ),
            },

            "speed": {
                "x_key": "mean_speed_mps",
                "x_label": "Mean UAV Speed (m/s)",
                "y_label": "PDR (%)",

                "runs": speed_runs,

                "points": aggregate_points(
                    speed_runs,
                    "mean_speed_mps",
                ),
            },

            "nodes": {
                "x_key": "num_uavs",
                "x_label": "Number of UAV Nodes",
                "y_label": "PDR (%)",

                "runs": node_runs,

                "points": aggregate_points(
                    node_runs,
                    "num_uavs",
                ),
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run FANET sensitivity studies and create "
            "experiment_results.json."
        )
    )

    parser.add_argument(
        "--study",
        choices=[
            "all",
            "flow",
            "speed",
            "nodes",
        ],
        default="all",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "number of independent simulations per X value "
            "(use 3 or more for mean/std curves)"
        ),
    )

    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "only rebuild experiment_results.json from existing "
            "experiments/*.json files"
        ),
    )

    parser.add_argument(
        "--backend",
        choices=[
            "ollama",
            "groq",
            "mixed",
        ],
        default="ollama",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
    )

    parser.add_argument(
        "--base-uavs",
        type=int,
        default=DEFAULT_UAVS,
    )

    parser.add_argument(
        "--base-packet-rate-pps",
        type=float,
        default=DEFAULT_PACKET_RATE_PPS,
    )

    parser.add_argument(
        "--relay-capacity-pps",
        type=float,
        default=DEFAULT_RELAY_CAPACITY_PPS,
    )

    parser.add_argument(
        "--base-speed-mps",
        type=float,
        default=DEFAULT_SPEED_MPS,
    )

    parser.add_argument(
        "--packet-size-bytes",
        type=int,
        default=DEFAULT_PACKET_SIZE_BYTES,
    )

    parser.add_argument(
        "--hello-interval-ms",
        type=float,
        default=DEFAULT_HELLO_INTERVAL_MS,
    )

    parser.add_argument(
        "--hello-size-bytes",
        type=int,
        default=DEFAULT_HELLO_SIZE_BYTES,
    )

    parser.add_argument(
        "--fresh-learning",
        action="store_true",
        help=(
            "ignore persisted Q-tables for each run. "
            "Leave this OFF to evaluate a trained model."
        ),
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error(
            "--runs must be >= 1"
        )

    if args.steps < 2:
        parser.error(
            "--steps must be >= 2"
        )

    if args.base_uavs < 1:
        parser.error(
            "--base-uavs must be >= 1"
        )

    if args.base_packet_rate_pps <= 0:
        parser.error(
            "--base-packet-rate-pps must be > 0"
        )

    if args.relay_capacity_pps <= 0:
        parser.error(
            "--relay-capacity-pps must be > 0"
        )

    if args.base_speed_mps < 0:
        parser.error(
            "--base-speed-mps must be >= 0"
        )

    return args


def main():
    args = parse_args()

    if not args.collect_only:
        studies = (
            ["flow", "speed", "nodes"]
            if args.study == "all"
            else [args.study]
        )

        for study in studies:
            execute_study(
                study,
                args.runs,
                args,
            )

    aggregate = build_aggregate()

    write_json_utf8_atomic(
        aggregate,
        AGGREGATE_PATH,
    )

    print()
    print("=" * 78)
    print(
        "[comparaison.py] Aggregate generated:"
    )
    print(
        AGGREGATE_PATH
    )
    print()
    print(
        "Start dashboard with:"
    )
    print(
        "python -m http.server 8000"
    )
    print()
    print(
        "Open:"
    )
    print(
        "http://localhost:8000/Dashboard.html"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
