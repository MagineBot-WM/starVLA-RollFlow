#!/usr/bin/env python3
"""Summarise RollFlow ablation training logs for paper figures.

The trainer writes a Python-literal metrics dictionary every logging interval,
for example ``Step 100, Loss: {...}``.  This script extracts those records
without importing the training stack, writes a long-form CSV, and produces a
JSON summary plus an optional moving-average plot.  It is safe to run while a
job is still appending to ``train.log``; the parser simply uses the complete
records seen at invocation time.

Example::

    python scripts/analyze_rollflow_ablation_logs.py \
      /data/tzq/starVLA_checkpoints/libero10hz_pick_up_the_apple_ablation_no_ot_no_gate_55k_20k_v1/train.log \
      /data/tzq/starVLA_checkpoints/libero10hz_pick_up_the_apple_ablation_ot_no_gate_55k_20k_v1/train.log \
      /data/tzq/starVLA_checkpoints/libero10hz_pick_up_the_apple_ablation_ot_gate_55k_20k_v1/train.log \
      --output-dir /data/tzq/starVLA_checkpoints/rollflow_ablation_analysis
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


_OSC_RE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ESC_RE = re.compile(r"\x1b[@-_]")
_STEP_RE = re.compile(r"Step\s+(\d+),\s+Loss:\s*")

METRIC_KEYS = (
    "loss",
    "fm_loss",
    "fm_loss_active",
    "lsd_loss",
    "lsd_loss_raw",
    "lsd_loss_metric",
    "lsd_jvp_enabled",
    "lsd_scaling_enabled",
    "lsd_gate_active",
    "lsd_gate_enabled",
    "lsd_finite_frac",
    "lsd_keep_frac",
    "lsd_budget",
    "active_lsd_frac",
    "teacher_clip_frac",
    "v_local_abs",
    "v_tangent_abs",
    "v_teacher_abs",
    "p_fm",
    "fm_only_frac",
    "source_ratio",
    "s_mean",
    "t_mean",
    "use_ot",
)

# Keep the CSV/JSON keys lossless, but use concise labels in figures so the
# three ablations remain readable in a paper-sized legend.
PLOT_LABELS = {
    "libero10hz_pick_up_the_apple_ablation_no_ot_no_gate_55k_20k_v1": "No OT + No Gate",
    "libero10hz_pick_up_the_apple_ablation_ot_no_gate_55k_20k_v1": "OT + No Gate",
    "libero10hz_pick_up_the_apple_ablation_ot_gate_55k_20k_v1": "OT + Gate",
    "libero10hz_pick_up_the_apple_ablation_ot_no_scaling_no_gate_55k_20k_v1": "OT + No Scaling + No Gate",
    "libero10hz_pick_up_the_apple_ablation_no_ot_no_gate_no_scaling_55k_20k_v1": "No OT + No Gate + No Scaling",
    "libero10hz_pick_up_the_apple_ablation_ot_no_gate_no_scaling_55k_20k_v1": "OT + No Gate + No Scaling",
    "libero10hz_pick_up_the_apple_ablation_ot_gate_no_scaling_55k_20k_v1": "OT + Gate + No Scaling",
    "libero10hz_pick_up_the_apple_jvp_ot_gate_no_scaling_55k_20k_v1": "JVP + OT + Gate + No Scaling",
    "libero10hz_pick_up_the_apple_jvp_ot_no_gate_no_scaling_55k_20k_v1": "JVP + OT + No Gate + No Scaling",
}


def strip_terminal_controls(line: str) -> str:
    """Remove ANSI colour/progress and OSC hyperlinks from a terminal log."""

    line = _OSC_RE.sub("", line)
    line = _CSI_RE.sub("", line)
    return _ESC_RE.sub("", line)


def _as_scalar(value: Any) -> float | int | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    return None


def parse_log(path: Path) -> list[dict[str, Any]]:
    """Parse and deduplicate ``(step, embodiment)`` records from one log.

    Rich may wrap the dictionary over dozens of physical lines in a durable
    log.  The small brace-balanced collector below handles both wrapped and
    single-line formats.
    """

    records: dict[tuple[int, str], dict[str, Any]] = {}

    def consume(step: int, text: str) -> None:
        try:
            payload = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return
        if not isinstance(payload, dict):
            return

        # The top-level action loss is useful as a sanity check.  Per-
        # embodiment RollFlow metrics are the rows used for comparisons.
        top = {
            key: _as_scalar(payload.get(key))
            for key in ("action_dit_loss", "action_loss")
            if key in payload
        }
        found_embodiment = False
        for key, value in payload.items():
            if not isinstance(key, str) or not key.startswith("rollflow/"):
                continue
            prefix, separator, metric = key[len("rollflow/") :].partition("/")
            if not separator or not metric:
                continue
            found_embodiment = True
            scalar = _as_scalar(value)
            if scalar is None and value is not None:
                continue
            row = {"step": step, "embodiment": prefix, **top}
            row.update({name: None for name in METRIC_KEYS})
            row[metric] = scalar
            existing = records.setdefault((step, prefix), row)
            existing.update({name: value for name, value in row.items() if value is not None})

        # A malformed or partial metric dictionary should not make the entire
        # step disappear; keep a row if only a top-level loss was emitted.
        if not found_embodiment:
            row = {"step": step, "embodiment": "__global__", **top}
            row.update({name: None for name in METRIC_KEYS})
            records[(step, "__global__")] = row

    pending_step: int | None = None
    pending_text = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = strip_terminal_controls(raw_line).rstrip()
            if pending_step is not None:
                # Rich inserts indentation and can split a quoted key at the
                # terminal width.  Concatenating trimmed physical lines
                # reconstructs both tokens and split strings.
                pending_text += line.strip()
                if pending_text.count("{") > 0 and pending_text.count("{") == pending_text.count("}"):
                    start = pending_text.index("{")
                    end = pending_text.rfind("}") + 1
                    consume(pending_step, pending_text[start:end])
                    pending_step = None
                    pending_text = ""
                continue

            match = _STEP_RE.search(line)
            if match is None:
                continue
            pending_step = int(match.group(1))
            pending_text = line[match.end() :].strip()
            if "{" not in pending_text:
                continue
            if pending_text.count("{") == pending_text.count("}"):
                start = pending_text.index("{")
                end = pending_text.rfind("}") + 1
                consume(pending_step, pending_text[start:end])
                pending_step = None
                pending_text = ""

    # Ignore an incomplete final record; it is normal if the trainer was
    # writing while this script was invoked.

    return sorted(records.values(), key=lambda row: (int(row["step"]), str(row["embodiment"])))


def merge_metric_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge metric fragments from the same step/embodiment."""

    merged: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["step"]), str(row["embodiment"]))
        target = merged.setdefault(key, {"step": key[0], "embodiment": key[1]})
        for name, value in row.items():
            if name in target and target[name] is not None and value is None:
                continue
            target[name] = value
    return sorted(merged.values(), key=lambda row: (row["step"], row["embodiment"]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["embodiment"])].append(row)

    result: dict[str, Any] = {"records": len(rows), "embodiments": {}}
    for embodiment, values in sorted(grouped.items()):
        entry: dict[str, Any] = {"records": len(values), "first_step": values[0]["step"], "last_step": values[-1]["step"]}
        for metric in METRIC_KEYS:
            numeric = [float(row[metric]) for row in values if isinstance(row.get(metric), (int, float)) and not isinstance(row.get(metric), bool)]
            if not numeric:
                continue
            finite = [x for x in numeric if math.isfinite(x)]
            metric_summary: dict[str, Any] = {"count": len(numeric), "finite_count": len(finite)}
            if finite:
                metric_summary.update(
                    mean=mean(finite),
                    std=pstdev(finite),
                    first=finite[0],
                    last=finite[-1],
                    minimum=min(finite),
                    maximum=max(finite),
                )
            entry[metric] = metric_summary
        result["embodiments"][embodiment] = entry
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], run: str) -> None:
    fields = ["run", "step", "embodiment", "action_dit_loss", "action_loss", *METRIC_KEYS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({"run": run, **row})


def moving_average(values: list[float | None], window: int) -> list[float | None]:
    output: list[float | None] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        current = [float(x) for x in values[start : index + 1] if isinstance(x, (int, float))]
        output.append(sum(current) / len(current) if current else None)
    return output


def write_plot(path: Path, runs: list[tuple[str, list[dict[str, Any]]]], window: int) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    embodiments = sorted({str(row["embodiment"]) for _, rows in runs for row in rows if row["embodiment"] != "__global__"})
    # Main paper figure: keep only the two optimization losses for each
    # embodiment. Gate/keep diagnostics remain available in the CSV/JSON
    # outputs, but should not occupy panels in the loss comparison figure.
    metrics = ["fm_loss", "lsd_loss"]
    if not embodiments:
        return False
    figure, axes = plt.subplots(len(metrics), len(embodiments), squeeze=False, figsize=(5.2 * len(embodiments), 3.2 * len(metrics)), sharex="col")
    for column, embodiment in enumerate(embodiments):
        for metric_index, metric in enumerate(metrics):
            axis = axes[metric_index][column]
            for run, rows in runs:
                selected = [row for row in rows if row["embodiment"] == embodiment]
                selected.sort(key=lambda row: row["step"])
                steps = [int(row["step"]) for row in selected]
                values = [row.get(metric) for row in selected]
                smooth = moving_average(values, window)
                if steps:
                    axis.plot(steps, smooth, label=PLOT_LABELS.get(run, run), linewidth=1.5)
            axis.set_title(f"{embodiment}: {metric}")
            axis.grid(alpha=0.25)
            if metric in {"lsd_gate_active", "lsd_keep_frac"}:
                axis.set_ylim(-0.02, 1.02)
            if metric_index == len(metrics) - 1:
                axis.set_xlabel("optimization step")
    axes[0][0].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def resolve_log(path: Path) -> Path:
    if path.is_dir():
        path = path / "train.log"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="train.log files or run directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=25, help="moving-average window for the PNG")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, list[dict[str, Any]]]] = []
    summary: dict[str, Any] = {"logs": []}
    all_rows: list[dict[str, Any]] = []
    for supplied in args.logs:
        log_path = resolve_log(supplied)
        run = log_path.parent.name
        rows = merge_metric_rows(parse_log(log_path))
        runs.append((run, rows))
        write_csv(args.output_dir / f"{run}.csv", rows, run)
        run_summary = {"run": run, "log": str(log_path), **summarize(rows)}
        summary["logs"].append(run_summary)
        all_rows.extend({"run": run, **row} for row in rows)

    write_csv(args.output_dir / "all_runs.csv", all_rows, "mixed")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_written = write_plot(args.output_dir / "paper_curves.png", runs, args.window)
    print(f"Parsed {len(runs)} logs and {len(all_rows)} metric rows")
    print(f"CSV/JSON output: {args.output_dir}")
    if plot_written:
        print(f"Plot: {args.output_dir / 'paper_curves.png'}")
    else:
        print("Plot not written (matplotlib is unavailable or logs contain no embodiment metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
