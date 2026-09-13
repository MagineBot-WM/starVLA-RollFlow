#!/usr/bin/env python3
"""Plot the G1 OT scaling/mask ablation without hiding LSD instability.

The old plot used only ``lsd_loss`` (the post-gate contribution).  That is
misleading for a stability analysis: an enabled gate can turn an exploding
``lsd_loss_raw`` into an apparently perfect zero.  This script keeps both
quantities and records the provenance of every run (in particular whether it
starts from a checkpoint).

Example
-------
python scripts/plot_rollflow_g1_ot_paper.py
python scripts/plot_rollflow_g1_ot_paper.py --max-step 2000
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(__file__).resolve().with_name("g1_stability_all_manifest.json")


def _float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _moving_average(values: list[float], window: int) -> list[float]:
    """Causal average; non-finite observations remain visible as gaps."""
    output: list[float] = []
    history: list[float] = []
    for value in values:
        if not math.isfinite(value):
            output.append(math.nan)
            continue
        history.append(value)
        if len(history) > window:
            history.pop(0)
        output.append(sum(history) / len(history) if history else math.nan)
    return output


def _config_metadata(run: str) -> dict[str, Any]:
    """Read config metadata when the historical run directory is available."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - matplotlib environments normally include yaml
        yaml = None

    config = Path("/data/tzq/starVLA_checkpoints") / run / "config.yaml"
    data: dict[str, Any] = {}
    if yaml is not None and config.exists():
        try:
            data = yaml.safe_load(config.read_text()) or {}
        except Exception:
            data = {}
    action = data.get("framework", {}).get("action_model", {})
    trainer = data.get("trainer", {})
    # Older configs predate these explicit switches.  The run name is the
    # only safe fallback for the two ablation factors in that case.
    no_scaling = "no_scaling" in run or "no_scaling" in run.replace("-", "_")
    no_gate = "no_gate" in run or "no_mask" in run.replace("-", "_")
    scaling = action.get("use_lsd_scaling")
    gate = action.get("use_lsd_gate")
    return {
        "config": str(config) if config.exists() else None,
        "pretrained_checkpoint": trainer.get("pretrained_checkpoint"),
        "initialization": (
            "55K-pretrained fine-tune"
            if trainer.get("pretrained_checkpoint")
            else "scratch"
        ),
        "use_lsd_scaling": bool(scaling) if scaling is not None else not no_scaling,
        "use_lsd_gate": bool(gate) if gate is not None else not no_gate,
        "fm_only_steps": action.get("fm_only_steps"),
        "w_lsd": action.get("w_lsd"),
        "fm_curriculum_steps": action.get("fm_curriculum_steps"),
        "p_fm": action.get("p_fm"),
        "p_k1": action.get("p_k1"),
        "finite_difference_delta": action.get("finite_difference_delta"),
        "action_horizon": action.get("action_horizon"),
        "execution_horizon": action.get("execution_horizon"),
        "seed": data.get("seed"),
    }


def _strip_ansi(text: str) -> str:
    """Remove Rich/terminal control sequences from a training log."""
    text = re.sub(r"\x1b\][^\x1b]*(?:\x1b\\|\x07)", "", text)
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _parse_log_dict(text: str, start: int) -> tuple[dict[str, Any] | None, int]:
    """Parse one pretty-printed ``Loss`` dictionary from a Rich log."""
    depth = 0
    quote: str | None = None
    escaped = False
    end = None
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        return None, start
    snippet = text[start:end]
    # Rich's pretty printer inserts spaces/newlines inside long quoted keys.
    # They are visual wrapping, not part of the metric name.
    snippet = re.sub(
        r"'([^']*)'", lambda match: "'" + re.sub(r"\s+", "", match.group(1)) + "'", snippet
    )
    try:
        return ast.literal_eval(snippet), end
    except (SyntaxError, ValueError):
        # A few old logs contain bare nan/inf values.  Keep parsing safe by
        # exposing only these numeric constants to eval.
        try:
            return eval(snippet, {"__builtins__": {}}, {"nan": math.nan, "inf": math.inf}), end
        except Exception:
            return None, end


def _read_log_rows(run_info: dict[str, Any], metadata: dict[str, Any], max_step: int | None) -> list[dict[str, Any]]:
    """Extract G1 metrics directly from the source log when available."""
    log_name = run_info.get("log")
    path = Path(log_name) if log_name else Path()
    if not path.is_file():
        return []
    text = _strip_ansi(path.read_text(errors="ignore"))
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"Step\s+(\d+),\s*Loss:", text):
        step = int(match.group(1))
        if max_step is not None and step > max_step:
            continue
        start = text.find("{", match.end())
        if start < 0:
            continue
        metrics, _ = _parse_log_dict(text, start)
        if not metrics:
            continue
        prefix = "rollflow/agibot-g1/"
        row: dict[str, Any] = {
            "run": run_info["run"],
            "label": run_info["label"],
            "step": step,
            "embodiment": "agibot-g1",
        }
        for name in (
            "use_ot",
            "fm_loss",
            "fm_loss_active",
            "lsd_loss",
            "lsd_loss_raw",
            "lsd_loss_metric",
            "lsd_gate_active",
            "lsd_gate_enabled",
            "lsd_scaling_enabled",
            "lsd_keep_frac",
            "lsd_active_sample_frac",
            "active_lsd_frac",
            "p_fm",
            "fm_only",
            "fm_only_steps",
        ):
            value = metrics.get(prefix + name, metrics.get(name))
            row[name] = value if isinstance(value, bool) else _float(value)
        # Older logging code did not expose a separate raw field.
        if not math.isfinite(row["lsd_loss_raw"]):
            row["lsd_loss_raw"] = row["lsd_loss"]
        row.update(metadata)
        fm, raw = row["fm_loss"], row["lsd_loss_raw"]
        row["lsd_over_fm"] = raw / fm if math.isfinite(raw) and math.isfinite(fm) and fm > 0 else math.nan
        active_fm = row.get("fm_loss_active", math.nan)
        budget = float(metadata.get("w_lsd") or 0.1) * active_fm
        row["lsd_over_budget"] = (
            raw / budget
            if math.isfinite(raw) and math.isfinite(budget) and budget > 0
            else math.nan
        )
        rows.append(row)
    return rows


def _load_rows(manifest: dict[str, Any], max_step: int | None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows_path = Path(manifest.get("rows", DEFAULT_MANIFEST.with_name("g1_ot_loss_rows.csv")))
    if not rows_path.exists():
        raise FileNotFoundError(f"loss rows not found: {rows_path}")
    selected = {item["run"]: item for item in manifest["runs"]}
    metadata = {run: _config_metadata(run) for run in selected}
    rows: list[dict[str, Any]] = []
    # Prefer source logs so rerunning the script cannot silently reuse a stale
    # CSV.  The CSV remains a portable fallback for archived/deleted logs.
    for run_info in manifest["runs"]:
        parsed = _read_log_rows(run_info, metadata[run_info["run"]], max_step)
        if parsed:
            rows.extend(parsed)
    parsed_runs = {row["run"] for row in rows}
    with rows_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("run") not in selected or row.get("run") in parsed_runs or row.get("embodiment") != "agibot-g1":
                continue
            row.setdefault("label", selected[row["run"]].get("label", row["run"]))
            step = int(float(row["step"]))
            if max_step is not None and step > max_step:
                continue
            row["step"] = step
            for key, value in list(row.items()):
                if key not in {"run", "label", "embodiment"}:
                    row[key] = _float(value)
            row.update(metadata[row["run"]])
            fm = row.get("fm_loss", math.nan)
            raw = row.get("lsd_loss_raw", math.nan)
            row["lsd_over_fm"] = raw / fm if math.isfinite(raw) and math.isfinite(fm) and fm > 0 else math.nan
            active_fm = row.get("fm_loss_active", math.nan)
            budget = float(metadata[row["run"]].get("w_lsd") or 0.1) * active_fm
            row["lsd_over_budget"] = (
                raw / budget
                if math.isfinite(raw) and math.isfinite(budget) and budget > 0
                else math.nan
            )
            rows.append(row)
    if not rows:
        raise RuntimeError("no agibot-g1 rows selected")
    return rows, metadata


def _run_label(run: str, label: str) -> str:
    """Use a compact, factorized label that is readable in one legend."""
    no_ot = "no_ot" in run or "no OT" in label
    no_scaling = "no_scaling" in run or "w/o scaling" in label
    no_gate = "no_gate" in run or "no mask" in label
    collapsed = no_scaling and no_gate
    prefix = "no-OT" if no_ot else "OT"
    if collapsed:
        return f"{prefix} · unstable (−S−G)"
    if no_scaling:
        return f"{prefix} · gate only (−S+G)"
    if no_gate:
        return f"{prefix} · scaling only (+S−G)"
    return f"{prefix} · full (+S+G)"


def _run_style(run: str, label: str) -> tuple[str, str, float, str]:
    """Stable color/style identity independent of manifest ordering."""
    no_ot = "no_ot" in run or "no OT" in label
    no_scaling = "no_scaling" in run or "w/o scaling" in label
    no_gate = "no_gate" in run or "no mask" in label
    prefix = "no-OT" if no_ot else "OT"
    # Removing both protections is the collapse control even when an older
    # manifest did not include the parenthetical "(collapsed)" label.
    collapsed = no_scaling and no_gate
    if collapsed and no_ot:
        return "#7b2cbf", "--", 2.4, "no-OT · unstable (−S−G)"
    if collapsed:
        return "#d55e00", "-", 2.4, "OT · unstable (−S−G)"
    if no_scaling:
        return "#e69f00", "--" if no_ot else "-", 2.0, f"{prefix} · gate only (−S+G)"
    if no_gate:
        return "#0072b2", "--" if no_ot else "-", 1.9, f"{prefix} · scaling only (+S−G)"
    return "#009e73", "--" if no_ot else "-", 2.5, f"{prefix} · full (+S+G)"


def _finite_values(group: list[dict[str, Any]], key: str) -> list[float]:
    return [row[key] for row in group if math.isfinite(row.get(key, math.nan))]


def _value_at_or_before(group: list[dict[str, Any]], key: str, step: int) -> float:
    """Use the last finite observation at or before a common step."""
    candidates = [
        row for row in group
        if row["step"] <= step and math.isfinite(row.get(key, math.nan))
    ]
    return candidates[-1][key] if candidates else math.nan


def _plot(rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], out: Path, window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order: list[str] = []
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        run = row["run"]
        if run not in by_run:
            order.append(run)
            by_run[run] = []
        by_run[run].append(row)

    # Runs often stop at slightly different logging steps. Clip the visual
    # comparison to the common observed horizon; the report still records each
    # run's true last step.
    common_end = min(
        max(row["step"] for row in group)
        for group in by_run.values()
        if group
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), sharex=True, constrained_layout=False)
    ax_fm, ax_raw, ax_gate, ax_ratio = axes.flat
    legend_handles = []
    for index, run in enumerate(order):
        group = sorted(by_run[run], key=lambda row: row["step"])
        original_label = group[0]["label"]
        color, line_style, linewidth, label = _run_style(run, original_label)
        x = [row["step"] for row in group]
        fm = [row.get("fm_loss", math.nan) for row in group]
        raw = [row.get("lsd_loss_raw", math.nan) for row in group]
        reject = [
            100.0 * row.get("lsd_gate_active", math.nan)
            if math.isfinite(row.get("lsd_gate_active", math.nan))
            else math.nan
            for row in group
        ]
        ratio = [row.get("lsd_over_budget", math.nan) for row in group]
        # FM is smoothed for readability. Raw LSD and pressure remain unsmoothed
        # in a faint trace, with a thicker causal trend overlaid.
        line, = ax_fm.plot(x, _moving_average(fm, window), color=color, lw=linewidth, ls=line_style, label=label)
        legend_handles.append(line)
        ax_raw.plot(x, raw, color=color, lw=0.8, ls=line_style, alpha=0.22)
        ax_raw.plot(x, _moving_average(raw, window), color=color, lw=linewidth, ls=line_style, alpha=0.95)
        ax_gate.plot(x, reject, color=color, lw=linewidth, ls=line_style)
        ax_ratio.plot(x, ratio, color=color, lw=0.8, ls=line_style, alpha=0.22)
        ax_ratio.plot(x, _moving_average(ratio, window), color=color, lw=linewidth, ls=line_style, alpha=0.95)

        # Make the two collapse claims explicit on the FM panel. The annotation
        # is derived from the observed minimum rather than a hard-coded step.
        no_scaling = "no_scaling" in run or "w/o scaling" in original_label
        no_gate = "no_gate" in run or "no mask" in original_label
        window_group = [row for row in group if row["step"] <= common_end]
        if no_scaling and no_gate and _finite_values(window_group, "fm_loss"):
            finite_fm = [(row["step"], row["fm_loss"]) for row in window_group if math.isfinite(row.get("fm_loss", math.nan))]
            min_step, min_fm = min(finite_fm, key=lambda item: item[1])
            no_ot = "no_ot" in run or "no OT" in original_label
            text_x = min_step + (210 if no_ot else 95)
            text_y = min_fm + (0.07 if no_ot else 0.18)
            ax_fm.annotate(
                f"rebound ({'no-OT' if no_ot else 'OT'})",
                xy=(min_step, min_fm),
                xytext=(text_x, text_y),
                color=color,
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            )

    ax_fm.set_title("A  FM objective", loc="left", fontweight="bold")
    ax_fm.set_ylabel("FM velocity MSE")
    ax_raw.set_title("B  Raw LSD magnitude", loc="left", fontweight="bold")
    ax_raw.set_ylabel("LSD loss (pre-gate)")
    ax_raw.set_yscale("log")
    ax_raw.text(0.02, 0.95, "faint = individual logs\nthick = causal trend", transform=ax_raw.transAxes, va="top", fontsize=8.5)
    ax_gate.set_title("C  Gate behavior", loc="left", fontweight="bold")
    ax_gate.set_ylabel("rejection among active samples (%)")
    ax_gate.set_ylim(-2.0, 102.0)
    ax_gate.set_yticks([0, 25, 50, 75, 100])
    # A zero rejection value is ambiguous when no LSD token was active. Show
    # active-token coverage as a neutral reference on the same 0–100 scale.
    active_by_step: dict[int, list[float]] = {}
    for row in rows:
        value = row.get("active_lsd_frac", math.nan)
        if math.isfinite(value):
            active_by_step.setdefault(int(row["step"]), []).append(100.0 * value)
    if active_by_step:
        active_steps = sorted(active_by_step)
        active_values = [sum(active_by_step[step]) / len(active_by_step[step]) for step in active_steps]
        ax_gate.plot(active_steps, active_values, color="#777777", lw=1.4, ls=":", alpha=0.9)
        ax_gate.text(0.02, 0.90, "gray dotted = active-token coverage", transform=ax_gate.transAxes, fontsize=8.5)
    ax_ratio.set_title("D  Raw LSD relative to its budget", loc="left", fontweight="bold")
    ax_ratio.set_ylabel("raw LSD / (w_lsd · FM_active)")
    ax_ratio.set_yscale("log")
    ax_ratio.axhline(1.0, color="#555555", lw=1, ls=":", alpha=0.8)
    ax_ratio.text(0.02, 0.95, "reference = 1; global proxy\nactual gate is sample-wise", transform=ax_ratio.transAxes, va="top", fontsize=8.5)
    for axis in axes.flat:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.set_xlabel("training step")
        axis.set_xlim(0, common_end)

    initializations = sorted({info["initialization"] for info in metadata.values()})
    caveat = "; ".join(initializations)
    scope = "OT only" if all(not ("no_ot" in run) for run in order) else "OT and no-OT stress controls"
    fig.suptitle(
        f"RollFlow stability ablation on G1 ({scope})",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.945,
        f"Agibot G1 · H{next(iter(metadata.values())).get('action_horizon') or '?'} / C{next(iter(metadata.values())).get('execution_horizon') or '?'} · {caveat} · seed {next(iter(metadata.values())).get('seed')}",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.text(
        0.5,
        0.915,
        "S = central-difference scaling (2δ)   ·   G = sample-wise LSD gate   ·   lower FM/pressure is better",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=max(1, len(legend_handles)),
        fontsize=9.5,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.6,
    )
    fig.text(
        0.5,
        0.035,
        f"Matched historical window: {caveat}; common comparison through step {common_end}; FM = causal moving average (window={window}); raw LSD is unsmoothed.",
        ha="center",
        fontsize=9.0,
        color="#444444",
    )
    fig.text(
        0.5,
        0.012,
        "Both −S−G controls show FM rebound. This is a training-loss stability diagnostic, not a closed-loop success evaluation.",
        ha="center",
        fontsize=9.0,
        color="#444444",
    )
    fig.subplots_adjust(top=0.755, bottom=0.125, left=0.075, right=0.985, hspace=0.28, wspace=0.22)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def _write_manifest(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    out: Path,
    window: int,
    max_step: int | None,
    destination: Path | None = None,
    report: Path | None = None,
) -> Path:
    corrected = dict(manifest)
    corrected["rows"] = str(Path(manifest.get("rows", DEFAULT_MANIFEST.with_name("g1_ot_loss_rows.csv"))))
    corrected["filter"] = dict(corrected.get("filter", {}))
    if max_step is not None:
        corrected["filter"]["max_step"] = max_step
    if rows:
        by_run: dict[str, list[int]] = {}
        for row in rows:
            by_run.setdefault(row["run"], []).append(int(row["step"]))
        corrected["filter"]["common_end_step"] = min(max(steps) for steps in by_run.values())
    corrected["smoothing"] = {
        "fm": f"causal moving average, window={window}",
        "lsd_raw": "none (spikes are stability evidence)",
        "lsd_accepted": "none",
    }
    corrected["metric"] = {
        "fm": "fm_loss",
        "fm_active": "fm_loss_active",
        "lsd_raw": "lsd_loss_raw (pre-gate, includes 2*delta scaling when enabled)",
        "lsd_accepted": "lsd_loss (post-gate contribution)",
        "gate_rejection": "lsd_gate_active (active-sample rejection fraction)",
        "pressure_ratio": "lsd_loss_raw / (w_lsd * fm_loss_active); global proxy for sample-wise gate",
    }
    corrected["provenance"] = {
        "runs": metadata,
        "limitations": [
            "All selected ablations initialize from the same H32 55K checkpoint; they are not scratch runs.",
            "A post-gate LSD curve can be zero while raw LSD is large; never use it alone for stability claims.",
            "The selected window is a training-loss diagnostic, not a closed-loop trajectory evaluation.",
        ],
        "interpretation": "Scaling controls central-difference gradient magnitude; the gate suppresses non-finite or over-budget sample updates. They are complementary safety mechanisms.",
    }
    corrected["outputs"] = {
        "figure": str(out),
        "pdf": str(out.with_suffix(".pdf")),
        "records": len(rows),
    }
    if report is not None:
        corrected["outputs"]["report"] = str(report)
    out_manifest = destination or out.with_name(f"{out.stem}_manifest.json")
    out_manifest.write_text(json.dumps(corrected, indent=2, ensure_ascii=False) + "\n")
    return out_manifest


def _write_report(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    figure: Path,
    destination: Path,
) -> Path:
    """Write a short, reviewer-facing interpretation next to the figure."""
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    common_end = min(
        max(row["step"] for row in group)
        for group in by_run.values()
        if group
    )

    def fmt(value: float) -> str:
        return "—" if not math.isfinite(value) else f"{value:.3g}"

    table: list[str] = []
    for run_info in manifest["runs"]:
        run = run_info["run"]
        group = sorted(by_run.get(run, []), key=lambda row: row["step"])
        if not group:
            continue
        window_group = [row for row in group if row["step"] <= common_end]
        fm = _finite_values(window_group, "fm_loss")
        pressure = _finite_values(window_group, "lsd_over_budget")
        rejection = _finite_values(window_group, "lsd_gate_active")
        min_fm = min(fm) if fm else math.nan
        min_step = next(
            (row["step"] for row in group if row.get("fm_loss") == min_fm),
            math.nan,
        )
        table.append(
            "| "
            + " | ".join(
                (
                    _run_label(run, run_info["label"]),
                    "no" if ("no_ot" in run or "no OT" in run_info["label"]) else "yes",
                    "on" if metadata[run]["use_lsd_scaling"] else "off",
                    "on" if metadata[run]["use_lsd_gate"] else "off",
                    fmt(min_fm),
                    str(int(min_step)) if math.isfinite(min_step) else "—",
                    fmt(_value_at_or_before(group, "fm_loss", common_end)),
                    str(group[-1]["step"]),
                    fmt(max(pressure) if pressure else math.nan),
                    fmt(100.0 * max(rejection) if rejection else math.nan) + "%",
                )
            )
            + " |"
        )

    limitations = [
        "The selected runs start from the same H32 55K checkpoint; they are fine-tunes, not scratch self-bootstrap runs.",
        "The figure uses training losses. It does not replace held-out or closed-loop trajectory evaluation.",
        "A post-gate LSD curve can be zero because the gate rejected the update; raw LSD and rejection rate are therefore shown separately.",
    ]
    figure_name = figure.name
    settings = next(iter(metadata.values()))
    scope_line = (
        f"Configuration scope: H{settings.get('action_horizon') or '?'} / C{settings.get('execution_horizon') or '?'}, "
        f"`fm_curriculum_steps={settings.get('fm_curriculum_steps')}`, `p_fm={settings.get('p_fm')}`, "
        f"`w_lsd={settings.get('w_lsd')}`, seed `{settings.get('seed')}`."
    )
    text = "\n".join(
        [
            "# RollFlow G1 stability ablation",
            "",
            "## Reading the figure",
            "",
            f"The figure (`{figure_name}`) separates the two safety mechanisms that are easy to conflate:",
            "",
            "- **S (scaling)** multiplies the central-difference LSD metric by `2δ`, controlling its gradient magnitude.",
            "- **G (gate/mask)** rejects non-finite or over-budget per-sample LSD updates.",
            "- **OT** denotes optimal-transport matching of noise to target trajectories.",
            "",
            scope_line,
            "",
            "Panel A is the training FM objective. Panels B and D expose the raw LSD pressure before the gate; Panel C shows how many active samples are rejected (an intervention signal, not a quantity to minimize blindly). The faint traces are individual log values and the thick traces are causal moving averages. The two `−S−G` controls are explicitly marked because both show FM rebound after their minimum.",
            "",
            "## Observed window",
            "",
            f"Common comparison step: `{common_end}`. `FM @ common` is the last observation at or before that step; `last step` is the run's actual endpoint.",
            "",
            "| run | OT | S | G | min FM | min step | FM @ common | last step | peak raw LSD/budget | peak rejection (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *table,
            "",
            "## Interpretation",
            "",
            "In this matched historical window, scaling keeps the global raw LSD/budget proxy well below one. Removing scaling drives the proxy above one; the gate then rejects the offending sample updates. Removing both protections produces the two observed unstable controls (OT and no-OT), where FM first improves and then rebounds.",
            "",
            "Both available scaling-only `+S−G` controls (OT and no-OT) remain stable in this window, so the observed rebound is not attributable to OT alone. The gate is not always active once scaling is present. This does **not** establish that removing the gate is safe from scratch; a factorial scratch ablation is required for that claim.",
            "",
            "## Scope and limitations",
            "",
            *[f"- {item}" for item in limitations],
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/tzq/starVLA_checkpoints/rollflow_ablation_analysis/g1_ot_paper/g1_stability_all_paper_0_2500.png"),
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    manifest = json.loads(args.manifest.read_text())
    max_step = args.max_step
    if max_step is None:
        max_step = manifest.get("filter", {}).get("max_step")
    rows, metadata = _load_rows(manifest, max_step)
    _plot(rows, metadata, args.output, args.window)
    report_output = args.report_output or args.output.with_suffix(".md")
    manifest_output = _write_manifest(
        manifest,
        rows,
        metadata,
        args.output,
        args.window,
        max_step,
        args.manifest_output,
        report_output,
    )
    report_output = _write_report(rows, metadata, manifest, args.output, report_output)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.pdf')}")
    print(f"wrote {manifest_output}")
    print(f"wrote {report_output}")


if __name__ == "__main__":
    main()
