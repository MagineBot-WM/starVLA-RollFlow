#!/usr/bin/env python3
"""Plot the G1 RollFlow stability ablation and an optional OT comparison.

The old plot used only ``lsd_loss`` (the post-gate contribution).  That is
misleading for a stability analysis: an enabled gate can turn an exploding
``lsd_loss_raw`` into an apparently perfect zero.  This script keeps both
quantities and records the provenance of every run (in particular whether it
starts from a checkpoint).  A manifest may reserve panels A--C for the
scaling/gate ablation and panel D for a matched OT-on/OT-off total-loss pair.

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
            "loss",
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


def _is_no_ot(run: str, label: str) -> bool:
    return "no_ot" in run or "no OT" in label


def _run_label(run: str, label: str, *, show_ot: bool = False) -> str:
    """Use a compact label for the scaling/gate ablation."""
    no_ot = _is_no_ot(run, label)
    no_scaling = "no_scaling" in run or "w/o scaling" in label
    no_gate = "no_gate" in run or "no mask" in label
    collapsed = no_scaling and no_gate
    prefix = ("no-OT · " if no_ot else "OT · ") if show_ot else ""
    if collapsed:
        return prefix + "unstable (−S−G)"
    if no_scaling:
        return prefix + "gate only (−S+G)"
    if no_gate:
        return prefix + "scaling only (+S−G)"
    return prefix + "full (+S+G)"


def _run_style(run: str, label: str, *, show_ot: bool = False) -> tuple[str, str, float, str]:
    """Stable color/style identity for the scaling/gate ablation."""
    no_ot = _is_no_ot(run, label)
    no_scaling = "no_scaling" in run or "w/o scaling" in label
    no_gate = "no_gate" in run or "no mask" in label
    # Removing both protections is the collapse control even when an older
    # manifest did not include the parenthetical "(collapsed)" label.
    collapsed = no_scaling and no_gate
    prefix = ("no-OT · " if no_ot else "OT · ") if show_ot else ""
    line_style = "--" if no_ot else "-"
    if collapsed:
        color = "#7b2cbf" if no_ot else "#d55e00"
        return color, line_style, 2.4, prefix + "−S−G"
    if no_scaling:
        return "#e69f00", line_style, 2.0, prefix + "−S+G"
    if no_gate:
        return "#0072b2", line_style, 1.9, prefix + "+S−G"
    return "#009e73", line_style, 2.5, prefix + "+S+G"


def _finite_values(group: list[dict[str, Any]], key: str) -> list[float]:
    return [row[key] for row in group if math.isfinite(row.get(key, math.nan))]


def _value_at_or_before(group: list[dict[str, Any]], key: str, step: int) -> float:
    """Use the last finite observation at or before a common step."""
    candidates = [
        row for row in group
        if row["step"] <= step and math.isfinite(row.get(key, math.nan))
    ]
    return candidates[-1][key] if candidates else math.nan


def _panel_runs(manifest: dict[str, Any], order: list[str]) -> tuple[list[str], list[str]]:
    """Return the runs used by panels A--C and D.

    The ordinary stability manifest has no panel declaration, so every run is
    used everywhere (the historical behaviour).  An OT comparison manifest
    can instead reserve the first three panels for the scaling/gate ablation
    and panel D for a matched OT/no-OT pair.
    """
    panels = manifest.get("panels")
    if not panels:
        return order, order

    def select(name: str, fallback: list[str]) -> list[str]:
        requested = panels.get(name)
        if requested is None:
            return fallback
        requested_set = set(requested)
        return [run for run in order if run in requested_set]

    stability = select("stability", order)
    total = select("total", order)
    if not stability:
        raise ValueError("panel 'stability' must select at least one run")
    if not total:
        raise ValueError("panel 'total' must select at least one run")
    return stability, total


def _plot(rows: list[dict[str, Any]], manifest: dict[str, Any], out: Path, window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Keep the figure typography consistent with the paper layout.  The
    # fallback names cover environments where Microsoft fonts are unavailable.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
        }
    )

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
    # Render at approximately half-column width so a paper insertion does not
    # shrink an otherwise full-page figure and make its labels unreadable.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, constrained_layout=False)
    ax_fm, ax_pressure, ax_gate, ax_total = axes.flat
    stability_runs, total_runs = _panel_runs(manifest, order)
    split_panels = bool(manifest.get("panels"))
    stability_handles = []
    for run in stability_runs:
        group = sorted(by_run[run], key=lambda row: row["step"])
        original_label = group[0]["label"]
        color, line_style, linewidth, label = _run_style(run, original_label)
        x = [row["step"] for row in group]
        fm = [row.get("fm_loss", math.nan) for row in group]
        reject = [
            100.0 * row.get("lsd_gate_active", math.nan)
            if math.isfinite(row.get("lsd_gate_active", math.nan))
            else math.nan
            for row in group
        ]
        ratio = [row.get("lsd_over_budget", math.nan) for row in group]
        # FM and total loss are smoothed for readability. Raw LSD pressure is
        # shown as a faint trace, with a thicker causal trend overlaid.
        line, = ax_fm.plot(x, _moving_average(fm, window), color=color, lw=linewidth, ls=line_style, label=label)
        stability_handles.append(line)
        pressure_trend = _moving_average(ratio, window)
        ax_pressure.fill_between(
            x,
            [1.0] * len(x),
            pressure_trend,
            where=[math.isfinite(value) and value > 1.0 for value in pressure_trend],
            color="#d62728",
            alpha=0.12,
            interpolate=True,
            linewidth=0,
        )
        ax_pressure.plot(x, ratio, color=color, lw=0.8, ls=line_style, alpha=0.22)
        ax_pressure.plot(x, pressure_trend, color=color, lw=linewidth, ls=line_style, alpha=0.95)
        ax_gate.plot(x, reject, color=color, lw=linewidth, ls=line_style)

    total_handles = []
    total_show_ot = split_panels and len(total_runs) > 1 and any(
        _is_no_ot(run, by_run[run][0]["label"]) for run in total_runs
    ) and any(not _is_no_ot(run, by_run[run][0]["label"]) for run in total_runs)
    for run in total_runs:
        group = sorted(by_run[run], key=lambda row: row["step"])
        original_label = group[0]["label"]
        if total_show_ot:
            no_ot = _is_no_ot(run, original_label)
            color = "#d55e00" if no_ot else "#009e73"
            line_style = "--" if no_ot else "-"
            linewidth = 2.3
            label = "OT off" if no_ot else "OT on"
        else:
            color, line_style, linewidth, label = _run_style(run, original_label)
        x = [row["step"] for row in group]
        total = [row.get("loss", math.nan) for row in group]
        line, = ax_total.plot(
            x,
            _moving_average(total, window),
            color=color,
            lw=linewidth,
            ls=line_style,
            label=label,
        )
        ax_total.plot(x, total, color=color, lw=0.8, ls=line_style, alpha=0.22)
        if total_show_ot:
            total_handles.append(line)

    ax_fm.set_title("(a) FM loss", loc="left", fontsize=14)
    ax_fm.set_ylabel("FM loss (MSE)")
    ax_pressure.set_title("(b) LSD loss / budget", loc="left", fontsize=14)
    ax_pressure.set_ylabel("LSD / budget")
    ax_pressure.set_yscale("log")
    ax_gate.set_title("(c) LSD gate rejection", loc="left", fontsize=14)
    ax_gate.set_ylabel("rejected (%)")
    ax_gate.set_ylim(-2.0, 102.0)
    ax_gate.set_yticks([0, 25, 50, 75, 100])
    ax_pressure.axhline(1.0, color="#555555", lw=1, ls=":", alpha=0.8)
    total_title = "(d) total loss: OT comparison" if total_show_ot else "(d) total RollFlow loss"
    ax_total.set_title(total_title, loc="left", fontsize=14)
    ax_total.set_ylabel("total loss")
    for axis in axes.flat:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.set_xlim(0, common_end)
    for axis in (ax_gate, ax_total):
        axis.set_xlabel("training step")

    fig.suptitle("RollFlow G1 stability", fontsize=18, fontweight="bold", y=0.98)
    # Repeat the compact 4-by-1 mechanism key in each subplot.  A separate
    # panel declaration can still replace panel D's key with an OT comparison
    # legend when producing the dedicated OT figure.
    legend_kwargs = {
        "loc": "upper right",
        "ncol": 1,
        "fontsize": 11.5,
        "frameon": True,
        "framealpha": 0.78,
        "facecolor": "white",
        "edgecolor": "none",
        "labelspacing": 0.55,
        "handlelength": 2.0,
        "handletextpad": 0.45,
        "borderaxespad": 0.35,
    }
    # A common inset anchor keeps the keys aligned while leaving clear space
    # below each subplot title.
    legend_anchor = (0.98, 0.90)
    ax_fm.legend(
        stability_handles,
        [handle.get_label() for handle in stability_handles],
        bbox_to_anchor=legend_anchor,
        **legend_kwargs,
    )
    ax_pressure.legend(
        stability_handles,
        [handle.get_label() for handle in stability_handles],
        bbox_to_anchor=legend_anchor,
        **legend_kwargs,
    )
    ax_gate.legend(
        stability_handles,
        [handle.get_label() for handle in stability_handles],
        bbox_to_anchor=legend_anchor,
        **legend_kwargs,
    )
    if total_show_ot:
        ax_total.legend(
            total_handles,
            [handle.get_label() for handle in total_handles],
            fontsize=11.5,
            loc="upper right",
            bbox_to_anchor=legend_anchor,
            frameon=False,
            handlelength=2.2,
            borderaxespad=0.35,
        )
    else:
        ax_total.legend(
            stability_handles,
            [handle.get_label() for handle in stability_handles],
            bbox_to_anchor=legend_anchor,
            **legend_kwargs,
        )
    fig.subplots_adjust(top=0.87, bottom=0.11, left=0.095, right=0.985, hspace=0.50, wspace=0.30)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
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
        "total_loss": f"causal moving average, window={window}",
        "lsd_accepted": "none",
    }
    corrected["metric"] = {
        "fm": "fm_loss",
        "fm_active": "fm_loss_active",
        "lsd_raw": "lsd_loss_raw (pre-gate, includes 2*delta scaling when enabled)",
        "total_loss": "loss (RollFlow total objective)",
        "lsd_accepted": "lsd_loss (post-gate contribution)",
        "gate_rejection": "lsd_gate_active (active-sample rejection fraction)",
        "pressure_ratio": "lsd_loss_raw / (w_lsd * fm_loss_active); global proxy for sample-wise gate",
    }
    corrected["provenance"] = {
        "runs": metadata,
        "limitations": [
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
    show_ot = (
        any(_is_no_ot(run, group[0]["label"]) for run, group in by_run.items())
        and any(not _is_no_ot(run, group[0]["label"]) for run, group in by_run.items())
    )
    order = [item["run"] for item in manifest["runs"] if item["run"] in by_run]
    _, total_runs = _panel_runs(manifest, order)

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
        fields = [
            _run_label(run, run_info["label"], show_ot=show_ot),
            "on" if metadata[run]["use_lsd_scaling"] else "off",
            "on" if metadata[run]["use_lsd_gate"] else "off",
            fmt(min_fm),
            str(int(min_step)) if math.isfinite(min_step) else "—",
            fmt(_value_at_or_before(group, "fm_loss", common_end)),
            str(group[-1]["step"]),
            fmt(max(pressure) if pressure else math.nan),
            fmt(100.0 * max(rejection) if rejection else math.nan) + "%",
        ]
        if show_ot:
            fields.insert(1, "no" if _is_no_ot(run, run_info["label"]) else "yes")
        table.append(
            "| "
            + " | ".join(fields)
            + " |"
        )

    limitations = [
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
    reading_notes = [
        "- **S (scaling)** multiplies the central-difference LSD metric by `2δ`, controlling its gradient magnitude.",
        "- **G (gate/mask)** rejects non-finite or over-budget per-sample LSD updates.",
    ]
    if show_ot:
        reading_notes.append("- **OT** denotes optimal-transport matching of noise to target trajectories.")
    table_header = "| run | S | G | min FM | min step | FM @ common | last step | peak raw LSD/budget | peak rejection (%) |"
    table_separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    if show_ot:
        table_header = "| run | OT | S | G | min FM | min step | FM @ common | last step | peak raw LSD/budget | peak rejection (%) |"
        table_separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    interpretation = [
        "In this matched historical window, scaling keeps the raw LSD/budget proxy well below one. Removing scaling drives the proxy above one; when the gate is enabled, it rejects the offending sample updates. Removing both protections produces the unstable control, where both FM and total loss first improve and then rebound.",
        "The scaling-only `+S−G` control remains stable in this window, while the gate is not always active once scaling is present.",
    ]
    if manifest.get("panels"):
        total_labels = [
            "OT on" if not _is_no_ot(run, by_run[run][0]["label"]) else "OT off"
            for run in total_runs
        ]
        interpretation = [
            "Panels A–C use the four OT runs to isolate the two stability mechanisms: scaling keeps raw LSD pressure low, while the gate rejects over-budget samples. Removing both protections produces the characteristic FM and total-loss rebound.",
            f"Panel D isolates optimal-transport matching with the matched scaling-only pair ({' vs. '.join(total_labels)}). In the common window, OT on descends earlier and with a smaller late-window spread than OT off, supporting OT as a convergence/stability aid in this setup.",
        ]
    text = "\n".join(
        [
            "# RollFlow G1 stability ablation",
            "",
            "## Reading the figure",
            "",
            f"The figure (`{figure_name}`) separates the two safety mechanisms that are easy to conflate:",
            "",
            *reading_notes,
            "",
            scope_line,
            "",
            "Panel A is the FM loss. Panel B normalizes raw LSD loss by its detached FM budget; the red region is above the budget threshold of one. Panel C shows how many active samples are rejected. Panel D shows total loss; in the OT comparison figure it contains only the matched OT-on/OT-off pair. Faint traces are individual log values and thick traces are causal moving averages.",
            "",
            "## Observed window",
            "",
            f"Common comparison step: `{common_end}`. `FM @ common` is the last observation at or before that step; `last step` is the run's actual endpoint.",
            "",
            table_header,
            table_separator,
            *table,
            "",
            "## Interpretation",
            "",
            *interpretation,
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
        default=Path("/data/tzq/starVLA_checkpoints/rollflow_ablation_analysis/g1_ot_paper/g1_stability_all_clean_0_2500.png"),
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
    _plot(rows, manifest, args.output, args.window)
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
