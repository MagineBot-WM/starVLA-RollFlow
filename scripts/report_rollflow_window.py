#!/usr/bin/env python3
"""Create a reproducible 0..N RollFlow analysis bundle.

The log parser in :mod:`analyze_rollflow_ablation_logs` intentionally keeps
all records.  This small companion filters one run to a requested window and
emits a CSV, JSON summary, Markdown notes, and a responsive D3 HTML fragment.
It is safe to run while ``train.log`` is still being appended: the caller
only needs to refresh the parser CSV first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


FOCUS = (
    "fm_loss",
    "fm_loss_active",
    "lsd_loss",
    "lsd_loss_raw",
    "lsd_loss_metric",
    "lsd_gate_active",
    "lsd_gate_enabled",
    "lsd_scaling_enabled",
    "lsd_finite_frac",
    "lsd_keep_frac",
    "active_lsd_frac",
    "teacher_clip_frac",
    "p_fm",
    "fm_only_frac",
    "source_ratio",
    "s_mean",
    "t_mean",
)


def number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    if value in {"True", "true"}:
        return 1.0
    if value in {"False", "false"}:
        return 0.0
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def read_rows(path: Path, max_step: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            try:
                step = int(float(raw.get("step", "")))
            except ValueError:
                continue
            if step < 0 or step > max_step:
                continue
            row: dict[str, Any] = {
                "step": step,
                "embodiment": raw.get("embodiment", ""),
            }
            for key in FOCUS:
                row[key] = number(raw.get(key))
            rows.append(row)
    rows.sort(key=lambda row: (row["step"], row["embodiment"]))
    return rows


def stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def summarize(rows: list[dict[str, Any]], requested_max: int, source_csv: Path) -> dict[str, Any]:
    embodiments = sorted({str(row["embodiment"]) for row in rows})
    by_embodiment: dict[str, Any] = {}
    for embodiment in embodiments:
        selected = [row for row in rows if row["embodiment"] == embodiment]
        by_embodiment[embodiment] = {
            "records": len(selected),
            "first_step": selected[0]["step"],
            "last_step": selected[-1]["step"],
            "metrics": {key: stats(selected, key) for key in FOCUS if stats(selected, key) is not None},
        }

    observed = max((int(row["step"]) for row in rows), default=None)
    result: dict[str, Any] = {
        "source_csv": str(source_csv),
        "requested_window": [0, requested_max],
        "observed_window": [min((int(row["step"]) for row in rows), default=None), observed],
        "observed_max_step": observed,
        "complete_through_requested_max": observed is not None and observed >= requested_max,
        "records": len(rows),
        "embodiments": by_embodiment,
    }

    # Equal-width bins are useful for a short progress report while the run is
    # live.  Keep them based on observed steps; a future refresh will replace
    # these values with the newly available records.
    if rows:
        lower = min(int(row["step"]) for row in rows)
        upper = max(int(row["step"]) for row in rows)
        edges = [lower, lower + (upper - lower + 1) // 3, lower + 2 * (upper - lower + 1) // 3, upper + 1]
        bins: list[dict[str, Any]] = []
        for left, right in zip(edges[:-1], edges[1:]):
            selected = [row for row in rows if left <= int(row["step"]) < right]
            if not selected:
                continue
            entry: dict[str, Any] = {"step_range": [left, min(right - 1, upper)], "records": len(selected)}
            for key in ("fm_loss", "lsd_loss_raw", "lsd_loss_metric", "lsd_gate_active", "lsd_keep_frac", "active_lsd_frac", "p_fm"):
                value = stats(selected, key)
                if value is not None:
                    entry[key] = {name: value[name] for name in ("first", "last", "minimum", "maximum", "mean")}
            bins.append(entry)
        result["observed_bins"] = bins
    return result


def write_filtered_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["step", "embodiment", *FOCUS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    observed = summary.get("observed_max_step")
    requested = summary["requested_window"][1]
    complete = summary["complete_through_requested_max"]
    lines = [
        "# RollFlow scale + mask window report",
        "",
        f"- Requested window: steps 0–{requested}",
        f"- Observed at report time: {summary['observed_window'][0]}–{observed if observed is not None else 'n/a'}",
        f"- Complete through {requested}: {'yes' if complete else 'no (the run is still writing)'}",
        f"- Records: {len(rows)}; embodiment(s): {', '.join(sorted({str(row['embodiment']) for row in rows})) or 'n/a'}",
        "",
        "## What is measured",
        "",
        "`fm_loss` is the instantaneous flow-matching loss. `lsd_loss_raw` is the scaled LSD value before the budget mask, while `lsd_loss` is the actual contribution after masking. With delta=0.01 and scaling enabled, `lsd_loss_raw` should equal `0.02 * lsd_loss_metric`; `lsd_gate_active`/`lsd_keep_frac` expose the budget mask, while `p_fm` and `active_lsd_frac` expose the curriculum.",
        "",
        "## Observed statistics",
        "",
        "| Metric | First | Last | Min | Max | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selected = [row for row in rows if row.get("embodiment") != "__global__"]
    for key, label in (
        ("fm_loss", "FM loss"),
        ("fm_loss_active", "FM loss (active)"),
        ("lsd_loss_raw", "LSD raw (scaled, pre-mask)"),
        ("lsd_loss", "LSD contribution (post-mask)"),
        ("lsd_loss_metric", "LSD metric (pre-scale)"),
        ("lsd_gate_active", "Gate active"),
        ("lsd_keep_frac", "Keep fraction"),
        ("active_lsd_frac", "Active LSD fraction"),
        ("p_fm", "FM curriculum probability"),
        ("teacher_clip_frac", "Teacher clip fraction"),
    ):
        value = stats(selected, key)
        if value is not None:
            lines.append("| {} | {} | {} | {} | {} | {} |".format(label, *(fmt(value[name]) for name in ("first", "last", "minimum", "maximum", "mean"))))

    if selected:
        ratios = []
        for row in selected:
            raw = row.get("lsd_loss_raw")
            metric = row.get("lsd_loss_metric")
            if raw is not None and metric is not None and metric > 1e-12 and raw > 0:
                ratios.append(raw / metric)
        lines.extend(["", "## Checks", ""])
        if ratios:
            lines.append(f"- Scaling check on {len(ratios)} non-zero records: raw/metric = {fmt(min(ratios), 8)}–{fmt(max(ratios), 8)} (target `2*delta = 0.02`).")
        finite = [row.get("lsd_finite_frac") for row in selected if row.get("lsd_finite_frac") is not None]
        if finite:
            lines.append(f"- Finite LSD fraction: min={fmt(min(finite))}, max={fmt(max(finite))}.")
        gated = [int(row["step"]) for row in selected if row.get("lsd_gate_active") == 1.0]
        if gated:
            lines.append(f"- Budget gate active at logged steps: {', '.join(map(str, gated))}; these snapshots should be read as deliberate LSD suppression, not NaN/Inf failure.")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Treat this as a live-window result until `observed_max_step` reaches 2000. The HTML intentionally leaves the unobserved portion of the requested axis blank.",
        "- A stable FM curve together with finite LSD metrics and a raw/metric ratio near 0.02 is consistent with the scaled-LSD design reducing the self-distillation gradient pressure.",
        "- Gate activity is a separate mechanism: it removes over-budget LSD rows from the update. The report distinguishes this from finite-value checks.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def html_fragment(data: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    requested = int(summary["requested_window"][1])
    observed = summary.get("observed_max_step")
    observed_js = "null" if observed is None else str(int(observed))
    title = "RollFlow scale + mask | pure Libero | steps 0–{}".format(requested)
    return f'''<div class="rollflow-window-report" data-report="rollflow-scale-mask-0-2000">
  <style>
    .rollflow-window-report {{
      color: var(--foreground);
      background: var(--background);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      width: 100%;
      box-sizing: border-box;
    }}
    .rollflow-window-report h2 {{ margin: 0 0 3px; font-size: 18px; }}
    .rollflow-window-report .sub {{ color: var(--muted-foreground); margin-bottom: 12px; }}
    .rollflow-window-report .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .rollflow-window-report .panel {{ min-width: 0; border: 1px solid var(--border); border-radius: 8px; padding: 7px; }}
    .rollflow-window-report svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
    .rollflow-window-report .axis path, .rollflow-window-report .axis line {{ stroke: var(--border); }}
    .rollflow-window-report .axis text, .rollflow-window-report .label, .rollflow-window-report .legend {{ fill: var(--foreground); font-size: 11px; }}
    .rollflow-window-report .gridline {{ stroke: var(--border); stroke-dasharray: 2 3; opacity: .55; }}
    .rollflow-window-report .observed-line {{ stroke: var(--border); stroke-dasharray: 5 4; }}
    .rollflow-window-report .raw {{ opacity: .30; }}
    .rollflow-window-report .series-1 {{ stroke: var(--viz-series-1); fill: none; }}
    .rollflow-window-report .series-2 {{ stroke: var(--viz-series-2); fill: none; }}
    .rollflow-window-report .series-3 {{ stroke: var(--viz-series-3); fill: none; }}
    .rollflow-window-report .series-4 {{ stroke: var(--viz-series-4); fill: none; }}
    .rollflow-window-report .dot-1 {{ fill: var(--viz-series-1); }}
    .rollflow-window-report .dot-2 {{ fill: var(--viz-series-2); }}
    .rollflow-window-report .dot-3 {{ fill: var(--viz-series-3); }}
    .rollflow-window-report .dot-4 {{ fill: var(--viz-series-4); }}
    @media (max-width: 720px) {{ .rollflow-window-report .grid {{ grid-template-columns: 1fr; }} .rollflow-window-report {{ padding: 9px; }} }}
  </style>
  <h2>{title}</h2>
  <div class="sub">Observed through step {observed if observed is not None else 'n/a'}; the remaining requested range is intentionally blank. Lines show raw points (faint) and a 7-record rolling mean (bold).</div>
  <div class="grid">
    <div class="panel"><svg data-panel="fm" viewBox="0 0 560 300" role="img" aria-label="FM loss over optimization step"></svg></div>
    <div class="panel"><svg data-panel="raw" viewBox="0 0 560 300" role="img" aria-label="Raw and post-mask scaled LSD contribution over optimization step"></svg></div>
    <div class="panel"><svg data-panel="metric" viewBox="0 0 560 300" role="img" aria-label="LSD metric before scaling over optimization step"></svg></div>
    <div class="panel"><svg data-panel="diag" viewBox="0 0 560 300" role="img" aria-label="Mask and curriculum diagnostics over optimization step"></svg></div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
  (() => {{
    const data = {payload};
    const requestedMax = {requested};
    const observedMax = {observed_js};
    const margin = {{top: 34, right: 18, bottom: 42, left: 54}};
    const width = 560 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;
    const clean = key => data.map(d => ({{step: +d.step, value: d[key] == null ? null : +d[key]}}));
    const rolling = (values, span=7) => values.map((d, i) => {{
      const sample = values.slice(Math.max(0, i-span+1), i+1).filter(x => x.value != null);
      return {{step: d.step, value: sample.length ? d3.mean(sample, x => x.value) : null}};
    }});
    const draw = (selector, title, yLabel, series, domainOverride=null) => {{
      const svg = d3.select(selector);
      const root = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);
      const x = d3.scaleLinear().domain([0, requestedMax]).range([0, width]);
      const all = series.flatMap(s => s.values.map(d => d.value)).filter(v => v != null && Number.isFinite(v));
      let ymin = domainOverride ? domainOverride[0] : (all.length ? d3.min(all) : 0);
      let ymax = domainOverride ? domainOverride[1] : (all.length ? d3.max(all) : 1);
      if (ymin === ymax) {{ ymax = ymin + (Math.abs(ymin) || 1); }}
      if (!domainOverride) {{ const pad = (ymax-ymin)*0.10 || 0.01; ymin = Math.max(0, ymin-pad); ymax += pad; }}
      const y = d3.scaleLinear().domain([ymin, ymax]).nice().range([height, 0]);
      root.append('text').attr('class','label').attr('x',0).attr('y',-13).text(title);
      root.append('text').attr('class','label').attr('transform','rotate(-90)').attr('x',-height/2).attr('y',-40).attr('text-anchor','middle').text(yLabel);
      const yTicks = y.ticks(5);
      root.selectAll('.gridline').data(yTicks).join('line').attr('class','gridline').attr('x1',0).attr('x2',width).attr('y1',d=>y(d)).attr('y2',d=>y(d));
      root.append('g').attr('class','axis').attr('transform',`translate(0,${{height}})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format('~s')));
      root.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3~g')));
      root.append('text').attr('class','label').attr('x',width/2).attr('y',height+35).attr('text-anchor','middle').text('optimization step');
      if (observedMax != null && observedMax < requestedMax) {{
        root.append('line').attr('class','observed-line').attr('x1',x(observedMax)).attr('x2',x(observedMax)).attr('y1',0).attr('y2',height);
        root.append('text').attr('class','label').attr('x',Math.min(x(observedMax)+4,width-4)).attr('y',12).attr('text-anchor',x(observedMax)>width-70?'end':'start').text(`observed ${{observedMax}}`);
      }}
      const line = d3.line().defined(d=>d.value != null).x(d=>x(d.step)).y(d=>y(d.value));
      series.forEach((s, idx) => {{
        const n = idx+1;
        root.append('path').datum(s.values).attr('class',`series-${{n}} raw`).attr('d',line).attr('stroke-width',1.2);
        root.append('path').datum(rolling(s.values)).attr('class',`series-${{n}}`).attr('d',line).attr('stroke-width',2.4);
        root.selectAll(`.dot-${{n}}`).data(s.values.filter(d=>d.value != null)).join('circle').attr('class',`dot-${{n}}`).attr('cx',d=>x(d.step)).attr('cy',d=>y(d.value)).attr('r',1.5);
        const last = [...s.values].reverse().find(d=>d.value != null);
        if (last) root.append('text').attr('class','legend').attr('x',Math.min(x(last.step)+6,width-70)).attr('y',y(last.value)-5-idx*14).text(`${{s.label}} ${{d3.format('.3~g')(last.value)}}`);
      }});
    }};
    draw('[data-panel="fm"]','FM loss','loss',[{{label:'fm_loss', values:clean('fm_loss')}}]);
    draw('[data-panel="raw"]','LSD raw vs after mask','loss',[
      {{label:'raw (pre-mask)', values:clean('lsd_loss_raw')}},
      {{label:'after mask', values:clean('lsd_loss')}}
    ]);
    draw('[data-panel="metric"]','LSD metric (pre-scale)','metric',[{{label:'lsd_loss_metric', values:clean('lsd_loss_metric')}}]);
    draw('[data-panel="diag"]','Mask + curriculum diagnostics','fraction',[
      {{label:'keep_frac', values:clean('lsd_keep_frac')}},
      {{label:'active_lsd', values:clean('active_lsd_frac')}},
      {{label:'p_fm', values:clean('p_fm')}}
    ],[0,1]);
  }})();
  </script>
</div>\n'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV emitted by analyze_rollflow_ablation_logs.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=2000)
    args = parser.parse_args()
    if args.max_step <= 0:
        parser.error("--max-step must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv, args.max_step)
    summary = summarize(rows, args.max_step, args.csv)
    write_filtered_csv(args.output_dir / f"metrics_0_{args.max_step}.csv", rows)
    (args.output_dir / f"summary_0_{args.max_step}.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.output_dir / f"analysis_0_{args.max_step}.md", summary, rows)
    html = html_fragment(rows, summary)
    (args.output_dir / f"scale-mask-0-{args.max_step}.html").write_text(html, encoding="utf-8")
    print(f"Wrote {len(rows)} records through observed step {summary.get('observed_max_step')} to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
