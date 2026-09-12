#!/usr/bin/env python3
"""Create a paper-ready G1 loss figure for the scaling/mask ablation.

OT is held fixed as the default transport coupling; it is not a plotted
factor.  The figure compares the four directly comparable finite-difference
runs while varying only the two stabilisers of interest: the ``2*h`` LSD
scaling and the aggregate LSD mask/gate.  The plotted LSD metric is
``lsd_loss`` (the contribution after the mask/gate), i.e. the value that
actually enters the optimisation loss.

The parser is shared with :mod:`analyze_rollflow_ablation_logs`, so this
script can be run while logs are still being appended.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_rollflow_ablation_logs import (  # noqa: E402
    merge_metric_rows,
    moving_average,
    parse_log,
)


RUN_SPECS = (
    {
        "run": "libero10hz_pick_up_the_apple_ablation_ot_gate_55k_20k_v1",
        "label": "Full design (scaling + mask)",
        "color": "#009E73",
    },
    {
        "run": "libero10hz_pick_up_the_apple_ablation_ot_no_gate_55k_20k_v1",
        "label": "w/o mask",
        "color": "#0072B2",
    },
    {
        "run": "libero10hz_pick_up_the_apple_ablation_ot_gate_no_scaling_55k_20k_v1",
        "label": "w/o scaling",
        "color": "#CC79A7",
    },
    {
        "run": "libero10hz_pick_up_the_apple_ablation_ot_no_gate_no_scaling_55k_20k_v1",
        "label": "w/o scaling & mask",
        "color": "#D55E00",
    },
)


def resolve_log(path: Path) -> Path:
    if path.is_dir():
        path = path / "train.log"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_rows(log_path: Path, max_step: int) -> list[dict[str, Any]]:
    rows = merge_metric_rows(parse_log(resolve_log(log_path)))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("embodiment") != "agibot-g1":
            continue
        step = int(row["step"])
        if step > max_step:
            continue
        # Keep the OT filter explicit in the data path.  It prevents an
        # accidental non-OT run from entering a figure labelled as OT-only.
        if row.get("use_ot") is not True:
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: int(row["step"]))


def write_long_csv(path: Path, all_rows: list[dict[str, Any]]) -> None:
    fields = ["run", "label", "step", "embodiment", "use_ot", "fm_loss", "lsd_loss", "lsd_loss_raw", "lsd_gate_active", "lsd_gate_enabled", "lsd_scaling_enabled"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def make_figure(
    path_png: Path,
    path_pdf: Path,
    datasets: list[tuple[dict[str, str], list[dict[str, Any]]]],
    window: int,
    max_step: int,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharex=True)
    metric_info = (
        ("fm_loss", "FM loss", "(a)"),
        ("lsd_loss", "LSD loss after mask", "(b)"),
    )

    for axis, (metric, ylabel, panel) in zip(axes, metric_info):
        for spec, rows in datasets:
            steps = [int(row["step"]) for row in rows if isinstance(row.get(metric), (int, float))]
            values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
            if not steps:
                continue
            # A faint raw trace documents the observed points, while the
            # thicker moving average keeps the paper figure readable.
            axis.plot(steps, values, color=spec["color"], alpha=0.15, linewidth=0.65)
            smooth = moving_average(values, window)
            axis.plot(
                steps,
                smooth,
                color=spec["color"],
                linewidth=2.0,
                label=spec["label"],
                solid_capstyle="round",
            )

        axis.set_xlim(0, max_step)
        axis.set_xlabel("Optimization step")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#B0B0B0", alpha=0.22, linewidth=0.55)
        axis.text(0.02, 1.03, panel, transform=axis.transAxes, fontweight="bold", va="bottom")

    # LSD spans nearly three orders of magnitude between the stable and
    # unstable ablations.  Symlog keeps small stable values and the runaway
    # no-scaling/no-mask trajectory visible in one panel.
    axes[1].set_yscale("symlog", linthresh=1e-3, linscale=1.0)
    axes[1].set_ylabel("LSD loss after mask (symlog)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.22), ncol=2, frameon=False, handlelength=2.5, columnspacing=1.8)
    figure.subplots_adjust(wspace=0.28, bottom=0.25, left=0.08, right=0.99, top=0.92)
    figure.savefig(path_png, dpi=350)
    figure.savefig(path_pdf)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("/data/tzq/starVLA_checkpoints"),
        help="directory containing the four ablation run directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/tzq/starVLA_checkpoints/rollflow_ablation_analysis/g1_ot_paper"),
    )
    parser.add_argument("--max-step", type=int, default=2000)
    parser.add_argument("--window", type=int, default=9, help="moving-average window in logged points")
    args = parser.parse_args()
    if args.max_step <= 0:
        parser.error("--max-step must be positive")
    if args.window <= 0:
        parser.error("--window must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[tuple[dict[str, str], list[dict[str, Any]]]] = []
    long_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "filter": {"embodiment": "agibot-g1", "use_ot": True, "max_step": args.max_step},
        "smoothing": {"method": "causal moving average", "window_logged_points": args.window},
        "metric": {"fm": "fm_loss", "lsd": "lsd_loss (post-mask/gate contribution)"},
        "ot": "fixed default (all selected logs have use_ot=true; not a plotted factor)",
        "runs": [],
    }
    for spec in RUN_SPECS:
        log_path = resolve_log(args.checkpoints_dir / spec["run"])
        rows = load_rows(log_path, args.max_step)
        if not rows:
            raise RuntimeError(f"no OT G1 rows found in {log_path}")
        datasets.append((spec, rows))
        manifest["runs"].append(
            {
                "run": spec["run"],
                "label": spec["label"],
                "log": str(log_path),
                "first_step": int(rows[0]["step"]),
                "last_step": int(rows[-1]["step"]),
                "records": len(rows),
            }
        )
        for row in rows:
            long_rows.append(
                {
                    "run": spec["run"],
                    "label": spec["label"],
                    **row,
                }
            )

    write_long_csv(args.output_dir / "g1_ot_loss_rows.csv", long_rows)
    (args.output_dir / "g1_ot_loss_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_figure(
        args.output_dir / "g1_stability_scaling_mask_0_2000.png",
        args.output_dir / "g1_stability_scaling_mask_0_2000.pdf",
        datasets,
        args.window,
        args.max_step,
    )
    print(f"Wrote {len(long_rows)} G1 OT rows from {len(datasets)} runs")
    print(f"PNG: {args.output_dir / 'g1_stability_scaling_mask_0_2000.png'}")
    print(f"PDF: {args.output_dir / 'g1_stability_scaling_mask_0_2000.pdf'}")
    print(f"CSV/manifest: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
