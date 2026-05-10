"""Post-collection validation: overlay recorded episode trajectories on the figure-8 map.

Reads pose_x / pose_y from each episode's parquet file, overlays them on the expert
reference paths, and saves a PNG. Exits 1 if too many episodes have high track error.

Usage:
    python scripts/validate_figure8_collection.py --data_dir data/05102026_ACT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Figure-8 waypoints — duplicated from mountain_cliff_scene.py so no Isaac Lab dep needed.
_COMMON_ARM = ((0.00, -1.45), (0.00, -0.68), (0.00, 0.10), (0.00, 0.88))
_LEFT_LOOP = ((0.00, 0.88), (-0.91, 0.72), (-1.58, 0.30), (-1.82, -0.28),
              (-1.58, -0.87), (-0.91, -1.29), (0.00, -1.45))
_RIGHT_LOOP = ((0.00, 0.88), (0.91, 0.72), (1.58, 0.30), (1.82, -0.28),
               (1.58, -0.87), (0.91, -1.29), (0.00, -1.45))

FIGURE8_LEFT_ROUTE = _COMMON_ARM + _LEFT_LOOP[1:]
FIGURE8_RIGHT_ROUTE = _COMMON_ARM + _RIGHT_LOOP[1:]
TASK_ROUTES = {"go_left": FIGURE8_LEFT_ROUTE, "go_right": FIGURE8_RIGHT_ROUTE}


def _read_task(parquet_path: Path) -> str:
    try:
        df = pd.read_parquet(parquet_path, columns=["task"])
        return str(df["task"].iloc[0])
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate figure-8 collection trajectories.")
    ap.add_argument("--data_dir", required=True,
                    help="Pipeline data root (e.g. data/05102026_ACT)")
    ap.add_argument("--out", default=None,
                    help="Output PNG path (default: <data_dir>/validation_tracks.png)")
    ap.add_argument("--max_track_error", type=float, default=0.12,
                    help="Mean track error (m) above which an episode is flagged high-error")
    ap.add_argument("--fail_fraction", type=float, default=0.30,
                    help="Exit 1 if this fraction of episodes exceeds --max_track_error")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else data_dir / "validation_tracks.png"
    parquet_files = sorted(data_dir.glob("**/episode_*/data.parquet"))

    if not parquet_files:
        print(f"[validate] ERROR: no parquet files found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[validate] found {len(parquet_files)} episodes in {data_dir}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    task_color = {"go_left": "tab:blue", "go_right": "tab:orange"}
    summary: list[tuple[str, str, float]] = []

    for ax, task_name in zip(axes, ("go_left", "go_right")):
        route = np.array(TASK_ROUTES[task_name])
        ax.plot(route[:, 0], route[:, 1], "k--", lw=2.5, label="expert path", zorder=4)
        ax.scatter(route[0, 0], route[0, 1], c="green", s=80, zorder=6, label="start")

        task_files = [f for f in parquet_files if _read_task(f) == task_name]
        for f in task_files:
            try:
                df = pd.read_parquet(f, columns=["pose_x", "pose_y", "track_error"])
            except Exception:
                continue
            xs = df["pose_x"].to_numpy()
            ys = df["pose_y"].to_numpy()
            mean_err = float(df["track_error"].mean()) if "track_error" in df.columns else float("nan")
            if not np.all(np.isnan(xs)):
                ax.plot(xs, ys, alpha=0.55, lw=0.9, color=task_color[task_name])
            summary.append((f.parent.name, task_name, mean_err))

        label_name = task_name.replace("_", " ")
        ax.set_title(f"{label_name}  ({len(task_files)} episodes)", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x  (m)")
        ax.set_ylabel("y  (m)")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Figure-8 Collection Validation — {len(parquet_files)} episodes",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[validate] saved track overlay → {out_path}", flush=True)

    errors = [e for _, _, e in summary if not (e != e)]  # drop nan
    mean_err_all = float(np.mean(errors)) if errors else float("nan")
    max_err_all = float(np.max(errors)) if errors else float("nan")
    high_error = [(ep, t, e) for ep, t, e in summary if e > args.max_track_error]

    print()
    print(f"{'episode':<22} {'task':<12} {'mean_err_m':>12}")
    for ep, t, e in summary:
        flag = "  *** HIGH" if e > args.max_track_error else ""
        print(f"{ep:<22} {t:<12} {e:>12.4f}{flag}")
    print()
    print(
        f"[validate] episodes={len(summary)}  mean_error={mean_err_all:.4f}m  "
        f"max_error={max_err_all:.4f}m  high_error={len(high_error)}"
    )

    fail_thresh = int(len(summary) * args.fail_fraction)
    if len(high_error) > fail_thresh:
        print(
            f"[validate] FAIL: {len(high_error)}/{len(summary)} episodes exceed "
            f"{args.max_track_error}m (threshold {args.fail_fraction*100:.0f}%). "
            "Check physics / controller settings.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[validate] Track error within acceptable limits. ✓", flush=True)


if __name__ == "__main__":
    main()
