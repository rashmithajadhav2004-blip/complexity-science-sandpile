#!/usr/bin/env python3
"""
2D Sandpile Model for Self-Organized Criticality (SOC)

This script:
1. Simulates a 2D sandpile with 4-neighbor and 8-neighbor connectivity.
2. Records avalanche sizes.
3. Generates figures for the report.
4. Writes a LaTeX table with summary statistics.

Output folder:
sandpile_outputs/
    figures/
    tables/
    data/
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
L = 64                 # Grid size: L x L
STEPS = 30000          # Grain additions per run
BURN_IN = 5000         # Discard early transient
RUNS = 3               # Independent runs per connectivity
SEED = 7               # Base seed
MIN_FIT_SIZE = 2       # Tail-fit lower bound for log-log slope

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "sandpile_outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
DATA_DIR = OUT_DIR / "data"

for d in (FIG_DIR, TABLE_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Model
# -----------------------------
def neighbor_offsets(connectivity: int) -> list[tuple[int, int]]:
    """Return neighbor offsets for 4-neighbor or 8-neighbor connectivity."""
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
    raise ValueError("Connectivity must be 4 or 8.")


def simulate_sandpile(
    L: int,
    steps: int,
    burn_in: int,
    connectivity: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate a 2D sandpile with open boundaries.

    Avalanche size = number of topplings triggered by one grain addition.
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((L, L), dtype=np.int32)
    offsets = neighbor_offsets(connectivity)
    threshold = len(offsets)

    avalanche_sizes: list[int] = []

    for step in range(steps):
        i = rng.integers(0, L)
        j = rng.integers(0, L)
        grid[i, j] += 1

        avalanche = 0
        stack: list[tuple[int, int]] = []

        if grid[i, j] >= threshold:
            stack.append((i, j))

        while stack:
            x, y = stack.pop()

            if grid[x, y] < threshold:
                continue

            # Topple as many times as needed in one visit
            n_topples = grid[x, y] // threshold
            if n_topples <= 0:
                continue

            grid[x, y] -= n_topples * threshold
            avalanche += n_topples

            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < L and 0 <= ny < L:
                    before = grid[nx, ny]
                    grid[nx, ny] += n_topples
                    if before < threshold and grid[nx, ny] >= threshold:
                        stack.append((nx, ny))
                # open boundary: grains leaving the grid are lost

        if step >= burn_in:
            avalanche_sizes.append(avalanche)

    return np.array(avalanche_sizes, dtype=np.int64)


# -----------------------------
# Analysis helpers
# -----------------------------
def size_frequency(sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique sizes and their frequencies, excluding zero avalanches."""
    positive = sizes[sizes > 0]
    if positive.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    vals, counts = np.unique(positive, return_counts=True)
    return vals, counts


def tail_slope(values: np.ndarray, counts: np.ndarray, min_size: int = 2) -> float:
    """Estimate slope of log-log frequency distribution for the tail."""
    mask = values >= min_size
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x = np.log10(values[mask])
    y = np.log10(counts[mask])
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def summarize(sizes: np.ndarray, connectivity: int) -> dict:
    """Compute summary statistics for one connectivity case."""
    positive = sizes[sizes > 0]
    vals, counts = size_frequency(sizes)

    summary = {
        "connectivity": connectivity,
        "total_events": int(sizes.size),
        "nonzero_events": int(positive.size),
        "nonzero_fraction": float(positive.size / sizes.size) if sizes.size else 0.0,
        "mean_size": float(np.mean(positive)) if positive.size else 0.0,
        "median_size": float(np.median(positive)) if positive.size else 0.0,
        "max_size": int(np.max(positive)) if positive.size else 0,
        "tail_slope": tail_slope(vals, counts, MIN_FIT_SIZE),
        "values": vals,
        "counts": counts,
    }
    return summary


def combine_runs(connectivity: int) -> np.ndarray:
    """Run several independent simulations and concatenate avalanche sizes."""
    all_sizes = []
    for r in range(RUNS):
        seed = SEED + 1000 * connectivity + r
        sizes = simulate_sandpile(L=L, steps=STEPS, burn_in=BURN_IN,
                                  connectivity=connectivity, seed=seed)
        all_sizes.append(sizes)
    return np.concatenate(all_sizes)


# -----------------------------
# Plotting
# -----------------------------
def plot_distribution(results: dict[int, dict]) -> None:
    plt.figure(figsize=(7.2, 5.2))
    for conn in sorted(results):
        vals = results[conn]["values"]
        counts = results[conn]["counts"]
        if vals.size == 0:
            continue
        plt.plot(vals, counts, marker="o", linestyle="-", markersize=3, label=f"{conn}-neighbor")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Avalanche size")
    plt.ylabel("Frequency")
    plt.title("Avalanche size-frequency distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "avalanche_distribution.png", dpi=300)
    plt.close()


def plot_connectivity_effect(results: dict[int, dict]) -> None:
    conns = sorted(results)
    means = [results[c]["mean_size"] for c in conns]

    plt.figure(figsize=(6.2, 4.6))
    plt.bar([str(c) for c in conns], means)
    plt.xlabel("Connectivity")
    plt.ylabel("Mean avalanche size")
    plt.title("Effect of connectivity on avalanche size")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "connectivity_mean_size.png", dpi=300)
    plt.close()


# -----------------------------
# Output files
# -----------------------------
def write_summary_table(results: dict[int, dict]) -> None:
    tex_path = TABLE_DIR / "summary_table.tex"
    csv_path = DATA_DIR / "summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "connectivity",
            "total_events",
            "nonzero_events",
            "nonzero_fraction",
            "mean_size",
            "median_size",
            "max_size",
            "tail_slope",
        ])
        for conn in sorted(results):
            r = results[conn]
            writer.writerow([
                r["connectivity"],
                r["total_events"],
                r["nonzero_events"],
                f'{r["nonzero_fraction"]:.6f}',
                f'{r["mean_size"]:.6f}',
                f'{r["median_size"]:.6f}',
                r["max_size"],
                f'{r["tail_slope"]:.6f}',
            ])

    lines = []
    lines.append(r"\begin{tabular}{rrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Connectivity & Events & Nonzero & Nonzero fraction & Mean size & Median size & Max size & Tail slope \\")
    lines.append(r"\midrule")
    for conn in sorted(results):
        r = results[conn]
        lines.append(
            f'{r["connectivity"]} & {r["total_events"]} & {r["nonzero_events"]} & '
            f'{r["nonzero_fraction"]:.4f} & {r["mean_size"]:.3f} & {r["median_size"]:.3f} & '
            f'{r["max_size"]} & {r["tail_slope"]:.3f} \\\\'
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path.write_text("\n".join(lines), encoding="utf-8")


def write_readme_hint(results: dict[int, dict]) -> None:
    txt = []
    txt.append("Run order:")
    txt.append("1. python sandpile_soc.py")
    txt.append("2. compile the LaTeX report")
    txt.append("")
    txt.append("Generated files:")
    txt.append(str(FIG_DIR / "avalanche_distribution.png"))
    txt.append(str(FIG_DIR / "connectivity_mean_size.png"))
    txt.append(str(TABLE_DIR / "summary_table.tex"))
    txt.append(str(DATA_DIR / "summary.csv"))
    (OUT_DIR / "run_instructions.txt").write_text("\n".join(txt), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    results = {}

    for conn in (4, 8):
        sizes = combine_runs(conn)
        np.save(DATA_DIR / f"avalanche_sizes_{conn}n.npy", sizes)
        results[conn] = summarize(sizes, conn)

    plot_distribution(results)
    plot_connectivity_effect(results)
    write_summary_table(results)
    write_readme_hint(results)

    print("Done.")
    for conn in sorted(results):
        r = results[conn]
        print(
            f"{conn}-neighbor: mean={r['mean_size']:.3f}, "
            f"median={r['median_size']:.3f}, max={r['max_size']}, "
            f"tail_slope={r['tail_slope']:.3f}"
        )
    print(f"Outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()