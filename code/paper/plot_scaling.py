# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Figure 2: prompt length and NDCG@5 under the history and slate interventions.

Panel (a) plots mean stage-1 prompt tokens against history length for the SID and text renderings
(diagnosis/prompt-tokens.json, written by measure_prompt_tokens.py). Panels (b) and (c) plot NDCG@5
against history length k (N=30) and slate size N (k=64) for Base 8B (SID), RLVR (NDCG@5) 8B (SID),
the Gemini-3.6-Flash text reference and seeded Random. Every value is read from the result files.

    python code/paper/plot_scaling.py [--results downloads/rerankid/results] [--out PATH]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "downloads/rerankid/results"
K_GRID = [0, 4, 8, 16, 32, 64, 128, 256]
N_GRID = [10, 20, 30, 50, 100]

SERIES = {
    "base_k": "diagnosis/onerec-8b/sid/k{k}_n30.json",
    "base_n": "diagnosis/onerec-8b/sid/k64_n{n}.json",
    "rlvr_k": "posttraining/rlvr-ndcg5/onerec-8b/test/k{k}_n30.json",
    "rlvr_n": "posttraining/rlvr-ndcg5/onerec-8b/test/k64_n{n}.json",
    "gem_k": "diagnosis/gemini-3.6-flash/k{k}_n30.json",
    "gem_n": "diagnosis/gemini-3.6-flash/k64_n{n}.json",
    "random_n": "diagnosis/random/k64_n{n}.json",
}
TOKENS = "diagnosis/prompt-tokens.json"

LINE_STYLES = {
    "Base-8B (SID)": dict(color="#56B4E9", marker="o", linestyle="--", linewidth=1.5,
                          markersize=2.6),
    "RLVR-8B (SID)": dict(color="#0072B2", marker="s", linestyle="-", linewidth=1.7,
                          markersize=2.4),
    "Gemini-3.6 (text)": dict(color="#D55E00", marker="^", linestyle="-.", linewidth=1.4,
                              markersize=2.8),
    "Random": dict(color="#555555", linestyle=":", linewidth=1.1),
}

TOKEN_STYLES = {
    "SID": dict(color="#0072B2", marker="D", linestyle="-", linewidth=1.7, markersize=2.5),
    "Text": dict(color="#D55E00", marker="^", linestyle="-.", linewidth=1.4, markersize=2.8),
}


def draw(axis, positions, values, label, style, sigma=0.0):
    """Smoothed trend curve over the measured values (markers and a faint polyline).

    sigma > 0 applies a Gaussian filter with pinned endpoints before monotone interpolation.
    """
    pos = np.asarray(positions, dtype=float)
    val = np.asarray(values, dtype=float)
    anchors = val
    if sigma > 0:
        anchors = gaussian_filter1d(val, sigma, mode="nearest")
        anchors[0], anchors[-1] = val[0], val[-1]
    dense = np.linspace(pos[0], pos[-1], (len(pos) - 1) * 20 + 1)
    smooth = PchipInterpolator(pos, anchors)(dense)
    axis.plot(pos, val, color=style["color"], linewidth=0.8, alpha=0.3,
              linestyle=style.get("linestyle", "-"))
    line_style = {k: v for k, v in style.items() if k not in ("marker", "markersize")}
    axis.plot(dense, smooth, label=label, **line_style)
    axis.plot(pos, val, color=style["color"], linestyle="none",
              marker=style.get("marker"), markersize=style.get("markersize", 3.5))


def ndcg5(results, relative):
    with (results / relative).open() as handle:
        return float(json.load(handle)["metrics"]["ndcg@5"])


def series(results, key, axis, values):
    return [ndcg5(results, SERIES[key].format(**{axis: value})) for value in values]


def main():
    parser = argparse.ArgumentParser(description="Render the scaling figure (Fig. 2).")
    parser.add_argument("--results", type=Path, default=RESULTS, help="results root")
    parser.add_argument("--tokens", type=Path, default=None,
                        help=f"prompt-token file (default: <results>/{TOKENS})")
    parser.add_argument("--out", type=Path, default=ROOT / "figures/scaling.pdf")
    parser.add_argument("--no-tex", action="store_true",
                        help="use matplotlib text rendering instead of LaTeX with newtx")
    args = parser.parse_args()
    results = args.results

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    if not args.no_tex:
        matplotlib.rcParams.update({"text.usetex": True,
                                    "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}"})

    with open(args.tokens or results / TOKENS) as handle:
        tokens = json.load(handle)
    assert tokens["k_grid"] == K_GRID

    fig, (ax_t, ax_k, ax_n) = plt.subplots(1, 3, figsize=(7.0, 2.0), constrained_layout=True)
    k_pos = list(range(len(K_GRID)))
    n_pos = list(range(len(N_GRID)))

    sid_tokens = tokens["mean_prompt_tokens"]["sid"]
    text_tokens = tokens["mean_prompt_tokens"]["text"]
    draw(ax_t, k_pos, sid_tokens, "SID", TOKEN_STYLES["SID"])
    draw(ax_t, k_pos, text_tokens, "Text", TOKEN_STYLES["Text"])
    ratio = text_tokens[-1] / sid_tokens[-1]
    ax_t.annotate(
        "", xy=(k_pos[-1], text_tokens[-1] * 0.95), xytext=(k_pos[-1], sid_tokens[-1] + 1200),
        arrowprops=dict(arrowstyle="<->", color="#333333", linewidth=0.9),
    )
    ax_t.annotate(
        rf"${ratio:.0f}\times$", xy=(k_pos[-1] - 0.42, 24000),
        ha="right", va="center", fontsize=8, color="#333333",
    )
    ax_t.set_ylim(0, 55000)
    ax_t.set_yticks([0, 10000, 20000, 30000, 40000, 50000],
                    ["0", "10k", "20k", "30k", "40k", "50k"])
    ax_t.set_title("(a) Prompt tokens vs. history")
    ax_t.set_ylabel("Mean prompt tokens")
    ax_t.legend(loc="upper left", frameon=False, borderaxespad=0.2, handlelength=1.9)

    random_anchor = ndcg5(results, SERIES["random_n"].format(n=30))
    panels = (
        (ax_k, k_pos, "k", K_GRID,
         [("Base-8B (SID)", "base_k"), ("RLVR-8B (SID)", "rlvr_k"),
          ("Gemini-3.6 (text)", "gem_k")],
         [random_anchor] * len(K_GRID)),
        (ax_n, n_pos, "n", N_GRID,
         [("Base-8B (SID)", "base_n"), ("RLVR-8B (SID)", "rlvr_n"),
          ("Gemini-3.6 (text)", "gem_n")],
         [ndcg5(results, SERIES["random_n"].format(n=n)) for n in N_GRID]),
    )
    for axis, positions, var, grid, lines, random_line in panels:
        for label, key in lines:
            draw(axis, positions, series(results, key, var, grid), label, LINE_STYLES[label],
                 sigma=1.2)
        axis.plot(positions, random_line, label="Random", **LINE_STYLES["Random"])
        axis.set_ylabel("NDCG@5")

    # Shade the no-history setting and the long-history region discussed in the text.
    ax_k.add_patch(plt.Rectangle((-0.32, 0.072), 0.64, 0.125, fill=True,
                                 facecolor="#888888", alpha=0.12,
                                 edgecolor="#666666", linestyle=(0, (2, 2)),
                                 linewidth=0.7, zorder=0.5))
    ax_k.annotate("no history", xy=(0.42, 0.089), fontsize=6.5,
                  color="#444444", style="italic", ha="left", va="center")
    ax_k.add_patch(plt.Rectangle((4.68, 0.217), 2.6, 0.038, fill=True,
                                 facecolor="#D55E00", alpha=0.10,
                                 edgecolor="#B24E00", linestyle=(0, (2, 2)),
                                 linewidth=0.7, zorder=0.5))
    ax_k.annotate("long-history region", xy=(6.0, 0.207), fontsize=6.5,
                  color="#8A4000", style="italic", ha="center", va="center")
    ax_k.set_title("(b) NDCG@5 vs. history ($N{=}30$)")
    ax_n.set_title("(c) NDCG@5 vs. slate size ($k{=}64$)")
    ax_k.set_ylim(0.06, 0.30)
    ax_n.set_ylim(0.0, 0.42)
    # One legend for panels (b) and (c), outside on the right.
    handles, labels = ax_n.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside center right", frameon=False,
               handlelength=1.9)

    for axis, positions, grid, xlabel in (
        (ax_t, k_pos, K_GRID, "History length $k$"),
        (ax_k, k_pos, K_GRID, "History length $k$"),
        (ax_n, n_pos, N_GRID, "Slate size $N$"),
    ):
        axis.set_xticks(positions, [str(v) for v in grid])
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
