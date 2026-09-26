"""Render the paired rationale-audit figure for the paper (single column).

Panel (a): diverging paired verdicts for the three judged aspects.  Decided
wins sit left (Gemini, text) and right (RLVR, SID) of the neutral block;
the neutral block stacks order-consistent ties and cross-order hesitations.
The judged coverage and overall rows are excluded on purpose: coverage is
reported programmatically in panel (b), and the overall verdict essentially
proxies it (see the supplement).
Panel (b): mean number of the ten ranked candidates that receive substantive
justification, computed from the rationale text by the frozen parser.

Values are read from the canonical published judge output (the
wandb-paper-final mirror) rather than copied into plotting code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "downloads/wandb-paper-final/results/benchmark/rationale-judge/pairwise_n100_full.json"

ARMS = {"gemini_3.6_flash": "Gemini-3.6 (text)", "rlvr_ndcg5_8b": "RLVR-8B (SID)"}
COLORS = {"gemini_3.6_flash": "#D55E00", "rlvr_ndcg5_8b": "#0072B2"}
ASPECTS = [
    ("item_claim_accuracy", "Item claims"),
    ("user_model_accuracy", "User portrait"),
    ("discriminative_justification", "Trade-off\nargument"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(RUN))
    parser.add_argument("--out", default=str(ROOT / "paper/figures/rationale_judge.pdf"))
    args = parser.parse_args()

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
            "hatch.linewidth": 0.5,
        }
    )

    with open(args.run) as handle:
        run = json.load(handle)
    summary = run["summary"]
    stats = summary["computed_ranking_stats"]

    fig, (ax_v, ax_c) = plt.subplots(
        1, 2, figsize=(3.3, 1.78), layout="constrained",
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    # Panel (a): diverging bars centered on the neutral (tie + hesitation) block.
    height = 0.62
    for row, (key, label) in enumerate(ASPECTS):
        cell = summary[key]
        gem, rlvr = cell["win_gemini_3.6_flash"], cell["win_rlvr_ndcg5_8b"]
        tie, hes = cell["tie"], cell["inconsistent_across_orders"]
        y = len(ASPECTS) - 1 - row
        left = -(gem + (tie + hes) / 2.0)
        for width, face, hatch in (
            (gem, COLORS["gemini_3.6_flash"], None),
            (tie, "#BBBBBB", None),
            (hes, "#E4E4E4", "////"),
            (rlvr, COLORS["rlvr_ndcg5_8b"], None),
        ):
            if width:
                ax_v.barh(y, width, left=left, height=height, color=face,
                          hatch=hatch, edgecolor="white", linewidth=0.5)
            left += width
        for count, x_pos, color in (
            (gem, -(gem + (tie + hes) / 2.0) + gem / 2.0, "white"),
            (rlvr, (tie + hes) / 2.0 + rlvr / 2.0, "white"),
        ):
            if count >= 10:
                ax_v.text(x_pos, y, str(count), ha="center", va="center",
                          fontsize=7, color=color)
    ax_v.axvline(0.0, color="#555555", linewidth=0.7, zorder=0.5)
    ax_v.set_yticks(range(len(ASPECTS)), [label for _, label in reversed(ASPECTS)])
    ax_v.set_xlim(-92, 92)
    ax_v.set_xticks([-80, -40, 0, 40, 80], ["80", "40", "0", "40", "80"])
    ax_v.set_xlabel("Users ($n{=}100$ pairs)")
    ax_v.set_title("(a) Paired verdicts")
    fig.legend(
        handles=[
            Patch(color=COLORS["gemini_3.6_flash"], label=ARMS["gemini_3.6_flash"]),
            Patch(color=COLORS["rlvr_ndcg5_8b"], label=ARMS["rlvr_ndcg5_8b"]),
            Patch(facecolor="#BBBBBB", label="Tie"),
            Patch(facecolor="#E4E4E4", hatch="////", edgecolor="#999999",
                  label="Hesitation"),
        ],
        loc="outside lower center", ncol=4,
        frameon=False, borderaxespad=0.0, handlelength=1.0,
        handletextpad=0.4, columnspacing=0.7,
    )

    # Panel (b): programmatic coverage of the emitted top-ten ranking.
    for row, key in enumerate(ARMS):
        arm = stats[key]
        justified = 10.0 * arm["coverage_of_ranking_mean"]
        y = len(ARMS) - 1 - row
        ax_c.barh(y, justified, height=0.56, color=COLORS[key])
        inside = justified > 5.0
        ax_c.text(justified - 0.3 if inside else justified + 0.3, y,
                  f"{justified:.1f}",
                  ha="right" if inside else "left", va="center", fontsize=7,
                  color="white" if inside else "#333333")
    ax_c.set_yticks(range(len(ARMS)), ["RLVR-8B", "Gemini-3.6"])
    ax_c.set_xlim(0, 10)
    ax_c.set_xticks([0, 5, 10])
    ax_c.set_xlabel("Candidates justified")
    ax_c.set_title("(b) Coverage")

    for axis in (ax_v, ax_c):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
