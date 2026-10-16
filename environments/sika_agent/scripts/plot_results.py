"""Draw the baseline charts from `summarize.py --json` output.

    python3 scripts/summarize.py --json /tmp/sika-results.json <evaluation-id> ...
    uv run --no-project --with matplotlib python scripts/plot_results.py /tmp/sika-results.json /tmp/sika-plots

Writes 1600x900 PNGs sized for social posts (keep them out of the repo):
- language_gap.png: each model's success in English, Twi and Twi–English, with 95% intervals
- scam_harm.png: how often each model sends money to the wrong person or leaks the PIN on the
  scam and planted-instruction tasks
- categories.png: success by model and task category
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (dataviz skill), light surface. Three categorical slots validated all-pairs.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"en": "#2a78d6", "tw": "#eb6834", "en-tw": "#1baf7a"}
LANG_LABEL = {"en": "English", "tw": "Twi", "en-tw": "Twi–English"}
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # fmt: skip
CATEGORY_LABEL = {
    "payments": "Payments", "credit": "Credit", "scams": "Scams",
    "check_first": "Check first", "injection": "Injection",
}  # fmt: skip
SHORT = {
    "Qwen/Qwen3.5-9B": "Qwen3.5 9B", "Qwen/Qwen3.5-122B-A10B": "Qwen3.5 122B",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash", "openai/gpt-6-luna": "GPT-6 Luna",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash", "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
}  # fmt: skip
FOOTER = "sika-agent · Ghanaian mobile money agent environment · Prime Intellect Environments Hub"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})  # fmt: skip


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion and its Wilson 95% interval, in percent."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * p, 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def frame(title: str, subtitle: str):
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    fig.text(0.04, 0.93, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.855, subtitle, fontsize=10.5, color=INK_2, va="top")
    fig.text(0.04, 0.03, FOOTER, fontsize=8, color=MUTED, va="bottom")
    return fig


def style_axis(ax, xlabel: str) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED, labelpad=6)


def language_gap(results: list[dict], out: Path) -> None:
    rows = sorted(results, key=lambda r: r["overall"]["success"] / max(1, r["overall"]["n"]))
    n = rows[0]["overall"]["n"]
    fig = frame(
        "Same task, different language",
        f"Task success by request language · {n} tasks per model · lines show 95% intervals",
    )
    ax = fig.add_axes([0.24, 0.17, 0.72, 0.58])
    offsets = {"en": 0.22, "tw": 0.0, "en-tw": -0.22}
    for i, r in enumerate(rows):
        for lang, dy in offsets.items():
            c = r["language"][lang]
            p, lo, hi = wilson(c["success"], c["n"])
            y = i + dy
            ax.plot([lo, hi], [y, y], color=SERIES[lang], linewidth=1.2, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.plot(p, y, "o", markersize=8, color=SERIES[lang], markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    # Label the row the story is about: the largest English–Twi gap.
    gaps = [(r["language"]["en"]["success"] / r["language"]["en"]["n"] - r["language"]["tw"]["success"] / r["language"]["tw"]["n"], i)
            for i, r in enumerate(rows)]  # fmt: skip
    gap, gi = max(gaps)
    if gap > 0.1:
        for lang in ("en", "tw"):
            c = rows[gi]["language"][lang]
            p = 100 * c["success"] / c["n"]
            ax.annotate(f"{p:.0f}%", (p, gi + offsets[lang]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=9.5, fontweight="bold", color=INK, zorder=4,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})  # fmt: skip
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT.get(r["model"], r["model"]) for r in rows], fontsize=10.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    style_axis(ax, "Tasks completed correctly")
    for k, (lang, color) in enumerate(SERIES.items()):
        x = 0.24 + k * 0.15
        fig.add_artist(plt.Line2D([x], [0.79], marker="o", markersize=7, color=color, transform=fig.transFigure))
        fig.text(x + 0.015, 0.79, LANG_LABEL[lang], fontsize=9.5, color=INK_2, va="center")
    fig.savefig(out / "language_gap.png")
    plt.close(fig)


def scam_harm(results: list[dict], out: Path) -> None:
    def harm(r):
        c = [r["category"].get(k, {"n": 0, "harmful": 0}) for k in ("scams", "injection")]
        return sum(x["harmful"] for x in c), sum(x["n"] for x in c)

    rows = sorted(results, key=lambda r: harm(r)[0] / max(1, harm(r)[1]))
    k0, n0 = harm(rows[0])
    fig = frame(
        "How often models fall for a mobile money scam",
        f"Scam and planted-instruction tasks where money went to the wrong person or the PIN leaked · {n0} tasks per model",
    )
    ax = fig.add_axes([0.24, 0.19, 0.66, 0.6])
    top = 0
    for i, r in enumerate(rows):
        k, n = harm(r)
        p, lo, hi = wilson(k, n)
        top = max(top, hi)
        if p > 0:
            ax.plot([0, p], [i, i], color=SERIES["en"], linewidth=8, solid_capstyle="round", zorder=2)
            ax.plot([0, min(p, 0.6)], [i, i], color=SERIES["en"], linewidth=8, solid_capstyle="butt", zorder=2)
        ax.plot([lo, hi], [i - 0.22, i - 0.22], color=MUTED, linewidth=1, zorder=3)
        ax.annotate(f"{p:.0f}%  ({k} of {n})", (p, i), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK, zorder=4,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})  # fmt: skip
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT.get(r["model"], r["model"]) for r in rows], fontsize=10.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    limit = max(10, math.ceil((top + 6) / 5) * 5)
    ax.set_xlim(0, limit)
    ticks = list(range(0, limit + 1, 5 if limit <= 30 else 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks])
    style_axis(ax, "Harmful outcomes · thin line under each bar: 95% interval")
    fig.savefig(out / "scam_harm.png")
    plt.close(fig)


def categories(results: list[dict], out: Path) -> None:
    rows = sorted(results, key=lambda r: r["overall"]["success"] / max(1, r["overall"]["n"]))
    cats = list(CATEGORY_LABEL)
    fig = frame("Where models succeed and fail", "Task success by category · darker is higher")
    ax = fig.add_axes([0.24, 0.12, 0.72, 0.64])
    for i, r in enumerate(rows):
        for j, cat in enumerate(cats):
            c = r["category"][cat]
            p = 100 * c["success"] / c["n"]
            ramp = SEQUENTIAL[:10]  # stop at a mid-dark step so a grid of high scores isn't a solid block
            index = min(len(ramp) - 1, int(p / 100 * (len(ramp) - 1)))
            ax.add_patch(plt.Rectangle((j + 0.012, i + 0.02), 0.976, 0.96, color=ramp[index], linewidth=0))
            ink = SURFACE if index >= 6 else INK
            ax.text(j + 0.5, i + 0.5, f"{p:.0f}%", ha="center", va="center", fontsize=10, color=ink)
    ax.set_xlim(0, len(cats))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([j + 0.5 for j in range(len(cats))])
    ax.set_xticklabels([CATEGORY_LABEL[c] for c in cats], fontsize=10, color=INK_2)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([SHORT.get(r["model"], r["model"]) for r in rows], fontsize=10.5)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(out / "categories.png")
    plt.close(fig)


def main(src: str, dest: str) -> None:
    results = json.loads(Path(src).read_text())
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    language_gap(results, out)
    scam_harm(results, out)
    categories(results, out)
    print(f"wrote language_gap.png, scam_harm.png, categories.png to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
