"""Figures, tables and the PDF report of an ``Analysis`` (matplotlib only, no display needed)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import Analysis


def _figure(*args, **kwargs):
    """A figure without pyplot: it needs no display and never touches the backend of a notebook or Napari."""
    from matplotlib.figure import Figure

    return Figure(*args, **kwargs)


def figure_png(fig, dpi: int = 110) -> bytes:
    """The figure as PNG bytes (for ``IPython.display.Image``)."""
    import io

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


def confusion_figure(a: Analysis):
    """The row-normalised confusion matrix: of the images of each true condition, the share guessed as each condition."""
    m = a.confusion_normalised
    fig = _figure(figsize=(1.2 * len(m.columns) + 3, 1.0 * len(m.index) + 2.5))
    ax = fig.subplots()
    image = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(m.index)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            n = int(a.confusion.iloc[i, j])
            ax.text(j, i, "%.0f%%\n(n = %d)" % (100 * m.iloc[i, j], n), ha="center", va="center", fontsize=8, color="white" if m.iloc[i, j] > 0.6 else "black")
    ax.set_xlabel("Guessed condition")
    ax.set_ylabel("True condition")
    ax.set_title("Normalised confusion matrix")
    fig.colorbar(image, ax=ax, label="share of the true condition's images")
    fig.tight_layout()
    return fig


def accuracy_figure(a: Analysis):
    """Accuracy of every biological repeat against its chance level, and the recognition of each condition per repeat."""
    fig = _figure(figsize=(12, 4.5))
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
    r = a.per_repeat
    x = np.arange(len(r))
    left.bar(x, 100 * r["accuracy"], color="#4c78a8")
    left.scatter(x, 100 * r["chance"], marker="_", s=900, color="black", label="chance")
    left.set_xticks(x, r["biological_repeat"])
    left.set_ylim(0, 100)
    left.set_ylabel("Accuracy (%)")
    left.set_title("Biological repeats (n = %d)" % len(r))
    left.legend(loc="lower right")
    d = a.per_condition_repeat
    conditions = list(a.per_condition["condition"])
    repeats = list(r["biological_repeat"])
    width = 0.8 / max(1, len(repeats))
    for i, repeat in enumerate(repeats):
        values = [100 * d[(d["condition"] == c) & (d["biological_repeat"] == repeat)]["recall"].mean() for c in conditions]
        right.bar(np.arange(len(conditions)) + i * width, np.nan_to_num(values), width, label=repeat)
    right.set_xticks(np.arange(len(conditions)) + 0.4 - width / 2, conditions, rotation=30, ha="right")
    right.set_ylim(0, 100)
    right.set_ylabel("Images recognised (%)")
    right.set_title("Condition-wise accuracy per repeat")
    right.legend(title="Repeat")
    fig.tight_layout()
    return fig


def decision_time_figure(table: pd.DataFrame):
    """Decision times per condition (a SuperPlot: every image a small dot, the median of each repeat a large one)."""
    import matplotlib

    table = table.dropna(subset=["decision_time_s"])
    fig = _figure(figsize=(1.6 * table["condition"].nunique() + 4, 4.5))
    ax = fig.subplots()
    conditions = sorted(table["condition"].astype(str).unique(), key=str.lower)
    repeats = sorted(table["biological_repeat"].astype(str).unique())
    colours = matplotlib.colormaps["Set2"]
    rng = np.random.default_rng(0)
    for i, condition in enumerate(conditions):
        for k, repeat in enumerate(repeats):
            values = table[(table["condition"].astype(str) == condition) & (table["biological_repeat"].astype(str) == repeat)]["decision_time_s"].to_numpy()
            if not len(values):
                continue
            x = i + (k - (len(repeats) - 1) / 2) * 0.18
            ax.scatter(x + rng.uniform(-0.04, 0.04, len(values)), values, s=14, alpha=0.5, color=colours(k % 8))
            ax.scatter([x], [np.median(values)], s=90, edgecolor="black", color=colours(k % 8), label=repeat if i == 0 else None, zorder=3)
    ax.set_xticks(range(len(conditions)), conditions, rotation=30, ha="right")
    ax.set_ylabel("Decision time (s)")
    ax.set_title("Decision time (small dots: images; large dots: median of a repeat)")
    ax.legend(title="Repeat")
    fig.tight_layout()
    return fig


def _text_page(pdf, title: str, text: str):
    fig = _figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, title, fontsize=15, fontweight="bold", va="top")
    for marker, word in (("✅", "[yes]"), ("⚠️", "[partly]"), ("➖", "[no conclusion]"), ("❌", "[check the data]")):
        text = text.replace(marker, word)  # the PDF's monospace font has no emoji
    wrapped = "\n".join(textwrap.fill(line, 95, subsequent_indent="   ") if line else "" for line in text.splitlines())
    fig.text(0.07, 0.91, wrapped, fontsize=8.5, family="monospace", va="top")
    pdf.savefig(fig)


def write_report(a: Analysis, table: pd.DataFrame, out_dir) -> dict:
    """Write ``analysis_results.pdf``, the figures (PNG), the tables (CSV) and ``summary.txt`` into ``out_dir``; returns their paths keyed by file name."""
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    text = a.text()
    (out / "summary.txt").write_text(text, encoding="utf-8")
    paths["summary.txt"] = out / "summary.txt"
    tables = {
        "per_repeat": a.per_repeat,
        "per_condition": a.per_condition,
        "per_condition_repeat": a.per_condition_repeat,
        "confusion_counts": a.confusion.reset_index().rename(columns={"condition": "true_condition"}),
        "confusion_misclassification_pairs": a.pairs,
        "decision_time": a.decision_time,
    }
    for name, frame in tables.items():
        frame.to_csv(out / (name + ".csv"), index=False)
        paths[name + ".csv"] = out / (name + ".csv")
    figures = {"confusion_matrix": confusion_figure(a), "condition_accuracy": accuracy_figure(a)}
    if len(a.decision_time):
        figures["decision_time"] = decision_time_figure(table)
    for name, fig in figures.items():
        fig.savefig(out / (name + ".png"), dpi=150, bbox_inches="tight")
        paths[name + ".png"] = out / (name + ".png")
    with PdfPages(out / "analysis_results.pdf") as pdf:
        _text_page(pdf, "Guess the Condition: results", text)
        for fig in figures.values():
            pdf.savefig(fig, bbox_inches="tight")
    paths["analysis_results.pdf"] = out / "analysis_results.pdf"
    return paths
