"""What the guesses say: accuracy against chance, across biological repeats, per condition, confusion, decision time.

The rules follow the other MorphoBricks tools (MorphoCoverage):

- **The biological repeat is the replicate.** Images of one repeat are not independent, so the answer to "can the conditions be
  told apart?" is the accuracy of every repeat compared with chance across the repeats (a one-sided t-test on the repeats'
  accuracy minus chance, from 3 repeats), reported with n and the 95% interval. The test on all images pooled (exact binomial)
  is shown beside it as *exploratory*: it counts every image as independent, so its false-positive chance is higher than stated.
- **Words say what was found**: ✅ distinguishable across repeats, ⚠️ trend only, ➖ no conclusion possible (or not above chance),
  ❌ check the data first. It never says "confirmed". A null result is a result, said as one.
- **Pilots get advice, not verdicts.** With one or two repeats the numbers are shown and the design is said, not judged.
- **A usual design is not flagged.** Only a measured property of these data is noted (a repeat with very few guesses).
- Nothing is pooled silently: per-repeat and per-condition tables come with the pooled figures.

Chance is the accuracy of an observer who ignores the images and always names the most common condition of what was tested
(1 / number of conditions when it is balanced, as the stratified order makes it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MARKERS = {"yes": "✅", "partly": "⚠️", "none": "➖", "check": "❌"}
MIN_REPEATS = 3          # the repeats a test across repeats needs
FEW_GUESSES = 5          # a repeat with fewer guessed images is noted
ALPHA = 0.05


@dataclass
class Readout:
    marker: str
    chip: str
    text: str


@dataclass
class Analysis:
    n: int
    correct: int
    accuracy: float
    chance: float
    ci_low: float
    ci_high: float
    p_images: float
    per_repeat: pd.DataFrame
    across: dict
    per_condition: pd.DataFrame
    per_condition_repeat: pd.DataFrame
    confusion: pd.DataFrame
    confusion_normalised: pd.DataFrame
    pairs: pd.DataFrame
    decision_time: pd.DataFrame
    repeat_heterogeneity_p: float
    readout: Readout
    notes: list = field(default_factory=list)

    def text(self) -> str:
        return summary_text(self)


def _chance(conditions: pd.Series) -> float:
    return float(conditions.value_counts(normalize=True).max())


def _exact_interval(correct: int, n: int, level: float = 0.95) -> tuple[float, float]:
    from scipy import stats

    if n == 0:
        return float("nan"), float("nan")
    ci = stats.binomtest(int(correct), int(n)).proportion_ci(confidence_level=level, method="exact")
    return float(ci.low), float(ci.high)


def _paired_t(differences: np.ndarray) -> float:
    """One-sided (greater) t-test of ``differences`` (one per repeat) against 0; NaN when they do not vary or one is missing
    (the same rule as MorphoCoverage's ``paired_t``, here one-sided because the question is 'better than chance')."""
    from scipy import stats

    differences = np.asarray(differences, dtype=float)
    if not len(differences) or np.isnan(differences).any() or np.ptp(differences) == 0:
        return float("nan")
    return float(stats.ttest_1samp(differences, 0.0, alternative="greater").pvalue)


def _heterogeneity_p(table: pd.DataFrame, seed: int, draws: int = 5000) -> float:
    """Exploratory: do the repeats differ in how often the guess was right? A permutation test of the chi-square statistic of
    the (repeat x right/wrong) counts, shuffling right/wrong over the images (seeded). NaN with fewer than 2 repeats."""
    repeats = table["biological_repeat"].to_numpy()
    right = (table["condition"] == table["guess"]).to_numpy().astype(float)
    levels = np.unique(repeats)
    if len(levels) < 2 or right.min() == right.max():
        return float("nan")
    codes = np.searchsorted(levels, repeats)
    sizes = np.bincount(codes).astype(float)

    def statistic(r: np.ndarray) -> float:
        hits = np.bincount(codes, weights=r)
        expected = sizes * r.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.nansum((hits - expected) ** 2 / expected + ((sizes - hits) - (sizes - expected)) ** 2 / (sizes - expected)))

    observed = statistic(right)
    rng = np.random.default_rng(seed)
    greater = sum(statistic(rng.permutation(right)) >= observed - 1e-12 for _ in range(draws))
    return float((greater + 1) / (draws + 1))


def analyze(table: pd.DataFrame, seed: int = 0) -> Analysis:
    """Analyse the guessed images: a table with ``condition``, ``biological_repeat``, ``guess`` and (optionally)
    ``decision_time_s``. Rows without a guess are ignored. Raises ``ValueError`` when nothing was guessed."""
    from scipy import stats

    table = table.dropna(subset=["guess"]).copy()
    if table.empty:
        raise ValueError("No image has been guessed yet: there is nothing to analyse.")
    for column in ("condition", "biological_repeat", "guess"):
        table[column] = table[column].astype(str)
    table["right"] = table["condition"] == table["guess"]
    n, correct = len(table), int(table["right"].sum())
    chance = _chance(table["condition"])
    low, high = _exact_interval(correct, n)
    p_images = float(stats.binomtest(correct, n, chance, alternative="greater").pvalue)

    rows = []
    for repeat, group in table.groupby("biological_repeat", sort=True):
        c = _chance(group["condition"])
        rows.append(
            {
                "biological_repeat": repeat,
                "n": len(group),
                "correct": int(group["right"].sum()),
                "accuracy": float(group["right"].mean()),
                "chance": c,
                "accuracy_minus_chance": float(group["right"].mean() - c),
                "p_exact_this_repeat": float(stats.binomtest(int(group["right"].sum()), len(group), c, alternative="greater").pvalue),
            }
        )
    per_repeat = pd.DataFrame(rows)

    differences = per_repeat["accuracy_minus_chance"].to_numpy()
    k = len(per_repeat)
    across = {"n_repeats": k, "mean_difference": float(differences.mean()), "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan"), "repeats_above_chance": int((differences > 0).sum())}
    if k >= 2 and not np.isnan(differences).any():
        half = stats.t.ppf(0.975, k - 1) * stats.sem(differences)
        across["ci_low"], across["ci_high"] = float(differences.mean() - half), float(differences.mean() + half)
    if k >= MIN_REPEATS:
        across["p"] = _paired_t(differences)

    per_condition = (
        table.groupby("condition", sort=True)["right"].agg(n="size", correct="sum").assign(recall=lambda d: d["correct"] / d["n"]).reset_index()
    )
    per_condition["ci_low"], per_condition["ci_high"] = zip(*(_exact_interval(int(c), int(m)) for c, m in zip(per_condition["correct"], per_condition["n"])))
    per_condition_repeat = (
        table.groupby(["condition", "biological_repeat"], sort=True)["right"].agg(n="size", correct="sum").assign(recall=lambda d: d["correct"] / d["n"]).reset_index()
    )

    true_labels = sorted(table["condition"].unique(), key=str.lower)
    guessed_labels = sorted(set(true_labels) | set(table["guess"].unique()), key=str.lower)
    confusion = pd.crosstab(table["condition"], table["guess"]).reindex(index=true_labels, columns=guessed_labels, fill_value=0)
    confusion_normalised = confusion.div(confusion.sum(axis=1), axis=0)
    pairs = [
        {"true_condition": t, "guessed_as": g, "count": int(confusion.loc[t, g]), "proportion_of_true": float(confusion_normalised.loc[t, g])}
        for t in true_labels
        for g in guessed_labels
        if t != g and confusion.loc[t, g] > 0
    ]
    pairs = pd.DataFrame(pairs, columns=["true_condition", "guessed_as", "count", "proportion_of_true"]).sort_values(["count", "proportion_of_true"], ascending=False).reset_index(drop=True)

    if "decision_time_s" in table.columns and table["decision_time_s"].notna().any():
        times = table.dropna(subset=["decision_time_s"])
        decision_time = (
            times.groupby(["condition", "biological_repeat"], sort=True)["decision_time_s"]
            .agg(n="size", median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
            .reset_index()
        )
    else:
        decision_time = pd.DataFrame(columns=["condition", "biological_repeat", "n", "median", "q1", "q3"])

    notes = [
        "Repeat %s has only %d guessed image(s)." % (r["biological_repeat"], r["n"]) for _, r in per_repeat.iterrows() if r["n"] < FEW_GUESSES
    ]
    untested = sorted(set(table["guess"]) - set(table["condition"]))
    if untested:
        notes.append("%s was guessed but none of its images was tested." % ", ".join(untested))

    analysis = Analysis(
        n=n, correct=correct, accuracy=correct / n, chance=chance, ci_low=low, ci_high=high, p_images=p_images,
        per_repeat=per_repeat, across=across, per_condition=per_condition, per_condition_repeat=per_condition_repeat,
        confusion=confusion, confusion_normalised=confusion_normalised, pairs=pairs, decision_time=decision_time,
        repeat_heterogeneity_p=_heterogeneity_p(table, seed), readout=Readout("", "", ""), notes=notes,
    )  # fmt: skip
    analysis.readout = _readout(analysis)
    return analysis


def _fmt_p(p: float) -> str:
    return "= n/a" if p != p else ("< 0.001" if p < 0.001 else "= %.3f" % p)


def _readout(a: Analysis) -> Readout:
    k, ac = a.across["n_repeats"], a.across
    repeats = "%d biological repeat%s" % (k, "" if k == 1 else "s")
    if k < MIN_REPEATS:
        return Readout(
            MARKERS["none"],
            "No conclusion (n = %d)" % k,
            "With %s the guesses cannot say whether the conditions can be told apart across repeats: the repeat is the replicate and a "
            "test across repeats needs at least %d. Overall accuracy was %.0f%% against %.0f%% by chance (95%% interval %.0f-%.0f%%). "
            "Next time, guess images from every repeat you have, or add repeats." % (repeats, MIN_REPEATS, 100 * a.accuracy, 100 * a.chance, 100 * a.ci_low, 100 * a.ci_high),
        )
    p, d = ac["p"], ac["mean_difference"]
    above = "%d of %d repeats were above chance" % (ac["repeats_above_chance"], k)
    effect = "accuracy minus chance, mean %.0f points (95%% interval %.0f to %.0f), n = %d repeats" % (100 * d, 100 * ac["ci_low"], 100 * ac["ci_high"], k)
    if p == p and p <= ALPHA:
        return Readout(MARKERS["yes"], "Distinguishable across repeats", "Significant across repeats: %s, p %s. %s." % (effect, _fmt_p(p), above.capitalize()))
    if p != p and d > 0 and ac["repeats_above_chance"] == k:
        worst = float(a.per_repeat["p_exact_this_repeat"].max())
        if worst <= ALPHA:  # no variation across repeats: the t-test is undefined, but every repeat is above chance on its own
            return Readout(
                MARKERS["yes"], "Distinguishable in every repeat",
                "Every repeat was above chance on its own (the largest exact p of a repeat is %s; within a repeat the images are counted as independent). "
                "The t-test across repeats is undefined because the repeats do not vary: %s." % (_fmt_p(worst), effect),
            )  # fmt: skip
        return Readout(
            MARKERS["partly"], "Trend only",
            "Every repeat was above chance (%s) but not clearly so on its own (largest exact p of a repeat %s), and the repeats do not vary, so the t-test is undefined: %s. Look at the table of repeats." % (above, _fmt_p(worst), effect),
        )  # fmt: skip
    if d > 0:
        return Readout(MARKERS["partly"], "Trend only", "Above chance on average but not significant across repeats: %s, p %s. %s." % (effect, _fmt_p(p), above.capitalize()))
    return Readout(
        MARKERS["none"], "Not above chance",
        "The guesses were not above chance across repeats: %s, p %s. %s. A null result is a result; with %s a small difference could still be missed." % (effect, _fmt_p(p), above.capitalize(), repeats),
    )  # fmt: skip


def summary_text(a: Analysis) -> str:
    """The results as plain text, the answer first."""
    ac = a.across
    lines = [
        "%s %s" % (a.readout.marker, a.readout.chip),
        a.readout.text,
        "",
        "Images guessed: %d, correct: %d (accuracy %.1f%%, exact 95%% interval %.1f-%.1f%%); chance %.1f%% (always naming the most common condition)."
        % (a.n, a.correct, 100 * a.accuracy, 100 * a.ci_low, 100 * a.ci_high, 100 * a.chance),
        "Exploratory, all images pooled: exact binomial p %s. It counts every image as independent, so its false-positive chance is higher than stated; the test across repeats above is the one to read."
        % _fmt_p(a.p_images),
        "",
        "By biological repeat (n = %d):" % ac["n_repeats"],
    ]
    lines += [
        "- %s: %d/%d correct, %.1f%% (chance %.1f%%)" % (r.biological_repeat, r.correct, r.n, 100 * r.accuracy, 100 * r.chance) for r in a.per_repeat.itertuples()
    ]
    if a.repeat_heterogeneity_p == a.repeat_heterogeneity_p:
        lines.append("Exploratory: do the repeats differ in accuracy? permutation p %s." % _fmt_p(a.repeat_heterogeneity_p))
    lines += ["", "By condition (pooled over repeats):"]
    lines += [
        "- %s: %d/%d recognised, %.1f%% (95%% interval %.1f-%.1f%%)" % (r.condition, r.correct, r.n, 100 * r.recall, 100 * r.ci_low, 100 * r.ci_high)
        for r in a.per_condition.itertuples()
    ]
    if len(a.pairs):
        lines += ["", "Most frequent confusions:"]
        lines += [
            "- %s guessed as %s: %d time(s), %.1f%% of its images" % (r.true_condition, r.guessed_as, r.count, 100 * r.proportion_of_true) for r in a.pairs.head(5).itertuples()
        ]
    if a.notes:
        lines += ["", "Notes:"] + ["- " + n for n in a.notes]
    return "\n".join(lines)
