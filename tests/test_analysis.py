import numpy as np
import pandas as pd
import pytest
from scipy import stats

from guessthecondition import analyze
from guessthecondition.analysis import _heterogeneity_p


def frame(spec):
    """spec: {(condition, repeat): [(guess, n), ...]} -> one row per guess."""
    rows = []
    for (condition, repeat), guesses in spec.items():
        for guess, n in guesses:
            rows += [{"condition": condition, "biological_repeat": repeat, "guess": guess, "decision_time_s": 3.0}] * n
    return pd.DataFrame(rows)


def balanced(correct_by_repeat, conditions=("A", "B"), per_cell=10):
    """Each repeat holds ``per_cell`` images of every condition; ``correct_by_repeat[r]`` of them are guessed right."""
    spec = {}
    for r, right in enumerate(correct_by_repeat, 1):
        for i, c in enumerate(conditions):
            wrong = conditions[(i + 1) % len(conditions)]
            n_right = int(round(right * per_cell))
            spec[(c, "R%d" % r)] = [(c, n_right), (wrong, per_cell - n_right)]
    return frame(spec)


def test_numbers_match_hand_computation():
    a = analyze(balanced([0.9, 0.8, 0.7]))
    assert (a.n, a.correct, a.chance) == (60, 48, 0.5)
    assert a.accuracy == pytest.approx(0.8)
    assert a.per_repeat["accuracy"].tolist() == pytest.approx([0.9, 0.8, 0.7])
    d = np.array([0.4, 0.3, 0.2])
    assert a.across["mean_difference"] == pytest.approx(d.mean())
    assert a.across["p"] == pytest.approx(stats.ttest_1samp(d, 0, alternative="greater").pvalue)
    half = stats.t.ppf(0.975, 2) * stats.sem(d)
    assert (a.across["ci_low"], a.across["ci_high"]) == pytest.approx((d.mean() - half, d.mean() + half))
    assert a.p_binomial == pytest.approx(stats.binomtest(48, 60, 0.5, alternative="greater").pvalue)


def test_the_exact_correct_count_is_used_not_a_truncated_one():
    # 29 right out of 100 (0.29 * 100 is 28.999... in floating point): the p-value must be the one for 29
    spec = {("A", "R1"): [("A", 29), ("B", 21)], ("B", "R1"): [("B", 0), ("A", 50)]}
    a = analyze(frame(spec))
    assert a.correct == 29
    assert a.p_binomial == pytest.approx(stats.binomtest(29, 100, 0.5, alternative="greater").pvalue)


def test_distinguishable_across_repeats():
    a = analyze(balanced([0.9, 0.85, 0.95]))
    assert a.readout.marker == "✅" and a.readout.chip == "Distinguishable across repeats"
    assert "n = 3 repeats" in a.readout.text and "confirmed" not in a.text().lower()


def test_pilot_with_one_or_two_repeats_is_answered_by_the_randomization_test_for_these_images():
    for repeats in ([0.9], [0.9, 0.8]):
        a = analyze(balanced(repeats))
        assert a.readout.marker == "⚠️" and a.readout.chip == "Above chance in a pilot (n = %d)" % len(repeats)
        assert np.isnan(a.across["p"]) and "at least 3" in a.readout.text and "for the images you saw" in a.readout.text
        assert a.p_randomization < 0.01
    null = analyze(balanced([0.5]))
    assert null.readout.marker == "➖" and null.readout.chip == "Not above chance in a pilot (n = 1)" and null.p_randomization > 0.3


def test_trend_only_when_above_chance_but_not_significant():
    a = analyze(balanced([0.8, 0.5, 0.6]))
    assert a.readout.marker == "⚠️" and a.readout.chip == "Trend only"


def test_null_result_is_said_as_a_result():
    a = analyze(balanced([0.4, 0.5, 0.45]))
    assert a.readout.chip == "Not above chance" and "null result is a result" in a.readout.text


def test_perfect_identical_repeats_are_distinguishable_in_every_repeat():
    a = analyze(balanced([1.0, 1.0, 1.0]))  # the t-test is undefined (no variation), each repeat is clearly above chance on its own
    assert np.isnan(a.across["p"]) and a.readout.marker == "✅" and a.readout.chip == "Distinguishable in every repeat"
    assert "undefined" in a.readout.text


def test_identical_but_weak_repeats_stay_a_trend():
    a = analyze(balanced([0.75, 0.75, 0.75], per_cell=4))  # 6 of 8 in each repeat: p = 0.145 on its own
    assert np.isnan(a.across["p"]) and a.readout.chip == "Trend only" and "undefined" in a.readout.text


def test_chance_is_the_most_common_condition_when_unbalanced():
    spec = {("A", "R1"): [("A", 6)], ("B", "R1"): [("B", 2)]}
    assert analyze(frame(spec)).chance == pytest.approx(0.75)


def test_confusion_matrix_and_pairs():
    spec = {("A", "R1"): [("A", 3), ("B", 1)], ("B", "R1"): [("B", 2), ("A", 2)]}
    a = analyze(frame(spec))
    assert a.confusion.loc["A", "B"] == 1 and a.confusion.loc["B", "A"] == 2
    assert a.confusion_normalised.loc["B", "A"] == pytest.approx(0.5)
    assert a.confusion_normalised.sum(axis=1).tolist() == pytest.approx([1, 1])
    assert a.pairs.iloc[0].to_dict() == {"true_condition": "B", "guessed_as": "A", "count": 2, "proportion_of_true": 0.5}


def test_condition_names_with_spaces_are_fine():
    spec = {("Wild type", "Repeat 1"): [("Wild type", 3)], ("Mutant A", "Repeat 1"): [("Wild type", 3)]}
    a = analyze(frame(spec))
    assert set(a.per_condition["condition"]) == {"Wild type", "Mutant A"}
    assert "Mutant A guessed as Wild type" in a.text()


def test_unguessed_rows_are_ignored_and_nothing_to_analyse_is_explained():
    table = balanced([0.9, 0.8, 0.7])
    table = pd.concat([table, table.assign(guess=None)], ignore_index=True)
    assert analyze(table).n == 60
    with pytest.raises(ValueError, match="nothing to analyse"):
        analyze(table.assign(guess=None))


def test_a_usual_design_is_not_flagged_but_a_measured_property_is():
    assert analyze(balanced([0.9, 0.8, 0.7])).notes == []
    spec = {("A", "R1"): [("A", 8)], ("B", "R1"): [("B", 8)], ("A", "R2"): [("A", 2)], ("B", "R2"): [("B", 1)]}
    assert any("Repeat R2 has only 3 guessed" in n for n in analyze(frame(spec)).notes)


def test_repeat_heterogeneity_is_seeded_and_detects_a_bad_repeat():
    good = balanced([0.9, 0.9, 0.5], per_cell=20)
    p = _heterogeneity_p(good.assign(right=None), seed=1)
    assert p < 0.01 and p == _heterogeneity_p(good, seed=1)
    assert np.isnan(_heterogeneity_p(balanced([0.9]), seed=1))


def test_decision_time_summary():
    a = analyze(balanced([0.9, 0.8, 0.7]))
    assert set(a.decision_time.columns) == {"condition", "biological_repeat", "n", "median", "q1", "q3"}
    assert (a.decision_time["median"] == 3.0).all()
    assert analyze(balanced([0.9, 0.8, 0.7]).drop(columns="decision_time_s")).decision_time.empty


def _exact_stratified_p(rows):
    """The exact p by enumerating every distinct arrangement of the true conditions within each repeat (tiny data only)."""
    import itertools
    from collections import Counter

    by_repeat = {}
    for r in rows:
        by_repeat.setdefault(r["biological_repeat"], []).append(r)
    observed = sum(r["condition"] == r["guess"] for r in rows)
    distributions = []
    for rs in by_repeat.values():
        guesses = [r["guess"] for r in rs]
        counts = Counter()
        for arrangement in set(itertools.permutations([r["condition"] for r in rs])):
            counts[sum(a == g for a, g in zip(arrangement, guesses))] += 1
        total = sum(counts.values())
        distributions.append({k: v / total for k, v in counts.items()})
    p, state = 0.0, {0: 1.0}
    for d in distributions:                      # convolve the repeats
        new = {}
        for a, pa in state.items():
            for b, pb in d.items():
                new[a + b] = new.get(a + b, 0) + pa * pb
        state = new
    return sum(v for k, v in state.items() if k >= observed)


@pytest.mark.parametrize("seed", range(6))
def test_randomization_p_matches_the_exact_enumeration_within_repeats(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in ("R1", "R2"):
        for condition in ("A", "B", "C"):
            for _ in range(rng.integers(1, 3)):
                guess = condition if rng.random() < 0.6 else rng.choice(["A", "B", "C"])
                rows.append({"condition": condition, "biological_repeat": repeat, "guess": guess, "decision_time_s": 1.0})
    a = analyze(pd.DataFrame(rows))
    exact = _exact_stratified_p(rows)
    # Monte Carlo with 10 000 shuffles: within 3 standard errors of the exact value (and never 0)
    se = np.sqrt(max(exact * (1 - exact), 1e-4) / a.n_draws)
    assert abs(a.p_randomization - exact) <= 3 * se + 1 / (a.n_draws + 1) and a.p_randomization > 0


def test_randomization_is_seeded_and_never_zero():
    table = balanced([1.0])
    assert analyze(table, seed=3).p_randomization == analyze(table, seed=3).p_randomization
    assert analyze(table).p_randomization == pytest.approx(1 / 10001)


def test_chance_comes_from_the_whole_experiment_not_from_the_few_images_guessed():
    # 3 conditions x 3 repeats x 4 images; only 4 images guessed (a perfect observer): two repeats hold a single guessed condition
    population = pd.DataFrame([{"condition": c, "biological_repeat": r, "guess": None, "decision_time_s": np.nan} for c in "ABC" for r in ("R1", "R2", "R3") for _ in range(4)])
    guessed = [("A", "R1"), ("B", "R1"), ("C", "R2"), ("A", "R3")]
    table = population.copy()
    for k, (c, r) in enumerate(guessed):
        row = table.index[(table["condition"] == c) & (table["biological_repeat"] == r)][0]
        table.loc[row, ["guess", "decision_time_s"]] = [c, 3.0]
    full = analyze(table)
    only = analyze(table.dropna(subset=["guess"]))
    assert full.per_repeat["chance"].tolist() == pytest.approx([1 / 3] * 3) and full.chance == pytest.approx(1 / 3)
    assert full.across["mean_difference"] == pytest.approx(2 / 3)          # a perfect observer is 67 points above chance
    assert only.across["mean_difference"] < full.across["mean_difference"]  # read from the guessed rows alone, chance is distorted


def test_a_repeat_that_holds_one_condition_cannot_be_compared_with_chance():
    rows = []
    for c, r in (("A", "R1"), ("B", "R1"), ("A", "R2"), ("B", "R2"), ("A", "R3"), ("B", "R3"), ("B", "R4"), ("B", "R4")):
        rows.append({"condition": c, "biological_repeat": r, "guess": c, "decision_time_s": 2.0})
    a = analyze(pd.DataFrame(rows))
    assert a.across["n_repeats"] == 3 and a.per_repeat.set_index("biological_repeat").loc["R4", "testable"] == False  # noqa: E712
    assert any("Repeat R4 holds images of one condition only" in n for n in a.notes)
    assert "n/a: one condition only" in a.text()
