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
    assert a.p_images == pytest.approx(stats.binomtest(48, 60, 0.5, alternative="greater").pvalue)


def test_the_exact_correct_count_is_used_not_a_truncated_one():
    # 29 right out of 100 (0.29 * 100 is 28.999... in floating point): the p-value must be the one for 29
    spec = {("A", "R1"): [("A", 29), ("B", 21)], ("B", "R1"): [("B", 0), ("A", 50)]}
    a = analyze(frame(spec))
    assert a.correct == 29
    assert a.p_images == pytest.approx(stats.binomtest(29, 100, 0.5, alternative="greater").pvalue)


def test_distinguishable_across_repeats():
    a = analyze(balanced([0.9, 0.85, 0.95]))
    assert a.readout.marker == "✅" and a.readout.chip == "Distinguishable across repeats"
    assert "n = 3 repeats" in a.readout.text and "confirmed" not in a.text().lower()


def test_pilot_with_one_or_two_repeats_gets_no_verdict():
    for repeats in ([0.9], [0.9, 0.8]):
        a = analyze(balanced(repeats))
        assert a.readout.marker == "➖" and a.readout.chip.startswith("No conclusion (n = %d)" % len(repeats))
        assert np.isnan(a.across["p"]) and "at least 3" in a.readout.text


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
