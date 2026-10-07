import numpy as np
import pandas as pd
import pytest

from guessthecondition import GameError, Session, scan_experiment
from guessthecondition.session import play_order


def test_play_order_is_a_stratified_permutation(experiment):
    table = scan_experiment(experiment)
    order = play_order(table, seed=3)
    assert sorted(order) == list(range(len(table)))
    first = table.assign(o=order).sort_values("o").head(9)  # 9 strata (3 conditions x 3 repeats): each once in the first round
    assert first.groupby(["condition", "biological_repeat"]).size().eq(1).all() and len(first) == 9
    assert np.array_equal(order, play_order(table, seed=3)) and not np.array_equal(order, play_order(table, seed=4))


def test_unequal_groups_still_spread_the_first_images():
    table = pd.DataFrame(
        {"image_file": [str(i) for i in range(12)], "condition": ["A"] * 8 + ["B"] * 4, "biological_repeat": ["R1"] * 12}
    )
    first = table.assign(o=play_order(table, 0)).sort_values("o").head(4)
    assert first["condition"].value_counts().to_dict() == {"A": 2, "B": 2}


def test_play_through_and_resume(experiment, tmp_path):
    results = tmp_path / "Results"
    s = Session.create(experiment, results, "Ana", percentage_to_test=25, seed=1)
    assert s.total == 36 and s.target == 9 and s.remaining == 9
    seen = []
    for k in range(3):
        item = s.serve(now=100.0 + 10 * k)
        assert item["number"] == k + 1 and item["of"] == 9
        assert set(item) == {"number", "of", "path"}  # nothing that names the condition
        seen.append(item["path"])
        result = s.answer("control", now=100.0 + 10 * k + 4.5)  # name in any case
        assert result == {"guess": "Control", "decision_time_s": 4.5}
    assert len(set(seen)) == 3
    again = Session.open(results, "Ana")
    assert again.n_answered == 3 and again.remaining == 6
    assert again.serve()["number"] == 4
    assert (results / "Ana" / "game_results.csv").exists() and (results / "Ana" / "session.json").exists()


def test_a_shown_image_stays_pending_until_answered(experiment, tmp_path):
    s = Session.create(experiment, tmp_path, "Ana", 50, seed=2)
    first = s.serve(now=10)
    reopened = Session.open(tmp_path, "Ana")
    again = reopened.serve(now=500)
    assert again["path"] == first["path"] and again["number"] == 1
    assert reopened.answer("1", now=502)["decision_time_s"] == 2.0  # the clock restarted when the image was shown again
    assert Session.open(tmp_path, "Ana").n_answered == 1


def test_guess_by_number_and_unknown_guess(experiment, tmp_path):
    s = Session.create(experiment, tmp_path, "Ana", 50, seed=2)
    s.serve()
    assert s.conditions == ["Control", "Drug", "Mutant"] and s.conditions_text() == "1: Control | 2: Drug | 3: Mutant"
    with pytest.raises(GameError) as caught:
        s.answer("Mutnat")
    assert caught.value.code == "unknown_condition" and "1: Control" in caught.value.message
    assert s.answer("3")["guess"] == "Mutant"


def test_answer_without_image_and_finished(experiment, tmp_path):
    s = Session.create(experiment, tmp_path, "Ana", 3, seed=2)  # 3% of 36 -> 2 images (rounded up)
    assert s.target == 2
    with pytest.raises(GameError) as caught:
        s.answer("Control")
    assert caught.value.code == "no_image"
    for _ in range(2):
        s.serve()
        s.answer("Control")
    with pytest.raises(GameError) as caught:
        s.serve()
    assert caught.value.code == "finished" and "Thank you" in caught.value.message
    s2 = Session.create(experiment, tmp_path, "Ana", 10)  # resuming with a higher percentage goes on
    assert s2.target == 4 and s2.n_answered == 2 and s2.serve()["number"] == 3


def test_restart_keeps_a_backup(experiment, tmp_path):
    s = Session.create(experiment, tmp_path, "Ana", 50, seed=1)
    s.serve(); s.answer("Control")
    fresh = Session.create(experiment, tmp_path, "Ana", 50, seed=1, restart=True)
    assert fresh.n_answered == 0
    assert list((tmp_path / "Ana").glob("game_results.*.bak.csv"))


def test_resuming_without_restart_keeps_results_even_if_images_are_added(experiment, tmp_path):
    s = Session.create(experiment, tmp_path, "Ana", 50, seed=1)
    s.serve(); s.answer("Control")
    assert Session.create(experiment, tmp_path, "Ana", 50).n_answered == 1


def test_undo_last_guess_shows_the_image_again(experiment, tmp_path):
    s = Session.create(experiment, tmp_path, "Ana", 50, seed=1)
    first = s.serve()["path"]
    s.answer("Control")
    assert s.undo_last() == {"forgotten_guess": "Control"}
    assert s.n_answered == 0 and s.serve()["path"] == first
    with pytest.raises(GameError):
        s.undo_last()  # nothing left to undo


@pytest.mark.parametrize("name", ["", "  ", "..", "a/b", "a\\b", "x:y"])
def test_user_name_cannot_escape_the_results_folder(experiment, tmp_path, name):
    with pytest.raises(GameError) as caught:
        Session.create(experiment, tmp_path / "R", name, 20)
    assert caught.value.code == "bad_user_name"


def test_bad_inputs_are_explained(experiment, tmp_path):
    for percentage in (0, 101):
        with pytest.raises(GameError) as caught:
            Session.create(experiment, tmp_path, "Ana", percentage)
        assert caught.value.code == "bad_percentage"
    with pytest.raises(GameError) as caught:
        Session.create(tmp_path / "missing", tmp_path, "Ana", 20)
    assert caught.value.code == "no_experiment"
    with pytest.raises(GameError) as caught:
        Session.open(tmp_path, "Nobody")
    assert caught.value.code == "no_session"


def test_results_of_the_first_versions_can_be_resumed(experiment, tmp_path):
    table = scan_experiment(experiment)
    legacy = pd.DataFrame(
        {
            "Filename": table["image_file"].map(lambda f: f.split("/")[-1]),
            "Condition": table["condition"],
            "Repeat": table["biological_repeat"],
            "FilePath": [str(experiment / f) for f in table["image_file"]],
            "UserGuess": [None] * len(table),
            "DecisionTime": [None] * len(table),
        }
    )
    legacy.loc[0, ["UserGuess", "DecisionTime"]] = ["Control", 3.5]
    folder = tmp_path / "Old"
    folder.mkdir()
    legacy.to_csv(folder / "game_results.csv", index=False)
    s = Session.create(experiment, tmp_path, "Old", 25)
    assert s.n_answered == 1 and s.answered().loc[0, "decision_time_s"] == 3.5
    assert s.answered().loc[0, "image_file"] == "Control/R1/FOV1.tif"
    assert s.serve()["number"] == 2
