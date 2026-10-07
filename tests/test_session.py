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
        assert set(item) == {"number", "of", "path", "fresh"}  # nothing that names the condition
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
    again = Session.open(tmp_path, "Ana").serve(now=20)
    assert again["path"] == first["path"] and again["number"] == 1
    assert Session.open(tmp_path, "Ana").n_answered == 0


def test_the_clock_runs_from_the_first_display_unless_the_player_came_back_after_a_break(experiment, tmp_path):
    s = Session.create(experiment, tmp_path, "Ana", 50, seed=2)
    s.serve(now=100)
    s.serve(now=130)                      # "show it again": the player has been looking at it all along
    assert s.answer("1", now=145)["decision_time_s"] == 45
    s.serve(now=200)
    s.serve(now=200 + 601)                # came back after more than ten minutes: the clock starts again
    assert s.answer("1", now=801 + 7)["decision_time_s"] == 7


def test_two_front_ends_do_not_erase_each_others_guesses(experiment, tmp_path):
    napari = Session.create(experiment, tmp_path, "Ana", 100, seed=2)
    notebook = Session.open(tmp_path, "Ana")      # opened before the other one plays
    for k in range(3):
        napari.serve(now=10 * k); napari.answer("1", now=10 * k + 1)
    notebook.serve(now=40)
    notebook.answer("2", now=41)                  # a stale copy must not overwrite the three guesses of the other
    final = Session.open(tmp_path, "Ana")
    assert final.n_answered == 4 and list(final.answered()["guess"]) == ["Control", "Control", "Control", "Drug"]


def test_numeric_condition_names_disable_numbers(tmp_path):
    import tifffile
    for c in ("10", "2"):
        for r in ("R1", "R2"):
            (tmp_path / "E" / c / r).mkdir(parents=True)
            tifffile.imwrite(tmp_path / "E" / c / r / "a.tif", np.zeros((4, 4), np.uint8))
    s = Session.create(tmp_path / "E", tmp_path / "R", "Ana", 100)
    assert not s.numbered and s.conditions_text() == "10 | 2"
    s.serve()
    assert s.resolve_guess("2") == "2" and s.resolve_guess("10") == "10"
    with pytest.raises(GameError):
        s.resolve_guess("3")


def test_conditions_that_differ_only_in_case_are_refused(tmp_path):
    import tifffile
    for c in ("Control", "control", "Other"):
        (tmp_path / "E" / c / "R1").mkdir(parents=True)
        tifffile.imwrite(tmp_path / "E" / c / "R1" / "a.tif", np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="upper/lower case"):
        scan_experiment(tmp_path / "E")


def test_an_unreadable_image_is_left_out_not_a_dead_end(experiment, tmp_path):
    from guessthecondition import load_for_display
    s = Session.create(experiment, tmp_path, "Ana", 100, seed=2)
    first = s.serve()["path"]
    first.write_bytes(b"not a tiff")
    item, data, skipped = s.serve_readable(load_for_display)
    assert skipped == [first.name] and item["path"] != first and data.ndim == 3
    reopened = Session.open(tmp_path, "Ana")
    assert reopened.total == 35 and reopened.status()["unreadable_images_left_out"] == 1
    assert reopened.table["skipped"].notna().sum() == 1


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


def test_numeric_folder_names_survive_a_resume_and_the_analysis(tmp_path):
    import tifffile
    from guessthecondition import analyze, read_results
    for c in ("10", "20"):
        for r in ("1", "2", "3"):
            (tmp_path / "E" / c / r).mkdir(parents=True)
            for i in range(2):
                tifffile.imwrite(tmp_path / "E" / c / r / ("a%d.tif" % i), np.zeros((4, 4), np.uint8))
    s = Session.create(tmp_path / "E", tmp_path / "R", "Ana", 100, seed=1)
    for _ in range(4):
        s.serve(now=0)
        truth = s.table.loc[s.table["image_file"] == s.state["pending"]["image_file"], "condition"].iloc[0]
        s.answer(truth, now=1)
    again = Session.open(tmp_path / "R", "Ana")           # re-read from the CSV
    assert again.conditions == ["10", "20"] and set(again.table["biological_repeat"]) == {"1", "2", "3"}
    saved = read_results(tmp_path / "R" / "Ana" / "game_results.csv")
    assert analyze(saved).accuracy == 1.0 and analyze(again.table).correct == 4
