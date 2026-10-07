"""Guess the Condition for Napari, Fiji and the command line (LabConstrictor tools bridge).

This module only *declares* the tools: it imports `labconstrictor_tools` and the standard library at the top and the
`guessthecondition` package inside the functions (hosts import it to list the tools, which must stay fast). Its name must be
`<package>_lc_tools` for the installer to register it.

A game is played in three steps, each one run of a tool; the state lives in the results folder, so the steps are independent:

    labconstrictor-tools run GuessTheCondition create_demo_experiment output_folder=.      # optional: a demo experiment to try
    labconstrictor-tools run GuessTheCondition prepare_game experiment_folder=Experiments results_folder=Results user_name=Ana
    labconstrictor-tools run GuessTheCondition play_round   results_folder=Results user_name=Ana              # shows an image
    labconstrictor-tools run GuessTheCondition play_round   results_folder=Results user_name=Ana guess=Mutant # records it, shows the next
    labconstrictor-tools run GuessTheCondition analyze_results results_folder=Results user_name=Ana

    labconstrictor-tools check --module guessthecondition_lc_tools
    labconstrictor-tools test  --module guessthecondition_lc_tools --cases lc_tests/cases.json
"""

import time
from typing import Annotated, Optional

from labconstrictor_tools import (
    Advanced,
    ChoicesFrom,
    ClearAfterRun,
    Description,
    FileOut,
    Folder,
    Group,
    ImageOut,
    Label,
    Max,
    Min,
    Name,
    Axes,
    Replace,
    Scalars,
    TableOut,
    ToolError,
    check_cancel,
    progress,
    tool,
)

_RESULTS = Annotated[
    Folder, Group("Game"), Description("Folder that holds the results of every player (it must exist); a folder with the user's name is made inside it")
]
_USER = Annotated[str, Group("Game"), Label("Your name"), Description("Names your game: your results are kept in a folder with this name, so you can stop and resume")]


def _game_error(error):
    return ToolError(error.code, error.message)


@tool("Create a demo experiment")
def create_demo_experiment(
    output_folder: Annotated[Folder, Group("Demo"), Description("An existing folder; the demo experiment is made inside it, in a folder called GuessTheCondition_demo")],
    images_per_repeat: Annotated[int, Min(1), Max(50), Group("Demo"), Description("Images per condition and repeat (2 conditions x 3 repeats)")] = 4,
) -> Scalars:
    """Make a small synthetic experiment (two conditions, three repeats) to try the game before using your own images."""
    from guessthecondition import make_demo_experiment

    try:
        root = make_demo_experiment(output_folder, images_per_repeat)
    except FileExistsError as error:
        raise ToolError("already_exists", str(error)) from error
    return {"experiment_folder": str(root), "images": 2 * 3 * int(images_per_repeat), "next": "Use this folder as the experiment folder of 'Prepare the game'"}


@tool("Prepare the game")
def prepare_game(
    experiment_folder: Annotated[
        Folder, Group("Game"), Description("One folder per condition, inside it one folder per biological repeat holding the TIFF images")
    ],
    results_folder: _RESULTS,
    user_name: _USER = "YourName",
    percentage_to_test: Annotated[
        int, Min(1), Max(100), Group("Game"), Description("Share of the images you will guess (rounded up); images are spread over the conditions and repeats")
    ] = 20,
    restart: Annotated[bool, Group("Game"), Description("Start again: your earlier results are kept as a backup. Unticked, an existing game is resumed")] = False,
    random_seed: Annotated[Optional[int], Min(0), Group("Game"), Advanced(), Description("Seed of the order of the images; unset = a random one, kept with the game")] = None,
) -> Scalars:
    """Scan the experiment folder and start (or resume) your game."""
    progress(0.1, "reading the experiment folder")
    from guessthecondition import Session, describe, scan_experiment
    from guessthecondition.session import GameError

    try:
        session = Session.create(experiment_folder, results_folder, user_name, percentage_to_test, random_seed, restart)
    except GameError as error:
        raise _game_error(error) from error
    check_cancel()
    status = session.status()
    return {
        **status,
        "game_folder": str(session.folder),
        "experiment": describe(scan_experiment(session.experiment_dir)).replace("\n", " "),
        "next": "Run 'Play a round' to see the first image" if session.remaining else "Nothing left to guess: analyse the results, or raise the percentage",
    }


@tool("Show the conditions")
def list_conditions(results_folder: _RESULTS, user_name: _USER = "YourName") -> Scalars:
    """The conditions you can name in 'Play a round' (needs a game prepared for this name). Hosts use it to fill the guess list."""
    from guessthecondition import Session
    from guessthecondition.session import GameError

    try:
        session = Session.open(results_folder, user_name)
    except GameError as error:
        raise _game_error(error) from error
    return {"choices": list(session.conditions), "conditions": session.conditions_text()}


@tool("Play a round")
def play_round(
    results_folder: _RESULTS,
    user_name: _USER = "YourName",
    guess: Annotated[
        Optional[str],
        Group("Game"),
        ChoicesFrom("list_conditions", depends=["results_folder", "user_name"]),
        ClearAfterRun(),
        Description("Your guess for the image on screen: a condition name or its number in the list. Leave unset to see the first image. Change it every round"),
    ] = None,
) -> tuple[Annotated[ImageOut, Name("image"), Axes("YX"), Replace()], Scalars]:
    """Record your guess (when you give one) and show the next image. The condition of an image is never shown."""
    from guessthecondition import Session, load_for_display, tile_channels
    from guessthecondition.session import GameError

    called_at = time.time()  # the guess is timed when the tool starts, not after the imports and the image loading below
    recorded = None
    try:
        session = Session.open(results_folder, user_name)
        if guess is not None and not str(guess).strip():
            raise GameError("empty_guess", "The guess is ticked but empty: type a condition name or its number (%s), or untick it to see the image again." % session.conditions_text())
        recorded = session.answer(guess, now=called_at) if guess is not None else None
        progress(0.4, "loading the next image")
        item, tiled, skipped = session.serve_readable(lambda path: tile_channels(load_for_display(path)))
    except GameError as error:
        if error.code == "finished" and recorded is not None:
            raise ToolError(
                "no_result",
                "Your guess (%s, %.1f s) is saved. That was the last image: all %d images are guessed. Thank you! Run 'Analyse the results'."
                % (recorded["guess"], recorded["decision_time_s"], session.target),
            ) from error
        raise _game_error(error) from error
    image = tiled
    if item["fresh"]:
        session.start_clock()  # the image is ready: from here on the player is looking at it
    values = {
        "image": "%d of %d" % (item["number"], item["of"]),
        "conditions": session.conditions_text(),
        "how_to_answer": "Run again with your guess: a condition name or its number",
    }
    if skipped:
        values["left_out_unreadable"] = ", ".join(skipped)
    if recorded is not None:
        values["recorded"] = "%s (%.1f s)" % (recorded["guess"], recorded["decision_time_s"])
    return image, values


@tool("Undo my last guess")
def undo_last_guess(results_folder: _RESULTS, user_name: _USER = "YourName") -> Scalars:
    """Forget the last guess you recorded (for example one made by mistake); that image is shown again by 'Play a round'."""
    from guessthecondition import Session
    from guessthecondition.session import GameError

    try:
        session = Session.open(results_folder, user_name)
        undone = session.undo_last()
    except GameError as error:
        raise _game_error(error) from error
    return {"forgotten_guess": undone["forgotten_guess"], "guessed": session.n_answered, "next": "Run 'Play a round' to see that image again"}


@tool("Analyse the results")
def analyze_results(
    results_folder: _RESULTS,
    user_name: _USER = "YourName",
    random_seed: Annotated[Optional[int], Min(0), Group("Game"), Advanced(), Description("Seed of the randomization tests; unset = 0")] = None,
) -> tuple[
    Scalars,
    Annotated[TableOut, Name("by_repeat")],
    Annotated[TableOut, Name("by_condition")],
    Annotated[FileOut, Name("report_pdf")],
]:
    """Can the conditions be told apart? Accuracy against chance across the biological repeats, per condition, the confusion
    matrix and decision times; the PDF holds the text, the figures and the tables."""
    from guessthecondition import Session, analyze
    from guessthecondition.report import write_report
    from guessthecondition.session import GameError

    try:
        session = Session.open(results_folder, user_name)
        answered = session.answered()
        if answered.empty:
            raise GameError("no_guesses", "No image has been guessed yet: play some rounds first.")
    except GameError as error:
        raise _game_error(error) from error
    progress(0.2, "computing")
    analysis = analyze(session.table, seed=0 if random_seed is None else int(random_seed))  # the whole table: chance comes from the experiment
    check_cancel()
    progress(0.6, "drawing the figures and writing the report")
    paths = write_report(analysis, answered, session.folder / "analysis")
    across = analysis.across
    values = {
        "answer": "%s %s" % (analysis.readout.marker, analysis.readout.chip),
        "reading": analysis.readout.text,
        "images_guessed": analysis.n,
        "accuracy": round(analysis.accuracy, 4),
        "chance": round(analysis.chance, 4),
        "biological_repeats": across["n_repeats"],
        "repeats_above_chance": across["repeats_above_chance"],
        "report_folder": str(session.folder / "analysis"),
    }
    if across["p"] == across["p"]:
        values["p_across_repeats"] = round(across["p"], 5)
    values["p_randomization_these_images_exploratory"] = round(analysis.p_randomization, 5)
    if analysis.notes:
        values["notes"] = " ".join(analysis.notes)
    return values, analysis.per_repeat, analysis.per_condition, paths["analysis_results.pdf"]
