"""The tool declarations must stay valid, cheap to import and blind to the answers.

    pip install pytest labconstrictor-tools
    pytest tests
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("labconstrictor_tools")


def _tools():
    from labconstrictor_tools.introspection import describe_tools

    importlib.import_module("guessthecondition_lc_tools")  # the declarations register their tools when imported
    return {t["id"]: t for t in describe_tools("guessthecondition_lc_tools")["tools"]}


def test_declarations_are_valid():
    assert set(_tools()) == {"prepare_game", "play_round", "undo_last_guess", "analyze_results"}


def test_importing_the_declarations_stays_light():
    """In a fresh interpreter (purging modules in this one would break pandas for the other tests)."""
    code = (
        "import sys, guessthecondition_lc_tools\n"
        "heavy = [m for m in ('guessthecondition', 'pandas', 'scipy', 'matplotlib', 'tifffile') if m in sys.modules]\n"
        "assert not heavy, heavy"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": src})
    assert done.returncode == 0, done.stderr


def test_the_image_shown_has_no_condition_in_its_outputs():
    play = _tools()["play_round"]
    assert [o["name"] for o in play["outputs"]] == ["image", "values"]
    assert next(o for o in play["outputs"] if o["name"] == "image")["axes"] == "YX"


def test_a_guess_is_optional_and_every_other_parameter_has_a_default_or_is_a_folder():
    for tool in _tools().values():
        for p in tool["inputs"]:
            assert p["type"] == "folder" or "default" in p or p.get("nullable"), (tool["id"], p["name"])
    guess = next(p for p in _tools()["play_round"]["inputs"] if p["name"] == "guess")
    assert guess.get("nullable") and "default" not in guess


def test_percentage_is_bounded():
    p = next(p for p in _tools()["prepare_game"]["inputs"] if p["name"] == "percentage_to_test")
    assert (p["minimum"], p["maximum"], p["default"]) == (1, 100, 20)
