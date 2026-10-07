"""Guess the Condition: can you tell the experimental conditions apart by looking at the images?

The package is the API behind the notebook, the Napari and Fiji tools (``guessthecondition_lc_tools``) and any script::

    from guessthecondition import Session, analyze

    session = Session.create("Experiments", "Results", user_name="Ana", percentage_to_test=20)
    item = session.serve()          # a blinded image: its path and its number, never its condition
    session.answer("Mutant")        # records the guess and the time taken
    analysis = analyze(session.answered())

Importing it stays light (numpy, pandas and tifffile only when an image is read; scipy and matplotlib only to analyse or report).
"""

__version__ = "0.1.0"

from .analysis import Analysis, Readout, analyze
from .dataset import describe, scan_experiment
from .demo import make_demo_experiment
from .images import load_for_display, tile_channels
from .session import GameError, Session

__all__ = [
    "Analysis",
    "GameError",
    "Readout",
    "Session",
    "analyze",
    "describe",
    "load_for_display",
    "make_demo_experiment",
    "scan_experiment",
    "tile_channels",
]
