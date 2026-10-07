"""Reading an experiment folder: ``<experiment>/<condition>/<biological repeat>/*.tif``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = (".tif", ".tiff")
# Column names shared with the other MorphoBricks tables (condition, biological_repeat, image_file).
DATASET_COLUMNS = ["image_file", "condition", "biological_repeat"]

STRUCTURE_HELP = (
    "Expected one folder per condition, and inside it one folder per biological repeat holding the TIFF files:\n"
    "  Experiment/Control/R1/FOV1.tif, Experiment/Control/R2/FOV1.tif, Experiment/Mutant/R1/FOV1.tif, ..."
)


def _visible_folders(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))),
        key=lambda p: p.name.lower(),
    )


def scan_experiment(folder) -> pd.DataFrame:
    """One row per image: ``image_file`` (relative to ``folder``, with ``/``), ``condition`` and ``biological_repeat``.

    Hidden folders and files (starting with ``.`` or ``_``, such as macOS ``._x.tif``) are ignored. The rows are sorted, so the
    scan is the same on every machine. Raises ``ValueError`` (saying what was expected) when no image is found or when there is
    only one condition (there would be nothing to guess).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("The experiment folder does not exist: %s" % folder)
    rows = []
    for condition in _visible_folders(folder):
        for repeat in _visible_folders(condition):
            for image in sorted(repeat.iterdir(), key=lambda p: p.name.lower()):
                if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES and not image.name.startswith((".", "_")):
                    rows.append(
                        {
                            "image_file": image.relative_to(folder).as_posix(),
                            "condition": condition.name,
                            "biological_repeat": repeat.name,
                        }
                    )
    if not rows:
        raise ValueError("No TIFF image was found in %s.\n%s" % (folder, STRUCTURE_HELP))
    table = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    names = pd.Series(table["condition"].unique())
    clash = names[names.str.lower().duplicated(keep=False)]
    if len(clash):
        raise ValueError("Conditions that differ only in upper/lower case cannot be told apart in a guess: %s. Rename one folder." % ", ".join(sorted(clash)))
    if table["condition"].nunique() < 2:
        raise ValueError(
            "Only one condition was found (%s): the game needs at least two.\n%s" % (table["condition"].iloc[0], STRUCTURE_HELP)
        )
    return table


def describe(table: pd.DataFrame) -> str:
    """The conditions, repeats and number of images, as text."""
    lines = ["%d conditions, %d images" % (table["condition"].nunique(), len(table))]
    for condition, group in table.groupby("condition", sort=True):
        repeats = group.groupby("biological_repeat", sort=True).size()
        lines.append(
            "- %s: %d repeat(s), %d images (%s)"
            % (condition, len(repeats), len(group), ", ".join("%s: %d" % (r, n) for r, n in repeats.items()))
        )
    return "\n".join(lines)
