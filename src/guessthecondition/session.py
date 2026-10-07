"""A game session: which images are shown, in which order, what was guessed and how long it took.

Everything lives in the session folder ``<results>/<user>/``, so a session survives closing the notebook, Napari or Fiji and
every tool call is independent of the others:

- ``game_results.csv``: one row per image (``image_file``, ``condition``, ``biological_repeat``, ``play_order``, ``guess``,
  ``decision_time_s``, ``answered_at``). The name is the one the first versions used.
- ``session.json``: the experiment folder, the percentage to test, the seed and the image waiting for an answer.

The images are shown in a stratified order (see ``play_order``): the first images are spread over the conditions and the
repeats, and the order is fixed by the seed, so a resumed game goes on where it stopped. Nothing returned to the player holds
the condition or the file name of an image.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import DATASET_COLUMNS, scan_experiment

TABLE_NAME = "game_results.csv"
STATE_NAME = "session.json"
TABLE_COLUMNS = [*DATASET_COLUMNS, "play_order", "guess", "decision_time_s", "answered_at", "skipped"]
RESUME_GAP_S = 600  # an image shown again within this time keeps its clock (the player never looked away); later, the clock restarts
_LEGACY = {"Filename": None, "Condition": "condition", "Repeat": "biological_repeat", "UserGuess": "guess", "DecisionTime": "decision_time_s"}


def read_results(path) -> pd.DataFrame:
    """Read a ``game_results.csv``. The text columns are read as text: folders named with digits (conditions ``10``, ``20``,
    repeats ``1``, ``2``) must not turn into numbers, and a guess column holding numbers and gaps must not turn into floats."""
    return pd.read_csv(path, dtype={c: str for c in ("image_file", "condition", "biological_repeat", "guess", "answered_at", "skipped")})


class GameError(Exception):
    """A problem the player can fix; ``code`` is a short machine-readable name."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def play_order(table: pd.DataFrame, seed: int) -> np.ndarray:
    """The play order (0 = first): the images are drawn round by round, one from every (condition, repeat) group that still has
    images, the groups in a random order each round, the images of a group in a random order. Whatever number of images is
    stopped at, the groups are represented as evenly as their sizes allow. Fixed by ``seed``."""
    rng = np.random.default_rng(seed)
    groups = {key: list(rng.permutation(idx.to_numpy())) for key, idx in table.groupby(["condition", "biological_repeat"], sort=True).groups.items()}
    order = np.empty(len(table), dtype=int)
    position = 0
    while any(groups.values()):
        keys = [k for k, v in groups.items() if v]
        for k in rng.permutation(len(keys)):
            order[groups[keys[k]].pop()] = position
            position += 1
    return order


def _check_user_name(user_name: str) -> str:
    name = (user_name or "").strip()
    if not name or name in (".", "..") or re.search(r'[\\/:*?"<>|\x00]', name):
        raise GameError(
            "bad_user_name", "The name is used as a folder name: it cannot be empty, '.' or '..' or contain / \\ : * ? \" < > |"
        )
    return name


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class Session:
    """One player's game on one experiment (see the module docstring)."""

    def __init__(self, folder: Path, table: pd.DataFrame, state: dict):
        self.folder, self.table, self.state = Path(folder), table, state

    # ---- creating and opening ------------------------------------------------
    @classmethod
    def create(cls, experiment_dir, results_dir, user_name: str, percentage_to_test: float = 20, seed: int | None = None, restart: bool = False):
        """Start a session, or resume the one of ``user_name`` in ``results_dir`` (``restart=True`` starts it again, and the
        earlier results are kept as ``game_results.<time>.bak.csv``). Resuming accepts a new ``percentage_to_test``."""
        if not (0 < percentage_to_test <= 100):
            raise GameError("bad_percentage", "The percentage of images to test must be above 0 and at most 100.")
        experiment_dir = Path(experiment_dir)
        if not experiment_dir.is_dir():
            raise GameError("no_experiment", "The experiment folder does not exist: %s" % experiment_dir)
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        folder = results_dir / _check_user_name(user_name)
        folder.mkdir(parents=True, exist_ok=True)
        table_path, state_path = folder / TABLE_NAME, folder / STATE_NAME
        if table_path.exists() and not restart:
            session = cls._resume(folder, experiment_dir)
            session.state["percentage_to_test"] = float(percentage_to_test)
            session.save()
            return session
        if table_path.exists():
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            os.replace(table_path, folder / ("game_results.%s.bak.csv" % stamp))
            if state_path.exists():
                os.replace(state_path, folder / ("session.%s.bak.json" % stamp))
        try:
            table = scan_experiment(experiment_dir)
        except ValueError as error:
            raise GameError("bad_experiment", str(error)) from error
        seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
        table["play_order"] = play_order(table, seed)
        table["guess"] = None
        table["decision_time_s"] = np.nan
        table["answered_at"] = None
        table["skipped"] = None
        state = {
            "version": 1,
            "experiment_dir": str(experiment_dir.resolve()),
            "user_name": user_name.strip(),
            "percentage_to_test": float(percentage_to_test),
            "seed": seed,
            "pending": None,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        session = cls(folder, table.sort_values("play_order").reset_index(drop=True), state)
        session.save()
        return session

    @classmethod
    def open(cls, results_dir, user_name: str):
        """The session of ``user_name`` in ``results_dir`` (``GameError('no_session')`` when there is none)."""
        folder = Path(results_dir) / _check_user_name(user_name)
        if not (folder / TABLE_NAME).exists():
            raise GameError("no_session", "There is no game for '%s' in %s: prepare the game first." % (user_name, results_dir))
        return cls._resume(folder, None)

    @classmethod
    def _resume(cls, folder: Path, experiment_dir: Path | None):
        table = read_results(folder / TABLE_NAME)
        state_path = folder / STATE_NAME
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:  # results of the first versions: only the table
            if experiment_dir is None:
                raise GameError("no_session", "%s has no %s: prepare the game again with the experiment folder." % (folder, STATE_NAME))
            table = cls._from_legacy(table, experiment_dir)
            state = {
                "version": 1,
                "experiment_dir": str(experiment_dir.resolve()),
                "user_name": folder.name,
                "percentage_to_test": 20.0,
                "seed": 0,
                "pending": None,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        if "skipped" not in table.columns:
            table["skipped"] = None
        for column in TABLE_COLUMNS:
            if column not in table.columns:
                raise GameError("bad_results", "%s lacks the column '%s'." % (folder / TABLE_NAME, column))
        for column in ("guess", "answered_at", "skipped"):  # an empty column is read as float: it must be able to hold text
            table[column] = table[column].astype(object).where(table[column].notna(), None)
        table["decision_time_s"] = pd.to_numeric(table["decision_time_s"], errors="coerce")
        return cls(folder, table.sort_values("play_order").reset_index(drop=True), state)

    @staticmethod
    def _from_legacy(table: pd.DataFrame, experiment_dir: Path) -> pd.DataFrame:
        """Results written by the first versions (Filename, Condition, Repeat, FilePath, UserGuess, DecisionTime)."""
        table = table.rename(columns={k: v for k, v in _LEGACY.items() if v})
        if "FilePath" not in table.columns:
            raise GameError("bad_results", "These results are not in a format this version can read.")
        root = experiment_dir.resolve()

        def relative(path: str) -> str:
            try:
                return Path(path).resolve().relative_to(root).as_posix()
            except ValueError:
                return Path(path).as_posix()

        table["image_file"] = table["FilePath"].map(relative)
        fresh = scan_experiment(experiment_dir)  # images added since are appended, unanswered
        known = table.set_index("image_file")
        missing = fresh[~fresh["image_file"].isin(known.index)]
        table = pd.concat([table.drop(columns=["FilePath", "Filename"], errors="ignore"), missing], ignore_index=True)
        table["answered_at"] = None
        table["skipped"] = None
        done = table["guess"].notna().to_numpy()
        order = np.empty(len(table), dtype=int)
        order[np.concatenate([np.flatnonzero(done), np.flatnonzero(~done)])] = np.arange(len(table))
        table["play_order"] = order
        return table[TABLE_COLUMNS]

    # ---- state ---------------------------------------------------------------
    def reload(self) -> None:
        """Read the table and the state from the folder again: another front end (Napari, Fiji, the notebook) may have played
        since this object was made, and saving a stale copy would erase its guesses. Called before every change."""
        fresh = self._resume(self.folder, None)
        self.table, self.state = fresh.table, fresh.state

    def save(self) -> None:
        text = self.table[TABLE_COLUMNS].to_csv(index=False)
        _write_atomic(self.folder / TABLE_NAME, text)
        _write_atomic(self.folder / STATE_NAME, json.dumps(self.state, indent=2))

    @property
    def experiment_dir(self) -> Path:
        return Path(self.state["experiment_dir"])

    @property
    def conditions(self) -> list[str]:
        return sorted(self.table["condition"].unique(), key=str.lower)

    @property
    def total(self) -> int:
        """The images that can be played (those that could not be read are left out)."""
        return int(self.table["skipped"].isna().sum())

    @property
    def target(self) -> int:
        """Images to guess: the percentage of the images, rounded up, at least 1 and at most all of them."""
        return min(self.total, max(1, math.ceil(self.state["percentage_to_test"] / 100 * self.total - 1e-9)))

    @property
    def n_answered(self) -> int:
        return int(self.table["guess"].notna().sum())

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.n_answered)

    def answered(self) -> pd.DataFrame:
        """The images guessed so far (for ``analyze``), with the true ``condition``."""
        return self.table[self.table["guess"].notna()].sort_values("play_order").reset_index(drop=True)

    @property
    def numbered(self) -> bool:
        """Conditions can be guessed by number, unless a condition is itself named with digits (then only names count)."""
        return not any(c.strip().isdigit() for c in self.conditions)

    def conditions_text(self) -> str:
        if not self.numbered:
            return " | ".join(self.conditions)
        return " | ".join("%d: %s" % (i, c) for i, c in enumerate(self.conditions, 1))

    def status(self) -> dict:
        return {
            "images": self.total,
            "to_guess": self.target,
            "guessed": self.n_answered,
            "remaining": self.remaining,
            "conditions": self.conditions_text(),
            **({"unreadable_images_left_out": int(self.table["skipped"].notna().sum())} if self.table["skipped"].notna().any() else {}),
        }

    # ---- playing -------------------------------------------------------------
    def serve(self, now: float | None = None) -> dict:
        """The image to guess: the one still waiting for an answer, else the next in the play order. Returns ``{"number",
        "path", "of"}`` (``number`` counts from 1, ``of`` is the number of images to guess); starts the clock for the guess.
        ``GameError('finished')`` when the percentage to test has been reached."""
        self.reload()
        now = time.time() if now is None else now
        pending = self.state.get("pending")
        if pending is None:
            if self.remaining == 0:
                raise GameError(
                    "finished",
                    "All %d images to guess have been guessed. Thank you! Analyse the results, or raise the percentage to go on." % self.target,
                )
            unanswered = self.table[self.table["guess"].isna() & self.table["skipped"].isna()]
            if unanswered.empty:
                raise GameError("finished", "Every image of the experiment has been guessed.")
            pending = {"image_file": unanswered.iloc[0]["image_file"], "served_at": now}
            fresh = True
        elif now - float(pending.get("last_seen", pending["served_at"])) > RESUME_GAP_S:
            pending["served_at"] = now  # the player came back after a break: the clock starts again
            fresh = True
        else:
            fresh = False
        pending["last_seen"] = now
        self.state["pending"] = pending
        self.save()
        return {
            "number": self.n_answered + 1,
            "of": self.target,
            "path": self.experiment_dir / pending["image_file"],
            "fresh": fresh,   # the clock of this image was (re)started now
        }

    def start_clock(self, now: float | None = None) -> None:
        """(Re)start the clock of the image on screen. A front end calls it when the image is ready to be displayed, so that the
        time to read the file is not counted as thinking time."""
        self.reload()
        pending = self.state.get("pending")
        if pending is not None:
            now = time.time() if now is None else now
            pending["served_at"] = pending["last_seen"] = now
            self.save()

    def serve_readable(self, loader, now: float | None = None):
        """``serve`` and read the image with ``loader(path)``; an image that cannot be read is left out of the game (its reason
        is kept in the ``skipped`` column) and the next one is served. Returns ``(item, data, skipped_names)``."""
        skipped = []
        while True:
            item = self.serve(now)
            try:
                return item, loader(item["path"]), skipped
            except Exception as error:  # noqa: BLE001 - any unreadable file must not block the game
                self.skip_pending("%s: %s" % (type(error).__name__, error))
                skipped.append(item["path"].name)

    def skip_pending(self, reason: str) -> None:
        """Leave the image on screen out of the game (it could not be read)."""
        self.reload()
        pending = self.state.get("pending")
        if pending is None:
            return
        row = self.table.index[self.table["image_file"] == pending["image_file"]]
        if len(row) == 1:
            self.table.loc[row[0], "skipped"] = str(reason)[:300]
        self.state["pending"] = None
        self.save()

    def resolve_guess(self, guess: str) -> str:
        """A condition from its name (any case) or, when the conditions are numbered, its number."""
        text = (guess or "").strip()
        names = {c.lower(): c for c in self.conditions}
        if text.lower() in names:
            return names[text.lower()]
        if self.numbered and text.isdigit() and 1 <= int(text) <= len(self.conditions):
            return self.conditions[int(text) - 1]
        raise GameError("unknown_condition", "'%s' is not a condition. Choose one of: %s" % (guess, self.conditions_text()))

    def answer(self, guess: str, now: float | None = None) -> dict:
        """Record the guess for the image being shown and the time it took. Returns ``{"guess", "decision_time_s"}``."""
        self.reload()
        now = time.time() if now is None else now
        pending = self.state.get("pending")
        if pending is None:
            raise GameError("no_image", "No image is waiting for a guess: show the next image first.")
        condition = self.resolve_guess(guess)
        row = self.table.index[self.table["image_file"] == pending["image_file"]]
        if len(row) != 1:
            raise GameError("bad_results", "The image waiting for a guess is not in the results table any more.")
        seconds = max(0.0, now - float(pending["served_at"]))
        self.table.loc[row[0], "guess"] = condition
        self.table.loc[row[0], "decision_time_s"] = seconds
        self.table.loc[row[0], "answered_at"] = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
        self.state["pending"] = None
        self.state["last_answered"] = pending["image_file"]
        self.save()
        return {"guess": condition, "decision_time_s": seconds}

    def undo_last(self) -> dict:
        """Forget the last guess (the image is shown again next, its clock restarting)."""
        self.reload()
        last = self.state.get("last_answered")
        row = self.table.index[self.table["image_file"] == last] if last else []
        if len(row) != 1 or pd.isna(self.table.loc[row[0], "guess"]):
            raise GameError("nothing_to_undo", "There is no guess to undo.")
        forgotten = self.table.loc[row[0], "guess"]
        self.table.loc[row[0], ["guess", "decision_time_s", "answered_at"]] = [None, np.nan, None]
        now = time.time()
        self.state["pending"] = {"image_file": last, "served_at": now, "last_seen": now}
        self.state["last_answered"] = None
        self.save()
        return {"forgotten_guess": forgotten}
