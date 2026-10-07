import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_experiment(root: Path, conditions=("Control", "Mutant", "Drug"), repeats=("R1", "R2", "R3"), per_repeat=4, seed=0):
    """A small experiment: every condition has its own mean brightness, so the images can really be told apart."""
    import tifffile

    rng = np.random.default_rng(seed)
    for ci, condition in enumerate(conditions):
        for repeat in repeats:
            folder = root / condition / repeat
            folder.mkdir(parents=True)
            for i in range(per_repeat):
                image = rng.normal(100 + 400 * ci, 20, (24, 24)).clip(0, 4095).astype(np.uint16)
                tifffile.imwrite(folder / ("FOV%d.tif" % (i + 1)), image)
    return root


@pytest.fixture
def experiment(tmp_path):
    return make_experiment(tmp_path / "Experiment")
