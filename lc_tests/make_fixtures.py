"""Host-side fixture generator for the tool cases (numpy + tifffile only): a small experiment and an empty results folder.

Two conditions that differ in brightness, three biological repeats, two images each (12 images), the first one two-channel
and the second a z-stack so that every kind of image is shown at least once.

    python lc_tests/make_fixtures.py lc_tests/fixtures
"""

import shutil
import sys
from pathlib import Path

import numpy as np
from tifffile import imwrite

out = Path(sys.argv[1])
shutil.rmtree(out, ignore_errors=True)
rng = np.random.default_rng(5)
for ci, condition in enumerate(("Control", "Mutant")):
    for repeat in ("R1", "R2", "R3"):
        folder = out / "experiment" / condition / repeat
        folder.mkdir(parents=True)
        base = lambda shape: rng.normal(200 + 600 * ci, 30, shape).clip(0, 4095).astype(np.uint16)  # noqa: E731
        imwrite(folder / "FOV1.tif", base((2, 32, 32)), metadata={"axes": "CYX"})
        imwrite(folder / "FOV2.tif", base((6, 32, 32)), metadata={"axes": "ZYX"})
(out / "results").mkdir()
print("wrote the experiment (12 images) and an empty results folder to", out)
