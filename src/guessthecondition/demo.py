"""A small synthetic experiment to try the game: two conditions that differ in what the images show (not in brightness)."""

from __future__ import annotations

from pathlib import Path

DEMO_FOLDER = "GuessTheCondition_demo"
CONDITIONS = ("Control", "Treated")
REPEATS = ("R1", "R2", "R3")


def make_demo_experiment(folder, images_per_repeat: int = 4, size: int = 128, seed: int = 0) -> Path:
    """Write ``<folder>/GuessTheCondition_demo/<condition>/<repeat>/FOVn.tif`` and return that folder.

    *Control* images hold a few small bright spots, *Treated* images many larger blobs; every image is normalised for display,
    so the two are told apart by the pattern, not by the brightness. Fixed by ``seed``. ``FileExistsError`` if the demo
    folder already exists (nothing is overwritten)."""
    import numpy as np
    import tifffile
    from scipy.ndimage import gaussian_filter

    root = Path(folder) / DEMO_FOLDER
    if root.exists():
        raise FileExistsError("%s already exists: choose another folder, or delete it." % root)
    rng = np.random.default_rng(seed)
    for k, condition in enumerate(CONDITIONS):
        for repeat in REPEATS:
            (root / condition / repeat).mkdir(parents=True)
            for i in range(images_per_repeat):
                n_spots = int(rng.integers(8, 14)) if k == 0 else int(rng.integers(55, 80))
                sigma = 1.6 if k == 0 else 3.4
                canvas = np.zeros((size, size))
                canvas[rng.integers(0, size, n_spots), rng.integers(0, size, n_spots)] = rng.uniform(0.6, 1.0, n_spots)
                image = gaussian_filter(canvas, sigma) * (60 if k == 0 else 30) * sigma**2
                image = image + rng.normal(0.02, 0.01, image.shape) * image.max()
                scaled = (image / image.max() * rng.uniform(1500, 3500)).clip(0, 4095).astype(np.uint16)
                tifffile.imwrite(root / condition / repeat / ("FOV%d.tif" % (i + 1)), scaled)
    return root
