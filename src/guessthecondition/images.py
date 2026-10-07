"""Reading an image for display: every channel on its own, normalised like ImageJ's "Auto"."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_channel(channel: np.ndarray, lower_percentile: float = 0.01, upper_percentile: float = 99.99) -> np.ndarray:
    """Percentile-based scaling to [0, 1] (float32). A flat channel becomes all zeros instead of dividing by zero."""
    channel = np.asarray(channel, dtype=np.float64)
    lower, upper = np.nanpercentile(channel, [lower_percentile, upper_percentile])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.zeros(channel.shape, dtype=np.float32)
    return ((np.clip(channel, lower, upper) - lower) / (upper - lower)).astype(np.float32)


def _channels_first(data: np.ndarray, axes: str) -> np.ndarray:
    """(C, Y, X) from an array whose axes are named (TIFF metadata). Z and T are max-projected, the other axes dropped by max."""
    axes = axes.upper()
    if len(axes) != data.ndim:  # unnamed axes: guess below
        axes = "Q" * (data.ndim - 2) + "YX"
    if "C" not in axes:
        # without a channel axis: a short leading axis (up to 5) is read as channels, anything longer as z / time
        if data.ndim == 3 and data.shape[0] <= 5 and axes[0] in "QS":
            axes = "C" + axes[1:]
    for letter in [a for a in axes if a not in "CYX"]:
        index = axes.index(letter)
        data = data.max(axis=index)
        axes = axes[:index] + axes[index + 1 :]
    if "C" not in axes:
        return data[np.newaxis]
    order = [axes.index(a) for a in "CYX" if a in axes]
    return np.transpose(data, order)


def tile_channels(channels: np.ndarray, gap_fraction: float = 0.02) -> np.ndarray:
    """One 2D picture (Y, X') from (C, Y, X): the channels side by side, separated by a dark gap. A single channel is
    returned as it is. Used where a host shows one image per result (Napari, Fiji)."""
    channels = np.asarray(channels)
    if channels.ndim == 2:
        return channels
    if len(channels) == 1:
        return channels[0]
    gap = np.zeros((channels.shape[1], max(1, round(gap_fraction * channels.shape[2]))), dtype=channels.dtype)
    parts = []
    for i, channel in enumerate(channels):
        if i:
            parts.append(gap)
        parts.append(channel)
    return np.concatenate(parts, axis=1)


def load_for_display(path) -> np.ndarray:
    """(C, Y, X) float32 in [0, 1], one normalised plane per channel; a 2D image is one channel.

    z-stacks and time series are projected (maximum) so that one picture is shown per channel. A 3D file without axis
    metadata is read as channels when its first axis has at most 5 planes, else as a z-stack.
    """
    import tifffile

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("image not found: %s" % path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data, axes = series.asarray(), series.axes
    if data.ndim < 2:
        raise ValueError("%s is not an image (shape %s)" % (path.name, data.shape))
    channels = _channels_first(data, axes)
    return np.stack([normalize_channel(c) for c in channels])
