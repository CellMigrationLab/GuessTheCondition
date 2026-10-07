import numpy as np
import pytest
import tifffile

from guessthecondition import describe, load_for_display, scan_experiment
from guessthecondition.images import normalize_channel


def test_scan_reads_condition_repeat_and_is_sorted(experiment):
    table = scan_experiment(experiment)
    assert list(table.columns) == ["image_file", "condition", "biological_repeat"]
    assert len(table) == 3 * 3 * 4
    assert table["image_file"].iloc[0] == "Control/R1/FOV1.tif"
    assert set(table["condition"]) == {"Control", "Mutant", "Drug"}
    assert table.equals(scan_experiment(experiment))
    assert "3 conditions, 36 images" in describe(table)


def test_scan_ignores_hidden_files_and_folders(experiment):
    (experiment / "Control" / "R1" / "._FOV1.tif").write_bytes(b"junk")
    (experiment / ".hidden" / "R1").mkdir(parents=True)
    (experiment / ".hidden" / "R1" / "x.tif").write_bytes(b"junk")
    assert len(scan_experiment(experiment)) == 36


def test_scan_accepts_upper_case_suffix_and_tiff(experiment):
    tifffile.imwrite(experiment / "Control" / "R1" / "extra.TIFF", np.zeros((4, 4), np.uint8))
    assert "Control/R1/extra.TIFF" in set(scan_experiment(experiment)["image_file"])


def test_scan_says_what_was_expected(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(ValueError, match="No TIFF image"):
        scan_experiment(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        scan_experiment(tmp_path / "nope")


def test_one_condition_is_refused(tmp_path):
    (tmp_path / "A" / "R1").mkdir(parents=True)
    tifffile.imwrite(tmp_path / "A" / "R1" / "a.tif", np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError, match="at least two"):
        scan_experiment(tmp_path)


def _write(path, array, **kw):
    tifffile.imwrite(path, array, **kw)
    return path


def test_2d_image_is_one_channel_not_one_per_row(tmp_path):
    out = load_for_display(_write(tmp_path / "a.tif", np.random.default_rng(0).integers(0, 4000, (64, 48)).astype(np.uint16)))
    assert out.shape == (1, 64, 48) and out.dtype == np.float32
    assert out.min() == 0 and out.max() == 1


def test_channels_are_separate_and_normalised_each(tmp_path):
    data = np.stack([np.arange(100).reshape(10, 10), 1000 + 5 * np.arange(100).reshape(10, 10)]).astype(np.uint16)
    out = load_for_display(_write(tmp_path / "c.tif", data, metadata={"axes": "CYX"}))
    assert out.shape == (2, 10, 10)
    assert np.allclose(out[0].max(), 1) and np.allclose(out[1].max(), 1)


def test_z_stack_is_max_projected_to_one_picture(tmp_path):
    data = np.zeros((30, 8, 8), np.uint16)
    data[17, 3, 3] = 4000
    out = load_for_display(_write(tmp_path / "z.tif", data, metadata={"axes": "ZYX"}))
    assert out.shape == (1, 8, 8)
    assert out[0, 3, 3] == 1


def test_hyperstack_tzcyx_gives_channels(tmp_path):
    data = np.random.default_rng(1).integers(0, 4000, (2, 5, 3, 16, 16)).astype(np.uint16)
    out = load_for_display(_write(tmp_path / "h.tif", data, metadata={"axes": "TZCYX"}, photometric="minisblack"))
    assert out.shape == (3, 16, 16)


def test_unlabelled_short_stack_is_channels_long_is_z(tmp_path):
    short = load_for_display(_write(tmp_path / "s.tif", np.random.default_rng(2).integers(0, 255, (3, 9, 9)).astype(np.uint8), photometric="minisblack"))
    long = load_for_display(_write(tmp_path / "l.tif", np.random.default_rng(2).integers(0, 255, (40, 9, 9)).astype(np.uint8), photometric="minisblack"))
    assert short.shape[0] == 3 and long.shape[0] == 1


def test_flat_channel_does_not_divide_by_zero():
    assert not normalize_channel(np.full((5, 5), 7)).any()


def test_missing_image_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="image not found"):
        load_for_display(tmp_path / "nope.tif")
