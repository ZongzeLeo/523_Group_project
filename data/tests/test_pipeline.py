"""Synthetic contract tests; these fixtures are not CAMUS research results."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from data.dataset import CamusDataset
from data.native import read_pair
from data.prepare import ef_group, inventory, prepare, read_splits, verify_prepared
from data.transforms import make_transform, preprocess, restore_to_native


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "raw"
    split_dir = root / "database_split"
    split_dir.mkdir(parents=True)
    for number, split in enumerate(("training", "validation", "testing"), 1):
        patient = f"patient{number:04d}"
        (split_dir / f"subgroup_{split}.txt").write_text(patient + "\n", encoding="utf-8")
        directory = root / "database_nifti" / patient
        directory.mkdir(parents=True)
        yy, xx = np.indices((40, 60))
        for view_index, view in enumerate(("2CH", "4CH")):
            (directory / f"Info_{view}.cfg").write_text(
                f"ED: 1\nES: 9\nNbFrame: 9\nEF: {40 + 5 * number}\nImageQuality: Good\n", encoding="utf-8")
            for phase_index, phase in enumerate(("ED", "ES")):
                pixels = ((xx * 2 + yy + number * 7 + view_index * 3 + phase_index) % 256).astype(np.float32)
                labels = np.zeros_like(pixels, dtype=np.uint8)
                labels[8:32, 16:44] = 2
                labels[10:30, 18:42] = 1
                labels[32:38, 22:38] = 3
                for suffix, values in (("", pixels), ("_gt", labels)):
                    image = sitk.GetImageFromArray(values)
                    image.SetSpacing((0.4, 0.8))
                    image.SetOrigin((3, -7))
                    image.SetDirection((0, -1, 1, 0))
                    sitk.WriteImage(image, str(directory / f"{patient}_{view}_{phase}{suffix}.nii.gz"))
    return root


def test_complete_pipeline_and_patient_pairing(source, tmp_path):
    output = tmp_path / "prepared"
    report = prepare(source, output, (64, 64), expected_patients=3)
    assert report["status"] == "passed"
    assert report["patient_counts_by_split"] == {"train": 1, "validation": 1, "test": 1}
    assert report["frame_count"] == 12
    assert report["missing_reference_edv"] == report["missing_reference_esv"] == 3
    dataset = CamusDataset(output, "train", verify=True)
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["image"].shape == (1, 64, 64)
    assert sample["image"].dtype == np.float32
    assert set(np.unique(sample["mask"])) == {0, 1}
    assert "reference_ef_pct" not in sample
    paired = dataset.patient_frames("patient0001")
    assert set(paired) == {"2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"}
    native = paired["2CH_ED"]
    np.testing.assert_array_equal(native["mask"], native["multiclass_mask"] == 1)
    np.testing.assert_allclose(native["geometry"]["spacing_xy_mm"], [0.4, 0.8])
    np.testing.assert_allclose(native["geometry"]["direction"], [0, -1, 1, 0])
    with pytest.raises(KeyError):
        dataset.patient_frames("patient0002")
    with (output / "patients.csv").open() as stream:
        patients = list(csv.DictReader(stream))
    assert [row["reference_ef_group"] for row in patients] == ["le45", "gt45_lt55", "ge55"]
    assert all(row["reference_edv_ml"] == row["reference_esv_ml"] == "" for row in patients)


def test_patient_leakage_is_rejected(source):
    path = source / "database_split/subgroup_testing.txt"
    path.write_text("patient0003\npatient0001\n")
    with pytest.raises(ValueError, match="leakage"):
        read_splits(path.parent)


def test_missing_pair_fails_with_receipt(source, tmp_path):
    (source / "database_nifti/patient0002/patient0002_4CH_ES_gt.nii.gz").unlink()
    with pytest.raises(ValueError, match="audit failed"):
        prepare(source, tmp_path / "prepared", expected_patients=3)
    audit = json.loads((tmp_path / "prepared.failed_audit.json").read_text())
    assert audit["status"] == "failed"
    assert any(error["code"] == "incomplete_view_phase_pairs" for error in audit["errors"])
    assert not (tmp_path / "prepared/run_metadata.json").exists()


def test_equal_shape_but_wrong_geometry_is_rejected(source):
    directory = source / "database_nifti/patient0001"
    path = directory / "patient0001_2CH_ED_gt.nii.gz"
    mask = sitk.ReadImage(str(path))
    mask.SetOrigin((13, -7))
    sitk.WriteImage(mask, str(path))
    with pytest.raises(ValueError, match="origin"):
        read_pair(directory / "patient0001_2CH_ED.nii.gz", path)


def test_native_axes_and_pixels_match_sitk_for_valid_data(source):
    directory = source / "database_nifti/patient0001"
    image_path = directory / "patient0001_2CH_ED.nii.gz"
    pixels, _, _ = read_pair(image_path, directory / "patient0001_2CH_ED_gt.nii.gz")
    np.testing.assert_array_equal(pixels, sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))))


def test_partial_dataset_requires_explicit_opt_in(source, tmp_path):
    split_dir = source / "database_split"
    with (split_dir / "subgroup_training.txt").open("a") as stream:
        stream.write("patient0004\n")
    with pytest.raises(ValueError, match="audit failed"):
        prepare(source, tmp_path / "failed", expected_patients=4)
    output = tmp_path / "partial"
    report = prepare(source, output, (32, 32), expected_patients=4, allow_partial=True)
    assert report["partial_population"] is True
    with pytest.raises(ValueError, match="partial"):
        CamusDataset(output)
    assert len(CamusDataset(output, allow_partial=True)) == 4


@pytest.mark.parametrize("bad_value", [4, 1.5, float("nan")])
def test_invalid_mask_labels_are_rejected(source, bad_value):
    directory = source / "database_nifti/patient0001"
    path = directory / "patient0001_2CH_ED_gt.nii.gz"
    old = sitk.ReadImage(str(path))
    values = sitk.GetArrayFromImage(old).astype(np.float32)
    values[0, 0] = bad_value
    mask = sitk.GetImageFromArray(values)
    mask.CopyInformation(old)
    sitk.WriteImage(mask, str(path))
    with pytest.raises(ValueError, match="labels"):
        read_pair(directory / "patient0001_2CH_ED.nii.gz", path)


def test_physical_aspect_and_center_coordinate_mapping():
    yy, xx = np.indices((81, 41))
    labels = (((xx - 20) * 2) ** 2 + (yy - 40) ** 2 <= 20 ** 2).astype(np.uint8)
    image = (xx + yy).astype(np.float32)
    _, mask, valid, transform = preprocess(image, labels, (2, 1), (82, 82))
    ys, xs = np.where(mask)
    assert abs(np.ptp(xs) - np.ptp(ys)) <= 1  # A physical circle remains circular.
    matrix = np.array(transform["model_to_native_xy"])
    np.testing.assert_allclose(matrix @ [40.5, 40.5, 1], [20, 40, 1])
    np.testing.assert_allclose(np.array(transform["native_to_model_xy"]) @ matrix, np.eye(3), atol=1e-12)
    assert valid.dtype == bool


def test_probability_restore_uses_correct_xy_order():
    transform = make_transform((37, 61), (0.2, 0.7), (96, 96))
    yy, xx = np.indices((96, 96))
    matrix = np.array(transform["model_to_native_xy"])
    # Independent analytic field in native coordinates, sampled on the model grid.
    field = 3 * (matrix[0, 0] * xx + matrix[0, 2]) + 5 * (matrix[1, 1] * yy + matrix[1, 2])
    restored = restore_to_native(field, transform)
    ny, nx = np.indices((37, 61))
    np.testing.assert_allclose(restored[2:-2, 2:-2], (3 * nx + 5 * ny)[2:-2, 2:-2], atol=5e-5)


def test_frozen_manifests_and_arrays(source, tmp_path):
    output = tmp_path / "prepared"
    prepare(source, output, (48, 48), expected_patients=3)
    metadata = (output / "run_metadata.json").read_bytes()
    prepare(source, output, (48, 48), expected_patients=3)
    assert metadata == (output / "run_metadata.json").read_bytes()
    with pytest.raises(ValueError, match="Frozen preparation differs"):
        prepare(source, output, (64, 64), expected_patients=3)
    array_path = next((output / "arrays").glob("*.npz"))
    array_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Prepared array changed"):
        verify_prepared(output)


def test_json_metadata_uses_portable_lf_bytes(source, tmp_path):
    output = tmp_path / "prepared"
    prepare(source, output, (32, 32), expected_patients=3)
    for path in output.glob("*.json"):
        assert b"\r\n" not in path.read_bytes(), path.name


def test_reference_conflict_is_missing_not_averaged(source, tmp_path):
    path = source / "database_nifti/patient0001/Info_4CH.cfg"
    path.write_text(path.read_text().replace("EF: 45", "EF: 65"))
    output = tmp_path / "prepared"
    report = prepare(source, output, (32, 32), expected_patients=3)
    assert report["missing_reference_ef"] == 1
    patient = CamusDataset(output).patient_metadata["patient0001"]
    assert patient["reference_ef_pct"] == ""
    assert patient["reference_ef_group"] == "missing"


def test_duplicate_images_across_patients_fail(source, tmp_path):
    original = source / "database_nifti/patient0001/patient0001_2CH_ED.nii.gz"
    duplicate = source / "database_nifti/patient0002/patient0002_2CH_ED.nii.gz"
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="audit failed"):
        prepare(source, tmp_path / "prepared", expected_patients=3)
    report = json.loads((tmp_path / "prepared.failed_audit.json").read_text())
    assert any(error["code"] == "duplicate_pixels_across_splits" for error in report["errors"])


def test_duplicates_within_one_split_are_flagged(source):
    original = source / "database_nifti/patient0001/patient0001_2CH_ED.nii.gz"
    duplicate = source / "database_nifti/patient0002/patient0002_2CH_ED.nii.gz"
    duplicate.write_bytes(original.read_bytes())
    (source / "database_split/subgroup_training.txt").write_text("patient0001\npatient0002\n")
    (source / "database_split/subgroup_validation.txt").write_text("patient0004\n")
    _, _, report, _, _ = inventory(source, 4, True)
    assert report["status"] == "passed"
    assert report["cross_split_duplicate_image_groups"] == 0
    assert report["warnings"][0]["code"] == "duplicate_pixels_within_split"


@pytest.mark.parametrize("value,expected", [(45, "le45"), (45.1, "gt45_lt55"), (54.9, "gt45_lt55"), (55, "ge55"), (None, "missing")])
def test_ef_boundary_definitions(value, expected):
    assert ef_group(value) == expected
