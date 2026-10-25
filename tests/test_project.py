"""Basic tests for the pork-rasher detection project."""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_config_has_five_classes():
    """Check that the dataset config declares five classes."""
    config_path = ROOT / "data" / "data.yaml"
    assert config_path.is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["nc"] == 5
    assert len(config["names"]) == 5


def test_expected_source_files_exist():
    """Check that the main project scripts are present."""
    expected = [
        "train.py",
        "evaluate.py",
        "predict.py",
        "inspect_dataset.py",
        "profile_inference.py",
    ]
    for filename in expected:
        assert (ROOT / "src" / filename).is_file()


def test_dataset_config_has_all_splits():
    """Check that train, validation and test paths are configured."""
    config = yaml.safe_load(
        (ROOT / "data" / "data.yaml").read_text(encoding="utf-8")
    )
    assert config.get("train")
    assert config.get("val") or config.get("valid")
    assert config.get("test")
