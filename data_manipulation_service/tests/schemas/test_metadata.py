import pytest
from datetime import datetime
from pydantic import ValidationError

from app.api.schemas.metadata import DatasetMetadata, Version, Source


def test_fixture_loads(dsm_a: DatasetMetadata):
    assert isinstance(dsm_a, DatasetMetadata)
    assert len(dsm_a.versions) > 0
    assert dsm_a.updated_at >= dsm_a.created_at


def test_fixture_types(dsm_a: DatasetMetadata):
    assert isinstance(dsm_a.dataset_id, str)
    assert isinstance(dsm_a.class_names, list)
    assert isinstance(dsm_a.versions, list)


# ================ Source ======================

def test_source_structure(dsm_cd: DatasetMetadata):
    assert dsm_cd.source.kaggle.url
    assert dsm_cd.source.kaggle.description

# ================ Version ======================

def test_version_structure(dsm_a: DatasetMetadata):
    for v in dsm_a.versions:
        assert isinstance(v, Version)
        assert v.version_id
        assert isinstance(v.created_at, datetime)


def test_version_numeric_fields(dsm_a: DatasetMetadata):
    for v in dsm_a.versions:
        assert v.size_bytes >= 0
        assert v.num_samples >= 0
        assert v.num_train >= 0
        assert v.num_val >= 0
        assert v.num_test >= 0


def test_version_split_consistency(dsm_a: DatasetMetadata):
    for v in dsm_a.versions:
        assert v.num_train + v.num_val + v.num_test == v.num_samples


# ================ DatasetMetadata ======================

def test_num_classes_consistency(dsm_cd: DatasetMetadata):
    assert dsm_cd.num_classes == len(dsm_cd.class_names)


def test_class_mapping_consistency(dsm_cd: DatasetMetadata):
    assert set(dsm_cd.class_names) == set(dsm_cd.class_to_idx.keys())


def test_class_indices_unique(dsm_cd: DatasetMetadata):
    indices = list(dsm_cd.class_to_idx.values())
    assert len(indices) == len(set(indices))

def test_default_version_exists(dsm_a: DatasetMetadata):
    version_ids = [v.version_id for v in dsm_a.versions]
    assert dsm_a.default_version_id in version_ids

def test_dataset_type_literal(dsm_cd: DatasetMetadata):
    assert dsm_cd.type in {"image", "text", "tabular", "other"}


def test_task_literal(dsm_cd: DatasetMetadata):
    assert dsm_cd.task in {
        "classification",
        "regression",
        "detection",
        "segmentation",
        "other",
    }

def test_validate_assignment_num_classes(dsm_cd: DatasetMetadata):
    with pytest.raises(ValidationError):
        dsm_cd.num_classes = 0

def test_updated_at_changes_on_update(dsm_cd: DatasetMetadata):
    old_updated = dsm_cd.updated_at
    dsm_cd.name = "New name"
    assert dsm_cd.updated_at > old_updated


def test_created_at_not_changed(dsm_cd: DatasetMetadata):
    created = dsm_cd.created_at
    dsm_cd.name = "another name"
    assert dsm_cd.created_at == created

def test_version_negative_values():
    with pytest.raises(ValidationError):
        Version(
            version_id="1",
            description="test",
            size_bytes=-1,
            num_samples=10,
            num_train=8,
            num_val=1,
            num_test=1,
        )

def test_roundtrip_serialization(dsm_a: DatasetMetadata):
    dumped = dsm_a.model_dump()
    new_model = DatasetMetadata.model_validate(dumped)

    assert new_model == dsm_a