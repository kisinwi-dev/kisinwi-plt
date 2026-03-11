import pytest
from datetime import datetime
from app.api.schemas.metadata import DatasetMetadata

def test_load_fixture(dsm_a: DatasetMetadata):
    assert isinstance(dsm_a, DatasetMetadata)
    assert isinstance(dsm_a.versions, list)
    assert len(dsm_a.versions) > 0
    assert dsm_a.updated_at >= dsm_a.created_at

def test_valid_version(dsm_a: DatasetMetadata):
    version = dsm_a.versions[0]
    assert version.version_id
    assert version.size_bytes >= 0
    assert version.num_samples >= 0
    assert version.num_train >= 0
    assert version.num_val >= 0
    assert version.num_test >= 0
    assert isinstance(version.created_at, datetime) 