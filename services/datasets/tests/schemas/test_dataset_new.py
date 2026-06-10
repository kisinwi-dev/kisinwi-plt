
from app.api.schemas.dataset_new import NewDataset


def test_fixture_loads(ds_new: NewDataset):
    assert isinstance(ds_new, NewDataset)

def test_task_literal(ds_new: NewDataset):
    assert ds_new.task in {
        "classification",
        "regression",
        "detection",
        "segmentation",
        "other",
    }

def test_type_literal(ds_new: NewDataset):
    assert ds_new.type in {"image", "text", "tabular", "other"}

def test_defaults():
    ds = NewDataset(
        name="minimal",
        description="датасет без явных type/task",
        version={
            "id_data": "upload-1",
            "name": "v1",
            "sources": [],
        },
    )
    assert ds.type == "image"
    assert ds.task == "classification"
    assert ds.version.description == "Нет описания"

def test_version_structure(ds_new: NewDataset):
    assert ds_new.version.id_data
    assert ds_new.version.name
    assert isinstance(ds_new.version.sources, list)
