
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

def test_class_unique(ds_new: NewDataset):
    assert len(ds_new.class_names) == len(set(ds_new.class_names))
