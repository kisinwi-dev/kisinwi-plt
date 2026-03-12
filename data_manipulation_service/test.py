import pytest
from app.api.schemas.dataset import DatasetMetadata
from app.core.services.dataset import Dataset
from app.logs import get_logger

loagger = get_logger(__name__)


d = Dataset()
print('\n')
print(d.get_datasets_id)
print('\n')

dsm = d.get_dataset_info('apple')
print('\n')
print('JSON load:')
print(dsm.model_dump_json(indent=2))
print('\n')

print('\n')
print('Create: ', dsm.created_at)
print('Update: ', dsm.updated_at)

dsm.name = 'apple'
print('🟩 change name 🟩')
print('Create: ', dsm.created_at)
print('Update: ', dsm.updated_at)

print('\n')
print(d.change_dataset_info('apple', dsm))
print('\n')

def test_update_time(dsm: DatasetMetadata):
    old_updated_at = dsm.updated_at
    dsm.name = 'new_name'
    new_updated_at = dsm.updated_at
    assert(old_updated_at != new_updated_at)
