from app.core.filesystem.fsm import FileSystemManager
from pydantic import HttpUrl
from app.api.schemas.dataset import Source, SourceItem
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.core.services.dataset import DatasetManager

nds = NewDataset(
    dataset_id="apple_new",
    name="apple",
    description='',
    class_names=['red', 'green'],
    source=Source(
        kaggle=SourceItem(
            url=HttpUrl("https://github.com/kisinwi-dev/kisinwi-plt/blob/main"),
            description="наверное"
        )
    ),
    type="image",
    task="classification",
    version=NewVersion(
        version_id="v_1",
        description="описание версии"
    ),
)

ds = DatasetManager()
ds.add_new_dataset(nds)
ds.get_dataset_info(nds.dataset_id)

nv = NewVersion(
    version_id='v_2',
    description='Типо описание'
)

ds.add_new_version(nds.dataset_id, nv)