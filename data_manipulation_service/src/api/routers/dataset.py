from fastapi import APIRouter, Depends

from core.dataset_module.store import Store
from api.deps import get_store

router = APIRouter()

@router.get(
    "/datasets",
    response_model=list[str],
    summary="List datasets"
)
def list_datasets(
        store: Store = Depends(get_store)
    ):
    return store.get_dataset_name()