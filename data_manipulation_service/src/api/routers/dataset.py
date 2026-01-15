from fastapi import APIRouter, Depends, HTTPException, status

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

@router.get(
    "/datasets/{dataset_name}",
    summary="Get dataset info",
)
def info_dataset(
        dataset_name: str,
        store: Store = Depends(get_store)
    ):
    try:
        info = store.get_dataset_info(dataset_name)
        return info
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )