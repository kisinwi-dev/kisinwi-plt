from fastapi import (
    APIRouter, HTTPException, 
    Depends, status, Query
)

from core.dataset_module import Store
from api.deps import get_store
from api.schemas.dataset import DATASET_NAME_PATH
from api.schemas.version import VERSION_NAME_PATH

router = APIRouter()

@router.get(
    "/duplicates",
    summary="Check if duplicate files exist",
    response_model=bool,
)
def has_duplicate_files(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str | None = Query(
            None,
            description="Optional class name to limit duplicate check"
        ),
        store: Store = Depends(get_store),
):
    try:
        return store.has_duplicate_files(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=class_name,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

@router.get(
    "/duplicates/list",
    summary="Get duplicate files",
)
def find_duplicate_files(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str | None = Query(
            None,
            description="Optional class name to search for duplicates"
        ),
        store: Store = Depends(get_store),
):
    try:
        duplicates = store.find_duplicate_files(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=class_name,
        )

        return duplicates

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

@router.post(
    "/duplicates/remove",
    summary="Remove duplicate files",
)
def remove_duplicate_files(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str | None = Query(
            None,
            description="Optional class name to remove duplicates from"
        ),
        store: Store = Depends(get_store),
):
    try:
        removed_files = store.remove_duplicates(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=class_name,
        )

        return removed_files

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
