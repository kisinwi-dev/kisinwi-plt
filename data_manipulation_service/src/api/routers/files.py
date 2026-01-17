from fastapi import APIRouter, Depends, HTTPException, status

from core.dataset_module import Store
from api.deps import get_store
from api.schemas import MessageResponse
from api.schemas.files import *

router = APIRouter()

@router.get(
    "/files",
    summary="Get all files in a class of a dataset version",
    response_model=FileListResponse,
)
def get_files(
        dataset_name: str,
        version_name: str,
        class_name: str,
        store: Store = Depends(get_store),
    ):
    try:
        files = store.get_dataset_files_name(
            dataset_name=dataset_name,
            dataset_version=version_name,
            class_name=class_name,
        )
        return FileListResponse(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=class_name,
            files=files,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )