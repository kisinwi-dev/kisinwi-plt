from fastapi import APIRouter, Depends, HTTPException, status

from core.dataset_module import Store
from api.deps import get_store
from api.schemas import MessageResponse
from api.schemas.dataset import *

router = APIRouter()

@router.get(
    "/",
    response_model=list[str],
    summary="List datasets"
)
def list_datasets(
        store: Store = Depends(get_store)
    ):
    return store.get_dataset_name()

@router.get(
    "/{dataset_name}",
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
    
@router.post(
    "/",
    summary="Create dataset from archive",
    response_model=MessageResponse,
)
def create_dataset(
        body: DatasetCreateRequest,
        store: Store = Depends(get_store),
    ):
    try:
        store.set_new_dataset(
            dataset_name=body.dataset_name,
            archive_name=body.archive_name,
            dataset_type=body.dataset_type,
            dataset_task=body.dataset_task
        )
        return MessageResponse(
            message=f"Dataset '{body.dataset_name}' successfully created"
        )

    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    
@router.delete(
    "/{dataset_name}",
    summary="Delete dataset",
    response_model=MessageResponse,
)
def delete_dataset(
        dataset_name: str,
        store: Store = Depends(get_store),
    ):
    try:
        store.drop_dataset(dataset_name)
        return MessageResponse(
            message=f"Dataset '{dataset_name}' successfully deleted"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    
@router.patch(
    "/{dataset_name}",
    summary="Rename dataset",
    response_model=MessageResponse,
)
def rename_dataset(
        dataset_name: str,
        body: DatasetRenameRequest,
        store: Store = Depends(get_store),
    ):
    try:
        store.rename_dataset(
            old_name=body.dataset_name,
            new_name=body.new_name,
        )
        return MessageResponse(
            message=(
                f"Dataset '{dataset_name}' renamed "
                f"to '{body.new_name}'"
            )
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )