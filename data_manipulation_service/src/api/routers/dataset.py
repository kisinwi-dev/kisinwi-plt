from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException, status

from core.dataset_module import Store
from api.deps import get_store
from api.schemas import MessageResponse
from api.schemas.dataset import (
    DATASET_NAME_PATH,
    DatasetCreateRequest,
    DatasetRenameRequest,
    DatasetInfoResponse,
    DatasetListResponse
)

router = APIRouter()


@router.get(
    "/",
    response_model=DatasetListResponse,
    summary="Get list of datasets",
    description="Returns a list of available dataset names from the storage."
)
def list_datasets(
        store: Store = Depends(get_store)
):
    datasets = store.get_dataset_name()
    return DatasetListResponse(datasets=datasets)


@router.get(
    "/{dataset_name}",
    summary="Get dataset info",
    description="Returns detailed information about a dataset by its name.",
    responses={
        200: {"description": "Dataset information successfully returned"},
        404: {"description": "Dataset not found"}
    },
)
def info_dataset(
        dataset_name: str = DATASET_NAME_PATH,
        store: Store = Depends(get_store)
):
    try:
        dataset_info = store.get_dataset_info(dataset_name)
        return DatasetInfoResponse.model_validate(asdict(dataset_info))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/",
    summary="Create dataset from archive",
    description="Creates a new dataset from the provided archive. Returns a success message or error details.",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Dataset successfully created"},
        400: {"description": "Invalid request data"},
        404: {"description": "Archive not found"},
        409: {"description": "Dataset already exists"},
        500: {"description": "Internal server error"},
    },
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
    description="Deletes a dataset by its name. Returns a success message if deletion was successful.",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Dataset successfully deleted"},
        404: {"description": "Dataset not found"},
        500: {"description": "Internal server error"}
    }
)
def delete_dataset(
        dataset_name: str = DATASET_NAME_PATH,
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
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.patch(
    "/{dataset_name}",
    summary="Rename dataset",
    description="Renames an existing dataset. Returns a message confirming the rename or an error.",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Dataset successfully renamed"},
        400: {"description": "Invalid request data"},
        404: {"description": "Dataset not found"},
        409: {"description": "Dataset with new name already exists"},
        500: {"description": "Internal server error"},
    }
)
def rename_dataset(
        body: DatasetRenameRequest,
        dataset_name: str = DATASET_NAME_PATH,
        store: Store = Depends(get_store),
):
    try:
        store.rename_dataset(
            dataset_name=dataset_name,
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
