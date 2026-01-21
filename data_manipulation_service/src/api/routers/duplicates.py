from fastapi import (
    APIRouter, HTTPException,
    Depends, status, Query
)

from core.dataset_module import Store
from api.deps import get_store
from api.schemas import MessageResponse
from api.schemas.dataset import DATASET_NAME_PATH
from api.schemas.version import VERSION_NAME_PATH
from api.schemas.duplicates import DuplicateFilesGroupResponse

router = APIRouter()


@router.get(
    "/duplicates",
    summary="Check for duplicate files in a dataset version",
    description=(
        "Checks whether duplicate files exist in a specific dataset version. "
        "The check is performed recursively across all files in the version.\n\n"
        "If `class_name` is provided, the check is limited to files within the specified class."
    ),
    response_model=bool,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Duplicate check completed successfully"},
        404: {"description": "Dataset, version, or class not found"},
        500: {"description": "Internal server error"},
    },
)
def has_duplicate_files(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str = Query(
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/duplicates/list",
    summary="List duplicate files in a dataset version",
    description=(
        "Finds and returns duplicate files in a specific dataset version. "
        "The search is performed recursively across all files in the version.\n\n"
        "If `class_name` is provided, the search is limited to files within that class."
    ),
    response_model=DuplicateFilesGroupResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Duplicate files found and returned successfully"},
        404: {"description": "Dataset, version, or class not found"},
        500: {"description": "Internal server error"},
    },
)
def find_duplicate_files(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str = Query(
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

        return DuplicateFilesGroupResponse(
            duplicates=[[path.name for path in group] for group in duplicates]
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/duplicates/remove",
    summary="Remove duplicate files in a dataset version",
    description=(
        "Removes duplicate files from a specific dataset version. "
        "The operation is performed recursively across all files in the version.\n\n"
        "If `class_name` is provided, duplicates are removed only within that class."
    ),
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Duplicate files removed successfully"},
        404: {"description": "Dataset, version, or class not found"},
        500: {"description": "Internal server error"},
    }
)
def remove_duplicate_files(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str = Query(
            None,
            description="Optional class name to remove duplicates from"
        ),
        store: Store = Depends(get_store),
):
    try:
        store.remove_duplicates(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=class_name,
        )

        return MessageResponse(
            message="Duplicates removed successfully"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
