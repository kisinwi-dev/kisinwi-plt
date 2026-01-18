from fastapi import APIRouter, Depends, HTTPException, status

from core.dataset_module import Store
from api.deps import get_store
from api.schemas import MessageResponse
from api.schemas.version import (
    VERSION_NAME_PATH,
    VersionCreateRequest,
    VersionRenameRequest,
    VersionListResponse
)
from api.schemas.dataset import DATASET_NAME_PATH

router = APIRouter()


@router.get(
    "/",
    summary="Get dataset versions",
    description="Returns a list of version names for a specific dataset.",
    response_model=VersionListResponse,
    responses={
        200: {"description": "Versions successfully returned"},
        404: {"description": "Dataset not found"},
    },
)
def get_versions(
        dataset_name: str = DATASET_NAME_PATH,
        store: Store = Depends(get_store),
):
    try:
        versions = store.get_dataset_version_name(
            dataset_name=dataset_name
        )
        return VersionListResponse(versions=versions)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/",
    summary="Create new version for a dataset",
    description="Creates a new version for a specific dataset.",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Version successfully created"},
        400: {"description": "Invalid request data"},
        404: {"description": "Dataset not found"},
        409: {"description": "Version already exists"},
        500: {"description": "Internal server error"},
    },
)
def create_version(
        dataset_name: str = DATASET_NAME_PATH,
        body: VersionCreateRequest = ...,
        store: Store = Depends(get_store),
):
    try:
        store.set_dataset_new_version(
            dataset_name=dataset_name,
            version_name=body.version_name,
        )
        return MessageResponse(
            message=f"Version '{body.version_name}' successfully created for dataset '{dataset_name}'"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete(
    "/{version_name}",
    summary="Delete a version of a dataset",
    description="Deletes a specific version of a dataset.",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Version successfully deleted"},
        404: {"description": "Dataset or version not found"},
        500: {"description": "Internal server error"},
    }
)
def delete_version(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        store: Store = Depends(get_store),
):
    try:
        store.drop_version(
            dataset_name=dataset_name,
            version_name=version_name
        )
        return MessageResponse(
            message=f"Version '{version_name}' successfully deleted from dataset '{dataset_name}'"
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
    "/{version_name}",
    summary="Rename a version of a dataset",
    description="Renames a specific version of a dataset.",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Version successfully renamed"},
        400: {"description": "Invalid request data"},
        404: {"description": "Dataset or version not found"},
        409: {"description": "Version with new name already exists"},
        500: {"description": "Internal server error"},
    }
)
def rename_version(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        body: VersionRenameRequest = ...,
        store: Store = Depends(get_store),
):
    try:
        store.rename_version(
            dataset_name=dataset_name,
            version_name=version_name,
            new_version=body.new_name,
        )
        return MessageResponse(
            message=f"Version '{version_name}' renamed to '{body.new_name}' in dataset '{dataset_name}'"
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
