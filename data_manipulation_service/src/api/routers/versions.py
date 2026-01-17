from fastapi import APIRouter, Depends, HTTPException, status

from core.dataset_module import Store
from api.deps import get_store
from api.schemas import MessageResponse
from api.schemas.version import *

router = APIRouter()

@router.get(
    "/",
    summary="Get dataset versions",
    response_model=VersionListResponse,
)
def get_versions(
        dataset_name: str,
        store: Store = Depends(get_store),
    ):
    versions = store.get_dataset_version_name(dataset_name)
    return VersionListResponse(versions=versions)



@router.post(
    "/",
    summary="Create new version for a dataset",
    response_model=MessageResponse,
)
def create_version(
        dataset_name: str,
        body: VersionCreateRequest,
        store: Store = Depends(get_store),
    ):
    try:
        store.set_new_dataset_version(
            dataset_name=dataset_name,
            version_name=body.version_name,
        )
        return MessageResponse(
            message=f"Version '{body.version_name}' successfully created for dataset '{dataset_name}'"
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
    
@router.delete(
    "/{version_name}",
    summary="Delete a version of a dataset",
    response_model=MessageResponse,
)
def delete_version(
        dataset_name: str,
        version_name: str,
        store: Store = Depends(get_store),
    ):
    try:
        store.drop_version(dataset_name, version_name)
        return MessageResponse(
            message=f"Version '{version_name}' successfully deleted from dataset '{dataset_name}'"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    
@router.patch(
    "/{version_name}",
    summary="Rename a version of a dataset",
    response_model=MessageResponse,
)
def rename_version(
        dataset_name: str,
        version_name: str,
        body: VersionRenameRequest,
        store: Store = Depends(get_store),
    ):
    try:
        store.rename_version(
            dataset_name=dataset_name,
            old_version=version_name,
            new_version=body.new_name,
        )
        return MessageResponse(
            message=(
                f"Version '{version_name}' renamed to '{body.new_name}' "
                f"in dataset '{dataset_name}'"
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