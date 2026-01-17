from fastapi import APIRouter, Depends, HTTPException, status

from core.dataset_module import Store
from api.deps import get_store
from api.schemas import MessageResponse
from api.schemas.classes import *

router = APIRouter()

@router.get(
    "/",
    summary="Get all classes in a dataset version",
    response_model=ClassListResponse,
)
def get_classes(
        dataset_name: str,
        version_name: str,
        store: Store = Depends(get_store),
    ):
    try:
        classes = store.get_dataset_classes_name(dataset_name, version_name)
        return ClassListResponse(
            dataset_name=dataset_name,
            version_name=version_name,
            classes=classes,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    
@router.post(
    "/",
    summary="Create a class in a dataset version",
    response_model=MessageResponse,
)
def create_class(
        dataset_name: str,
        version_name: str,
        body: ClassCreateRequest,
        store: Store = Depends(get_store),
    ):
    try:
        store.set_new_class(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=body.class_name,
        )
        return MessageResponse(
            message=(
                f"Class '{body.class_name}' successfully created in "
                f"version '{version_name}' of dataset '{dataset_name}'"
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
    
@router.delete(
    "/{class_name}",
    summary="Delete a class from a dataset version",
    response_model=MessageResponse,
)
def delete_class(
        dataset_name: str,
        version_name: str,
        class_name: str,
        store: Store = Depends(get_store),
    ):
    try:
        store.drop_class(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=class_name,
        )
        return MessageResponse(
            message=(
                f"Class '{class_name}' successfully deleted from "
                f"version '{version_name}' of dataset '{dataset_name}'"
            )
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    
@router.patch(
    "/{class_name}",
    summary="Rename a class in a dataset version",
    response_model=MessageResponse,
)
def rename_class(
        dataset_name: str,
        version_name: str,
        class_name: str,
        body: ClassRenameRequest,
        store: Store = Depends(get_store),
    ):
    try:
        store.rename_class(
            dataset_name=dataset_name,
            version_name=version_name,
            old_class=class_name,
            new_class=body.new_name,
        )
        return MessageResponse(
            message=(
                f"Class '{class_name}' renamed to '{body.new_name}' "
                f"in version '{version_name}' of dataset '{dataset_name}'"
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