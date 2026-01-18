from fastapi import APIRouter, Depends, HTTPException, status

from core.dataset_module import Store
from api.deps import get_store
from api.schemas import MessageResponse
from api.schemas.classes import *
from api.schemas.dataset import DATASET_NAME_PATH
from api.schemas.version import VERSION_NAME_PATH

router = APIRouter()

@router.get(
    "/",
    summary="Get all classes in a dataset version",
    description="Returns a list of all class names for a specific dataset version.",
    response_model=ClassListResponse,
    responses={
        200: {"description": "List of classes successfully returned"},
        404: {"description": "Dataset or version not found"},
        500: {"description": "Internal server error"},
    }
)
def get_classes(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        store: Store = Depends(get_store),
    ):
    try:
        classes = store.get_dataset_version_classes_name(
            dataset_name=dataset_name, 
            version_name=version_name
        )
        return ClassListResponse(classes=classes)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )
    
@router.post(
    "/",
    summary="Create a class in a dataset version",
    description="Creates a new class in a specific dataset version.",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Class successfully created"},
        404: {"description": "Dataset or version not found"},
        409: {"description": "Class already exists"},
        500: {"description": "Internal server error"},
    }
)
def create_class(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        body: ClassCreateRequest = ...,
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
            detail=str(e)
        )
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, 
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
@router.delete(
    "/{class_name}",
    summary="Delete a class from a dataset version",
    description="Deletes a specific class from a dataset version.",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Class successfully deleted"},
        404: {"description": "Dataset, version, or class not found"},
        500: {"description": "Internal server error"},
    }
)
def delete_class(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str = CLASS_NAME_PATH,
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
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )
    
@router.patch(
    "/{class_name}",
    summary="Rename a class in a dataset version",
    description="Renames a specific class in a dataset version.",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Class successfully renamed"},
        400: {"description": "Invalid request data"},
        404: {"description": "Dataset, version, or class not found"},
        409: {"description": "Class with new name already exists"},
        500: {"description": "Internal server error"},
    }
)
def rename_class(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str = CLASS_NAME_PATH,
        body: ClassRenameRequest = ...,
        store: Store = Depends(get_store),
    ):
    try:
        store.rename_class(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=class_name,
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