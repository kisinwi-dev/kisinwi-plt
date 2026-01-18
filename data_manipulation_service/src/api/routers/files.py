from fastapi import APIRouter, Depends, HTTPException, status

from core.dataset_module import Store
from api.deps import get_store
from api.schemas.files import FileListResponse
from api.schemas.dataset import DATASET_NAME_PATH
from api.schemas.version import VERSION_NAME_PATH
from api.schemas.classes import CLASS_NAME_PATH

router = APIRouter()


@router.get(
    "/files",
    summary="Get all files in a class of a dataset version",
    description="Returns a list of all file names in a specific class of a dataset version.",
    response_model=FileListResponse,
    responses={
        200: {"description": "List of files successfully returned"},
        404: {"description": "Dataset, version, or class not found"},
        500: {"description": "Internal server error"},
    }
)
def get_files(
        dataset_name: str = DATASET_NAME_PATH,
        version_name: str = VERSION_NAME_PATH,
        class_name: str = CLASS_NAME_PATH,
        store: Store = Depends(get_store),
):
    try:
        files = store.get_dataset_vesion_class_files_name(
            dataset_name=dataset_name,
            version_name=version_name,
            class_name=class_name,
        )
        return FileListResponse(files=files)

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
