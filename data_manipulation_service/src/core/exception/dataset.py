from .base import CoreException

class DatasetAlreadyExistsException(CoreException):
    def __init__(self, dataset_name: str):
        super().__init__(
            message=f"Dataset '{dataset_name}' already exists",
            status_code=409
        )

class DatasetNotFoundException(CoreException):
    def __init__(self, dataset_name: str):
        super().__init__(
            message=f"Dataset '{dataset_name}' not found",
            status_code=404
        )

class DatasetValidationException(CoreException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"Dataset validation failed: {reason}",
            status_code=400
        )

class UnsupportedDatasetTypeException(CoreException):
    def __init__(self, dataset_type: str):
        super().__init__(
            message=f"Unsupported dataset type: {dataset_type}",
            status_code=400
        )

class UnsupportedDatasetTaskException(CoreException):
    def __init__(self, dataset_task: str):
        super().__init__(
            message=f"Unsupported dataset task: {dataset_task}",
            status_code=400
        )
