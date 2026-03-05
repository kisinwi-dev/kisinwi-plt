class CoreException(Exception):
    """
        Base exception for all domain-level exceptions.
    """
    def __init__(
            self, 
            message: str, 
            status_code: int = 400
    ):
        super().__init__(message)
        self.status_code = status_code