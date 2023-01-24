class APIException(Exception):
    status: int
    error: str
    inner_exception: Exception | None
    line: int | None

    def __init__(self, status: int, error: str, inner_exception: Exception | None = None, line: int | None = None):
        self.status = status
        self.error = error
        self.inner_exception = inner_exception
        self.line = line
