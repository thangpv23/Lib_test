class ValidationError(Exception):
    def ___init__(self, error_msg: str, status_code: int):
        super().__inint__(error_msg)

        self.status_code = status_code
        self.error_msg = error_msg
