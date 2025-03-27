class PredictionException(Exception):
    def __init__(self, message: str, input_data=None):
        super().__init__(message)
        self.message = message
        self.input_data = input_data

    def __str__(self):
        details = f"Message: {self.message}"
        if self.input_data:
            details += f", Input Data: {self.input_data}"
        return f"PredictionException: {details}"


class UserNotFoundException(Exception):
    def __init__(self, message: str, user_id=None):
        super().__init__(message)
        self.message = message
        self.user_id = user_id

    def __str__(self):
        details = f"Message: {self.message}"
        if self.user_id:
            details += f", User Id: {self.user_id}"
        return f"UserNotFoundException: {details}"