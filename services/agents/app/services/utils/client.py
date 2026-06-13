import requests


class BaseServiceClient:
    """
    Базовый HTTP-клиент сервиса: общая переиспользуемая сессия и её закрытие.

    Стратегию обработки ошибок наследники задают сами (raise / bool / декоратор) —
    она зависит от критичности вызова и здесь намеренно не унифицируется.
    """

    def __init__(self, url: str) -> None:
        self.URL = url
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def __exit__(self, *args) -> None:
        self.close()
