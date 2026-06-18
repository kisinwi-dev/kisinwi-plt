import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _make_resilient_session() -> requests.Session:
    """
    Сессия с авто-повтором на временные сбои: connect/read-ошибки и 5xx.

    Закрывает единичные сетевые blip (сервис рестартнулся, сеть моргнула) для
    коротких критичных вызовов, не пробрасывая исключение в пайплайн.
    backoff_factor даёт паузы 0.5s, 1s, 2s между попытками.
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(502, 503, 504),
        allowed_methods=None,  # повторять и POST — наши POST идемпотентны на нашей стороне
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class BaseServiceClient:
    """
    Базовый HTTP-клиент сервиса: общая переиспользуемая сессия и её закрытие.

    Сессия с retry-адаптером (см. _make_resilient_session) — transient-сбои
    повторяются прозрачно. Стратегию обработки оставшихся ошибок наследники
    задают сами (raise / bool / декоратор) — она зависит от критичности вызова.
    """

    def __init__(self, url: str) -> None:
        self.URL = url
        self.session = _make_resilient_session()

    def close(self) -> None:
        self.session.close()

    def __exit__(self, *args) -> None:
        self.close()
