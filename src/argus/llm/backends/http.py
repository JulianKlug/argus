"""The only HTTP client in argus: no env proxies, no redirects, typed failures."""

import json
from collections.abc import Callable

import httpx

from argus.llm.errors import BackendUnavailableError

HTTP_OK = 200
HTTP_SERVER_ERROR = 500


class HttpStatusError(Exception):
    """Non-2xx answer. `body` stays inside the backend layer; it may echo request content."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


class HttpSession:
    """JSON over HTTP to one backend. `guard(status, backend)` vets every status (redirect policy)."""

    def __init__(
        self,
        backend: str,
        base_url: str,
        connect_timeout_s: float,
        request_timeout_s: float,
        guard: Callable[[int, str], None],
        transport: httpx.BaseTransport | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._backend = backend
        self._guard = guard
        self._client = httpx.Client(
            base_url=base_url,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s),
            transport=transport,
            headers=headers,
        )

    @property
    def trust_env(self) -> bool:
        return self._client.trust_env

    @property
    def follow_redirects(self) -> bool:
        return self._client.follow_redirects

    def get_json(self, path: str) -> dict:
        return self._send("GET", path, None)

    def post_json(self, path: str, body: bytes) -> dict:
        return self._send("POST", path, body)

    def _send(self, method: str, path: str, body: bytes | None) -> dict:
        try:
            response = self._client.request(
                method, path, content=body, headers={"Content-Type": "application/json"} if body else None,
            )
        except httpx.TimeoutException:
            raise BackendUnavailableError(f"backend {self._backend}: timed out") from None
        except httpx.TransportError:
            raise BackendUnavailableError(f"backend {self._backend}: unreachable") from None

        self._guard(response.status_code, self._backend)
        if response.status_code >= HTTP_SERVER_ERROR:
            raise BackendUnavailableError(f"backend {self._backend}: server error HTTP {response.status_code}")
        if response.status_code != HTTP_OK:
            raise HttpStatusError(response.status_code, response.text)
        return json.loads(response.content)
