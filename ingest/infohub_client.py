from __future__ import annotations

import time
import requests
from typing import Any


class InfoHubClient:
    def __init__(
        self,
        base_url: str = "https://infohubapi.rs.ge/api",
        language_code: str = "ka",
        cookie: str | None = None,
        delay_sec: float = 0.2,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.language_code = language_code
        self.cookie = cookie
        self.delay_sec = delay_sec
        self.timeout = timeout

        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        h = {
            "accept": "application/json",
            "languagecode": self.language_code,
        }
        if self.cookie:
            h["cookie"] = self.cookie
        return h

    def list_documents(self, *, species: str, skip: int, take: int = 99) -> dict[str, Any]:
        url = f"{self.base_url}/documents"
        params = {"skip": skip, "take": take, "species": species}

        r = self.session.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        r.raise_for_status()
        time.sleep(self.delay_sec)
        return r.json()

    def get_details_by_key(self, unique_key: str) -> dict[str, Any]:
        """
        Your discovery showed a details-by-key endpoint.
        Some environments might use /documents/{key}/... or /documents/e{key}/...
        We'll try both to be robust.
        """
        candidates = [
            f"{self.base_url}/documents/{unique_key}/details-by-key",
            f"{self.base_url}/documents/e{unique_key}/details-by-key",
        ]
        params = {"openFromSearch": "true"}

        last_err: Exception | None = None
        for url in candidates:
            try:
                r = self.session.get(url, headers=self._headers(), params=params, timeout=self.timeout)
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                time.sleep(self.delay_sec)
                return r.json()
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Failed to fetch details for uniqueKey={unique_key}") from last_err
