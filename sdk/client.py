"""
SyntheticDatasetClient - Python SDK for the Synthetic Dataset Generation platform.

Usage
-----
    from sdk import SyntheticDatasetClient

    sdk = SyntheticDatasetClient(api_key="sdk-xxxx")

    pages = [
        {"page_no": 1, "doc_id": "DOC-A", "text": "..."},
        {"page_no": 2, "doc_id": "DOC-A", "text": "..."},
    ]
    sdk.upload(pages)

All network errors raise SDKError with a clear, actionable message.
"""
from __future__ import annotations

from typing import Any, Optional

import requests


class SDKError(Exception):
    """Raised when the API returns a non-2xx response or a network error occurs."""


class SyntheticDatasetClient:
    """
    Client for the Synthetic Dataset Generation API.

    Parameters
    ----------
    api_key : str
        The API key issued when you created your project.
    base_url : str
        Base URL of the FastAPI backend (default: http://localhost:8000).
    timeout : int
        Request timeout in seconds (default: 60).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": self.api_key})

    # -------------------------------------------------------------------------
    # Documents
    # -------------------------------------------------------------------------

    def upload(
        self,
        pages: list[dict[str, Any]],
        batch_size: int = 5,
        overlap: int = 1,
    ) -> dict[str, Any]:
        """
        Upload pages and trigger the full pipeline (batch > QA > RAGAS > persist).

        Parameters
        ----------
        pages : list[dict]
            Each dict must contain:
              - page_no (int)  - 1-based page number
              - doc_id  (str)  - document identifier, e.g. "DOC-A"
              - text    (str)  - full page text

            Example::

                pages = [
                    {"page_no": 1, "doc_id": "DOC-A", "text": "..."},
                    {"page_no": 2, "doc_id": "DOC-A", "text": "..."},
                ]
                sdk.upload(pages)

        batch_size : int
            Pages per batch (default 5).
        overlap : int
            Overlapping pages between consecutive batches (default 1).

        Returns
        -------
        dict
            {"document_id": ..., "project_id": ..., "total_pages": ...,
             "job_status": "processing"}
        """
        if not pages:
            raise SDKError("pages list is empty.")

        doc_id: str = str(pages[0]["doc_id"])

        clean_pages = [
            {"page_no": int(p["page_no"]), "text": str(p["text"])}
            for p in pages
        ]

        payload = {
            "doc_id": doc_id,
            "pages": clean_pages,
            "batch_size": batch_size,
            "overlap": overlap,
        }
        return self._post("/ingest/", payload)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    def get_batch_status(self, doc_id: str) -> dict[str, Any]:
        """Poll pipeline progress for a document."""
        return self._get("/dataset/status", params={"doc_id": doc_id})

    def get_dataset(self, include_faulty: bool = False) -> dict[str, Any]:
        """
        Retrieve the generated QA dataset for your project.

        Parameters
        ----------
        include_faulty : bool
            If True, also returns entries that failed RAGAS evaluation.
        """
        params = {"include_faulty": str(include_faulty).lower()}
        return self._get("/dataset/", params=params)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict[str, Any]:
        url = self.base_url + path
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise SDKError(f"Network error: {exc}") from exc
        return self._handle(resp)

    def _get(self, path: str, params: Optional[dict] = None) -> dict[str, Any]:
        url = self.base_url + path
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise SDKError(f"Network error: {exc}") from exc
        return self._handle(resp)

    @staticmethod
    def _handle(resp: requests.Response) -> dict[str, Any]:
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}

        if not resp.ok:
            detail = body.get("detail", resp.text)
            raise SDKError(f"API error {resp.status_code}: {detail}")

        return body


# -------------------------------------------------------------------------
# Static helper: create a project (no API key needed yet)
# -------------------------------------------------------------------------

def create_project(
    name: str,
    base_url: str = "http://localhost:8000",
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Create a new project and receive its API key.

    Returns
    -------
    dict
        {"project_id": ..., "name": ..., "api_key": "sdk-xxxx", "created_at": ...}
    """
    url = base_url.rstrip("/") + "/projects/"
    try:
        resp = requests.post(url, json={"name": name}, timeout=timeout)
    except requests.RequestException as exc:
        raise SDKError(f"Network error: {exc}") from exc

    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}

    if not resp.ok:
        detail = body.get("detail", resp.text)
        raise SDKError(f"API error {resp.status_code}: {detail}")

    return body
