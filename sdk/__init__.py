"""
Synthetic Dataset SDK
=====================
A lightweight Python client for the Synthetic Dataset Generation SaaS API.

Quick start
-----------
    from sdk import SyntheticDatasetClient

    sdk = SyntheticDatasetClient(api_key="sdk-xxxx", base_url="http://localhost:8000")

    # Upload a document (list of pages loaded from your JSON file)
    result = sdk.upload(doc_id="DOC-A", pages=pages)

    # Later – fetch the validated dataset
    dataset = sdk.get_dataset()
"""
from sdk.client import SyntheticDatasetClient

__all__ = ["SyntheticDatasetClient"]
