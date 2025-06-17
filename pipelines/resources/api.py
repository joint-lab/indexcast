"""
API resources for Dagster pipelines.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ernold@uvm.edu>
"""
import dagster as dg
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ManifoldClient:
    """Manifold Market API client."""

    def __init__(self):
        """Initialize the Manifold API client."""
        self.base_url = "https://api.manifold.markets/v0"
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.1,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def execute_get_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Execute a GET request on the Manifold API.

        Args:
            endpoint (str): The API endpoint to call.
            params (dict, optional): Query parameters for the request.

        Returns:
            dict: The JSON response from the API.

        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request to {url} failed: {e}") from e

    def markets(self, limit=500, before=None) -> list:
        """
        Fetch markets from the Manifold API.

        Args:
            limit (int): Maximum number of markets to fetch.
            before (str, optional): Fetch markets created before this id.

        """
        params = {"limit": limit}
        if before:
            params["before"] = before
        return self.execute_get_request("markets", params=params)


@dg.resource
def manifold_api_resource(context):
    """Manifold API client as a resource."""
    return ManifoldClient()
