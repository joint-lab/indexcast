"""
API resources for Dagster pipelines.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ernold@uvm.edu>
"""

import time

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
            status_forcelist=[503],  # Retry on 503 errors
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        # Rate limiting state
        self._last_request_time = 0
        self._rate_limit = 500
        self._rate_period = 60  # seconds
        self._min_interval = self._rate_period / self._rate_limit

    def _respect_rate_limit(self):
        """
        Rate limiter.

        Ensures that requests are spaced by at least the minimum interval.
        """
        now = time.time()
        min_interval = self._min_interval
        time_since_last = now - self._last_request_time
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        self._last_request_time = time.time()

    def execute_get_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Execute a GET request on the Manifold API.

        Args:
            endpoint (str): The API endpoint to call.
            params (dict, optional): Query parameters for the request.

        Returns:
            dict: The JSON response from the API.

        """
        self._respect_rate_limit()
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request to {url} failed: {e}") from e

    def markets(self, limit=1000, before=None) -> list:
        """
        Fetch markets from the Manifold API.

        Args:
            limit (int): Number of markets to fetch. Maximum is 1000.
            before (str, optional): Fetch markets created before this id.

        """
        params = {"limit": limit}
        if before:
            params["before"] = before
        if limit > 1000:
            raise ValueError("Limit must be 1000 or less.")
        return self.execute_get_request("markets", params=params)

    def full_market(self, market_id: str) -> dict:
        """
        Fetch full market details by ID.

        Args:
            market_id (str): The ID of the market.

        Returns:
            dict: The full market data.

        """
        return self.execute_get_request(f"market/{market_id}")

    def bets(self, market_id: str, limit=1000, before=None) -> list:
        """
        Fetch bets for a specific market.

        Args:
            market_id (str): The ID of the market.
            limit (int): Maximum number of bets to fetch (default 1000).
            before (str, optional): Fetch markets created before this id.

        Returns:
            list: A list of bets.

        """
        params = {"contractId": market_id, "limit": limit}
        if before:
            params["before"] = before
        if limit > 1000:
            raise ValueError("Limit must be 1000 or less.")
        return self.execute_get_request("bets", params=params)

    def comments(self, market_id: str, limit=1000, before=None) -> list:
        """
        Fetch comments for a specific market.

        Args:
            market_id (str): The ID of the market.
            limit (int): Number of markets to fetch. Maximum is 1000.
            before (str, optional): Fetch markets created before this id.

        Returns:
            list: A list of comments.

        """
        params = {"contractId": market_id, "limit": limit}
        if before:
            params["before"] = before
        if limit > 1000:
            raise ValueError("Limit must be 1000 or less.")
        return self.execute_get_request("comments", params=params)

@dg.resource
def manifold_api_resource(context):
    """Manifold API client as a resource."""
    return ManifoldClient()
