"""
API resources for Dagster pipelines.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ernold@uvm.edu>
"""

import time
from collections import deque

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
        # Rate limiting state
        self._request_timestamps = deque()
        self._rate_limit = 500
        self._rate_period = 60  # seconds
        self._min_interval = self._rate_period / self._rate_limit

    def _respect_rate_limit(self):
        """
        Rate limiter.
        
        Ensures that the number of requests does not exceed the rate limit, using
        a queue to store the timestamps of the last requests. Also spreads requests
        more evenly by enforcing a minimum interval between requests.
        """
        now = time.time()
        min_interval = self._min_interval

        # Enforce minimum interval between requests
        if self._request_timestamps:
            time_since_last = now - self._request_timestamps[-1]
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)
                now = time.time()

        # Remove timestamps older than 60 seconds
        while (
            self._request_timestamps and
            now - self._request_timestamps[0] > self._rate_period
        ):
            self._request_timestamps.popleft()

        # Sleep if rate limit is exceeded
        if len(self._request_timestamps) >= self._rate_limit:
            sleep_time = self._rate_period - (now - self._request_timestamps[0]) + 0.01
            time.sleep(sleep_time)
            now = time.time()
            while (
                self._request_timestamps and
                now - self._request_timestamps[0] > self._rate_period
            ):
                self._request_timestamps.popleft()

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
            self._request_timestamps.append(time.time())
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


@dg.resource
def manifold_api_resource(context):
    """Manifold API client as a resource."""
    return ManifoldClient()
