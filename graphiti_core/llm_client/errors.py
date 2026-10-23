"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class RateLimitError(Exception):
    """Exception raised when the rate limit is exceeded.

    ``retry_after`` (seconds) carries the provider's Retry-After hint when one
    was present on the triggering response, so retry logic and the storm
    throttle can honour it instead of hammering the gateway as it asks us to
    slow down. ``None`` when no hint was conveyed.
    """

    def __init__(
        self,
        message: str = 'Rate limit exceeded. Please try again later.',
        retry_after: float | None = None,
    ):
        self.message = message
        self.retry_after = float(retry_after) if retry_after is not None else None
        super().__init__(self.message)


class RefusalError(Exception):
    """Exception raised when the LLM refuses to generate a response."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class EmptyResponseError(Exception):
    """Exception raised when the LLM returns an empty response."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
