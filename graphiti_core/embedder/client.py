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

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))


class EmbedderConfig(BaseModel):
    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True)


class EmbedderClient(ABC):
    @abstractmethod
    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        pass

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        raise NotImplementedError()

    def set_channel_rate_limiter(self, limiter: Any) -> None:
        """Attach (or clear, with None) the per-channel rate limiter (duck-typed).

        Mirrors ``LLMClient.set_channel_rate_limiter``: every embedding call goes
        through ``limiter.acquire(channel_key())`` before hitting the wire when the
        limiter is enabled. Embeddings are the largest request source at the
        gateway, so they get their own sliding-window ceiling instead of burning
        the shared chat bucket.
        """
        self.channel_rate_limiter = limiter

    def channel_key(self) -> str:
        """The (base_url, embedding_model) channel key used by the rate limiter.

        ``getattr`` keeps it safe for embedders whose config lacks one of the
        fields (e.g. Voyage/Gemini/Azure); a missing piece degrades to
        ``'unknown'`` rather than raising.
        """
        config = getattr(self, 'config', None)
        base = getattr(config, 'base_url', None) or 'unknown'
        model = getattr(config, 'embedding_model', None) or 'unknown'
        return f'{base}|{model}'

    async def _channel_pace(self) -> None:
        """Wait for a free slot on this channel's rate limiter, if one is set.

        No-op when no limiter is attached or its limit is 0 (OFF). The wait is
        async on the caller's coroutine, so a saturated channel parks only the
        current embedding call, never the whole pool.
        """
        limiter = getattr(self, 'channel_rate_limiter', None)
        if limiter is None:
            return
        if not getattr(limiter, 'enabled', False):
            return
        await limiter.acquire(self.channel_key())
