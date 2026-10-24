"""Unit tests: optional ``llm.small_model`` routes ModelSize.small slots to a cheap model.

``ModelSize.small`` is used by time-sensitive/simple prompts such as
``timestamps_batch`` (graphiti_core/utils/maintenance/combined_extraction.py)
and the node/edge date-estimation prompts. Without ``llm.small_model`` the
factory keeps the historical behaviour — the same (main) model for every slot.
When set, the cheap model is used for small slots and the main model for
everything else.

Both production client types are covered:
- the official OpenAI path (``OpenAIClient``): small slot resolves through
  ``_get_model_for_size``;
- the OpenAI-compatible generic path used by llm-gateway-proton
  (``OpenAIGenericClient``): small slot resolves through ``_resolve_wire_model``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from config.schema import LLMConfig, LLMProvidersConfig, OpenAIProviderConfig
from services.factories import LLMClientFactory

OPENAI_API = 'https://api.openai.com/v1'
PROTON_API = 'https://llm-gateway-proton.t.o3.ru/v1'


def _config(small_model: str | None, api_url: str) -> LLMConfig:
    return LLMConfig(
        provider='openai',
        model='gpt-5.5',
        small_model=small_model,
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='test-key', api_url=api_url)
        ),
    )


def _assert_small_resolution(client, cheap_model: str | None) -> None:
    """Assert the small-slot resolution for any factory-built client."""
    if isinstance(client, OpenAIClient):
        if cheap_model is None:
            # Historical behaviour: the small slot is pinned to the main model.
            assert client.small_model == 'gpt-5.5'
            assert client._get_model_for_size(ModelSize.small) == 'gpt-5.5'
        else:
            assert client.small_model == cheap_model
            assert client._get_model_for_size(ModelSize.small) == cheap_model
        assert client._get_model_for_size(ModelSize.medium) == 'gpt-5.5'
    else:
        assert isinstance(client, OpenAIGenericClient)
        expected = cheap_model if cheap_model is not None else 'gpt-5.5'
        assert client._resolve_wire_model(ModelSize.small) == expected
        assert client._resolve_wire_model(ModelSize.medium) == 'gpt-5.5'


class TestSmallModelSchema:
    """The mcp_server LLMConfig carries the optional small_model field."""

    def test_defaults_to_none(self):
        assert LLMConfig().small_model is None

    def test_accepts_small_model(self):
        assert LLMConfig(small_model='gpt-4.1-nano').small_model == 'gpt-4.1-nano'


class TestSmallModelFactory:
    """The factory wires small_model into the core client for all providers."""

    @pytest.mark.parametrize('api_url', [OPENAI_API, PROTON_API])
    def test_small_model_none_keeps_main_model(self, api_url):
        client = LLMClientFactory.create(_config(small_model=None, api_url=api_url))
        assert client.model == 'gpt-5.5'
        _assert_small_resolution(client, cheap_model=None)

    @pytest.mark.parametrize('api_url', [OPENAI_API, PROTON_API])
    def test_small_model_set_uses_cheap_model(self, api_url):
        client = LLMClientFactory.create(
            _config(small_model='gpt-4.1-nano', api_url=api_url)
        )
        assert client.model == 'gpt-5.5'  # main slot unchanged
        _assert_small_resolution(client, cheap_model='gpt-4.1-nano')


class TestSmallModelGenericWire:
    """End-to-end: the model actually sent on the wire follows the slot size."""

    @staticmethod
    def _fake_client():
        class _FakeCompletions:
            def __init__(self):
                self.sent_models: list[str] = []

            async def create(self, **kwargs):
                self.sent_models.append(kwargs['model'])
                message = type('M', (), {'content': '{"ok": true}'})
                choice = type('C', (), {'message': message, 'finish_reason': 'stop'})
                return type('R', (), {'choices': [choice], 'usage': None})

        class _FakeChat:
            def __init__(self):
                self.completions = _FakeCompletions()

        class _FakeOpenAIClient:
            def __init__(self):
                self.chat = _FakeChat()

        return _FakeOpenAIClient()

    @pytest.mark.asyncio
    async def test_default_none_sends_main_model_for_both_slots(self):
        from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.prompts.models import Message

        fake = self._fake_client()
        client = OpenAIGenericClient(
            config=CoreLLMConfig(api_key='k', model='gpt-5.5', base_url=PROTON_API),
            client=fake,
            structured_output_mode='json_object',
        )
        msg = [Message(role='user', content='hi')]
        await client._generate_response(msg, model_size=ModelSize.small)
        await client._generate_response(msg, model_size=ModelSize.medium)
        assert fake.chat.completions.sent_models == ['gpt-5.5', 'gpt-5.5']

    @pytest.mark.asyncio
    async def test_small_model_set_sends_cheap_model_for_small_slot(self):
        from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.prompts.models import Message

        fake = self._fake_client()
        client = OpenAIGenericClient(
            config=CoreLLMConfig(
                api_key='k', model='gpt-5.5', small_model='gpt-4.1-nano', base_url=PROTON_API
            ),
            client=fake,
            structured_output_mode='json_object',
        )
        msg = [Message(role='user', content='hi')]
        await client._generate_response(msg, model_size=ModelSize.small)
        await client._generate_response(msg, model_size=ModelSize.medium)
        assert fake.chat.completions.sent_models == ['gpt-4.1-nano', 'gpt-5.5']

    @pytest.mark.asyncio
    async def test_failover_override_wins_over_small_model(self):
        from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.prompts.models import Message

        fake = self._fake_client()
        client = OpenAIGenericClient(
            config=CoreLLMConfig(
                api_key='k', model='gpt-5.5', small_model='gpt-4.1-nano', base_url=PROTON_API
            ),
            client=fake,
            structured_output_mode='json_object',
        )
        client.set_model_override('zai/glm-5.3-flash')
        msg = [Message(role='user', content='hi')]
        await client._generate_response(msg, model_size=ModelSize.small)
        await client._generate_response(msg, model_size=ModelSize.medium)
        assert fake.chat.completions.sent_models == [
            'zai/glm-5.3-flash',
            'zai/glm-5.3-flash',
        ]
