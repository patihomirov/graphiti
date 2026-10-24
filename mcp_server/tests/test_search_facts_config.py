"""Regression tests: Graphiti.search() must not mutate the global search recipes.

Discovered during the sqlite-journal work: the server's ``search_memory_facts``
handler calls ``client.search(..., num_results=max_facts)`` and graphiti-core's
``Graphiti.search`` wrote ``num_results`` straight onto the module-level
``EDGE_HYBRID_SEARCH_RRF`` / ``EDGE_HYBRID_SEARCH_NODE_DISTANCE`` recipe:

    search_config = EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None else ...
    search_config.limit = num_results  # mutates the global recipe object

Any other consumer of the same recipe — notably the dedup / edge-invalidation
searches in graphiti-core ``utils/maintenance/edge_operations.py``, which pass
``config=EDGE_HYBRID_SEARCH_RRF`` — then inherited the last caller's limit and
returned far more results than the recipe's default.

The fix copies the recipe (``.model_copy()``) before setting ``limit``, so the
globals stay untouched. These tests pin that behaviour on the client path
(``Graphiti.search``), the edge_operations reuse path, and the server path
(``search_memory_facts``).
"""

from __future__ import annotations

from types import SimpleNamespace

import graphiti_core.graphiti as graphiti_mod
import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti import Graphiti
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT, SearchResults
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)

import graphiti_mcp_server as server


class FakeDriver:
    """Non-Falkor driver stand-in: keeps handle_multiple_group_ids on the
    plain (normal-execution) path."""

    provider = GraphProvider.NEO4J

    def clone(self, database: str):  # pragma: no cover - unused in tests
        return self


class FakeClients(SimpleNamespace):
    """Stands in for the pydantic GraphitiClients holder (all members are
    untyped here because Graphiti.search only reads ``clients``)."""


def make_search_client() -> Graphiti:
    """A real Graphiti instance whose search() never touches a backend.

    ``object.__new__`` skips Graphiti.__init__, whose pydantic
    ``GraphitiClients`` holder rejects bare stand-ins. search() reads nothing
    but ``self.clients``, so a namespace with a driver is enough.
    """
    client = object.__new__(Graphiti)
    client.clients = FakeClients(driver=FakeDriver())
    return client


async def _async_return(value):
    return value


@pytest.fixture(autouse=True)
def reset_recipe_limits():
    """Start every test from a clean global-recipe state and restore it after."""
    original_rrf = EDGE_HYBRID_SEARCH_RRF.limit
    original_node_distance = EDGE_HYBRID_SEARCH_NODE_DISTANCE.limit
    EDGE_HYBRID_SEARCH_RRF.limit = DEFAULT_SEARCH_LIMIT
    EDGE_HYBRID_SEARCH_NODE_DISTANCE.limit = DEFAULT_SEARCH_LIMIT
    try:
        yield
    finally:
        EDGE_HYBRID_SEARCH_RRF.limit = original_rrf
        EDGE_HYBRID_SEARCH_NODE_DISTANCE.limit = original_node_distance


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graphiti_search_does_not_mutate_global_recipe(monkeypatch):
    """A search with a large num_results applies its limit to a copy of the
    recipe, leaving the module-level EDGE_HYBRID_SEARCH_* recipes untouched."""
    seen: list[int] = []

    async def fake_search(
        clients,
        query,
        group_ids,
        config,
        search_filter=None,
        driver=None,
        center_node_uuid=None,
        query_vector=None,
    ):
        seen.append(config.limit)
        return SearchResults(edges=[])

    monkeypatch.setattr(graphiti_mod, 'search', fake_search)

    client = make_search_client()
    await client.search(query='needle', num_results=123)

    # The consumer call itself honored num_results…
    assert seen == [123]
    # …but the globals that other consumers share were not touched.
    assert EDGE_HYBRID_SEARCH_RRF.limit == DEFAULT_SEARCH_LIMIT
    assert EDGE_HYBRID_SEARCH_NODE_DISTANCE.limit == DEFAULT_SEARCH_LIMIT


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_operations_style_search_gets_default_limit(monkeypatch):
    """After a big-num_results search, a dedup-style search that passes the
    global EDGE_HYBRID_SEARCH_RRF recipe (as edge_operations.py does) still
    gets the recipe default limit, not the previous caller's limit."""
    seen: list[int] = []
    previous_rrf_limit = 123

    async def fake_search(
        clients,
        query,
        group_ids,
        config,
        search_filter=None,
        driver=None,
        center_node_uuid=None,
        query_vector=None,
    ):
        seen.append(config.limit)
        return SearchResults(edges=[])

    monkeypatch.setattr(graphiti_mod, 'search', fake_search)

    client = make_search_client()
    # search_memory_facts / search_memory path: big num_results.
    await client.search(query='needle', num_results=previous_rrf_limit)
    # edge_operations.py:413-443 / graphiti.py dedup path: global recipe object.
    await fake_search(None, 'edge.fact', ['g'], EDGE_HYBRID_SEARCH_RRF)

    # …the second call must see the default, never the leaked 123.
    assert seen == [previous_rrf_limit, DEFAULT_SEARCH_LIMIT]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_memory_facts_does_not_mutate_global_recipe(monkeypatch):
    """End-to-end over the server handler: search_memory_facts(max_facts=…)
    leaves the global EDGE_HYBRID_SEARCH_RRF recipe at its default."""
    seen: list[int] = []

    async def fake_search(
        clients,
        query,
        group_ids,
        config,
        search_filter=None,
        driver=None,
        center_node_uuid=None,
        query_vector=None,
    ):
        seen.append(config.limit)
        return SearchResults(edges=[])

    monkeypatch.setattr(graphiti_mod, 'search', fake_search)

    client = make_search_client()
    monkeypatch.setattr(
        server,
        'graphiti_service',
        SimpleNamespace(get_client=lambda: _async_return(client)),
    )
    monkeypatch.setattr(
        server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='g')),
        raising=False,
    )

    resp = await server.search_memory_facts(query='needle', max_facts=123, group_ids=None)

    assert seen == [123]
    assert EDGE_HYBRID_SEARCH_RRF.limit == DEFAULT_SEARCH_LIMIT
    # Successful handler path, no error surfaced.
    assert not (isinstance(resp, dict) and 'error' in resp)
