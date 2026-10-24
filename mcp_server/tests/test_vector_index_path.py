"""
EXPERIMENTAL vector-index node search path (feature F5).

Covers `node_similarity_search` routing behind `driver.use_vector_index`:

- flag False -> legacy linear in-Cypher cosine scan (queryNodes never called);
- flag True  -> native `db.index.vector.queryNodes` path, result mapped into the
  same EntityNode format as the legacy scan;
- flag True + index/query error -> fail-open: warning logged, fallback to the
  legacy linear scan (no exception propagates).

These are unit tests: the driver is a fake recording which queries it receives, so
no Neo4j instance is required.
"""

import logging
from datetime import datetime

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import node_similarity_search


def _node_record() -> dict:
    return {
        'uuid': 'node-1',
        'name': 'Alpha',
        'group_id': 'g1',
        'created_at': datetime(2024, 1, 1, 12, 0, 0),
        'summary': 'summary-alpha',
        'labels': ['Entity'],
        'attributes': {},
    }


class _RecordingNeo4jDriver:
    """Minimal GraphDriver stand-in that records every execute_query call."""

    provider = GraphProvider.NEO4J
    search_interface = None

    def __init__(self, use_vector_index: bool, fail_on_vector: bool = False):
        self.use_vector_index = use_vector_index
        self.fail_on_vector = fail_on_vector
        self.calls: list[tuple[str, dict]] = []

    async def execute_query(self, query: str, **kwargs: object):
        self.calls.append((query, kwargs))
        if self.fail_on_vector and 'db.index.vector.queryNodes' in query:
            raise RuntimeError('vector index missing')
        return [_node_record()], None, None


async def test_vector_index_disabled_uses_linear_scan():
    driver = _RecordingNeo4jDriver(use_vector_index=False)

    nodes = await node_similarity_search(
        driver, [0.1, 0.2, 0.3], SearchFilters(), limit=5, min_score=0.6
    )

    # (a) flag False: queryNodes is never called, the linear cosine scan runs.
    assert len(driver.calls) == 1
    query, kwargs = driver.calls[0]
    assert 'db.index.vector.queryNodes' not in query
    assert 'vector.similarity.cosine' in query
    assert kwargs['limit'] == 5
    assert len(nodes) == 1
    assert isinstance(nodes[0], EntityNode)


async def test_vector_index_enabled_uses_query_nodes_and_maps_result():
    driver = _RecordingNeo4jDriver(use_vector_index=True)

    search_vector = [0.1, 0.2, 0.3]
    nodes = await node_similarity_search(
        driver, search_vector, SearchFilters(), limit=5, min_score=0.6
    )

    # (b) flag True: queryNodes is called with the expected arguments and the
    # returned records are mapped into the same EntityNode format as the scan.
    assert len(driver.calls) == 1
    query, kwargs = driver.calls[0]
    assert "db.index.vector.queryNodes('entity_name_embedding'" in query
    assert 'vector.similarity.cosine' not in query
    assert kwargs['k'] == 5  # no filters: k == limit
    assert kwargs['search_vector'] == search_vector
    assert len(nodes) == 1
    assert isinstance(nodes[0], EntityNode)
    assert nodes[0].uuid == 'node-1'
    assert nodes[0].name == 'Alpha'


async def test_vector_index_enabled_k_scales_with_filters():
    driver = _RecordingNeo4jDriver(use_vector_index=True)

    # With extra predicates (group_ids here) the index is asked for more
    # candidates than the final LIMIT so in-Cypher filtering cannot starve it.
    await node_similarity_search(
        driver, [0.1, 0.2, 0.3], SearchFilters(), group_ids=['g1'], limit=5, min_score=0.6
    )

    assert len(driver.calls) == 1
    query, kwargs = driver.calls[0]
    assert 'db.index.vector.queryNodes' in query
    assert kwargs['k'] == 5 * 10


async def test_vector_index_enabled_fallback_on_error(caplog):
    driver = _RecordingNeo4jDriver(use_vector_index=True, fail_on_vector=True)

    # (c) flag True + queryNodes raises: warning is logged and the legacy linear
    # scan is used instead — the error never reaches the caller.
    with caplog.at_level(logging.WARNING, logger='graphiti_core.search.search_utils'):
        nodes = await node_similarity_search(
            driver, [0.1, 0.2, 0.3], SearchFilters(), limit=5, min_score=0.6
        )

    assert len(driver.calls) == 2  # queryNodes attempt + linear scan fallback
    assert 'db.index.vector.queryNodes' in driver.calls[0][0]
    assert 'db.index.vector.queryNodes' not in driver.calls[1][0]
    assert any(
        'falling back to linear cosine scan' in record.getMessage()
        for record in caplog.records
    )
    assert len(nodes) == 1
    assert isinstance(nodes[0], EntityNode)
