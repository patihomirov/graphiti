"""Regression tests: bulk_utils batches missing embeddings before the write tx.

Discovered during the sqlite-journal work: ``add_nodes_and_edges_bulk_tx`` embedded
each entity node/edge inside the Neo4j write transaction, one ``await`` round-trip at
a time, keeping the transaction open for N×0.5-1s. The fix collects every object whose
embedding is still ``None`` and sends their texts to the embedder as a single
``create_batch`` call *before* the transaction is opened, then distributes the results
back onto the objects.

These tests pin that behaviour on the ``add_nodes_and_edges_bulk`` path with a fake
driver whose session records the state of the objects at the moment ``execute_write``
is reached — proving the batch ran before the transaction, not inside it.
"""

from __future__ import annotations

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.datetime_utils import utc_now


class FakeEmbedder:
    """Embedder stand-in: records every create_batch call and returns distinct
    index-labelled vectors so result distribution is observable."""

    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(input_data_list))
        return [[float(i), 0.0] for i in range(len(input_data_list))]


class RecordingSession:
    """Driver session whose execute_write does not run the tx but snapshots the
    embedding state of the objects at the moment the tx would start."""

    def __init__(self) -> None:
        self.executed = False
        self.node_embeddings_at_tx: dict[str, list[float] | None] = {}
        self.edge_embeddings_at_tx: dict[str, list[float] | None] = {}

    async def close(self) -> None:
        return None

    async def execute_write(self, tx_fn, *args, **kwargs):
        # args = (episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder)
        entity_nodes = args[2]
        entity_edges = args[3]
        self.node_embeddings_at_tx = {n.uuid: n.name_embedding for n in entity_nodes}
        self.edge_embeddings_at_tx = {e.uuid: e.fact_embedding for e in entity_edges}
        self.executed = True
        return None


class FakeDriver:
    def __init__(self) -> None:
        self.session_obj = RecordingSession()

    def session(self) -> RecordingSession:
        return self.session_obj


def make_edge(name: str, fact: str) -> EntityEdge:
    return EntityEdge(
        name=name,
        fact=fact,
        source_node_uuid='source',
        target_node_uuid='target',
        created_at=utc_now(),
        group_id='g',
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_embeddings_batched_once_before_transaction() -> None:
    """N missing node/edge embeddings become exactly one create_batch call that runs
    before the write transaction, and its results are laid out onto the objects in
    node-first order. Preset objects are never re-embedded."""
    embedder = FakeEmbedder()
    driver = FakeDriver()

    node_missing_1 = EntityNode(name='Alpha Node', group_id='g')
    node_missing_2 = EntityNode(name='Gamma Node', group_id='g')
    node_preset = EntityNode(name='Beta Node', group_id='g')
    node_preset.name_embedding = [9.0, 9.0]

    edge_missing_1 = make_edge('relates_to', 'Alpha Node relates to Craft')
    edge_missing_2 = make_edge('relates_to', 'Gamma Node relates to Delta')
    edge_preset = make_edge('relates_to', 'Preset fact')
    edge_preset.fact_embedding = [8.0, 8.0]

    await add_nodes_and_edges_bulk(
        driver,
        [],
        [],
        [node_missing_1, node_preset, node_missing_2],
        [edge_missing_1, edge_preset, edge_missing_2],
        embedder,
    )

    # Exactly one batch call, containing only the missing objects, nodes first.
    assert embedder.batch_calls == [
        ['Alpha Node', 'Gamma Node', 'Alpha Node relates to Craft', 'Gamma Node relates to Delta']
    ]

    # Results distributed back onto the objects (node-first layout).
    assert node_missing_1.name_embedding == [0.0, 0.0]
    assert node_missing_2.name_embedding == [1.0, 0.0]
    assert edge_missing_1.fact_embedding == [2.0, 0.0]
    assert edge_missing_2.fact_embedding == [3.0, 0.0]

    # Preset objects left untouched.
    assert node_preset.name_embedding == [9.0, 9.0]
    assert edge_preset.fact_embedding == [8.0, 8.0]

    # Every object was already embedded by the time the write tx was reached.
    assert driver.session_obj.executed is True
    assert driver.session_obj.node_embeddings_at_tx == {
        node_missing_1.uuid: [0.0, 0.0],
        node_missing_2.uuid: [1.0, 0.0],
        node_preset.uuid: [9.0, 9.0],
    }
    assert driver.session_obj.edge_embeddings_at_tx == {
        edge_missing_1.uuid: [2.0, 0.0],
        edge_missing_2.uuid: [3.0, 0.0],
        edge_preset.uuid: [8.0, 8.0],
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_batch_when_all_embeddings_preset() -> None:
    """When nothing is missing, create_batch is never called and the tx still runs."""
    embedder = FakeEmbedder()
    driver = FakeDriver()

    node = EntityNode(name='Alpha Node', group_id='g')
    node.name_embedding = [1.0, 1.0]
    edge = make_edge('relates_to', 'Alpha Node relates to Craft')
    edge.fact_embedding = [2.0, 2.0]

    await add_nodes_and_edges_bulk(driver, [], [], [node], [edge], embedder)

    assert embedder.batch_calls == []
    assert driver.session_obj.executed is True
    assert node.name_embedding == [1.0, 1.0]
    assert edge.fact_embedding == [2.0, 2.0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_never_used_for_batches() -> None:
    """The batch path must not fall back to per-object embedder.create; a single
    batched call covers all missing objects even when there is only one of them."""
    embedder = FakeEmbedder()
    driver = FakeDriver()

    node = EntityNode(name='Solo Node', group_id='g')

    await add_nodes_and_edges_bulk(driver, [], [], [node], [], embedder)

    assert embedder.batch_calls == [['Solo Node']]
    assert node.name_embedding == [0.0, 0.0]
