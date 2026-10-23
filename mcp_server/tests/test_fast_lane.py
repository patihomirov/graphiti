"""Tests for the fast-lane episode path (2026-10-24).

Covers:
- queue_service routing: small bodies -> fast_lane=True, large -> False, disabled flag,
  and the journal worker actually injecting ``kwargs['fast_lane']``.
- graphiti.add_episode fast-lane branch: exactly ONE combined extraction call, no
  separate extract_nodes / resolve_extracted_nodes calls, narrow prev window, write
  through ``_process_episode_data``, result shape compatible with ``AddEpisodeResults``.
- deterministic exact-match: existing node name -> reused uuid + preserved summary,
  unknown name -> fresh uuid.
- ``build_entity_types(..., entity_attributes_doc_only=True)`` -> documentation-only
  models with no fields.
"""

from unittest.mock import AsyncMock, MagicMock

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults, Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from pydantic import BaseModel

from config.schema import EntityTypeConfig, ResilienceConfig
from services.queue_service import QueueService
from utils import type_config


def _make_clients(driver: MagicMock) -> GraphitiClients:
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()
    return GraphitiClients.model_construct(
        driver=driver,
        embedder=embedder,
        cross_encoder=cross_encoder,
        llm_client=llm_client,
        tracer=MagicMock(),
    )


def _make_graphiti(driver: MagicMock) -> Graphiti:
    """Build a Graphiti instance wired to test doubles, bypassing DB/LLM construction."""
    clients = _make_clients(driver)
    g = object.__new__(Graphiti)
    g.driver = driver
    g.clients = clients
    g.llm_client = clients.llm_client
    g.embedder = clients.embedder
    g.cross_encoder = clients.cross_encoder
    g.store_raw_episode_content = True
    g.max_coroutines = 2
    trac = MagicMock()
    trac.start_span.return_value.__enter__.return_value = MagicMock()
    # __exit__ must signal "handled" generously (no exception propagation issues).
    trac.start_span.return_value.__exit__.return_value = False
    g.tracer = trac
    # Namespace API used nowhere in the tested path, but keep it for safety.
    g.nodes = MagicMock()
    g.edges = MagicMock()
    return g


def _make_episode(uuid_suffix: str, group_id: str = '') -> EpisodicNode:
    return EpisodicNode(
        name=f'episode-{uuid_suffix}',
        group_id=group_id,
        labels=[],
        source=EpisodeType.message,
        content=f'content-{uuid_suffix}',
        source_description='test',
        created_at=utc_now(),
        valid_at=utc_now(),
    )


def _make_entity_node(name: str, group_id: str = '') -> EntityNode:
    return EntityNode(
        name=name,
        group_id=group_id,
        labels=['Entity'],
        created_at=utc_now(),
    )


# --------------------------------------------------------------------------- #
# B. queue_service routing                                                     #
# --------------------------------------------------------------------------- #


class TestFastLaneRouting:
    def test_small_body_routes_to_fast_lane(self):
        svc = QueueService(resilience=ResilienceConfig())  # fast_lane_enabled=True, 5120
        assert svc._should_fast_lane({'episode_body': 'x' * 3000}) is True
        # boundary inclusive
        assert svc._should_fast_lane({'episode_body': 'x' * 5120}) is True

    def test_large_body_stays_legacy(self):
        svc = QueueService(resilience=ResilienceConfig())
        assert svc._should_fast_lane({'episode_body': 'x' * 10000}) is False

    def test_disabled_flag_never_fast_lanes(self):
        svc = QueueService(resilience=ResilienceConfig(fast_lane_enabled=False))
        assert svc._should_fast_lane({'episode_body': 'x' * 10}) is False

    def test_missing_or_bad_plan_falls_back_to_legacy(self):
        svc = QueueService(resilience=ResilienceConfig())
        assert svc._should_fast_lane({}) is False
        assert svc._should_fast_lane({'episode_body': None}) is False
        # Unknown config (no fast_lane_enabled attribute) -> robust False.
        svc.resilience = object()
        assert svc._should_fast_lane({'episode_body': 'x'}) is False

    async def test_journal_worker_injects_fast_lane_flag(
        self, tmp_path, monkeypatch
    ):
        svc = QueueService(
            resilience=ResilienceConfig(
                journal_enabled=True,
                journal_path=str(tmp_path / 'journal.db'),
            )
        )
        svc._graphiti_client = MagicMock()

        def builder(plan):
            return {
                'name': plan['name'],
                'episode_body': plan['episode_body'],
                'group_id': plan['group_id'],
            }

        svc._episode_builder = builder

        captured: list[dict] = []

        async def fake_failover(row, kwargs):
            captured.append(dict(kwargs))

        monkeypatch.setattr(svc, '_process_with_failover', fake_failover)

        # small -> fast_lane
        small = {
            'group_id': 'g-fl',
            'name': 'small',
            'episode_body': 'x' * 3000,
        }
        await svc._journal.enqueue(small)
        assert await svc._process_journal_once('g-fl', leader=True)
        assert captured[-1]['fast_lane'] is True

        # large -> NO fast_lane key (legacy kwargs untouched)
        large = {
            'group_id': 'g-large',
            'name': 'large',
            'episode_body': 'y' * 10000,
        }
        await svc._journal.enqueue(large)
        assert await svc._process_journal_once('g-large', leader=True)
        assert 'fast_lane' not in captured[-1] or captured[-1].get('fast_lane') is False


# --------------------------------------------------------------------------- #
# A. add_episode fast-lane branch                                              #
# --------------------------------------------------------------------------- #


class TestAddEpisodeFastLane:
    async def _run(
        self,
        graphiti: Graphiti,
        *,
        entity_types=None,
        excluded_entity_types=None,
        edge_types=None,
        edge_type_map=None,
        custom_extraction_instructions=None,
        **over,
    ):
        return await graphiti.add_episode(
            name='episode',
            episode_body='x' * 1000,
            source_description='src',
            reference_time=utc_now(),
            source=EpisodeType.message,
            group_id=None,
            entity_types=entity_types,
            excluded_entity_types=excluded_entity_types,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
            custom_extraction_instructions=custom_extraction_instructions,
            fast_lane=True,
            **over,
        )

    async def test_single_combined_call_no_legacy_extractors(self, monkeypatch):
        driver = MagicMock()
        driver.provider = GraphProvider.NEO4J
        driver.execute_query = AsyncMock(return_value=([], None, None))
        graphiti = _make_graphiti(driver)

        prevs = [_make_episode(str(i)) for i in range(3)]
        monkeypatch.setattr(graphiti, 'retrieve_episodes', AsyncMock(return_value=prevs))

        n1 = _make_entity_node('Alice Smith')
        n2 = _make_entity_node('Bob Jones')
        e1 = EntityEdge(
            source_node_uuid=n1.uuid,
            target_node_uuid=n2.uuid,
            name='knows',
            group_id='',
            fact='Alice knows Bob',
            created_at=utc_now(),
        )

        combined = AsyncMock(return_value=([n1, n2], [e1], {n1.uuid: [0], n2.uuid: [0]}))
        monkeypatch.setattr('graphiti_core.graphiti.extract_nodes_and_edges', combined)

        legacy_extract = AsyncMock()
        legacy_resolve = AsyncMock()
        attributes = AsyncMock()
        monkeypatch.setattr('graphiti_core.graphiti.extract_nodes', legacy_extract)
        monkeypatch.setattr('graphiti_core.graphiti.resolve_extracted_nodes', legacy_resolve)
        monkeypatch.setattr(
            'graphiti_core.graphiti.extract_attributes_from_nodes', attributes
        )

        processed: dict = {}

        async def fake_process(
            episode,
            nodes,
            entity_edges,
            now,
            group_id,
            saga=None,
            saga_previous_episode_uuid=None,
            node_episode_index_map=None,
            clients=None,
        ):
            processed['episode'] = episode
            processed['nodes'] = nodes
            processed['edges'] = entity_edges
            processed['index_map'] = node_episode_index_map
            return [], episode

        monkeypatch.setattr(graphiti, '_process_episode_data', fake_process)

        result = await self._run(graphiti)

        # Exactly one combined extraction call, receiving the 3-episode prev window.
        assert combined.await_count == 1
        # previous_episodes is the 3rd positional arg (clients, episode, previous_episodes).
        assert combined.await_args.args[2] == prevs
        assert len(combined.await_args.args[2]) == 3
        # Legacy per-call extractors must NOT run on the fast-lane path.
        legacy_extract.assert_not_awaited()
        legacy_resolve.assert_not_awaited()
        # Default entity_types=None -> doc-only -> attribute extraction skipped.
        attributes.assert_not_awaited()
        # Write went through _process_episode_data with the resolved nodes/edges.
        assert processed['nodes'] == [n1, n2]
        assert processed['edges'] == [e1]
        # Result shape compatible with the legacy AddEpisodeResults.
        assert isinstance(result, AddEpisodeResults)
        assert result.nodes == [n1, n2]
        assert result.edges == [e1]
        assert result.episodic_edges == []
        assert result.communities == []
        assert result.episode is processed['episode']

    async def test_rich_types_run_attribute_extraction(self, monkeypatch):
        driver = MagicMock()
        driver.provider = GraphProvider.NEO4J
        driver.execute_query = AsyncMock(return_value=([], None, None))
        graphiti = _make_graphiti(driver)
        monkeypatch.setattr(graphiti, 'retrieve_episodes', AsyncMock(return_value=[]))

        n1 = _make_entity_node('Alice Smith')
        e1 = EntityEdge(
            source_node_uuid=n1.uuid,
            target_node_uuid=n1.uuid,
            name='self',
            group_id='',
            fact='Alice mentions herself',
            created_at=utc_now(),
        )
        combined = AsyncMock(return_value=([n1], [e1], {n1.uuid: [0]}))
        monkeypatch.setattr('graphiti_core.graphiti.extract_nodes_and_edges', combined)

        attributes = AsyncMock(side_effect=lambda clients, nodes, *a, **k: nodes)
        monkeypatch.setattr(
            'graphiti_core.graphiti.extract_attributes_from_nodes', attributes
        )

        class RichType(BaseModel):
            age: int

        async def fake_process(*args, **kwargs):
            return [], args[0]

        monkeypatch.setattr(graphiti, '_process_episode_data', fake_process)

        await self._run(graphiti, entity_types={'Person': RichType})

        # Rich model present -> doc-only shortcut must NOT apply.
        attributes.assert_awaited_once()

    async def test_exact_match_reuses_existing_node_uuid_and_summary(
        self, monkeypatch
    ):
        driver = MagicMock()
        driver.provider = GraphProvider.NEO4J
        now = utc_now()
        existing_record = {
            'uuid': 'existing-uuid',
            'name': 'Alice Smith',
            'group_id': '',
            'created_at': now,
            'summary': 'pre-existing summary',
            'labels': ['Entity'],
            'attributes': {},
        }
        driver.execute_query = AsyncMock(return_value=([existing_record], None, None))
        graphiti = _make_graphiti(driver)
        monkeypatch.setattr(graphiti, 'retrieve_episodes', AsyncMock(return_value=[]))

        n_alice = _make_entity_node('Alice Smith')  # matches existing by lowercased name
        n_bob = _make_entity_node('Bob Jones')  # no match -> fresh uuid
        edge = EntityEdge(
            source_node_uuid=n_alice.uuid,
            target_node_uuid=n_bob.uuid,
            name='knows',
            group_id='',
            fact='Alice knows Bob',
            created_at=now,
        )
        combined = AsyncMock(
            return_value=([n_alice, n_bob], [edge], {n_alice.uuid: [0], n_bob.uuid: [0]})
        )
        monkeypatch.setattr('graphiti_core.graphiti.extract_nodes_and_edges', combined)

        processed: dict = {}

        async def fake_process(*args, **kwargs):
            processed['nodes'] = args[1]
            processed['edges'] = args[2]
            return [], args[0]

        monkeypatch.setattr(graphiti, '_process_episode_data', fake_process)

        await self._run(graphiti)

        processed_nodes = processed['nodes']
        by_name = {n.name: n for n in processed_nodes}

        # Alice reuses the existing uuid and keeps the old summary (bulk save would
        # otherwise overwrite it with '') and the original created_at.
        alice = by_name['Alice Smith']
        assert alice.uuid == 'existing-uuid'
        assert alice.summary == 'pre-existing summary'
        assert alice.created_at == now
        # Bob is brand new -> fresh uuid, empty summary.
        bob = by_name['Bob Jones']
        assert bob.uuid != 'existing-uuid'
        assert bob.summary == ''
        # Edge was re-pointed onto the reused uuid.
        edge_resolved = processed['edges'][0]
        assert edge_resolved.source_node_uuid == 'existing-uuid'
        assert edge_resolved.target_node_uuid == bob.uuid

    async def test_no_candidates_keeps_all_fresh(self, monkeypatch):
        driver = MagicMock()
        driver.provider = GraphProvider.NEO4J
        driver.execute_query = AsyncMock(return_value=([], None, None))
        graphiti = _make_graphiti(driver)
        monkeypatch.setattr(graphiti, 'retrieve_episodes', AsyncMock(return_value=[]))

        n1 = _make_entity_node('Solo Entity')
        combined = AsyncMock(return_value=([n1], [], {n1.uuid: [0]}))
        monkeypatch.setattr('graphiti_core.graphiti.extract_nodes_and_edges', combined)

        async def fake_process(*args, **kwargs):
            return [], args[0]

        monkeypatch.setattr(graphiti, '_process_episode_data', fake_process)

        result = await self._run(graphiti)
        assert result.nodes == [n1]
        assert result.nodes[0].uuid == n1.uuid


# --------------------------------------------------------------------------- #
# D. exact-match helper (unit)                                                 #
# --------------------------------------------------------------------------- #


class TestExactMatch:
    async def test_queries_by_lowercased_names_in_group(self):
        driver = MagicMock()
        driver.provider = GraphProvider.NEO4J
        driver.execute_query = AsyncMock(return_value=([], None, None))
        graphiti = _make_graphiti(driver)

        nodes = [_make_entity_node('Alice Smith'), _make_entity_node('Bob Jones')]
        result = await graphiti._exact_match_existing_nodes(driver, nodes, 'grp-1')

        call = driver.execute_query.await_args
        assert call.kwargs['group_id'] == 'grp-1'
        assert call.kwargs['names'] == ['alice smith', 'bob jones']
        assert 'toLower(n.name)' in call.args[0]
        assert result == {}

    async def test_empty_input_no_query(self):
        driver = MagicMock()
        driver.provider = GraphProvider.NEO4J
        graphiti = _make_graphiti(driver)
        result = await graphiti._exact_match_existing_nodes(driver, [], 'grp')
        assert result == {}
        driver.execute_query.assert_not_called()


# --------------------------------------------------------------------------- #
# C/step-2b. build_entity_types with entity_attributes_doc_only                #
# --------------------------------------------------------------------------- #


class TestBuildEntityTypesDocOnly:
    def test_doc_only_forces_no_fields_even_for_rich_registered(self):
        cfg = [EntityTypeConfig(name='Requirement', description='A registered rich type')]
        result = type_config.build_entity_types(cfg, entity_attributes_doc_only=True)
        model = result['Requirement']
        assert len(model.model_fields) == 0
        assert 'A registered rich type' in (model.__doc__ or '')

    def test_default_still_prefers_rich_models(self):
        cfg = [EntityTypeConfig(name='Requirement', description='desc')]
        result = type_config.build_entity_types(cfg)
        model = result['Requirement']
        assert len(model.model_fields) > 0

    def test_none_config_returns_none(self):
        assert type_config.build_entity_types(None, entity_attributes_doc_only=True) is None
