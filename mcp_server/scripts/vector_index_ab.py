#!/usr/bin/env python3
"""A/B: node_similarity_search — exact linear scan (flag off) vs native vector
index (flag on). Feature F5 refresh-policy verification.

Method:
  - Query texts = real Entity names from the live 'main' graph (+ a couple of
    deliberately problematic / "outlier" nodes whose vectors drop out of HNSW).
  - Each query is embedded ONCE with the same wire embedding (Qwen/Qwen3-
    Embedding-8B via llm-gateway-proton, same as prod) then run through
    node_similarity_search with limit=5, group_ids=['main'].
  - Both drivers are real Neo4jDriver instances differing only in
    use_vector_index. Latency = median of 3 runs. Output: top-5 overlapping
    rank comparison + scores.

Run:  python3 scripts/vector_index_ab.py [--queries N] [--include-outliers]
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
from neo4j import GraphDatabase

SCRIPT_DIR = Path(__file__).resolve().parent
MCP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MCP_DIR / 'src'))

EMBED_URL = 'https://llm-gateway-proton.t.o3.ru/v1/embeddings'
EMBED_MODEL = 'Qwen/Qwen3-Embedding-8B'

# Deliberately problematic nodes found during self-check (vector drops out of
# HNSW / quantization applies) — must be probed, not hidden.
OUTLIER_UUIDS = [
    '5dcc4371-e07c-430e-9739-78f9aa7f5963',
    'b01592b2-672a-4daa-8dd9-e675f05bfdde',
]


def neo4j_creds() -> tuple[str, str, str]:
    uri, user, password = 'bolt://localhost:7687', 'neo4j', ''
    out = subprocess.run(['docker', 'inspect', 'graphiti-neo4j'],
                         capture_output=True, text=True, check=True, timeout=30)
    for item in json.loads(out.stdout):
        for kv in item.get('Config', {}).get('Env', []):
            if kv.startswith('NEO4J_AUTH='):
                _, auth = kv.split('=', 1)
                if '/' in auth:
                    user, password = auth.split('/', 1)
    return uri, user, password


def embed(texts: list[str], api_key: str, dim: int = 1024) -> list[list[float]]:
    """Embed with the same wire model as prod; matryoshka-truncate to 1024
    (graphiti's OpenAIClient does `embedding[: embedding_dim]`)."""
    resp = httpx.post(
        EMBED_URL,
        headers={'Authorization': f'Bearer {api_key}'},
        json={'model': EMBED_MODEL, 'input': texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    data['data'] = sorted(data['data'], key=lambda x: x['index'])
    return [item['embedding'][:dim] for item in data['data']]


def load_query_names(n: int, include_outliers: bool, uri, user, password) -> list[tuple[str, str | None]]:
    """Return [(uuid-or-None, name)] — real main-group entity names + outliers."""
    names: list[tuple[str, str | None]] = []
    drv = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with drv.session() as s:
            rows = s.run(
                "MATCH (n:Entity) WHERE n.group_id = 'main' AND n.name_embedding IS NOT NULL "
                "WITH n, rand() AS r ORDER BY r LIMIT $n "
                "RETURN n.uuid AS u, n.name AS name",
                n=n,
            ).data()
            names = [(r['u'], r['name']) for r in rows]
            if include_outliers:
                for uid in OUTLIER_UUIDS:
                    r = s.run(
                        "MATCH (n:Entity {uuid:$u}) RETURN n.name AS name", u=uid
                    ).single()
                    if r:
                        names.append((uid, r['name']))
    finally:
        drv.close()
    # de-dup, keep order
    seen, out = set(), []
    for u, nm in names:
        if nm not in seen:
            seen.add(nm)
            out.append((u, nm))
    return out


async def run_search(driver, vec, limit=5):
    from graphiti_core.search.search_filters import SearchFilters
    from graphiti_core.search.search_utils import node_similarity_search
    t0 = time.perf_counter()
    nodes = await node_similarity_search(
        driver, vec, SearchFilters(), group_ids=['main'], limit=limit, min_score=0.4
    )
    dt = time.perf_counter() - t0
    return dt, [(n.uuid, n.name) for n in nodes]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', type=int, default=10)
    ap.add_argument('--include-outliers', action='store_true',
                    help='add known HNSW-outlier nodes to the probe set')
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault('SSL_CERT_FILE', '/etc/ssl/certs/ca-certificates.crt')

    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        api_key = Path(os.environ['HOME']).joinpath('.o3_llm_token').read_text().strip()
    uri, user, password = neo4j_creds()

    queries = load_query_names(args.queries, args.include_outliers, uri, user, password)
    print(f'=== A/B vector index — {len(queries)} queries, {args.runs} runs each ===', flush=True)
    print(f'queries: {[q[1][:40] for q in queries]}', flush=True)

    texts = [q[1] for q in queries]
    embs = embed(texts, api_key)
    print(f'embedded {len(embs)} texts (dim={len(embs[0])})', flush=True)

    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    drv_linear = Neo4jDriver(uri, user, password, use_vector_index=False)
    drv_vector = Neo4jDriver(uri, user, password, use_vector_index=True)
    try:
        full_match = 0
        partial = 0
        empty = 0
        print(f'\n{"query":<44} {"linear(ms)":>10} {"vector(ms)":>10} {"top5_equal":>10} {"rank_pos":>9}', flush=True)
        for (_, name), emb in zip(queries, embs, strict=True):
            lats_l, lats_v, linear_res, vector_res = [], [], None, None
            for i in range(args.runs):
                dt, nodes = await run_search(drv_linear, emb)
                lats_l.append(dt * 1000)
                if i == 0:
                    linear_res = nodes
                dt, nodes = await run_search(drv_vector, emb)
                lats_v.append(dt * 1000)
                if i == 0:
                    vector_res = nodes
            l_uuids = [n[0] for n in linear_res]
            v_uuids = [n[0] for n in vector_res]
            max_rank_delta = 0
            v_set = {u: i for i, u in enumerate(v_uuids)}
            for i, u in enumerate(l_uuids):
                max_rank_delta = (
                    99  # missing from vector result
                    if u not in v_set
                    else max(max_rank_delta, abs(i - v_set[u]))
                )
            equal = l_uuids == v_uuids
            if l_uuids == v_uuids:
                full_match += 1
            elif set(l_uuids) == set(v_uuids):
                partial += 1
            else:
                empty += 1
            ms_lin = statistics.median(lats_l)
            ms_vec = statistics.median(lats_v)
            print(f'{name[:44]:<44} {ms_lin:>9.1f} {ms_vec:>9.1f} {str(equal):>10} {max_rank_delta:>7}<=', flush=True)
            if not equal:
                print(f'    linear: {[u[:8] for u in l_uuids]}', flush=True)
                print(f'    vector: {[u[:8] for u in v_uuids]}', flush=True)
        print('\n=== summary ===', flush=True)
        print(f'FULL(top5 exact)={full_match} PARTIAL(same set)={partial} DIFF={empty} / {len(queries)}', flush=True)
        print(f'not-equal queries: {empty + partial}', flush=True)
    finally:
        await drv_linear.client.close()
        await drv_vector.client.close()
    return 0 if empty == 0 else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
