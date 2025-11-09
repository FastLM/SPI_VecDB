# SPI_VecDB

**Towards Hyper-Efficient RAG Systems in VecDBs: Distributed Parallel Multi-Resolution Vector Search**

SPI_VecDB integrates Semantic Pyramid Indexing (SPI) into a production-ready vector database to provide query-adaptive, multi-resolution retrieval for Retrieval-Augmented Generation (RAG) systems. The design follows the paper by Dong Liu (Yale University) and Yanxuan Yu (Columbia University).

---

## Why Semantic Pyramid Indexing?

- **Multi-resolution hierarchy**: embeddings are refined across pyramid levels while preserving semantics, enabling coarse-to-fine retrieval with theoretical recall guarantees.
- **Entropy-driven control**: a lightweight classifier predicts search depth per query and activates uncertainty fallbacks when entropy variance is high.
- **Distributed parallelism**: each level can be sharded across FAISS/Qdrant executors; adaptive pruning reduces fan-out cost.
- **Cross-modal support**: the pyramid unifies text, image, audio and structured embeddings.

Empirically, SPI_VecDB achieves up to **5.7× lower latency**, **1.8× less memory**, and **+2.5 F1** increase compared to strong dense and hybrid baselines.

---

## Architecture Snapshot

```
Clients / RAG Pipelines
        │
        ▼
Proxy + SPI Controller
  • Parse DSL & filters
  • Estimate entropy / uncertainty
  • Select pyramid level
        │
        ▼
Semantic Pyramid Engines
  L1 (coarse) → … → Ln (fine)
  • FAISS / HNSW / IVF-PQ
  • Distributed shards + merge
  • Optional rerank / requery
        │
        ▼
Storage & Streaming Layer
  • Object store, WAL, CDC
  • INT8/PQ compression caches
```

### Key Files

- `internal/proxy/semantic_pyramid.go` – controller logic, entropy analytics, plan rewriting.
- `internal/proxy/task_search.go` – wiring SPI into the search pipeline.
- `pkg/metrics/proxy_metrics.go` – Prometheus counter `search_spi_level_total` for level telemetry.
- `internal/proxy/semantic_pyramid_test.go` – unit coverage for config parsing and runtime behaviour.

---

## Quickstart

```bash
git clone https://github.com/FastLM/SPI_VecDB.git
cd SPI_VecDB
./scripts/install_deps.sh
make
```

Run the standalone profile:

```bash
docker compose -f deployments/docker/docker-compose.yml up -d
docker compose -f deployments/docker/docker-compose.yml logs -f proxy
```

Send a SPI search request:

```json
{
  "anns_field": "embedding",
  "topk": 64,
  "metric_type": "COSINE",
  "search_params": {
    "metric_type": "COSINE",
    "nprobe": 16,
    "spi_config": {
      "levels": [
        {"name": "coarse",   "limit": 32,  "search_params": {"nprobe": 4,  "search_k": 512}},
        {"name": "balanced", "limit": 64,  "search_params": {"nprobe": 12, "search_k": 2048}},
        {"name": "fine",     "limit": 128, "search_params": {"nprobe": 32, "search_k": 8192}}
      ],
      "controller": {"type": "entropy", "thresholds": [0.25, 0.55], "uncertainty_epsilon": 0.08}
    }
  }
}
```

Monitor `search_spi_level_total{spi_level="coarse"}` via Prometheus at `http://localhost:9091/metrics`.

---

## Configuration Primer

| Setting | Description |
|---------|-------------|
| `spi_config.levels[].limit` | Maximum post-refinement topK for the level. |
| `spi_config.levels[].search_params` | Overrides merged into ANN params when the level is chosen. |
| `controller.thresholds` | Entropy cut points (length = levels − 1). |
| `controller.uncertainty_epsilon` | Forces an extra level probe when entropy stddev is high. |
| `spi_level` | Optional manual override (0-indexed). |

Additional details are documented inline in `internal/proxy/semantic_pyramid.go`.

---

## Observability

Prometheus metrics exposed by SPI_VecDB:

- `search_spi_level_total{spi_level}` – per-level query counts.
- `proxy_search_vectors_count` – vector comparisons processed.
- `sq_wait_result_latency`, `sq_reduce_result_latency` – pipeline latency segments.
- `scanned_remote_mb`, `scanned_total_mb` – IO/compute cost indicators.

Grafana dashboards under `deployments/monitor/` visualize entropy histograms, controller accuracy, and cost savings.

---

## Testing

```bash
make golang-test TEST_FLAGS='-run TestSemanticPyramid'

cd tests/python_client
python -m pytest --maxfail=1 --disable-warnings
```

> gofmt is required prior to submission. On macOS, install it via `brew install go` if it is not already available.

---

## Benchmark Highlights

| Dataset / Task           | Recall@10 | Latency (ms) | GPU Mem (GB) | Notes |
|-------------------------|-----------|--------------|--------------|-------|
| MS MARCO Passage        | **90.8**  | **22**       | **4.2**      | 5.7× faster than DenseX (76.9 / 122 ms / 7.3 GB). |
| Natural Questions       | **91.0**  | **24**       | **4.4**      | Matches Atlas recall with 5.7× latency reduction. |
| LAION-5B (text ↔ image) | 79.6      | 125          | 6.1          | Unified multi-modal pyramid. |
| Video-QA (clip)         | 76.8      | 138          | 6.3          | Handles long-form embeddings with adaptive depth. |

Experiments used a single NVIDIA RTX 4090 with FAISS backends configured for INT8 compression.

---

## Citation

```
@inproceedings{liu2025spi_vecdb,
  title={Towards Hyper-Efficient RAG Systems in VecDBs: Distributed Parallel Multi-Resolution Vector Search},
  author={Liu, Dong and Yu, Yanxuan},
  booktitle={Proceedings of the International Conference on Parallel and Distributed Systems (ICPADS 2025)},
  year={2025}
}
```

---

## License

SPI_VecDB is licensed under the Apache License 2.0. See `LICENSE` for details.

---

## Acknowledgements

Thanks to the SPI_VecDB community for extensive feedback on multi-resolution retrieval, distributed scheduling, and benchmarking methodology. Their real-world workloads shaped the adaptive controller, telemetry tooling, and evaluation suite.
