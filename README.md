# VECTRA — High-Performance In-Memory Vector Database & ML Indexing Engine

[![C++17](https://img.shields.io/badge/C++-17-00599C?style=flat-square&logo=c%2B%2B)](https://isocpp.org/)
[![Hardware SIMD](https://img.shields.io/badge/SIMD-AVX2%20%2B%20FMA-00f2fe?style=flat-square)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**VECTRA** is an ultra-fast in-memory vector database and approximate nearest neighbor (ANN) search engine built in modern **C++17** with hardware **AVX2 + FMA SIMD acceleration**, multi-algorithm spatial & graph indexing, unsupervised machine learning clustering, Reciprocal Rank Fusion (RRF) hybrid search, and Write-Ahead Log (WAL) persistence.

---

## 🗺️ Academic Roadmap & Milestone Coverage (Weeks 1 – 8)

| Week | Focus Area | Implemented Core Features | Machine Learning / Optimization Component |
| :--- | :--- | :--- | :--- |
| **Week 1 – 1.5** | **Core Engine & Spatial Trees** | Vector structures, Cosine / $\mathcal{L}_2$ / $\mathcal{L}_1$ metrics, Brute-Force $\mathcal{O}(N \cdot d)$, KD-Tree $\mathcal{O}(\log N)$, REST API & Web UI. | Baseline Mathematical Foundations & Spatial Partitioning |
| **Week 2** | **Graph Indexing (HNSW)** | Multi-layer graph index with probabilistic layer decay ($m_L = 1/\ln M$) and greedy entry point descent. | Graph-based Approximate Nearest Neighbors (ANN) |
| **Week 3** | **Unsupervised ML Clustering** | **K-Means Clustering & Inverted File Index (IVF)**: Lloyd's algorithm in C++ to cluster vectors into Voronoi cells, pruning &gt;85% of search space with configurable `nprobe`. | **ML 1**: K-Means Clustering & Centroid Assignment |
| **Week 4** | **ML Dimensionality Reduction** | **Principal Component Analysis (PCA)**: Covariance matrix computation ($\Sigma = \frac{1}{N} X^T X$) and Power Iteration with Hotelling Deflation for principal eigenvectors + **AVX2 SIMD** float projection. | **ML 2**: PCA Covariance & Eigenvector Projection |
| **Week 5** | **Hybrid Search Intelligence** | **Sparse ML (BM25) + Dense Vectors**: Combining statistical keyword weights with semantic vector embeddings using Reciprocal Rank Fusion (RRF). | **ML 3**: Inverted BM25 Index & Reciprocal Rank Fusion |
| **Week 6** | **Compression & Storage** | **Scalar Quantization (SQ8)**: FP32 $\to$ INT8 compression (75% RAM savings) + Write-Ahead Logging (WAL) binary disk persistence (`vectra.wal` & `vectra.vdb`). | Vector Quantization & ACID Durability Engine |
| **Week 7** | **Document Processing Pipeline** | Overlapping sliding-window chunker with configurable token overlap and local semantic embedding pipeline. | Information Extraction & Text Chunking |
| **Week 8** | **Local RAG & SIFT10K Profiler** | End-to-end Local **Retrieval-Augmented Generation (RAG)**, SIFT10K 128D Recall vs. QPS profiling arena across all 4 indexing algorithms. | Full Local RAG AI Pipeline & ANN Benchmark Arena |

---

## 🚀 Quick Launch

### 1. Instant 1-Click Launch:
Double-click `LAUNCH_UI.bat` to launch the server and open the live dashboard in your default browser at `http://localhost:8080`.

### 2. Manual Compile & Run (C++17 with AVX2/FMA):
```bash
# Compile with 64-bit MinGW GCC
g++ -std=c++17 -O3 -mavx2 -mfma main.cpp -o db.exe -lws2_32

# Run Server
./db.exe
```

Access the interactive dashboard at: **`http://localhost:8080`**

---

## 🌐 REST API Reference

### 🔍 Vector Search & Filtered Traversal
- `GET /search?v=...&k=5&algo=hnsw&metric=cosine&category=cs&nprobe=2`
  * Executes single-stage filtered vector search using HNSW, IVF, KD-Tree, or Brute-Force.
- `POST /insert`
  * Inserts a new vector item with metadata, category, and embedding.
- `DELETE /delete/{id}`
  * Deletes a vector item and logs the mutation to WAL.

### 🎯 Unsupervised Clustering (K-Means IVF)
- `POST /ivf/train` `{ "k": 4 }`
  * Clusters active vectors into $K$ Voronoi cells using Lloyd's algorithm.
- `GET /ivf/info`
  * Returns Voronoi cell centroids, member vector IDs, and within-cluster sum of squares (WCSS inertia).

### 🌌 Principal Component Analysis (PCA)
- `POST /pca/fit` `{ "components": 2 }`
  * Computes covariance matrix, extracts top principal eigenvectors via Power Iteration, and returns 2D/3D projections.
- `POST /pca/transform` `{ "vector": [0.8, 0.7, ...] }`
  * Projects high-dimensional query vector into the learned 2D latent space.

### 🔀 Hybrid Search (BM25 + Dense RRF)
- `POST /hybrid/search` `{ "query": "graph traversal", "k": 5, "alpha": 0.5 }`
  * Executes dual sparse BM25 and dense HNSW retrieval fused via Reciprocal Rank Fusion:
    $$\text{RRF}(d) = \alpha \cdot \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + (1 - \alpha) \cdot \frac{1}{60 + \text{rank}_{\text{BM25}}(d)}$$

### ⚡ Benchmark & Hardware Acceleration
- `GET /benchmark?v=...&k=5`
  * Benchmarks Brute-Force vs KD-Tree vs IVF-Flat vs HNSW head-to-head.
- `POST /benchmark/simd` `{ "iterations": 25000, "dims": 768 }`
  * Compares pure scalar loop vs 8-wide AVX2 hardware SIMD vectorization.
- `POST /benchmark/sift` `{ "numVectors": 500, "dims": 128, "numQueries": 20, "k": 10 }`
  * Runs 128D SIFT10K multi-algorithm Recall@K vs. QPS benchmark suite.

### 💾 Storage, WAL & Quantization
- `POST /storage/snapshot`
  * Checkpoints database memory to atomic binary `.vdb` snapshot file with magic header `VECTRA02` and truncates the WAL.
- `POST /storage/restore`
  * Restores database state from `.vdb` snapshot file.
- `GET /sq8/stats`
  * Returns Scalar Quantization memory savings and Mean Squared Error (MSE).

### 🤖 Local RAG & Document Pipeline
- `POST /doc/insert` `{ "title": "...", "text": "..." }`
  * Chunks text with sliding window and embeds into vector index.
- `POST /doc/ask` `{ "question": "...", "k": 3 }`
  * Retrieves relevant context chunks and synthesizes grounded answers.

---

## 🎨 Interactive Dashboard Tabs

1. **🌌 Semantic Space (PCA 2D Projection)**: Real-time interactive vector space visualizer with PCA projection, quadrant analysis, and single-stage category filtering.
2. **🧬 HNSW Graph Hierarchy**: Multi-layer skip-list graph visualizer ($L_2 \to L_1 \to L_0$) with animated query beam routing.
3. **🎯 K-Means & IVF Voronoi Cells**: Voronoi partition viewer, centroid inspector, and $nprobe$ prune slider.
4. **🔀 Hybrid Search Studio**: Dual sparse BM25 + dense vector query runner with dynamic $\alpha$ slider and rank breakdown waterfall.
5. **⚡ Benchmark Arena**: Multi-algorithm latency showdown, SIFT10K 128D Recall vs. QPS profiler, and AVX2 hardware SIMD tester.
6. **💾 Storage Engine & WAL**: Append-only transaction log inspector, `.vdb` binary snapshot manager, and SQ8 compression meter.
7. **🤖 Local RAG Studio**: Document ingestion with sliding-window chunker, context retrieval citations, and QA synthesizer.
8. **📐 Mathematical Inspector**: Formal LaTeX formulations for all 8 roadmap milestones.

---

## 📜 Repository
GitHub: [https://github.com/singhtheking060605/vectra.git](https://github.com/singhtheking060605/vectra.git)
