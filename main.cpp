#include "httplib.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#define VECTRA_AVX2_SUPPORTED 1
#else
#define VECTRA_AVX2_SUPPORTED 0
#endif

static const int DIMS = 16;   // demo vectors
// Doc embeddings dimension is determined at runtime from Ollama's model output

// Global runtime switch for SIMD acceleration (default: enabled if AVX2 is present)
static bool g_simd_enabled = (VECTRA_AVX2_SUPPORTED == 1);

// =====================================================================
//  DATA TYPES
// =====================================================================

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// =====================================================================
//  DISTANCE METRICS & HARDWARE AVX2 SIMD ACCELERATION
// =====================================================================

// Pure Scalar Distance Implementations (Baseline)
float euclidean_scalar(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

float cosine_scalar(const float* a, const float* b, size_t n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

float manhattan_scalar(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        s += std::abs(a[i] - b[i]);
    }
    return s;
}

// 8-Wide AVX2 + FMA SIMD Vectorized Distance Implementations
#if defined(__AVX2__)
float euclidean_avx2(const float* a, const float* b, size_t n) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
#if defined(__FMA__)
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
#else
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(diff, diff));
#endif
    }
    alignas(32) float buf[8];
    _mm256_storeu_ps(buf, sum256);
    float s = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    for (; i < n; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

float cosine_avx2(const float* a, const float* b, size_t n) {
    size_t i = 0;
    __m256 dot256 = _mm256_setzero_ps();
    __m256 na256  = _mm256_setzero_ps();
    __m256 nb256  = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        dot256 = _mm256_fmadd_ps(va, vb, dot256);
        na256  = _mm256_fmadd_ps(va, va, na256);
        nb256  = _mm256_fmadd_ps(vb, vb, nb256);
#else
        dot256 = _mm256_add_ps(dot256, _mm256_mul_ps(va, vb));
        na256  = _mm256_add_ps(na256,  _mm256_mul_ps(va, va));
        nb256  = _mm256_add_ps(nb256,  _mm256_mul_ps(vb, vb));
#endif
    }
    alignas(32) float bDot[8], bNa[8], bNb[8];
    _mm256_storeu_ps(bDot, dot256);
    _mm256_storeu_ps(bNa,  na256);
    _mm256_storeu_ps(bNb,  nb256);
    float dot = bDot[0]+bDot[1]+bDot[2]+bDot[3]+bDot[4]+bDot[5]+bDot[6]+bDot[7];
    float na  = bNa[0]+bNa[1]+bNa[2]+bNa[3]+bNa[4]+bNa[5]+bNa[6]+bNa[7];
    float nb  = bNb[0]+bNb[1]+bNb[2]+bNb[3]+bNb[4]+bNb[5]+bNb[6]+bNb[7];
    for (; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

float manhattan_avx2(const float* a, const float* b, size_t n) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 abs_diff = _mm256_and_ps(diff, sign_mask);
        sum256 = _mm256_add_ps(sum256, abs_diff);
    }
    alignas(32) float buf[8];
    _mm256_storeu_ps(buf, sum256);
    float s = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    for (; i < n; i++) s += std::abs(a[i] - b[i]);
    return s;
}
#endif

// Master Distance Dispatchers (Dynamic SIMD routing with scalar fallback)
float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
#if defined(__AVX2__)
    if (g_simd_enabled) return euclidean_avx2(a.data(), b.data(), n);
#endif
    return euclidean_scalar(a.data(), b.data(), n);
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
#if defined(__AVX2__)
    if (g_simd_enabled) return cosine_avx2(a.data(), b.data(), n);
#endif
    return cosine_scalar(a.data(), b.data(), n);
}

float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
#if defined(__AVX2__)
    if (g_simd_enabled) return manhattan_avx2(a.data(), b.data(), n);
#endif
    return manhattan_scalar(a.data(), b.data(), n);
}

DistFn getDistFn(const std::string& m) {
    if (m == "cosine")    return cosine;
    if (m == "manhattan") return manhattan;
    return euclidean;
}

// =====================================================================
//  SCALAR QUANTIZATION (SQ8) ENGINE
// =====================================================================

struct SQ8Vector {
    int id;
    float min_val;
    float diff;
    std::vector<uint8_t> qdata;

    static SQ8Vector quantize(int id, const std::vector<float>& vec) {
        SQ8Vector sq;
        sq.id = id;
        if (vec.empty()) { sq.min_val = 0; sq.diff = 1.0f; return sq; }
        float vmin = vec[0], vmax = vec[0];
        for (float v : vec) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        sq.min_val = vmin;
        sq.diff = (vmax - vmin < 1e-7f) ? 1.0f : (vmax - vmin);
        sq.qdata.resize(vec.size());
        for (size_t i = 0; i < vec.size(); i++) {
            float norm = (vec[i] - sq.min_val) / sq.diff;
            sq.qdata[i] = static_cast<uint8_t>(std::clamp(std::round(norm * 255.0f), 0.0f, 255.0f));
        }
        return sq;
    }

    std::vector<float> dequantize() const {
        std::vector<float> res(qdata.size());
        float inv = diff / 255.0f;
        for (size_t i = 0; i < qdata.size(); i++) {
            res[i] = min_val + static_cast<float>(qdata[i]) * inv;
        }
        return res;
    }

    // Fast Asymmetric Cosine Distance Computation (ADC)
    float asymmetricCosine(const std::vector<float>& q) const {
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        float inv = diff / 255.0f;
        size_t n = std::min(q.size(), qdata.size());
        for (size_t i = 0; i < n; i++) {
            float deq = min_val + static_cast<float>(qdata[i]) * inv;
            dot += q[i] * deq;
            na  += q[i] * q[i];
            nb  += deq * deq;
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
    }

    // Fast Asymmetric L2 Distance Computation (ADC)
    float asymmetricL2(const std::vector<float>& q) const {
        float s = 0.0f;
        float inv = diff / 255.0f;
        size_t n = std::min(q.size(), qdata.size());
        for (size_t i = 0; i < n; i++) {
            float deq = min_val + static_cast<float>(qdata[i]) * inv;
            float d = q[i] - deq;
            s += d * d;
        }
        return std::sqrt(s);
    }
};

struct QuantizationStats {
    size_t count = 0;
    size_t dims = 0;
    size_t fp32Bytes = 0;
    size_t sq8Bytes = 0;
    float compressionRatio = 0.0f;
    float memorySavedPercent = 0.0f;
    float meanSquaredError = 0.0f;
};

// =====================================================================
//  BRUTE FORCE
// =====================================================================

class BruteForce {
public:
    std::vector<VectorItem> items;

    void insert(const VectorItem& v) { items.push_back(v); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::vector<std::pair<float,int>> r;
        r.reserve(items.size());
        for (auto& v : items) r.push_back({dist(q, v.emb), v.id});
        std::sort(r.begin(), r.end());
        if ((int)r.size() > k) r.resize(k);
        return r;
    }

    std::vector<std::pair<float,int>> knnFiltered(
        const std::vector<float>& q, int k, DistFn dist,
        const std::function<bool(const VectorItem&)>& predicate)
    {
        std::vector<std::pair<float,int>> r;
        r.reserve(items.size());
        for (auto& v : items) {
            if (!predicate || predicate(v)) {
                r.push_back({dist(q, v.emb), v.id});
            }
        }
        std::sort(r.begin(), r.end());
        if ((int)r.size() > k) r.resize(k);
        return r;
    }

    void remove(int id) {
        items.erase(std::remove_if(items.begin(), items.end(),
            [id](const VectorItem& v){ return v.id == id; }), items.end());
    }
};

// =====================================================================
//  KD-TREE
// =====================================================================

struct KDNode {
    VectorItem item;
    KDNode* left  = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTree {
    KDNode* root = nullptr;
    int dims;

    void destroy(KDNode* n) {
        if (!n) return; destroy(n->left); destroy(n->right); delete n;
    }

    KDNode* ins(KDNode* n, const VectorItem& v, int d) {
        if (!n) return new KDNode(v);
        int ax = d % dims;
        if (v.emb[ax] < n->item.emb[ax]) n->left  = ins(n->left,  v, d+1);
        else                              n->right = ins(n->right, v, d+1);
        return n;
    }

    void knn(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
             std::priority_queue<std::pair<float,int>>& heap)
    {
        if (!n) return;
        float dn = dist(q, n->item.emb);
        if ((int)heap.size() < k || dn < heap.top().first) {
            heap.push({dn, n->item.id});
            if ((int)heap.size() > k) heap.pop();
        }
        int ax = d % dims;
        float diff = q[ax] - n->item.emb[ax];
        KDNode* closer  = diff < 0 ? n->left  : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        knn(closer, q, k, d+1, dist, heap);
        if ((int)heap.size() < k || std::abs(diff) < heap.top().first)
            knn(farther, q, k, d+1, dist, heap);
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { destroy(root); }

    void insert(const VectorItem& v) { root = ins(root, v, 0); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::priority_queue<std::pair<float,int>> heap;
        knn(root, q, k, 0, dist, heap);
        std::vector<std::pair<float,int>> r;
        while (!heap.empty()) { r.push_back(heap.top()); heap.pop(); }
        std::sort(r.begin(), r.end());
        return r;
    }

    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root); root = nullptr;
        for (auto& v : items) insert(v);
    }
};

// =====================================================================
//  HNSW — Hierarchical Navigable Small World (with Filtered Traversal)
// =====================================================================

class HNSW {
public:
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int    M, M0, ef_build;
    float  mL;
    int    topLayer = -1;
    int    entryPt  = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return (int)std::floor(-std::log(u(rng)) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if ((int)found.size() < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    // Hybrid Filtered Traversal: Traverses the graph spatially but admits only matching nodes into results
    std::vector<std::pair<float,int>> searchLayerFiltered(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist,
        const std::function<bool(const VectorItem&)>& predicate)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        if (!predicate || predicate(G[ep].item)) {
            found.push({d0, ep});
        }

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if (found.empty() || nd < found.top().first || (int)found.size() < ef) {
                    cands.push({nd, nid});
                    if (!predicate || predicate(G[nid].item)) {
                        found.push({nd, nid});
                        if ((int)found.size() > ef) found.pop();
                    }
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> selectNbrs(std::vector<std::pair<float,int>>& cands, int maxM) {
        std::vector<int> r;
        for (int i = 0; i < std::min((int)cands.size(), maxM); i++)
            r.push_back(cands[i].second);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2*m), ef_build(efBuild),
          mL(1.0f / std::log((float)m)), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G[id]   = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; lc--) {
            auto W   = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if ((int)G[nid].nbrs.size() <= lc) G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if ((int)conn.size() > maxM) {
                    std::vector<std::pair<float,int>> ds;
                    for (int c : conn) if (G.count(c))
                        ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn.clear();
                    for (int i = 0; i < maxM && i < (int)ds.size(); i++)
                        conn.push_back(ds[i].second);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    // Hybrid Filtered KNN Search
    std::vector<std::pair<float,int>> knnFiltered(
        const std::vector<float>& q, int k, int ef, DistFn dist,
        const std::function<bool(const VectorItem&)>& predicate)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        if (predicate && !predicate(G[ep].item)) {
            for (const auto& [nid, nd] : G) {
                if (predicate(nd.item)) {
                    ep = nid;
                    break;
                }
            }
        }
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayerFiltered(q, ep, std::max(ef, k), 0, dist, predicate);
        if (W.empty()) {
            for (const auto& [nid, nd] : G) {
                if (!predicate || predicate(nd.item)) {
                    W.push_back({dist(q, nd.item.emb), nid});
                }
            }
            std::sort(W.begin(), W.end());
        }
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    void remove(int id) {
        if (!G.count(id)) return;
        for (auto& [nid, nd] : G)
            for (auto& layer : nd.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
        if (entryPt == id) {
            entryPt = -1;
            for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }
        }
        G.erase(id);
    }

    void clear() {
        G.clear();
        topLayer = -1;
        entryPt  = -1;
    }

    // Binary serialization for snapshotting
    void serialize(std::ostream& out) const {
        int32_t tl = topLayer, ep = entryPt;
        int32_t m = M, m0 = M0, efb = ef_build;
        out.write(reinterpret_cast<const char*>(&tl), sizeof(tl));
        out.write(reinterpret_cast<const char*>(&ep), sizeof(ep));
        out.write(reinterpret_cast<const char*>(&m), sizeof(m));
        out.write(reinterpret_cast<const char*>(&m0), sizeof(m0));
        out.write(reinterpret_cast<const char*>(&efb), sizeof(efb));
        out.write(reinterpret_cast<const char*>(&mL), sizeof(mL));

        uint64_t nodeCount = G.size();
        out.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));
        for (const auto& [nid, nd] : G) {
            int32_t id = nid;
            int32_t maxLyr = nd.maxLyr;
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
            out.write(reinterpret_cast<const char*>(&maxLyr), sizeof(maxLyr));
            uint32_t layerCount = (uint32_t)nd.nbrs.size();
            out.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));
            for (uint32_t lc = 0; lc < layerCount; lc++) {
                uint32_t nbrCount = (uint32_t)nd.nbrs[lc].size();
                out.write(reinterpret_cast<const char*>(&nbrCount), sizeof(nbrCount));
                if (nbrCount > 0) {
                    out.write(reinterpret_cast<const char*>(nd.nbrs[lc].data()), nbrCount * sizeof(int));
                }
            }
        }
    }

    // Binary deserialization for restoring graph directly
    void deserialize(std::istream& in, const std::unordered_map<int, VectorItem>& items) {
        clear();
        int32_t tl = -1, ep = -1;
        int32_t m = 16, m0 = 32, efb = 200;
        in.read(reinterpret_cast<char*>(&tl), sizeof(tl));
        in.read(reinterpret_cast<char*>(&ep), sizeof(ep));
        in.read(reinterpret_cast<char*>(&m), sizeof(m));
        in.read(reinterpret_cast<char*>(&m0), sizeof(m0));
        in.read(reinterpret_cast<char*>(&efb), sizeof(efb));
        in.read(reinterpret_cast<char*>(&mL), sizeof(mL));
        topLayer = tl; entryPt = ep; M = m; M0 = m0; ef_build = efb;

        uint64_t nodeCount = 0;
        in.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
        for (uint64_t i = 0; i < nodeCount; i++) {
            int32_t id = 0, maxLyr = 0;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            in.read(reinterpret_cast<char*>(&maxLyr), sizeof(maxLyr));
            uint32_t layerCount = 0;
            in.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
            std::vector<std::vector<int>> nbrs(layerCount);
            for (uint32_t lc = 0; lc < layerCount; lc++) {
                uint32_t nbrCount = 0;
                in.read(reinterpret_cast<char*>(&nbrCount), sizeof(nbrCount));
                nbrs[lc].resize(nbrCount);
                if (nbrCount > 0) {
                    in.read(reinterpret_cast<char*>(nbrs[lc].data()), nbrCount * sizeof(int));
                }
            }
            auto it = items.find(id);
            if (it != items.end()) {
                G[id] = {it->second, maxLyr, nbrs};
            }
        }
    }

    struct GraphInfo {
        int topLayer, nodeCount;
        std::vector<int> nodesPerLayer, edgesPerLayer;
        struct NV { int id; std::string metadata, category; int maxLyr; };
        struct EV { int src, dst, lyr; };
        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo() {
        GraphInfo gi;
        gi.topLayer  = topLayer;
        gi.nodeCount = (int)G.size();
        int maxL = std::max(topLayer + 1, 1);
        gi.nodesPerLayer.assign(maxL, 0);
        gi.edgesPerLayer.assign(maxL, 0);
        for (auto& [id, nd] : G) {
            gi.nodes.push_back({id, nd.item.metadata, nd.item.category, nd.maxLyr});
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < (int)nd.nbrs.size())
                    for (int nid : nd.nbrs[lc])
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.push_back({id, nid, lc});
                        }
            }
        }
        return gi;
    }

    size_t size() const { return G.size(); }
};

// =====================================================================
//  WRITE-AHEAD LOGGING (WAL) ENGINE
// =====================================================================

class WALManager {
    std::string logPath;
    std::ofstream out;
    std::mutex mu;
    size_t entryCount = 0;

public:
    enum OpType : uint8_t {
        OP_INSERT = 1,
        OP_DELETE = 2
    };

    explicit WALManager(const std::string& path = "vectra.wal") : logPath(path) {
        // Open file in append binary mode
        out.open(logPath, std::ios::binary | std::ios::app);

        // Count existing records if file was already present
        std::ifstream in(logPath, std::ios::binary);
        if (in.is_open()) {
            while (in.peek() != EOF) {
                uint8_t op = 0;
                int64_t ts = 0;
                int32_t id = 0;
                if (!in.read(reinterpret_cast<char*>(&op), sizeof(op))) break;
                if (!in.read(reinterpret_cast<char*>(&ts), sizeof(ts))) break;
                if (!in.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
                if (op == OP_INSERT) {
                    uint32_t cLen = 0, mLen = 0, dLen = 0;
                    if (!in.read(reinterpret_cast<char*>(&cLen), sizeof(cLen))) break;
                    in.seekg(cLen, std::ios::cur);
                    if (!in.read(reinterpret_cast<char*>(&mLen), sizeof(mLen))) break;
                    in.seekg(mLen, std::ios::cur);
                    if (!in.read(reinterpret_cast<char*>(&dLen), sizeof(dLen))) break;
                    in.seekg(dLen * sizeof(float), std::ios::cur);
                }
                entryCount++;
            }
        }
    }

    ~WALManager() {
        if (out.is_open()) out.close();
    }

    void logInsert(const VectorItem& item) {
        std::lock_guard<std::mutex> lk(mu);
        if (!out.is_open()) return;
        uint8_t op = OP_INSERT;
        int64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        int32_t id = item.id;
        uint32_t cLen = (uint32_t)item.category.size();
        uint32_t mLen = (uint32_t)item.metadata.size();
        uint32_t dLen = (uint32_t)item.emb.size();

        out.write(reinterpret_cast<const char*>(&op), sizeof(op));
        out.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        out.write(reinterpret_cast<const char*>(&cLen), sizeof(cLen));
        if (cLen > 0) out.write(item.category.data(), cLen);
        out.write(reinterpret_cast<const char*>(&mLen), sizeof(mLen));
        if (mLen > 0) out.write(item.metadata.data(), mLen);
        out.write(reinterpret_cast<const char*>(&dLen), sizeof(dLen));
        if (dLen > 0) out.write(reinterpret_cast<const char*>(item.emb.data()), dLen * sizeof(float));
        out.flush();
        entryCount++;
    }

    void logDelete(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!out.is_open()) return;
        uint8_t op = OP_DELETE;
        int64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        int32_t vid = id;
        out.write(reinterpret_cast<const char*>(&op), sizeof(op));
        out.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
        out.write(reinterpret_cast<const char*>(&vid), sizeof(vid));
        out.flush();
        entryCount++;
    }

    void truncate() {
        std::lock_guard<std::mutex> lk(mu);
        if (out.is_open()) out.close();
        out.open(logPath, std::ios::binary | std::ios::trunc);
        entryCount = 0;
    }

    size_t getCount() const { return entryCount; }
    std::string getPath() const { return logPath; }
};

// =====================================================================
//  VECTOR DATABASE  (Demo 16D & Multi-Algorithm Core)
// =====================================================================

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree     kdt;
    HNSW       hnsw;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        return v.id;
    }

    // Direct insertion with predetermined ID (used during WAL replay and snapshot loading)
    void insertWithId(int id, const std::string& meta, const std::string& cat,
                      const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (store.count(id)) {
            bf.remove(id);
            hnsw.remove(id);
        }
        VectorItem v{id, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        if (id >= nextId) nextId = id + 1;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); bf.remove(id); hnsw.remove(id);
        std::vector<VectorItem> rem;
        for (auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
        return true;
    }

    struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
    struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };

    struct FilterOptions {
        std::string category;
        std::string keyword;
    };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo,
                     const FilterOptions& fOpt = {})
    {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn = getDistFn(metric);
        auto t0  = std::chrono::high_resolution_clock::now();

        auto predicate = [&](const VectorItem& item) -> bool {
            if (!fOpt.category.empty() && fOpt.category != "all") {
                if (item.category != fOpt.category) return false;
            }
            if (!fOpt.keyword.empty()) {
                std::string kw = fOpt.keyword;
                std::transform(kw.begin(), kw.end(), kw.begin(), ::tolower);
                std::string metaLower = item.metadata;
                std::transform(metaLower.begin(), metaLower.end(), metaLower.begin(), ::tolower);
                if (metaLower.find(kw) == std::string::npos) return false;
            }
            return true;
        };

        bool hasFilter = (!fOpt.category.empty() && fOpt.category != "all") || !fOpt.keyword.empty();

        std::vector<std::pair<float,int>> raw;
        if (hasFilter) {
            if (algo == "bruteforce") {
                raw = bf.knnFiltered(q, k, dfn, predicate);
            } else if (algo == "kdtree") {
                // KD-Tree filtered candidate fallback
                auto full = kdt.knn(q, std::min((int)store.size(), k * 10), dfn);
                for (auto& p : full) {
                    if (store.count(p.second) && predicate(store[p.second])) {
                        raw.push_back(p);
                        if ((int)raw.size() >= k) break;
                    }
                }
            } else {
                // Single-stage HNSW filtered beam search
                raw = hnsw.knnFiltered(q, k, 50, dfn, predicate);
            }
        } else {
            if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
            else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
            else                           raw = hnsw.knn(q, k, 50, dfn);
        }

        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchOut out; out.us = us; out.algo = algo; out.metric = metric;
        for (auto& [d, id] : raw)
            if (store.count(id))
                out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
        return out;
    }

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn  = getDistFn(metric);
        auto time = [&](auto fn) -> long long {
            auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&]{ bf.knn(q, k, dfn); }),
            time([&]{ kdt.knn(q, k, dfn); }),
            time([&]{ hnsw.knn(q, k, 50, dfn); }),
            (int)store.size()
        };
    }

    // Scalar Quantization Metrics Engine
    QuantizationStats getSQ8Stats() {
        std::lock_guard<std::mutex> lk(mu);
        QuantizationStats qs;
        qs.count = store.size();
        qs.dims = dims;
        qs.fp32Bytes = store.size() * dims * sizeof(float);
        // SQ8 stores: 1 byte per dimension + 2 floats (min, diff) + int id
        qs.sq8Bytes = store.size() * (dims * sizeof(uint8_t) + 2 * sizeof(float) + sizeof(int));
        if (qs.fp32Bytes > 0) {
            qs.compressionRatio = (float)qs.fp32Bytes / (float)qs.sq8Bytes;
            qs.memorySavedPercent = (1.0f - (float)qs.sq8Bytes / (float)qs.fp32Bytes) * 100.0f;
        }
        float totalMse = 0.0f;
        for (const auto& [id, item] : store) {
            auto sq = SQ8Vector::quantize(id, item.emb);
            auto deq = sq.dequantize();
            float mse = 0.0f;
            for (size_t i = 0; i < item.emb.size(); i++) {
                float diff = item.emb[i] - deq[i];
                mse += diff * diff;
            }
            totalMse += mse / (float)item.emb.size();
        }
        if (!store.empty()) qs.meanSquaredError = totalMse / (float)store.size();
        return qs;
    }

    std::vector<VectorItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<VectorItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    HNSW::GraphInfo hnswInfo() {
        std::lock_guard<std::mutex> lk(mu);
        return hnsw.getInfo();
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }

    const std::unordered_map<int, VectorItem>& getStore() const { return store; }
    const HNSW& getHNSW() const { return hnsw; }

    void restoreState(const std::unordered_map<int, VectorItem>& newStore, int nxtId) {
        std::lock_guard<std::mutex> lk(mu);
        store = newStore;
        nextId = nxtId;
        bf.items.clear();
        for (auto& [id, v] : store) bf.insert(v);
        std::vector<VectorItem> rem;
        for (auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
    }

    void deserializeHNSW(std::istream& in) {
        std::lock_guard<std::mutex> lk(mu);
        hnsw.deserialize(in, store);
    }
};

// =====================================================================
//  BINARY STORAGE ENGINE (.vdb binary format with magic header)
// =====================================================================

class StorageEngine {
public:
    static bool saveSnapshot(const std::string& filepath,
                             VectorDB& db, WALManager& wal)
    {
        std::string tmpPath = filepath + ".tmp";
        std::ofstream out(tmpPath, std::ios::binary);
        if (!out.is_open()) return false;

        char magic[8] = {'V', 'E', 'C', 'T', 'R', 'A', '0', '2'};
        out.write(magic, 8);

        uint32_t d = db.dims;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));

        const auto& store = db.getStore();
        uint64_t count = store.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (const auto& [id, v] : store) {
            int32_t vid = v.id;
            out.write(reinterpret_cast<const char*>(&vid), sizeof(vid));

            uint32_t catLen = (uint32_t)v.category.size();
            out.write(reinterpret_cast<const char*>(&catLen), sizeof(catLen));
            if (catLen > 0) out.write(v.category.data(), catLen);

            uint32_t metaLen = (uint32_t)v.metadata.size();
            out.write(reinterpret_cast<const char*>(&metaLen), sizeof(metaLen));
            if (metaLen > 0) out.write(v.metadata.data(), metaLen);

            if (!v.emb.empty()) {
                out.write(reinterpret_cast<const char*>(v.emb.data()), v.emb.size() * sizeof(float));
            }
        }

        // Serialize HNSW graph topology
        db.getHNSW().serialize(out);
        out.close();

        // Atomic replace
        std::remove(filepath.c_str());
        if (std::rename(tmpPath.c_str(), filepath.c_str()) != 0) {
            return false;
        }

        // Truncate WAL because all changes are safely snapshotted
        wal.truncate();
        return true;
    }

    static bool loadSnapshot(const std::string& filepath, VectorDB& db) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) return false;

        char magic[8];
        if (!in.read(magic, 8)) return false;
        if (std::memcmp(magic, "VECTRA02", 8) != 0) return false;

        uint32_t d = 0;
        in.read(reinterpret_cast<char*>(&d), sizeof(d));

        uint64_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));

        std::unordered_map<int, VectorItem> newStore;
        int maxId = 0;

        for (uint64_t i = 0; i < count; i++) {
            VectorItem v;
            int32_t vid = 0;
            in.read(reinterpret_cast<char*>(&vid), sizeof(vid));
            v.id = vid;
            if (v.id > maxId) maxId = v.id;

            uint32_t catLen = 0;
            in.read(reinterpret_cast<char*>(&catLen), sizeof(catLen));
            v.category.resize(catLen);
            if (catLen > 0) in.read(&v.category[0], catLen);

            uint32_t metaLen = 0;
            in.read(reinterpret_cast<char*>(&metaLen), sizeof(metaLen));
            v.metadata.resize(metaLen);
            if (metaLen > 0) in.read(&v.metadata[0], metaLen);

            v.emb.resize(d);
            if (d > 0) in.read(reinterpret_cast<char*>(v.emb.data()), d * sizeof(float));

            newStore[v.id] = v;
        }

        db.restoreState(newStore, maxId + 1);
        db.deserializeHNSW(in);
        return true;
    }

    static int replayWAL(const std::string& walPath, VectorDB& db, DistFn dist) {
        std::ifstream in(walPath, std::ios::binary);
        if (!in.is_open()) return 0;

        int replayed = 0;
        while (in.peek() != EOF) {
            uint8_t op = 0;
            int64_t ts = 0;
            int32_t id = 0;
            if (!in.read(reinterpret_cast<char*>(&op), sizeof(op))) break;
            if (!in.read(reinterpret_cast<char*>(&ts), sizeof(ts))) break;
            if (!in.read(reinterpret_cast<char*>(&id), sizeof(id))) break;

            if (op == WALManager::OP_INSERT) {
                uint32_t cLen = 0, mLen = 0, dLen = 0;
                if (!in.read(reinterpret_cast<char*>(&cLen), sizeof(cLen))) break;
                std::string cat(cLen, '\0');
                if (cLen > 0) in.read(&cat[0], cLen);

                if (!in.read(reinterpret_cast<char*>(&mLen), sizeof(mLen))) break;
                std::string meta(mLen, '\0');
                if (mLen > 0) in.read(&meta[0], mLen);

                if (!in.read(reinterpret_cast<char*>(&dLen), sizeof(dLen))) break;
                std::vector<float> emb(dLen);
                if (dLen > 0) in.read(reinterpret_cast<char*>(emb.data()), dLen * sizeof(float));

                db.insertWithId(id, meta, cat, emb, dist);
                replayed++;
            } else if (op == WALManager::OP_DELETE) {
                db.remove(id);
                replayed++;
            }
        }
        return replayed;
    }
};

// =====================================================================
//  JSON HELPERS
// =====================================================================

std::string jS(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else                o += c;
    }
    return o + '"';
}

std::string jVec(const std::vector<float>& v) {
    std::ostringstream ss; ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    return ss.str() + ']';
}

std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s); std::string t;
    while (std::getline(ss, t, ','))
        try { v.push_back(std::stof(t)); } catch (...) {}
    return v;
}

std::string extractStr(const std::string& body, const std::string& key) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return "";
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    if (p >= body.size() || body[p] != '"') return "";
    p++;
    std::string result;
    while (p < body.size()) {
        if (body[p] == '"') break;
        if (body[p] == '\\' && p + 1 < body.size()) {
            p++;
            switch (body[p]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += body[p]; break;
            }
        } else {
            result += body[p];
        }
        p++;
    }
    return result;
}

int extractInt(const std::string& body, const std::string& key, int def = 0) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return def;
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    try { return std::stoi(body.substr(p)); } catch (...) { return def; }
}

bool parseBody(const std::string& b, std::string& meta,
               std::string& cat, std::vector<float>& emb)
{
    meta = extractStr(b, "metadata");
    cat  = extractStr(b, "category");
    auto extractArr = [&](const std::string& key) -> std::vector<float> {
        size_t p = b.find('"' + key + '"');
        if (p == std::string::npos) return {};
        p = b.find('[', p);
        if (p == std::string::npos) return {};
        size_t e = b.find(']', p);
        if (e == std::string::npos) return {};
        return parseVec(b.substr(p + 1, e - p - 1));
    };
    emb = extractArr("embedding");
    return !meta.empty() && !emb.empty();
}

void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

// =====================================================================
//  TEXT CHUNKER
// =====================================================================

std::vector<std::string> chunkText(const std::string& text,
                                   int chunkWords = 250, int overlapWords = 30)
{
    std::istringstream ss(text);
    std::vector<std::string> words;
    std::string w;
    while (ss >> w) words.push_back(w);

    if (words.empty()) return {};
    if ((int)words.size() <= chunkWords) return {text};

    std::vector<std::string> chunks;
    int step = chunkWords - overlapWords;
    for (int i = 0; i < (int)words.size(); i += step) {
        int end = std::min(i + chunkWords, (int)words.size());
        std::string chunk;
        for (int j = i; j < end; j++) { if (j > i) chunk += ' '; chunk += words[j]; }
        chunks.push_back(chunk);
        if (end == (int)words.size()) break;
    }
    return chunks;
}

// =====================================================================
//  BUILT-IN SEMANTIC EMBEDDING ENGINE (Zero-Downtime Local Fallback)
// =====================================================================

std::vector<float> localSemanticEmbed(const std::string& text, int targetDims = 64) {
    if (targetDims <= 0) targetDims = 64;
    std::vector<float> v(targetDims, 0.02f);

    std::string lower;
    for (char c : text) {
        if (std::isalnum((unsigned char)c)) lower += (char)std::tolower((unsigned char)c);
        else lower += ' ';
    }
    std::istringstream ss(lower);
    std::string word;
    int wordCount = 0;

    static const std::unordered_map<std::string, int> domainClusters = {
        {"algorithm", 0}, {"data", 0}, {"tree", 0}, {"graph", 0}, {"array", 0}, {"hash", 0},
        {"database", 0}, {"vector", 0}, {"code", 0}, {"program", 0}, {"search", 0}, {"test", 0},
        {"calculus", 1}, {"matrix", 1}, {"probability", 1}, {"math", 1}, {"algebra", 1}, {"equation", 1},
        {"food", 2}, {"pizza", 2}, {"sushi", 2}, {"ramen", 2}, {"recipe", 2}, {"cook", 2}, {"coffee", 2},
        {"sport", 3}, {"basketball", 3}, {"football", 3}, {"tennis", 3}, {"game", 3}, {"match", 3}
    };

    while (ss >> word) {
        if (word.empty()) continue;
        wordCount++;

        uint32_t h = 2166136261u;
        for (char c : word) {
            h = (h ^ (uint8_t)c) * 16777619u;
        }
        int idx = (int)(h % targetDims);
        v[idx] += 1.4f;

        if (word.size() >= 3) {
            for (size_t i = 0; i + 3 <= word.size(); i++) {
                uint32_t gh = (uint32_t)(uint8_t)word[i] * 961 + (uint32_t)(uint8_t)word[i+1] * 31 + (uint32_t)(uint8_t)word[i+2];
                int gidx = (int)(gh % targetDims);
                v[gidx] += 0.5f;
            }
        }

        for (auto& [kw, clusterId] : domainClusters) {
            if (word.find(kw) != std::string::npos || kw.find(word) != std::string::npos) {
                int clusterOffset = (clusterId * (targetDims / 4)) % targetDims;
                for (int o = 0; o < std::max(1, targetDims / 8); o++) {
                    v[(clusterOffset + o) % targetDims] += 1.0f;
                }
            }
        }
    }

    if (wordCount == 0) {
        for (int i = 0; i < targetDims; i++) v[i] = 1.0f / std::sqrt((float)targetDims);
        return v;
    }

    float normSq = 0.0f;
    for (float val : v) normSq += val * val;
    float norm = std::sqrt(normSq);
    if (norm < 1e-7f) norm = 1.0f;
    for (float& val : v) val /= norm;

    return v;
}

// =====================================================================
//  OLLAMA CLIENT — Local LLM & Embedding Integration
// =====================================================================

class OllamaClient {
    std::string host;
    int         port;

    std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            if      (c == '"')  o += "\\\"";
            else if (c == '\\') o += "\\\\";
            else if (c == '\n') o += "\\n";
            else if (c == '\r') o += "\\r";
            else if (c == '\t') o += "\\t";
            else                o += c;
        }
        return o;
    }

    std::vector<float> parseEmbedding(const std::string& body) {
        size_t p = body.find("\"embedding\"");
        if (p == std::string::npos) return {};
        p = body.find('[', p);
        if (p == std::string::npos) return {};
        size_t e = p + 1, depth = 1;
        while (e < body.size() && depth > 0) {
            if (body[e] == '[') depth++;
            else if (body[e] == ']') depth--;
            e++;
        }
        return parseVec(body.substr(p + 1, e - p - 2));
    }

    std::string parseResponse(const std::string& body) {
        return extractStr(body, "response");
    }

public:
    std::string embedModel = "nomic-embed-text";
    std::string genModel   = "llama3.2";

    OllamaClient(const std::string& h = "127.0.0.1", int p = 11434)
        : host(h), port(p) {}

    bool isAvailable() {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    std::vector<float> embed(const std::string& text) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(30, 0);
        std::string body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
        auto res = cli.Post("/api/embeddings", body, "application/json");
        if (!res || res->status != 200) return {};
        return parseEmbedding(res->body);
    }

    std::string generate(const std::string& prompt) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(180, 0);
        std::string body = "{\"model\":\"" + genModel + "\","
                           "\"prompt\":\"" + esc(prompt) + "\","
                           "\"stream\":false}";
        auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200)
            return "ERROR: Ollama unavailable. Run: ollama serve";
        return parseResponse(res->body);
    }
};

// =====================================================================
//  DOCUMENT DATABASE — 768D Vector Engine
// =====================================================================

struct DocItem {
    int         id;
    std::string title;
    std::string text;
    std::vector<float> emb;
};

class DocumentDB {
    std::unordered_map<int, DocItem> store;
    HNSW       hnsw;
    BruteForce bf;
    std::mutex mu;
    int nextId = 1;
    int dims   = 0;

public:
    DocumentDB() : hnsw(16, 200) {}

    int insert(const std::string& title, const std::string& text,
               const std::vector<float>& emb)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (dims == 0) dims = (int)emb.size();
        DocItem item{nextId++, title, text, emb};
        store[item.id] = item;
        VectorItem vi{item.id, title, "doc", emb};
        hnsw.insert(vi, cosine);
        bf.insert(vi);
        return item.id;
    }

    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q, int k, float max_dist = 2.0f, const std::string& keyword = "")
    {
        std::lock_guard<std::mutex> lk(mu);
        if (store.empty()) return {};

        std::vector<std::pair<float,int>> raw;
        if (!keyword.empty()) {
            std::string kw = keyword;
            std::transform(kw.begin(), kw.end(), kw.begin(), ::tolower);
            auto pred = [&](const VectorItem& vi) {
                if (!store.count(vi.id)) return false;
                std::string t = store[vi.id].title + " " + store[vi.id].text;
                std::transform(t.begin(), t.end(), t.begin(), ::tolower);
                return t.find(kw) != std::string::npos;
            };
            raw = hnsw.knnFiltered(q, k, 50, cosine, pred);
        } else {
            raw = (store.size() < 10)
                       ? bf.knn(q, k, cosine)
                       : hnsw.knn(q, k, 50, cosine);
        }

        std::vector<std::pair<float, DocItem>> out;
        for (auto& [d, id] : raw)
            if (store.count(id) && d <= max_dist) out.push_back({d, store[id]});
        return out;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); hnsw.remove(id); bf.remove(id);
        return true;
    }

    std::vector<DocItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<DocItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }

    int getDims() { return dims; }
};

// =====================================================================
//  DEMO DATASET (16D Semantic Clusters)
// =====================================================================

void loadDemo(VectorDB& db, WALManager& wal) {
    auto dist = getDistFn("cosine");
    struct DemoItem { std::string meta, cat; std::vector<float> emb; };
    std::vector<DemoItem> data = {
        {"Linked List: nodes connected by pointers", "cs",
            {0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}},
        {"Binary Search Tree: O(log n) search and insert", "cs",
            {0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}},
        {"Dynamic Programming: memoization overlapping subproblems", "cs",
            {0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}},
        {"Graph BFS and DFS: breadth and depth first traversal", "cs",
            {0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}},
        {"Hash Table: O(1) lookup with collision chaining", "cs",
            {0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}},
        {"Calculus: derivatives integrals and limits", "math",
            {0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}},
        {"Linear Algebra: matrices eigenvalues eigenvectors", "math",
            {0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}},
        {"Probability: distributions random variables Bayes theorem", "math",
            {0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}},
        {"Number Theory: primes modular arithmetic RSA cryptography", "math",
            {0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}},
        {"Combinatorics: permutations combinations generating functions", "math",
            {0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}},
        {"Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}},
        {"Sushi: vinegared rice raw fish and nori rolls", "food",
            {0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}},
        {"Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            {0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}},
        {"Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            {0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}},
        {"Croissant: laminated pastry with buttery flaky layers", "food",
            {0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}},
        {"Basketball: fast-paced shooting dribbling slam dunks", "sports",
            {0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}},
        {"Football: tackles touchdowns field goals and strategy", "sports",
            {0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}},
        {"Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}},
        {"Chess: openings endgames tactics strategic board game", "sports",
            {0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}},
        {"Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            {0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}}
    };

    for (const auto& item : data) {
        int id = db.insert(item.meta, item.cat, item.emb, dist);
        VectorItem vi{id, item.meta, item.cat, item.emb};
        wal.logInsert(vi);
    }
}

// =====================================================================
//  HTTP REST SERVER & SERVICE DISPATCHER
// =====================================================================

int main() {
    VectorDB   db(DIMS);
    WALManager wal("vectra.wal");
    DocumentDB docDB;
    OllamaClient ollama;

    // Phase 2 Crash Recovery: Load Snapshot (.vdb) if present, then replay WAL
    bool snapshotLoaded = false;
    std::ifstream vdbCheck("vectra.vdb", std::ios::binary);
    if (vdbCheck.is_open()) {
        vdbCheck.close();
        snapshotLoaded = StorageEngine::loadSnapshot("vectra.vdb", db);
    }

    if (!snapshotLoaded && db.size() == 0) {
        loadDemo(db, wal);
    }

    // Replay any uncommitted changes from WAL
    int replayed = StorageEngine::replayWAL("vectra.wal", db, getDistFn("cosine"));

    bool ollamaUp = ollama.isAvailable();
    std::cout << "========================================================\n";
    std::cout << "  VECTRA ENGINE (C++17 High-Performance Vector Database)\n";
    std::cout << "========================================================\n";
    std::cout << "Server Listening:   http://localhost:8080\n";
    std::cout << "AVX2 SIMD Support:  " << (VECTRA_AVX2_SUPPORTED ? "HARDWARE ACCELERATED (256-bit)" : "SCALAR FALLBACK") << "\n";
    std::cout << "SIMD Status:        " << (g_simd_enabled ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "Active Vectors:     " << db.size() << " | Dims: " << DIMS << "\n";
    std::cout << "Storage Snapshot:   " << (snapshotLoaded ? "LOADED (vectra.vdb)" : "INITIALIZED IN-MEMORY") << "\n";
    std::cout << "WAL Replayed Ops:   " << replayed << " | Log Count: " << wal.getCount() << "\n";
    std::cout << "Ollama Backend:     " << (ollamaUp ? "ONLINE" : "OFFLINE (Deterministic Fallback Active)") << "\n";
    std::cout << "========================================================\n";

    httplib::Server svr;

    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res); res.status = 204;
    });

    // ── SEARCH ENDPOINTS (Single-Stage Filtered Search) ────────────────
    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo   = "hnsw";

        VectorDB::FilterOptions fOpt;
        fOpt.category = req.get_param_value("category");
        fOpt.keyword  = req.get_param_value("keyword");

        auto out = db.search(q, k, metric, algo, fOpt);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) ss << ',';
            auto& h = out.hits[i];
            ss << "{\"id\":"        << h.id
               << ",\"metadata\":"  << jS(h.meta)
               << ",\"category\":"  << jS(h.cat)
               << ",\"distance\":"  << std::fixed << std::setprecision(6) << h.dist
               << ",\"embedding\":" << jVec(h.emb) << '}';
        }
        ss << "],\"latencyUs\":" << out.us
           << ",\"algo\":"       << jS(out.algo)
           << ",\"metric\":"     << jS(out.metric)
           << ",\"simdActive\":" << (g_simd_enabled ? "true" : "false") << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string meta, cat; std::vector<float> emb;
        if (!parseBody(req.body, meta, cat, emb) || (int)emb.size() != DIMS) {
            res.set_content("{\"error\":\"invalid body\"}", "application/json"); return;
        }
        int id = db.insert(meta, cat, emb, getDistFn("cosine"));
        VectorItem vi{id, meta, cat, emb};
        wal.logInsert(vi);
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = db.remove(id);
        if (ok) wal.logDelete(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto items = db.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < items.size(); i++) {
            if (i) ss << ',';
            auto& v = items[i];
            ss << "{\"id\":"        << v.id
               << ",\"metadata\":"  << jS(v.metadata)
               << ",\"category\":"  << jS(v.category)
               << ",\"embedding\":" << jVec(v.emb) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto b = db.benchmark(q, k, metric);
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bfUs << ",\"kdtreeUs\":" << b.kdUs
           << ",\"hnswUs\":"       << b.hnswUs << ",\"itemCount\":" << b.n
           << ",\"simdActive\":"   << (g_simd_enabled ? "true" : "false") << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto gi = db.hnswInfo();
        std::ostringstream ss;
        ss << "{\"topLayer\":" << gi.topLayer << ",\"nodeCount\":" << gi.nodeCount
           << ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.nodesPerLayer[i];
        }
        ss << "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.edgesPerLayer[i];
        }
        ss << "],\"nodes\":[";
        for (size_t i = 0; i < gi.nodes.size(); i++) {
            if (i) ss << ',';
            auto& n = gi.nodes[i];
            ss << "{\"id\":" << n.id << ",\"metadata\":" << jS(n.metadata)
               << ",\"category\":" << jS(n.category) << ",\"maxLyr\":" << n.maxLyr << '}';
        }
        ss << "],\"edges\":[";
        for (size_t i = 0; i < gi.edges.size(); i++) {
            if (i) ss << ',';
            auto& e = gi.edges[i];
            ss << "{\"src\":" << e.src << ",\"dst\":" << e.dst << ",\"lyr\":" << e.lyr << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── MILESTONE 2: AVX2 SIMD BENCHMARK & TOGGLE ──────────────────────

    // POST /simd/toggle {"enabled": true/false}
    svr.Post("/simd/toggle", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        if (req.body.find("false") != std::string::npos) {
            g_simd_enabled = false;
        } else {
            g_simd_enabled = (VECTRA_AVX2_SUPPORTED == 1);
        }
        std::ostringstream ss;
        ss << "{\"simdEnabled\":"   << (g_simd_enabled ? "true" : "false")
           << ",\"avx2Supported\":" << (VECTRA_AVX2_SUPPORTED ? "true" : "false") << '}';
        res.set_content(ss.str(), "application/json");
    });

    // POST /benchmark/simd
    // Evaluates pure scalar loop vs 8-wide AVX2 vectorization over high-dimensional vectors
    svr.Post("/benchmark/simd", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int iters = extractInt(req.body, "iterations", 25000);
        int dims  = extractInt(req.body, "dims", 768);
        if (iters <= 0) iters = 25000;
        if (dims <= 0)  dims = 768;

        std::vector<float> a(dims, 0.45f), b(dims, 0.32f);
        for (int i = 0; i < dims; i++) {
            a[i] = std::sin((float)i * 0.1f);
            b[i] = std::cos((float)i * 0.1f);
        }

        // Benchmark Scalar Distance
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile float sinkScalar = 0.0f;
        for (int i = 0; i < iters; i++) {
            sinkScalar += cosine_scalar(a.data(), b.data(), dims);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        long long scalarUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // Benchmark AVX2 SIMD Distance
        long long simdUs = scalarUs;
#if defined(__AVX2__)
        volatile float sinkSimd = 0.0f;
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            sinkSimd += cosine_avx2(a.data(), b.data(), dims);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        simdUs = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        if (simdUs <= 0) simdUs = 1;
#endif

        float speedup = (float)scalarUs / (float)simdUs;
        double opsSecScalar = (double)iters / ((double)scalarUs / 1e6);
        double opsSecSimd   = (double)iters / ((double)simdUs / 1e6);

        std::ostringstream ss;
        ss << "{\"avx2Supported\":" << (VECTRA_AVX2_SUPPORTED ? "true" : "false")
           << ",\"simdActive\":"    << (g_simd_enabled ? "true" : "false")
           << ",\"iterations\":"    << iters
           << ",\"dims\":"          << dims
           << ",\"scalarUs\":"      << scalarUs
           << ",\"simdUs\":"        << simdUs
           << ",\"speedupFactor\":" << std::fixed << std::setprecision(2) << speedup
           << ",\"opsSecScalar\":"  << (long long)opsSecScalar
           << ",\"opsSecSimd\":"    << (long long)opsSecSimd << '}';
        res.set_content(ss.str(), "application/json");
    });

    // ── MILESTONE 2: STORAGE ENGINE & WAL ENDPOINTS ───────────────────

    // POST /storage/snapshot
    svr.Post("/storage/snapshot", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool ok = StorageEngine::saveSnapshot("vectra.vdb", db, wal);
        std::ostringstream ss;
        ss << "{\"ok\":" << (ok ? "true" : "false")
           << ",\"file\":" << jS("vectra.vdb")
           << ",\"walEntries\":" << wal.getCount() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // POST /storage/restore
    svr.Post("/storage/restore", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool ok = StorageEngine::loadSnapshot("vectra.vdb", db);
        std::ostringstream ss;
        ss << "{\"ok\":" << (ok ? "true" : "false")
           << ",\"itemCount\":" << db.size()
           << ",\"walEntries\":" << wal.getCount() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // GET /storage/info
    svr.Get("/storage/info", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        size_t vdbBytes = 0;
        std::ifstream vdb("vectra.vdb", std::ios::binary | std::ios::ate);
        if (vdb.is_open()) {
            vdbBytes = (size_t)vdb.tellg();
            vdb.close();
        }
        size_t walBytes = 0;
        std::ifstream walF("vectra.wal", std::ios::binary | std::ios::ate);
        if (walF.is_open()) {
            walBytes = (size_t)walF.tellg();
            walF.close();
        }

        std::ostringstream ss;
        ss << "{\"vdbExists\":"    << (vdbBytes > 0 ? "true" : "false")
           << ",\"vdbSizeBytes\":" << vdbBytes
           << ",\"walSizeBytes\":" << walBytes
           << ",\"walCount\":"     << wal.getCount()
           << ",\"itemCount\":"    << db.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // ── MILESTONE 2: SCALAR QUANTIZATION (SQ8) ENDPOINTS ──────────────

    // GET /sq8/stats
    svr.Get("/sq8/stats", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto qs = db.getSQ8Stats();
        std::ostringstream ss;
        ss << "{\"itemCount\":"           << qs.count
           << ",\"dims\":"                << qs.dims
           << ",\"fp32Bytes\":"           << qs.fp32Bytes
           << ",\"sq8Bytes\":"            << qs.sq8Bytes
           << ",\"compressionRatio\":"    << std::fixed << std::setprecision(2) << qs.compressionRatio
           << ",\"memorySavedPercent\":"  << std::fixed << std::setprecision(1) << qs.memorySavedPercent
           << ",\"meanSquaredError\":"    << std::fixed << std::setprecision(6) << qs.meanSquaredError << '}';
        res.set_content(ss.str(), "application/json");
    });

    // POST /benchmark/sq8
    svr.Post("/benchmark/sq8", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto items = db.all();
        if (items.empty()) {
            res.set_content("{\"error\":\"no vectors stored\"}", "application/json");
            return;
        }

        std::vector<SQ8Vector> sqStore;
        for (const auto& item : items) {
            sqStore.push_back(SQ8Vector::quantize(item.id, item.emb));
        }

        auto q = items[0].emb;
        int k = 5;

        // FP32 HNSW / Exact KNN
        auto t0 = std::chrono::high_resolution_clock::now();
        auto groundTruth = db.search(q, k, "cosine", "bruteforce");
        auto t1 = std::chrono::high_resolution_clock::now();
        long long fp32Us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // SQ8 Asymmetric Distance Search
        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<float,int>> sqResults;
        for (const auto& sq : sqStore) {
            sqResults.push_back({sq.asymmetricCosine(q), sq.id});
        }
        std::sort(sqResults.begin(), sqResults.end());
        if ((int)sqResults.size() > k) sqResults.resize(k);
        auto t3 = std::chrono::high_resolution_clock::now();
        long long sq8Us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        // Compute Recall@K overlap
        std::set<int> gtIds;
        for (const auto& h : groundTruth.hits) gtIds.insert(h.id);
        int matched = 0;
        for (const auto& r : sqResults) if (gtIds.count(r.second)) matched++;
        float recall = gtIds.empty() ? 100.0f : ((float)matched / (float)gtIds.size()) * 100.0f;

        std::ostringstream ss;
        ss << "{\"fp32Us\":"       << fp32Us
           << ",\"sq8Us\":"        << sq8Us
           << ",\"k\":"            << k
           << ",\"recallPercent\":"<< std::fixed << std::setprecision(1) << recall << '}';
        res.set_content(ss.str(), "application/json");
    });

    // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    // POST /doc/insert {"title":"...","text":"..."}
    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto title = extractStr(req.body, "title");
        auto text  = extractStr(req.body, "text");
        if (title.empty() || text.empty()) {
            res.set_content("{\"error\":\"need title and text\"}", "application/json"); return;
        }

        auto chunks = chunkText(text, 250, 30);
        std::vector<int> ids;
        bool usedOllama = false;

        for (int i = 0; i < (int)chunks.size(); i++) {
            std::vector<float> emb;
            if (ollama.isAvailable()) {
                emb = ollama.embed(chunks[i]);
                if (!emb.empty()) usedOllama = true;
            }
            if (emb.empty()) {
                int curDims = docDB.getDims();
                emb = localSemanticEmbed(chunks[i], curDims > 0 ? curDims : 64);
            }
            std::string chunkTitle = (chunks.size() > 1)
                ? title + " [" + std::to_string(i+1) + "/" + std::to_string(chunks.size()) + "]"
                : title;
            ids.push_back(docDB.insert(chunkTitle, chunks[i], emb));
        }

        std::ostringstream ss;
        ss << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); i++) { if (i) ss << ','; ss << ids[i]; }
        ss << "],\"chunks\":" << chunks.size()
           << ",\"dims\":"    << docDB.getDims()
           << ",\"engine\":"  << jS(usedOllama ? "ollama" : "local_semantic_engine") << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = docDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto docs = docDB.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < docs.size(); i++) {
            if (i) ss << ',';
            std::string preview = docs[i].text.substr(0, 120);
            if (docs[i].text.size() > 120) preview += "...";
            ss << "{\"id\":" << docs[i].id
               << ",\"title\":" << jS(docs[i].title)
               << ",\"preview\":" << jS(preview)
               << ",\"words\":"  << (int)std::count(docs[i].text.begin(), docs[i].text.end(), ' ') + 1
               << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto question = extractStr(req.body, "question");
        int  k        = extractInt(req.body, "k", 3);
        auto keyword  = extractStr(req.body, "keyword");
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        std::vector<float> qEmb;
        if (ollama.isAvailable()) {
            qEmb = ollama.embed(question);
        }
        if (qEmb.empty()) {
            int curDims = docDB.getDims();
            qEmb = localSemanticEmbed(question, curDims > 0 ? curDims : 64);
        }

        auto hits = docDB.search(qEmb, k, 2.0f, keyword);

        std::ostringstream ss;
        ss << "{\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto question = extractStr(req.body, "question");
        int  k        = extractInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        std::vector<float> qEmb;
        bool isOllama = ollama.isAvailable();
        if (isOllama) {
            qEmb = ollama.embed(question);
        }
        if (qEmb.empty()) {
            int curDims = docDB.getDims();
            qEmb = localSemanticEmbed(question, curDims > 0 ? curDims : 64);
        }

        auto hits = docDB.search(qEmb, k);

        std::string answer;
        std::string modelName;
        if (isOllama) {
            std::ostringstream ctx;
            for (int i = 0; i < (int)hits.size(); i++) {
                ctx << "[" << (i+1) << "] " << hits[i].second.title << ":\n"
                    << hits[i].second.text << "\n\n";
            }
            std::string prompt =
                "You are a helpful assistant. Answer the user's question directly. "
                "Use the provided context if it contains relevant information. "
                "If it doesn't, just use your own general knowledge. "
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                "Just answer the question naturally.\n\n"
                "Context:\n" + ctx.str() +
                "Question: " + question + "\n\n"
                "Answer:";
            answer = ollama.generate(prompt);
            modelName = ollama.genModel;
        } else {
            modelName = "VectorDB-Local-RAG";
            if (hits.empty()) {
                answer = "No relevant context found in stored documents for: \"" + question + "\". Insert documents in the Documents tab first!";
            } else {
                std::ostringstream ans;
                ans << "Found " << hits.size() << " matching document chunk" << (hits.size() > 1 ? "s" : "") << " via HNSW Vector Search:\n\n";
                for (size_t i = 0; i < hits.size(); i++) {
                    ans << "• **" << hits[i].second.title << "** (distance: " << std::fixed << std::setprecision(4) << hits[i].first << ")\n"
                        << "  " << hits[i].second.text << "\n\n";
                }
                answer = ans.str();
            }
        }

        std::ostringstream ss;
        ss << "{\"answer\":" << jS(answer)
           << ",\"model\":"  << jS(modelName)
           << ",\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"text\":"     << jS(hits[i].second.text)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "],\"docCount\":" << docDB.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // ── STATUS & SYSTEM METRICS ───────────────────────────────────────
    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool up = ollama.isAvailable();
        std::ostringstream ss;
        ss << "{\"ollamaAvailable\":"  << (up ? "true" : "false")
           << ",\"embedModel\":"       << (up ? jS(ollama.embedModel) : jS("Local Semantic Engine"))
           << ",\"genModel\":"         << (up ? jS(ollama.genModel) : jS("Extractive RAG Synthesizer"))
           << ",\"docCount\":"         << docDB.size()
           << ",\"docDims\":"          << docDB.getDims()
           << ",\"demoDims\":"         << DIMS
           << ",\"demoCount\":"        << db.size()
           << ",\"avx2Supported\":"    << (VECTRA_AVX2_SUPPORTED ? "true" : "false")
           << ",\"simdActive\":"       << (g_simd_enabled ? "true" : "false")
           << ",\"walCount\":"         << wal.getCount() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        ss << "{\"count\":"           << db.size()
           << ",\"dims\":"            << DIMS
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]"
           << ",\"avx2Supported\":"   << (VECTRA_AVX2_SUPPORTED ? "true" : "false")
           << ",\"simdActive\":"      << (g_simd_enabled ? "true" : "false")
           << ",\"walCount\":"        << wal.getCount() << "}";
        res.set_content(ss.str(), "application/json");
    });

    // Serve index.html
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("index.html");
        if (!f.is_open()) { res.status = 404; return; }
        res.set_content(
            std::string(std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>()),
            "text/html");
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
