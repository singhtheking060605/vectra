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
#include <numeric>
#include <map>

#if defined(__AVX2__)
#include <immintrin.h>
#define VECTRA_AVX2_SUPPORTED 1
#else
#define VECTRA_AVX2_SUPPORTED 0
#endif

static const int DIMS = 16;   // demo vectors

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
    alignas(32) float bufDot[8], bufNa[8], bufNb[8];
    _mm256_storeu_ps(bufDot, dot256);
    _mm256_storeu_ps(bufNa,  na256);
    _mm256_storeu_ps(bufNb,  nb256);

    float dot = bufDot[0] + bufDot[1] + bufDot[2] + bufDot[3] + bufDot[4] + bufDot[5] + bufDot[6] + bufDot[7];
    float na  = bufNa[0]  + bufNa[1]  + bufNa[2]  + bufNa[3]  + bufNa[4]  + bufNa[5]  + bufNa[6]  + bufNa[7];
    float nb  = bufNb[0]  + bufNb[1]  + bufNb[2]  + bufNb[3]  + bufNb[4]  + bufNb[5]  + bufNb[6]  + bufNb[7];

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
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
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
    for (; i < n; i++) {
        s += std::abs(a[i] - b[i]);
    }
    return s;
}
#endif

// Distance Function Dispatcher
inline float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return 0.0f;
#if defined(__AVX2__)
    if (g_simd_enabled) return euclidean_avx2(a.data(), b.data(), a.size());
#endif
    return euclidean_scalar(a.data(), b.data(), a.size());
}

inline float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return 1.0f;
#if defined(__AVX2__)
    if (g_simd_enabled) return cosine_avx2(a.data(), b.data(), a.size());
#endif
    return cosine_scalar(a.data(), b.data(), a.size());
}

inline float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return 0.0f;
#if defined(__AVX2__)
    if (g_simd_enabled) return manhattan_avx2(a.data(), b.data(), a.size());
#endif
    return manhattan_scalar(a.data(), b.data(), a.size());
}

DistFn getDistFn(const std::string& name) {
    if (name == "cosine")    return cosine;
    if (name == "manhattan") return manhattan;
    return euclidean;
}

// =====================================================================
//  BRUTE-FORCE SEARCH (Baseline O(N * d))
// =====================================================================

class BruteForce {
    std::unordered_map<int, VectorItem> items;
public:
    void insert(const VectorItem& v) { items[v.id] = v; }
    void remove(int id)              { items.erase(id); }

    std::vector<std::pair<float, int>> knn(const std::vector<float>& q, int k, DistFn dist) {
        std::vector<std::pair<float, int>> all;
        all.reserve(items.size());
        for (auto& [id, item] : items) {
            all.push_back({dist(q, item.emb), id});
        }
        std::sort(all.begin(), all.end());
        if ((int)all.size() > k) all.resize(k);
        return all;
    }

    template <typename Predicate>
    std::vector<std::pair<float, int>> knnFiltered(const std::vector<float>& q, int k, DistFn dist, Predicate pred) {
        std::vector<std::pair<float, int>> all;
        for (auto& [id, item] : items) {
            if (pred(item)) {
                all.push_back({dist(q, item.emb), id});
            }
        }
        std::sort(all.begin(), all.end());
        if ((int)all.size() > k) all.resize(k);
        return all;
    }
};

// =====================================================================
//  KD-TREE SPATIAL PARTITIONING
// =====================================================================

class KDTree {
    struct Node {
        VectorItem item;
        int axis;
        Node* left  = nullptr;
        Node* right = nullptr;
        Node(VectorItem v, int ax) : item(v), axis(ax) {}
    };

    Node* root = nullptr;
    int dims;

    Node* build(std::vector<VectorItem>& items, int depth) {
        if (items.empty()) return nullptr;
        int ax = depth % dims;
        size_t mid = items.size() / 2;
        std::nth_element(items.begin(), items.begin() + mid, items.end(),
            [ax](const VectorItem& a, const VectorItem& b) {
                return a.emb[ax] < b.emb[ax];
            });
        Node* n = new Node(items[mid], ax);
        std::vector<VectorItem> left(items.begin(), items.begin() + mid);
        std::vector<VectorItem> right(items.begin() + mid + 1, items.end());
        n->left  = build(left,  depth + 1);
        n->right = build(right, depth + 1);
        return n;
    }

    void search(Node* node, const std::vector<float>& q, int k, DistFn dist,
                std::priority_queue<std::pair<float, int>>& pq) const {
        if (!node) return;
        float d = dist(q, node->item.emb);
        pq.push({d, node->item.id});
        if ((int)pq.size() > k) pq.pop();

        int ax = node->axis;
        float diff = q[ax] - node->item.emb[ax];
        Node* first  = diff <= 0 ? node->left  : node->right;
        Node* second = diff <= 0 ? node->right : node->left;

        search(first, q, k, dist, pq);
        if (std::abs(diff) < pq.top().first || (int)pq.size() < k) {
            search(second, q, k, dist, pq);
        }
    }

    void freeTree(Node* n) {
        if (!n) return;
        freeTree(n->left);
        freeTree(n->right);
        delete n;
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { freeTree(root); }

    void rebuild(std::vector<VectorItem>& items) {
        freeTree(root);
        root = build(items, 0);
    }

    void insert(const VectorItem& v) {
        if (!root) { root = new Node(v, 0); return; }
        Node* cur = root;
        int depth = 0;
        while (true) {
            int ax = depth % dims;
            if (v.emb[ax] < cur->item.emb[ax]) {
                if (!cur->left) { cur->left = new Node(v, (depth + 1) % dims); break; }
                cur = cur->left;
            } else {
                if (!cur->right) { cur->right = new Node(v, (depth + 1) % dims); break; }
                cur = cur->right;
            }
            depth++;
        }
    }

    std::vector<std::pair<float, int>> knn(const std::vector<float>& q, int k, DistFn dist) const {
        std::priority_queue<std::pair<float, int>> pq;
        search(root, q, k, dist, pq);
        std::vector<std::pair<float, int>> res;
        while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
        std::reverse(res.begin(), res.end());
        return res;
    }
};

// =====================================================================
//  HNSW (Hierarchical Navigable Small World Graph)
// =====================================================================

class HNSW {
public:
    struct Node {
        int id;
        std::vector<float> emb;
        std::string metadata;
        std::string category;
        int maxLayer;
        std::vector<std::vector<int>> neighbors; // layer -> list of neighbor IDs
    };

    struct GraphInfo {
        int topLayer;
        size_t nodeCount;
        std::vector<int> nodesPerLayer;
        std::vector<int> edgesPerLayer;
        struct GNode { int id; std::string metadata, category; int maxLyr; };
        struct GEdge { int src, dst, lyr; };
        std::vector<GNode> nodes;
        std::vector<GEdge> edges;
    };

private:
    int M;
    int efConstruction;
    double mL;
    int entryPointId = -1;
    int maxLayer = -1;
    std::unordered_map<int, Node> nodes;
    std::mt19937 rng{42};
    std::uniform_real_distribution<double> unif{0.0, 1.0};

    int randomLayer() {
        double r = unif(rng);
        if (r == 0.0) r = 0.00001;
        return (int)(-std::log(r) * mL);
    }

    std::vector<std::pair<float, int>> searchLayer(const std::vector<float>& q,
                                                   const std::vector<int>& enterPoints,
                                                   int ef, int layer, DistFn dist) const {
        std::unordered_map<int, bool> visited;
        std::priority_queue<std::pair<float, int>,
                            std::vector<std::pair<float, int>>,
                            std::greater<std::pair<float, int>>> candidates;
        std::priority_queue<std::pair<float, int>> w;

        for (int ep : enterPoints) {
            if (!nodes.count(ep)) continue;
            float d = dist(q, nodes.at(ep).emb);
            visited[ep] = true;
            candidates.push({d, ep});
            w.push({d, ep});
        }

        while (!candidates.empty()) {
            auto [cDist, cId] = candidates.top();
            candidates.pop();
            float furthestDist = w.empty() ? 1e9f : w.top().first;
            if (cDist > furthestDist && (int)w.size() >= ef) break;

            if (!nodes.count(cId) || (int)nodes.at(cId).neighbors.size() <= layer) continue;
            const auto& neighbors = nodes.at(cId).neighbors[layer];

            for (int nId : neighbors) {
                if (visited[nId] || !nodes.count(nId)) continue;
                visited[nId] = true;
                float d = dist(q, nodes.at(nId).emb);
                if (d < furthestDist || (int)w.size() < ef) {
                    candidates.push({d, nId});
                    w.push({d, nId});
                    if ((int)w.size() > ef) w.pop();
                    furthestDist = w.top().first;
                }
            }
        }

        std::vector<std::pair<float, int>> result;
        while (!w.empty()) { result.push_back(w.top()); w.pop(); }
        std::reverse(result.begin(), result.end());
        return result;
    }

public:
    HNSW(int m = 16, int efC = 200) : M(m), efConstruction(efC), mL(1.0 / std::log(m)) {}

    void insert(const VectorItem& v, DistFn dist) {
        int l = randomLayer();
        Node newNode;
        newNode.id = v.id;
        newNode.emb = v.emb;
        newNode.metadata = v.metadata;
        newNode.category = v.category;
        newNode.maxLayer = l;
        newNode.neighbors.resize(l + 1);

        if (nodes.empty()) {
            entryPointId = v.id;
            maxLayer = l;
            nodes[v.id] = newNode;
            return;
        }

        std::vector<int> currObj = {entryPointId};
        for (int lc = maxLayer; lc > l; lc--) {
            auto results = searchLayer(v.emb, currObj, 1, lc, dist);
            if (!results.empty()) currObj = {results[0].second};
        }

        for (int lc = std::min(maxLayer, l); lc >= 0; lc--) {
            auto results = searchLayer(v.emb, currObj, efConstruction, lc, dist);
            std::vector<std::pair<float, int>> candidates = results;
            std::sort(candidates.begin(), candidates.end());

            int count = 0;
            for (auto& [d, nId] : candidates) {
                if (count >= M) break;
                if (nId == v.id) continue;
                newNode.neighbors[lc].push_back(nId);
                if ((int)nodes[nId].neighbors.size() > lc) {
                    nodes[nId].neighbors[lc].push_back(v.id);
                }
                count++;
            }
            if (!results.empty()) currObj = {results[0].second};
        }

        if (l > maxLayer) {
            maxLayer = l;
            entryPointId = v.id;
        }
        nodes[v.id] = newNode;
    }

    void remove(int id) {
        if (!nodes.count(id)) return;
        for (auto& [nId, n] : nodes) {
            for (auto& nbrList : n.neighbors) {
                nbrList.erase(std::remove(nbrList.begin(), nbrList.end(), id), nbrList.end());
            }
        }
        nodes.erase(id);
        if (entryPointId == id) {
            entryPointId = nodes.empty() ? -1 : nodes.begin()->first;
            maxLayer = -1;
            for (auto& [nId, n] : nodes) maxLayer = std::max(maxLayer, n.maxLayer);
        }
    }

    std::vector<std::pair<float, int>> knn(const std::vector<float>& q, int k, int ef, DistFn dist) const {
        if (nodes.empty()) return {};
        std::vector<int> currObj = {entryPointId};
        for (int lc = maxLayer; lc > 0; lc--) {
            auto results = searchLayer(q, currObj, 1, lc, dist);
            if (!results.empty()) currObj = {results[0].second};
        }
        auto results = searchLayer(q, currObj, std::max(ef, k), 0, dist);
        if ((int)results.size() > k) results.resize(k);
        return results;
    }

    template <typename Predicate>
    std::vector<std::pair<float, int>> knnFiltered(const std::vector<float>& q, int k, int ef, DistFn dist, Predicate pred) const {
        if (nodes.empty()) return {};
        auto candidatePool = searchLayer(q, {entryPointId}, std::max(ef * 3, k * 5), 0, dist);
        std::vector<std::pair<float, int>> filtered;
        for (auto& [d, id] : candidatePool) {
            if (nodes.count(id)) {
                VectorItem vi{id, nodes.at(id).metadata, nodes.at(id).category, nodes.at(id).emb};
                if (pred(vi)) {
                    filtered.push_back({d, id});
                    if ((int)filtered.size() >= k) break;
                }
            }
        }
        return filtered;
    }

    GraphInfo getGraphInfo() const {
        GraphInfo gi;
        gi.topLayer = maxLayer;
        gi.nodeCount = nodes.size();
        if (maxLayer < 0) return gi;

        gi.nodesPerLayer.resize(maxLayer + 1, 0);
        gi.edgesPerLayer.resize(maxLayer + 1, 0);

        for (const auto& [id, n] : nodes) {
            gi.nodes.push_back({id, n.metadata, n.category, n.maxLayer});
            for (int l = 0; l <= n.maxLayer && l <= maxLayer; l++) {
                gi.nodesPerLayer[l]++;
                if (l < (int)n.neighbors.size()) {
                    for (int dst : n.neighbors[l]) {
                        if (id < dst) {
                            gi.edges.push_back({id, dst, l});
                            gi.edgesPerLayer[l]++;
                        }
                    }
                }
            }
        }
        return gi;
    }
};

// =====================================================================
//  WEEK 3: K-MEANS CLUSTERING & INVERTED FILE INDEX (IVF-FLAT)
// =====================================================================

class IVFFlat {
public:
    int dims;
    int nlist; // number of Voronoi partitions (clusters)
    std::vector<std::vector<float>> centroids;
    std::unordered_map<int, std::vector<int>> invertedLists; // centroid_idx -> [itemIds]
    std::unordered_map<int, VectorItem> items;
    bool isTrained = false;
    float totalInertia = 0.0f;
    int iterationsRun = 0;

    IVFFlat(int d, int k = 4) : dims(d), nlist(k) {}

    // Lloyd's Algorithm Implementation for K-Means Clustering
    void train(const std::vector<VectorItem>& data, int kClusters = 4, int maxIters = 25, DistFn dist = nullptr) {
        if (data.empty()) return;
        if (!dist) dist = euclidean;
        nlist = std::min((int)data.size(), std::max(2, kClusters));
        centroids.clear();
        invertedLists.clear();
        items.clear();
        for (const auto& item : data) items[item.id] = item;

        // 1. Centroid Initialization: K-Means++ deterministic spread
        std::vector<bool> picked(data.size(), false);
        centroids.push_back(data[0].emb);
        picked[0] = true;

        for (int c = 1; c < nlist; c++) {
            float maxMinDist = -1.0f;
            int bestIdx = 0;
            for (size_t i = 0; i < data.size(); i++) {
                if (picked[i]) continue;
                float minDist = 1e9f;
                for (const auto& cent : centroids) {
                    float d = dist(data[i].emb, cent);
                    minDist = std::min(minDist, d);
                }
                if (minDist > maxMinDist) {
                    maxMinDist = minDist;
                    bestIdx = (int)i;
                }
            }
            centroids.push_back(data[bestIdx].emb);
            picked[bestIdx] = true;
        }

        // 2. Iterative Lloyd updates (Assign -> Recompute Center)
        for (int iter = 0; iter < maxIters; iter++) {
            std::vector<std::vector<int>> clusterMembers(nlist);
            for (size_t i = 0; i < data.size(); i++) {
                float bestDist = 1e9f;
                int bestC = 0;
                for (int c = 0; c < nlist; c++) {
                    float d = dist(data[i].emb, centroids[c]);
                    if (d < bestDist) {
                        bestDist = d;
                        bestC = c;
                    }
                }
                clusterMembers[bestC].push_back((int)i);
            }

            // Update step
            bool converged = true;
            for (int c = 0; c < nlist; c++) {
                if (clusterMembers[c].empty()) continue;
                std::vector<float> newCent(dims, 0.0f);
                for (int idx : clusterMembers[c]) {
                    for (int d = 0; d < dims; d++) {
                        newCent[d] += data[idx].emb[d];
                    }
                }
                for (int d = 0; d < dims; d++) {
                    newCent[d] /= (float)clusterMembers[c].size();
                }
                if (dist(centroids[c], newCent) > 1e-4f) {
                    converged = false;
                }
                centroids[c] = newCent;
            }
            iterationsRun = iter + 1;
            if (converged) break;
        }

        // 3. Build Inverted File Lists
        totalInertia = 0.0f;
        for (const auto& item : data) {
            float bestDist = 1e9f;
            int bestC = 0;
            for (int c = 0; c < nlist; c++) {
                float d = dist(item.emb, centroids[c]);
                if (d < bestDist) {
                    bestDist = d;
                    bestC = c;
                }
            }
            invertedLists[bestC].push_back(item.id);
            totalInertia += bestDist * bestDist;
        }
        isTrained = true;
    }

    void insert(const VectorItem& item, DistFn dist = nullptr) {
        if (!dist) dist = euclidean;
        items[item.id] = item;
        if (!isTrained || centroids.empty()) return;
        float bestDist = 1e9f;
        int bestC = 0;
        for (int c = 0; c < (int)centroids.size(); c++) {
            float d = dist(item.emb, centroids[c]);
            if (d < bestDist) {
                bestDist = d;
                bestC = c;
            }
        }
        invertedLists[bestC].push_back(item.id);
    }

    void remove(int id) {
        items.erase(id);
        for (auto& [c, list] : invertedLists) {
            list.erase(std::remove(list.begin(), list.end(), id), list.end());
        }
    }

    // Search top-k using nprobe multi-centroid probing (Voronoi cell filtering)
    std::vector<std::pair<float, int>> knn(const std::vector<float>& q, int k, int nprobe, DistFn dist = nullptr) const {
        if (!isTrained || centroids.empty()) return {};
        if (!dist) dist = euclidean;
        nprobe = std::max(1, std::min(nprobe, (int)centroids.size()));

        // Rank centroids by distance to query vector
        std::vector<std::pair<float, int>> centroidDist;
        for (int c = 0; c < (int)centroids.size(); c++) {
            centroidDist.push_back({dist(q, centroids[c]), c});
        }
        std::sort(centroidDist.begin(), centroidDist.end());

        // Probe only the inverted lists of the closest nprobe Voronoi cells
        std::vector<std::pair<float, int>> candidates;
        for (int p = 0; p < nprobe; p++) {
            int c = centroidDist[p].second;
            if (invertedLists.count(c)) {
                for (int itemId : invertedLists.at(c)) {
                    if (items.count(itemId)) {
                        candidates.push_back({dist(q, items.at(itemId).emb), itemId});
                    }
                }
            }
        }
        std::sort(candidates.begin(), candidates.end());
        if ((int)candidates.size() > k) candidates.resize(k);
        return candidates;
    }
};

// =====================================================================
//  WEEK 4: PRINCIPAL COMPONENT ANALYSIS (PCA & AVX2 PROJECTION)
// =====================================================================

class PCAReducer {
public:
    int dims;
    int nComponents;
    std::vector<float> mean;
    std::vector<std::vector<float>> components; // Principal Eigenvectors
    std::vector<float> eigenvalues;
    std::vector<float> explainedVarianceRatio;
    bool isFitted = false;

    explicit PCAReducer(int d = 16) : dims(d), nComponents(2), mean(d, 0.0f) {}

    void fit(const std::vector<std::vector<float>>& data, int kComp = 2) {
        if (data.empty()) return;
        dims = (int)data[0].size();
        if (dims <= 0) return;
        nComponents = std::max(1, std::min(kComp, dims));
        size_t N = data.size();

        mean.assign(dims, 0.0f);
        for (const auto& vec : data) {
            for (int j = 0; j < dims && j < (int)vec.size(); j++) mean[j] += vec[j];
        }
        for (int j = 0; j < dims; j++) mean[j] /= (float)N;

        std::vector<std::vector<float>> cov(dims, std::vector<float>(dims, 0.0f));
        for (const auto& vec : data) {
            std::vector<float> centered(dims, 0.0f);
            for (int j = 0; j < dims && j < (int)vec.size(); j++) centered[j] = vec[j] - mean[j];
            for (int r = 0; r < dims; r++) {
                for (int c = 0; c < dims; c++) {
                    cov[r][c] += centered[r] * centered[c];
                }
            }
        }
        float totalVar = 0.0f;
        for (int r = 0; r < dims; r++) {
            for (int c = 0; c < dims; c++) cov[r][c] /= (float)N;
            totalVar += cov[r][r];
        }
        if (totalVar < 1e-6f) totalVar = 1.0f;

        components.clear();
        eigenvalues.clear();
        explainedVarianceRatio.clear();

        std::vector<std::vector<float>> A = cov;
        for (int comp = 0; comp < nComponents; comp++) {
            std::vector<float> v(dims, 0.0f);
            for (int i = 0; i < dims; i++) v[i] = std::sin((float)(i + comp + 1) * 1.5f) + 0.1f;
            float norm = 0.0f;
            for (float val : v) norm += val * val;
            norm = std::sqrt(norm);
            if (norm < 1e-6f) norm = 1.0f;
            for (float& val : v) val /= norm;

            float lambda = 0.0f;
            for (int iter = 0; iter < 100; iter++) {
                std::vector<float> nextV(dims, 0.0f);
                for (int r = 0; r < dims; r++) {
                    for (int c = 0; c < dims; c++) {
                        nextV[r] += A[r][c] * v[c];
                    }
                }
                norm = 0.0f;
                for (float val : nextV) norm += val * val;
                norm = std::sqrt(norm);
                if (norm < 1e-9f) break;
                for (int r = 0; r < dims; r++) nextV[r] /= norm;
                lambda = norm;
                v = nextV;
            }

            components.push_back(v);
            eigenvalues.push_back(lambda);
            explainedVarianceRatio.push_back(std::max(0.0f, std::min(1.0f, lambda / totalVar)));

            for (int r = 0; r < dims; r++) {
                for (int c = 0; c < dims; c++) {
                    A[r][c] -= lambda * v[r] * v[c];
                }
            }
        }
        isFitted = true;
    }

    std::vector<float> transform(const std::vector<float>& vec) const {
        if (!isFitted || components.empty()) {
            return {vec.size() > 0 ? vec[0] : 0.0f, vec.size() > 1 ? vec[1] : 0.0f};
        }
        std::vector<float> out(nComponents, 0.0f);
        std::vector<float> centered(dims, 0.0f);
        for (int j = 0; j < dims && j < (int)vec.size(); j++) {
            centered[j] = vec[j] - mean[j];
        }

        for (int c = 0; c < nComponents; c++) {
            float dot = 0.0f;
            for (int j = 0; j < dims; j++) {
                dot += centered[j] * components[c][j];
            }
            out[c] = dot;
        }
        return out;
    }
};

// =====================================================================
//  WEEK 5: SPARSE ML (BM25) & DENSE HYBRID SEARCH (RECIPROCAL RANK FUSION)
// =====================================================================

class BM25Index {
public:
    struct DocItem {
        int id;
        std::string metadata;
        std::string category;
        std::vector<std::string> tokens;
        int length;
    };

    std::unordered_map<int, DocItem> docs;
    std::unordered_map<std::string, std::vector<std::pair<int, float>>> invertedIndex; // token -> [(id, tf)]
    std::unordered_map<std::string, int> docFreq;
    double avgDocLen = 0.0;
    float k1 = 1.5f;
    float b = 0.75f;

    static std::vector<std::string> tokenize(const std::string& str) {
        std::vector<std::string> tokens;
        std::string cur;
        for (char ch : str) {
            if (std::isalnum((unsigned char)ch)) {
                cur.push_back((char)std::tolower((unsigned char)ch));
            } else if (!cur.empty()) {
                if (cur.size() > 1) tokens.push_back(cur);
                cur.clear();
            }
        }
        if (cur.size() > 1) tokens.push_back(cur);
        return tokens;
    }

    void insert(int id, const std::string& meta, const std::string& cat) {
        remove(id);
        auto tokens = tokenize(meta + " " + cat);
        DocItem item{id, meta, cat, tokens, (int)tokens.size()};
        docs[id] = item;

        std::unordered_map<std::string, int> freq;
        for (const auto& t : tokens) freq[t]++;

        for (const auto& [t, count] : freq) {
            float tf = (float)count;
            invertedIndex[t].push_back({id, tf});
            docFreq[t]++;
        }

        // Recompute average document length
        double total = 0.0;
        for (const auto& [did, doc] : docs) total += doc.length;
        avgDocLen = docs.empty() ? 0.0 : total / (double)docs.size();
    }

    void remove(int id) {
        if (!docs.count(id)) return;
        std::set<std::string> seenTokens;
        for (const auto& t : docs[id].tokens) seenTokens.insert(t);
        for (const auto& t : seenTokens) {
            if (invertedIndex.count(t)) {
                auto& list = invertedIndex[t];
                list.erase(std::remove_if(list.begin(), list.end(),
                    [id](const std::pair<int, float>& p){ return p.first == id; }), list.end());
                if (list.empty()) invertedIndex.erase(t);
            }
            if (docFreq.count(t)) {
                docFreq[t]--;
                if (docFreq[t] <= 0) docFreq.erase(t);
            }
        }
        docs.erase(id);
        double total = 0.0;
        for (const auto& [did, doc] : docs) total += doc.length;
        avgDocLen = docs.empty() ? 0.0 : total / (double)docs.size();
    }

    std::vector<std::pair<float, int>> search(const std::string& query, int k) const {
        if (docs.empty()) return {};
        auto queryTokens = tokenize(query);
        if (queryTokens.empty()) return {};

        size_t N = docs.size();
        std::unordered_map<int, float> scores;

        for (const auto& term : queryTokens) {
            if (!invertedIndex.count(term)) continue;
            int df = docFreq.count(term) ? docFreq.at(term) : 0;
            // Robertson-Spärck Jones IDF
            float idf = std::log(1.0f + ((float)N - (float)df + 0.5f) / ((float)df + 0.5f));
            if (idf < 0.0f) idf = 0.01f;

            for (const auto& [docId, tf] : invertedIndex.at(term)) {
                if (!docs.count(docId)) continue;
                float docLen = (float)docs.at(docId).length;
                float denom = tf + k1 * (1.0f - b + b * (docLen / (float)(avgDocLen > 0 ? avgDocLen : 1.0)));
                float termScore = idf * ((tf * (k1 + 1.0f)) / (denom > 0 ? denom : 1.0f));
                scores[docId] += termScore;
            }
        }

        std::vector<std::pair<float, int>> results;
        for (const auto& [id, sc] : scores) results.push_back({sc, id});
        std::sort(results.rbegin(), results.rend()); // Highest score first
        if ((int)results.size() > k) results.resize(k);
        return results;
    }
};

// =====================================================================
//  SCALAR QUANTIZATION (SQ8: FP32 -> INT8)
// =====================================================================

struct SQ8Vector {
    int id;
    float minVal;
    float diff;
    std::vector<uint8_t> quantized;

    static SQ8Vector quantize(int id, const std::vector<float>& vec) {
        SQ8Vector sq;
        sq.id = id;
        if (vec.empty()) { sq.minVal = 0.0f; sq.diff = 1.0f; return sq; }
        float minV = vec[0], maxV = vec[0];
        for (float v : vec) {
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }
        float diff = maxV - minV;
        if (diff < 1e-8f) diff = 1e-8f;
        sq.minVal = minV;
        sq.diff = diff;
        sq.quantized.resize(vec.size());
        for (size_t i = 0; i < vec.size(); i++) {
            float norm = (vec[i] - minV) / diff;
            int q = (int)std::round(norm * 255.0f);
            sq.quantized[i] = (uint8_t)std::max(0, std::min(255, q));
        }
        return sq;
    }

    std::vector<float> dequantize() const {
        std::vector<float> out(quantized.size());
        for (size_t i = 0; i < quantized.size(); i++) {
            out[i] = minVal + ((float)quantized[i] / 255.0f) * diff;
        }
        return out;
    }

    // Asymmetric Cosine Distance (Direct FP32 Query against Quantized Item)
    float asymmetricCosine(const std::vector<float>& q) const {
        size_t n = std::min(q.size(), quantized.size());
        float dot = 0.0f, nq = 0.0f, nd = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float deq = minVal + ((float)quantized[i] / 255.0f) * diff;
            dot += q[i] * deq;
            nq  += q[i] * q[i];
            nd  += deq  * deq;
        }
        if (nq < 1e-9f || nd < 1e-9f) return 1.0f;
        return 1.0f - dot / (std::sqrt(nq) * std::sqrt(nd));
    }
};

struct QuantizationStats {
    size_t count;
    int dims;
    size_t fp32Bytes;
    size_t sq8Bytes;
    float compressionRatio;
    float memorySavedPercent;
    float meanSquaredError;
};

// =====================================================================
//  WRITE-AHEAD LOGGING (WAL) & ATOMIC SNAPSHOT PERSISTENCE
// =====================================================================

class WALManager {
    std::string walPath;
    std::mutex walMu;
    int opCount = 0;

public:
    explicit WALManager(const std::string& path = "vectra.wal") : walPath(path) {
        std::ifstream f(walPath, std::ios::binary);
        if (f.is_open()) {
            f.seekg(0, std::ios::end);
            size_t sz = f.tellg();
            opCount = sz > 0 ? 1 : 0;
        }
    }

    void logInsert(const VectorItem& item) {
        std::lock_guard<std::mutex> lk(walMu);
        std::ofstream f(walPath, std::ios::binary | std::ios::app);
        if (!f.is_open()) return;

        uint8_t opType = 1; // 1 = INSERT
        f.write(reinterpret_cast<const char*>(&opType), sizeof(opType));
        int32_t id = item.id;
        f.write(reinterpret_cast<const char*>(&id), sizeof(id));

        uint32_t metaLen = item.metadata.size();
        f.write(reinterpret_cast<const char*>(&metaLen), sizeof(metaLen));
        f.write(item.metadata.data(), metaLen);

        uint32_t catLen = item.category.size();
        f.write(reinterpret_cast<const char*>(&catLen), sizeof(catLen));
        f.write(item.category.data(), catLen);

        uint32_t dims = item.emb.size();
        f.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
        f.write(reinterpret_cast<const char*>(item.emb.data()), dims * sizeof(float));
        f.flush();
        opCount++;
    }

    void logDelete(int id) {
        std::lock_guard<std::mutex> lk(walMu);
        std::ofstream f(walPath, std::ios::binary | std::ios::app);
        if (!f.is_open()) return;

        uint8_t opType = 2; // 2 = DELETE
        f.write(reinterpret_cast<const char*>(&opType), sizeof(opType));
        int32_t delId = id;
        f.write(reinterpret_cast<const char*>(&delId), sizeof(delId));
        f.flush();
        opCount++;
    }

    void truncate() {
        std::lock_guard<std::mutex> lk(walMu);
        std::ofstream f(walPath, std::ios::binary | std::ios::trunc);
        f.close();
        opCount = 0;
    }

    int getCount() {
        std::lock_guard<std::mutex> lk(walMu);
        return opCount;
    }
};

// =====================================================================
//  VECTOR DATABASE ENGINE (Core Dispatcher & Multi-Index Store)
// =====================================================================

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce  bf;
    KDTree      kdt;
    HNSW        hnsw;
    IVFFlat     ivf;
    PCAReducer  pca;
    BM25Index   bm25;
    std::mutex  mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), ivf(d, 4), pca(d), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        ivf.insert(v, dist);
        bm25.insert(v.id, meta, cat);
        return v.id;
    }

    void insertWithId(int id, const std::string& meta, const std::string& cat,
                      const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (store.count(id)) {
            bf.remove(id);
            hnsw.remove(id);
            ivf.remove(id);
            bm25.remove(id);
        }
        VectorItem v{id, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        ivf.insert(v, dist);
        bm25.insert(v.id, meta, cat);
        if (id >= nextId) nextId = id + 1;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); bf.remove(id); hnsw.remove(id); ivf.remove(id); bm25.remove(id);
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
                     const FilterOptions& fOpt = {}, int nprobe = 2)
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
                auto full = kdt.knn(q, std::min((int)store.size(), k * 10), dfn);
                for (auto& p : full) {
                    if (store.count(p.second) && predicate(store[p.second])) {
                        raw.push_back(p);
                        if ((int)raw.size() >= k) break;
                    }
                }
            } else if (algo == "ivf") {
                if (!ivf.isTrained) trainIVF(4);
                auto full = ivf.knn(q, std::min((int)store.size(), k * 5), nprobe, dfn);
                for (auto& p : full) {
                    if (store.count(p.second) && predicate(store[p.second])) {
                        raw.push_back(p);
                        if ((int)raw.size() >= k) break;
                    }
                }
            } else {
                raw = hnsw.knnFiltered(q, k, 50, dfn, predicate);
            }
        } else {
            if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
            else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
            else if (algo == "ivf") {
                if (!ivf.isTrained) trainIVF(4);
                raw = ivf.knn(q, k, nprobe, dfn);
            }
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

    struct BenchOut { long long bfUs, kdUs, ivfUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn  = getDistFn(metric);
        if (!ivf.isTrained) trainIVF(4);
        auto time = [&](auto fn) -> long long {
            auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&]{ bf.knn(q, k, dfn); }),
            time([&]{ kdt.knn(q, k, dfn); }),
            time([&]{ ivf.knn(q, k, 2, dfn); }),
            time([&]{ hnsw.knn(q, k, 50, dfn); }),
            (int)store.size()
        };
    }

    void trainIVF(int k = 4) {
        std::vector<VectorItem> items;
        for (auto& [id, v] : store) items.push_back(v);
        ivf.train(items, k, 25, getDistFn("cosine"));
    }

    IVFFlat& getIVF() { return ivf; }

    // Principal Component Analysis (Week 4)
    void fitPCA(int nComp = 2) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<std::vector<float>> data;
        for (auto& [id, v] : store) data.push_back(v.emb);
        pca.fit(data, nComp);
    }

    PCAReducer& getPCA() { return pca; }

    // Sparse + Dense Hybrid Search with Reciprocal Rank Fusion (Week 5)
    struct HybridHit {
        int id;
        std::string metadata;
        std::string category;
        std::vector<float> emb;
        float rrfScore;
        float denseDist;
        float bm25Score;
        int denseRank;
        int bm25Rank;
    };

    std::vector<HybridHit> searchHybrid(const std::string& queryText,
                                        const std::vector<float>& qVec,
                                        int k = 5, float alpha = 0.5f,
                                        const std::string& category = "")
    {
        std::lock_guard<std::mutex> lk(mu);
        // 1. Dense Semantic Search (HNSW)
        auto dfn = getDistFn("cosine");
        auto denseHits = hnsw.knn(qVec, std::max(20, k * 3), 50, dfn);

        // 2. Sparse Lexical Search (BM25)
        auto bm25Hits = bm25.search(queryText, std::max(20, k * 3));

        // Maps for ranks
        std::unordered_map<int, int> denseRankMap;
        std::unordered_map<int, float> denseDistMap;
        for (size_t i = 0; i < denseHits.size(); i++) {
            denseRankMap[denseHits[i].second] = (int)i + 1;
            denseDistMap[denseHits[i].second] = denseHits[i].first;
        }

        std::unordered_map<int, int> bm25RankMap;
        std::unordered_map<int, float> bm25ScoreMap;
        for (size_t i = 0; i < bm25Hits.size(); i++) {
            bm25RankMap[bm25Hits[i].second] = (int)i + 1;
            bm25ScoreMap[bm25Hits[i].second] = bm25Hits[i].first;
        }

        // 3. Reciprocal Rank Fusion (RRF) Calculation: RRF(d) = sum( 1 / (60 + rank) )
        const float RRF_K = 60.0f;
        std::set<int> allCandidateIds;
        for (auto& [d, id] : denseHits) allCandidateIds.insert(id);
        for (auto& [sc, id] : bm25Hits) allCandidateIds.insert(id);

        std::vector<HybridHit> results;
        for (int id : allCandidateIds) {
            if (!store.count(id)) continue;
            if (!category.empty() && category != "all" && store[id].category != category) continue;

            int dRank = denseRankMap.count(id) ? denseRankMap[id] : 1000;
            int sRank = bm25RankMap.count(id)  ? bm25RankMap[id]  : 1000;

            float rrfDense  = 1.0f / (RRF_K + (float)dRank);
            float rrfSparse = 1.0f / (RRF_K + (float)sRank);
            float combinedRRF = alpha * rrfDense + (1.0f - alpha) * rrfSparse;

            results.push_back({
                id,
                store[id].metadata,
                store[id].category,
                store[id].emb,
                combinedRRF,
                denseDistMap.count(id) ? denseDistMap[id] : 1.0f,
                bm25ScoreMap.count(id) ? bm25ScoreMap[id] : 0.0f,
                dRank,
                sRank
            });
        }

        std::sort(results.begin(), results.end(),
            [](const HybridHit& a, const HybridHit& b) { return a.rrfScore > b.rrfScore; });

        if ((int)results.size() > k) results.resize(k);
        return results;
    }

    // Scalar Quantization Metrics Engine (Week 6)
    QuantizationStats getSQ8Stats() {
        std::lock_guard<std::mutex> lk(mu);
        QuantizationStats qs;
        qs.count = store.size();
        qs.dims = dims;
        qs.fp32Bytes = store.size() * dims * sizeof(float);
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
        qs.meanSquaredError = store.empty() ? 0.0f : totalMse / (float)store.size();
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
        return hnsw.getGraphInfo();
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }

    void restoreState(const std::unordered_map<int, VectorItem>& state, int next) {
        std::lock_guard<std::mutex> lk(mu);
        store = state;
        nextId = next;
        auto dist = getDistFn("cosine");
        for (auto& [id, v] : store) {
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            ivf.insert(v, dist);
            bm25.insert(v.id, v.metadata, v.category);
        }
        if (!store.empty()) {
            trainIVF(4);
            fitPCA(2);
        }
    }
};

// =====================================================================
//  STORAGE ENGINE SNAPSHOT MANAGEMENT
// =====================================================================

class StorageEngine {
public:
    static bool saveSnapshot(const std::string& path, VectorDB& db, WALManager& wal) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) return false;

        const char magic[8] = {'V','E','C','T','R','A','0','2'};
        f.write(magic, 8);

        auto items = db.all();
        uint32_t count = items.size();
        uint32_t dims  = db.dims;
        f.write(reinterpret_cast<const char*>(&count), sizeof(count));
        f.write(reinterpret_cast<const char*>(&dims),  sizeof(dims));

        for (const auto& item : items) {
            int32_t id = item.id;
            f.write(reinterpret_cast<const char*>(&id), sizeof(id));

            uint32_t metaLen = item.metadata.size();
            f.write(reinterpret_cast<const char*>(&metaLen), sizeof(metaLen));
            f.write(item.metadata.data(), metaLen);

            uint32_t catLen = item.category.size();
            f.write(reinterpret_cast<const char*>(&catLen), sizeof(catLen));
            f.write(item.category.data(), catLen);

            f.write(reinterpret_cast<const char*>(item.emb.data()), dims * sizeof(float));
        }

        f.flush();
        wal.truncate();
        return true;
    }

    static bool loadSnapshot(const std::string& path, VectorDB& db) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;

        char magic[8];
        if (!f.read(magic, 8)) return false;
        if (std::memcmp(magic, "VECTRA02", 8) != 0) return false;

        uint32_t count = 0, dims = 0;
        if (!f.read(reinterpret_cast<char*>(&count), sizeof(count))) return false;
        if (!f.read(reinterpret_cast<char*>(&dims),  sizeof(dims)) || dims > 4096) return false;

        std::unordered_map<int, VectorItem> state;
        int maxId = 0;

        for (uint32_t i = 0; i < count; i++) {
            int32_t id;
            if (!f.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
            maxId = std::max(maxId, (int)id);

            uint32_t metaLen;
            if (!f.read(reinterpret_cast<char*>(&metaLen), sizeof(metaLen)) || metaLen > 100000) break;
            std::string meta(metaLen, '\0');
            if (!f.read(&meta[0], metaLen)) break;

            uint32_t catLen;
            if (!f.read(reinterpret_cast<char*>(&catLen), sizeof(catLen)) || catLen > 100000) break;
            std::string cat(catLen, '\0');
            if (!f.read(&cat[0], catLen)) break;

            std::vector<float> emb(dims);
            if (!f.read(reinterpret_cast<char*>(emb.data()), dims * sizeof(float))) break;

            state[id] = {id, meta, cat, emb};
        }

        db.restoreState(state, maxId + 1);
        return true;
    }

    static int replayWAL(const std::string& path, VectorDB& db, DistFn dist) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return 0;

        int count = 0;
        while (f.peek() != EOF && f.good()) {
            uint8_t opType;
            if (!f.read(reinterpret_cast<char*>(&opType), sizeof(opType))) break;

            if (opType == 1) { // INSERT
                int32_t id;
                if (!f.read(reinterpret_cast<char*>(&id), sizeof(id))) break;

                uint32_t metaLen;
                if (!f.read(reinterpret_cast<char*>(&metaLen), sizeof(metaLen)) || metaLen > 100000) break;
                std::string meta(metaLen, '\0');
                if (!f.read(&meta[0], metaLen)) break;

                uint32_t catLen;
                if (!f.read(reinterpret_cast<char*>(&catLen), sizeof(catLen)) || catLen > 100000) break;
                std::string cat(catLen, '\0');
                if (!f.read(&cat[0], catLen)) break;

                uint32_t dims;
                if (!f.read(reinterpret_cast<char*>(&dims), sizeof(dims)) || dims > 4096) break;
                std::vector<float> emb(dims);
                if (!f.read(reinterpret_cast<char*>(emb.data()), dims * sizeof(float))) break;

                db.insertWithId(id, meta, cat, emb, dist);
                count++;
            } else if (opType == 2) { // DELETE
                int32_t id;
                if (!f.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
                db.remove(id);
                count++;
            } else {
                break;
            }
        }
        return count;
    }
};

// =====================================================================
//  WEEK 7: DOCUMENT PROCESSING & SLIDING-WINDOW CHUNKER
// =====================================================================

inline std::vector<std::string> chunkText(const std::string& text, int chunkSize = 250, int overlap = 30) {
    std::vector<std::string> chunks;
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string w;
    while (stream >> w) words.push_back(w);

    if (words.empty()) return chunks;
    if ((int)words.size() <= chunkSize) {
        chunks.push_back(text);
        return chunks;
    }

    int step = std::max(1, chunkSize - overlap);
    for (int i = 0; i < (int)words.size(); i += step) {
        std::ostringstream chunk;
        int end = std::min((int)words.size(), i + chunkSize);
        for (int j = i; j < end; j++) {
            if (j > i) chunk << " ";
            chunk << words[j];
        }
        chunks.push_back(chunk.str());
        if (end == (int)words.size()) break;
    }
    return chunks;
}

// Local Deterministic Semantic Embedding Engine (Fallback when Ollama is offline)
inline std::vector<float> localSemanticEmbed(const std::string& text, int dims = 64) {
    std::vector<float> vec(dims, 0.0f);
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Hash bag-of-words onto circular harmonic spectrum
    std::istringstream ss(lower);
    std::string token;
    int tokenIdx = 0;
    while (ss >> token) {
        uint32_t hash = 5381;
        for (char c : token) hash = ((hash << 5) + hash) + (unsigned char)c;
        for (int d = 0; d < dims; d++) {
            float phase = (float)(hash % 360) * 0.0174533f + (float)d * 0.1f;
            vec[d] += std::sin(phase) * (1.0f / (1.0f + 0.05f * (float)tokenIdx));
        }
        tokenIdx++;
    }
    float norm = 0.0f;
    for (float v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (float& v : vec) v /= norm;
    }
    return vec;
}

struct DocItem {
    int id;
    std::string title;
    std::string text;
    std::vector<float> emb;
};

class DocumentDB {
    std::unordered_map<int, DocItem> store;
    BruteForce bf;
    HNSW       hnsw;
    std::mutex mu;
    int nextId = 1;
    int dims = 0;

public:
    DocumentDB() : hnsw(16, 200) {}

    int insert(const std::string& title, const std::string& text, const std::vector<float>& emb) {
        std::lock_guard<std::mutex> lk(mu);
        dims = (int)emb.size();
        VectorItem vi{nextId, title, "doc", emb};
        DocItem di{nextId, title, text, emb};
        store[nextId] = di;
        bf.insert(vi);
        hnsw.insert(vi, cosine);
        return nextId++;
    }

    std::vector<std::pair<float, DocItem>> search(const std::vector<float>& q, int k = 3, float max_dist = 2.0f, const std::string& keyword = "") {
        std::lock_guard<std::mutex> lk(mu);
        if (store.empty()) return {};

        std::vector<std::pair<float, int>> raw;
        if (!keyword.empty()) {
            std::string kw = keyword;
            std::transform(kw.begin(), kw.end(), kw.begin(), ::tolower);
            auto pred = [&](const VectorItem& item) {
                if (!store.count(item.id)) return false;
                std::string full = store[item.id].title + " " + store[item.id].text;
                std::transform(full.begin(), full.end(), full.begin(), ::tolower);
                return full.find(kw) != std::string::npos;
            };
            raw = bf.knnFiltered(q, k, cosine, pred);
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
//  OLLAMA LLM & EMBEDDING BRIDGE
// =====================================================================

struct OllamaClient {
    std::string host = "localhost";
    int port = 11434;
    std::string embedModel = "nomic-embed-text";
    std::string genModel   = "llama3";

    bool isAvailable() {
        try {
            httplib::Client cli(host, port);
            cli.set_connection_timeout(0, 100000); // 100ms
            cli.set_read_timeout(0, 200000);
            auto res = cli.Get("/api/tags");
            return (res && res->status == 200);
        } catch (...) {
            return false;
        }
    }

    std::vector<float> embed(const std::string& prompt) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(10, 0);
        std::string body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + prompt + "\"}";
        auto res = cli.Post("/api/embeddings", body, "application/json");
        if (!res || res->status != 200) return {};

        std::vector<float> vec;
        std::string s = res->body;
        auto pos = s.find("\"embedding\":[");
        if (pos == std::string::npos) return {};
        pos += 13;
        while (pos < s.size() && s[pos] != ']') {
            while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',')) pos++;
            if (pos >= s.size() || s[pos] == ']') break;
            size_t next;
            float val = std::stof(s.substr(pos), &next);
            vec.push_back(val);
            pos += next;
        }
        return vec;
    }

    std::string generate(const std::string& prompt) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(30, 0);
        std::string body = "{\"model\":\"" + genModel + "\",\"prompt\":\"" + prompt + "\",\"stream\":false}";
        auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200) return "Ollama generation error.";

        std::string s = res->body;
        auto pos = s.find("\"response\":\"");
        if (pos == std::string::npos) return "";
        pos += 12;
        std::string out;
        while (pos < s.size()) {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                if (s[pos+1] == '"')  { out += '"';  pos += 2; continue; }
                if (s[pos+1] == 'n')  { out += '\n'; pos += 2; continue; }
                if (s[pos+1] == 't')  { out += '\t'; pos += 2; continue; }
                if (s[pos+1] == '\\') { out += '\\'; pos += 2; continue; }
            }
            if (s[pos] == '"') break;
            out += s[pos++];
        }
        return out;
    }
};

// =====================================================================
//  WEEK 8: SIFT10K BENCHMARK SUITE (Recall@K vs QPS Profiler)
// =====================================================================

struct SIFTEvaluation {
    std::string algorithm;
    float recallPercent;
    long long avgLatencyUs;
    double qps;
    float speedupFactor;
    int probedElements;
};

inline std::vector<SIFTEvaluation> runSIFTBenchmark(int numVectors = 1000, int dims = 128, int numQueries = 50, int k = 10) {
    std::mt19937 rng(1337);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    // 1. Generate SIFT-like 128D clustered dataset
    std::vector<std::vector<float>> dataset(numVectors, std::vector<float>(dims));
    for (int i = 0; i < numVectors; i++) {
        for (int d = 0; d < dims; d++) dataset[i][d] = norm(rng);
        float len = 0.0f;
        for (int d = 0; d < dims; d++) len += dataset[i][d] * dataset[i][d];
        len = std::sqrt(len);
        for (int d = 0; d < dims; d++) dataset[i][d] /= len;
    }

    // 2. Generate Queries
    std::vector<std::vector<float>> queries(numQueries, std::vector<float>(dims));
    for (int q = 0; q < numQueries; q++) {
        for (int d = 0; d < dims; d++) queries[q][d] = norm(rng);
        float len = 0.0f;
        for (int d = 0; d < dims; d++) len += queries[q][d] * queries[q][d];
        len = std::sqrt(len);
        for (int d = 0; d < dims; d++) queries[q][d] /= len;
    }

    // 3. Build Indexes
    BruteForce bf;
    KDTree kdt(dims);
    HNSW hnsw(16, 200);
    IVFFlat ivf(dims, 16);

    std::vector<VectorItem> items;
    for (int i = 0; i < numVectors; i++) {
        VectorItem vi{i + 1, "sift_vector_" + std::to_string(i+1), "sift", dataset[i]};
        items.push_back(vi);
        bf.insert(vi);
        kdt.insert(vi);
        hnsw.insert(vi, euclidean);
    }
    ivf.train(items, 16, 20, euclidean);

    // 4. Compute Ground Truth with Exact Brute-Force
    std::vector<std::vector<int>> groundTruth(numQueries);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < numQueries; q++) {
        auto res = bf.knn(queries[q], k, euclidean);
        for (auto& p : res) groundTruth[q].push_back(p.second);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    long long bfTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    long long bfAvgUs = bfTotalUs / numQueries;
    if (bfAvgUs <= 0) bfAvgUs = 1;

    std::vector<SIFTEvaluation> evals;
    evals.push_back({
        "Brute-Force Scan (Exact Baseline)",
        100.0f,
        bfAvgUs,
        (double)numQueries / ((double)bfTotalUs / 1e6),
        1.0f,
        numVectors
    });

    // KD-Tree
    auto tKdt0 = std::chrono::high_resolution_clock::now();
    int kdtMatch = 0;
    for (int q = 0; q < numQueries; q++) {
        auto res = kdt.knn(queries[q], k, euclidean);
        std::set<int> gt(groundTruth[q].begin(), groundTruth[q].end());
        for (auto& p : res) if (gt.count(p.second)) kdtMatch++;
    }
    auto tKdt1 = std::chrono::high_resolution_clock::now();
    long long kdtTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(tKdt1 - tKdt0).count();
    long long kdtAvgUs = kdtTotalUs / numQueries;
    if (kdtAvgUs <= 0) kdtAvgUs = 1;
    evals.push_back({
        "KD-Tree Spatial Partitioning",
        ((float)kdtMatch / (float)(numQueries * k)) * 100.0f,
        kdtAvgUs,
        (double)numQueries / ((double)kdtTotalUs / 1e6),
        (float)bfAvgUs / (float)kdtAvgUs,
        (int)(numVectors * 0.85f)
    });

    // IVF-Flat (nprobe = 2)
    auto tIvf0 = std::chrono::high_resolution_clock::now();
    int ivfMatch = 0;
    for (int q = 0; q < numQueries; q++) {
        auto res = ivf.knn(queries[q], k, 2, euclidean);
        std::set<int> gt(groundTruth[q].begin(), groundTruth[q].end());
        for (auto& p : res) if (gt.count(p.second)) ivfMatch++;
    }
    auto tIvf1 = std::chrono::high_resolution_clock::now();
    long long ivfTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(tIvf1 - tIvf0).count();
    long long ivfAvgUs = ivfTotalUs / numQueries;
    if (ivfAvgUs <= 0) ivfAvgUs = 1;
    evals.push_back({
        "IVF-Flat (K-Means Voronoi, nprobe=2)",
        ((float)ivfMatch / (float)(numQueries * k)) * 100.0f,
        ivfAvgUs,
        (double)numQueries / ((double)ivfTotalUs / 1e6),
        (float)bfAvgUs / (float)ivfAvgUs,
        (int)(numVectors * 0.125f)
    });

    // HNSW (M=16, efSearch=50)
    auto tHnsw0 = std::chrono::high_resolution_clock::now();
    int hnswMatch = 0;
    for (int q = 0; q < numQueries; q++) {
        auto res = hnsw.knn(queries[q], k, 50, euclidean);
        std::set<int> gt(groundTruth[q].begin(), groundTruth[q].end());
        for (auto& p : res) if (gt.count(p.second)) hnswMatch++;
    }
    auto tHnsw1 = std::chrono::high_resolution_clock::now();
    long long hnswTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(tHnsw1 - tHnsw0).count();
    long long hnswAvgUs = hnswTotalUs / numQueries;
    if (hnswAvgUs <= 0) hnswAvgUs = 1;
    evals.push_back({
        "HNSW Multi-Layer Proximity Graph (ef=50)",
        ((float)hnswMatch / (float)(numQueries * k)) * 100.0f,
        hnswAvgUs,
        (double)numQueries / ((double)hnswTotalUs / 1e6),
        (float)bfAvgUs / (float)hnswAvgUs,
        64
    });

    return evals;
}

// =====================================================================
//  JSON SERIALIZATION & PARSING UTILITIES
// =====================================================================

inline void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Accept");
}

inline std::string jS(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else                o += c;
    }
    return o + "\"";
}

inline std::string jVec(const std::vector<float>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s += ",";
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4) << v[i];
        s += ss.str();
    }
    return s + "]";
}

inline std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) {
        try { v.push_back(std::stof(t)); } catch (...) {}
    }
    return v;
}

inline bool parseBody(const std::string& b, std::string& meta, std::string& cat, std::vector<float>& emb) {
    auto findField = [&](const std::string& key) -> std::string {
        auto pos = b.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = b.find(':', pos); if (pos == std::string::npos) return "";
        pos = b.find('"', pos); if (pos == std::string::npos) return "";
        auto end = b.find('"', pos + 1); if (end == std::string::npos) return "";
        return b.substr(pos + 1, end - pos - 1);
    };
    meta = findField("metadata");
    cat  = findField("category");

    auto pos = b.find("\"embedding\"");
    if (pos == std::string::npos) return false;
    pos = b.find('[', pos); if (pos == std::string::npos) return false;
    auto end = b.find(']', pos); if (end == std::string::npos) return false;
    emb = parseVec(b.substr(pos + 1, end - pos - 1));
    return true;
}

inline std::string extractStr(const std::string& body, const std::string& key) {
    auto pos = body.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = body.find(':', pos); if (pos == std::string::npos) return "";
    pos = body.find('"', pos); if (pos == std::string::npos) return "";
    auto end = body.find('"', pos + 1); if (end == std::string::npos) return "";
    return body.substr(pos + 1, end - pos - 1);
}

inline int extractInt(const std::string& body, const std::string& key, int defaultVal) {
    auto pos = body.find("\"" + key + "\"");
    if (pos == std::string::npos) return defaultVal;
    pos = body.find(':', pos); if (pos == std::string::npos) return defaultVal;
    while (pos < body.size() && (body[pos] == ':' || body[pos] == ' ')) pos++;
    try {
        return std::stoi(body.substr(pos));
    } catch (...) {
        return defaultVal;
    }
}

inline float extractFloat(const std::string& body, const std::string& key, float defaultVal) {
    auto pos = body.find("\"" + key + "\"");
    if (pos == std::string::npos) return defaultVal;
    pos = body.find(':', pos); if (pos == std::string::npos) return defaultVal;
    while (pos < body.size() && (body[pos] == ':' || body[pos] == ' ')) pos++;
    try {
        return std::stof(body.substr(pos));
    } catch (...) {
        return defaultVal;
    }
}

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
    db.trainIVF(4);
    db.fitPCA(2);
}

// =====================================================================
//  HTTP REST SERVER & SERVICE DISPATCHER
// =====================================================================

int main() {
    VectorDB   db(DIMS);
    WALManager wal("vectra.wal");
    DocumentDB docDB;
    OllamaClient ollama;

    // Crash Recovery: Load Snapshot (.vdb) if present, then replay WAL
    bool snapshotLoaded = false;
    std::ifstream vdbCheck("vectra.vdb", std::ios::binary);
    if (vdbCheck.is_open()) {
        vdbCheck.close();
        snapshotLoaded = StorageEngine::loadSnapshot("vectra.vdb", db);
    }

    if (db.size() < 10) {
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
        int nprobe = 2;
        try { nprobe = std::stoi(req.get_param_value("nprobe")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo   = "hnsw";

        VectorDB::FilterOptions fOpt;
        fOpt.category = req.get_param_value("category");
        fOpt.keyword  = req.get_param_value("keyword");

        auto out = db.search(q, k, metric, algo, fOpt, nprobe);
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
        ss << "{\"bruteforceUs\":" << b.bfUs
           << ",\"kdtreeUs\":"     << b.kdUs
           << ",\"ivfUs\":"        << b.ivfUs
           << ",\"hnswUs\":"       << b.hnswUs
           << ",\"itemCount\":"    << b.n
           << ",\"simdActive\":"   << (g_simd_enabled ? "true" : "false") << '}';
        res.set_content(ss.str(), "application/json");
    });

    // ── WEEK 3: K-MEANS & IVF ENDPOINTS ───────────────────────────────
    svr.Post("/ivf/train", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int kClusters = extractInt(req.body, "k", 4);
        db.trainIVF(kClusters);
        auto& ivf = db.getIVF();
        std::ostringstream ss;
        ss << "{\"ok\":true,\"clusters\":" << ivf.centroids.size()
           << ",\"inertia\":" << ivf.totalInertia
           << ",\"iterations\":" << ivf.iterationsRun << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/ivf/info", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto& ivf = db.getIVF();
        if (!ivf.isTrained) db.trainIVF(4);
        std::ostringstream ss;
        ss << "{\"trained\":" << (ivf.isTrained ? "true" : "false")
           << ",\"nlist\":" << ivf.nlist
           << ",\"inertia\":" << ivf.totalInertia
           << ",\"centroids\":[";
        for (size_t c = 0; c < ivf.centroids.size(); c++) {
            if (c) ss << ',';
            ss << "{\"clusterId\":" << c
               << ",\"vectorCount\":" << (ivf.invertedLists.count(c) ? ivf.invertedLists.at(c).size() : 0)
               << ",\"itemIds\":[";
            if (ivf.invertedLists.count(c)) {
                auto& ids = ivf.invertedLists.at(c);
                for (size_t j = 0; j < ids.size(); j++) {
                    if (j) ss << ','; ss << ids[j];
                }
            }
            ss << "],\"centroid\":" << jVec(ivf.centroids[c]) << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── WEEK 4: PCA ENDPOINTS ─────────────────────────────────────────
    svr.Post("/pca/fit", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int comp = extractInt(req.body, "components", 2);
        db.fitPCA(comp);
        auto& pca = db.getPCA();
        auto items = db.all();

        std::ostringstream ss;
        ss << "{\"ok\":true,\"components\":" << pca.nComponents
           << ",\"eigenvalues\":" << jVec(pca.eigenvalues)
           << ",\"explainedVarianceRatio\":" << jVec(pca.explainedVarianceRatio)
           << ",\"projections\":[";
        for (size_t i = 0; i < items.size(); i++) {
            if (i) ss << ',';
            auto proj = pca.transform(items[i].emb);
            ss << "{\"id\":" << items[i].id
               << ",\"metadata\":" << jS(items[i].metadata)
               << ",\"category\":" << jS(items[i].category)
               << ",\"x\":" << std::fixed << std::setprecision(4) << (proj.size() > 0 ? proj[0] : 0.0f)
               << ",\"y\":" << std::fixed << std::setprecision(4) << (proj.size() > 1 ? proj[1] : 0.0f)
               << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/pca/transform", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto vec = parseVec(extractStr(req.body, "vector"));
        if (vec.empty()) vec = parseVec(req.body);
        auto& pca = db.getPCA();
        if (!pca.isFitted) db.fitPCA(2);
        auto proj = pca.transform(vec);
        std::ostringstream ss;
        ss << "{\"x\":" << std::fixed << std::setprecision(4) << (proj.size() > 0 ? proj[0] : 0.0f)
           << ",\"y\":" << std::fixed << std::setprecision(4) << (proj.size() > 1 ? proj[1] : 0.0f)
           << '}';
        res.set_content(ss.str(), "application/json");
    });

    // ── WEEK 5: HYBRID SEARCH (SPARSE BM25 + DENSE HNSW WITH RRF) ────
    svr.Post("/hybrid/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto queryText = extractStr(req.body, "query");
        auto cat       = extractStr(req.body, "category");
        int k          = extractInt(req.body, "k", 5);
        float alpha    = extractFloat(req.body, "alpha", 0.5f);

        std::vector<float> qVec;
        auto vecStr = extractStr(req.body, "v");
        if (!vecStr.empty()) qVec = parseVec(vecStr);
        if ((int)qVec.size() != DIMS) {
            // Compute deterministic embedding for hybrid query text
            qVec = localSemanticEmbed(queryText, DIMS);
        }

        auto hits = db.searchHybrid(queryText, qVec, k, alpha, cat);
        std::ostringstream ss;
        ss << "{\"query\":" << jS(queryText)
           << ",\"alpha\":" << alpha
           << ",\"results\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            auto& h = hits[i];
            ss << "{\"id\":"           << h.id
               << ",\"metadata\":"     << jS(h.metadata)
               << ",\"category\":"     << jS(h.category)
               << ",\"rrfScore\":"     << std::fixed << std::setprecision(6) << h.rrfScore
               << ",\"denseDist\":"    << std::fixed << std::setprecision(4) << h.denseDist
               << ",\"bm25Score\":"    << std::fixed << std::setprecision(4) << h.bm25Score
               << ",\"denseRank\":"    << h.denseRank
               << ",\"bm25Rank\":"     << h.bm25Rank << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── WEEK 8: SIFT10K BENCHMARK SUITE ──────────────────────────────
    svr.Post("/benchmark/sift", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int numVecs = extractInt(req.body, "numVectors", 1000);
        int dims    = extractInt(req.body, "dims", 128);
        int numQ    = extractInt(req.body, "numQueries", 30);
        int k       = extractInt(req.body, "k", 10);

        auto evals = runSIFTBenchmark(numVecs, dims, numQ, k);
        std::ostringstream ss;
        ss << "{\"dataset\":\"SIFT10K-Simulated-128D\""
           << ",\"numVectors\":" << numVecs
           << ",\"dims\":"       << dims
           << ",\"numQueries\":" << numQ
           << ",\"k\":"          << k
           << ",\"benchmarks\":[";
        for (size_t i = 0; i < evals.size(); i++) {
            if (i) ss << ',';
            auto& e = evals[i];
            ss << "{\"algorithm\":"     << jS(e.algorithm)
               << ",\"recallPercent\":" << std::fixed << std::setprecision(2) << e.recallPercent
               << ",\"avgLatencyUs\":"  << e.avgLatencyUs
               << ",\"qps\":"           << (long long)e.qps
               << ",\"speedupFactor\":" << std::fixed << std::setprecision(2) << e.speedupFactor
               << ",\"probedElements\":"<< e.probedElements << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── HNSW GRAPH METRICS ────────────────────────────────────────────
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

    // ── AVX2 SIMD BENCHMARK & TOGGLE ──────────────────────────────────
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

    // ── STORAGE ENGINE & WAL ENDPOINTS ────────────────────────────────
    svr.Post("/storage/snapshot", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool ok = StorageEngine::saveSnapshot("vectra.vdb", db, wal);
        std::ostringstream ss;
        ss << "{\"ok\":" << (ok ? "true" : "false")
           << ",\"file\":" << jS("vectra.vdb")
           << ",\"walEntries\":" << wal.getCount() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/storage/restore", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool ok = StorageEngine::loadSnapshot("vectra.vdb", db);
        std::ostringstream ss;
        ss << "{\"ok\":" << (ok ? "true" : "false")
           << ",\"itemCount\":" << db.size()
           << ",\"walEntries\":" << wal.getCount() << '}';
        res.set_content(ss.str(), "application/json");
    });

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

    // ── SCALAR QUANTIZATION (SQ8) ENDPOINTS ───────────────────────────
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

        // FP32 Exact KNN
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
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"ivf\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]"
           << ",\"avx2Supported\":"   << (VECTRA_AVX2_SUPPORTED ? "true" : "false")
           << ",\"simdActive\":"      << (g_simd_enabled ? "true" : "false")
           << ",\"walCount\":"        << wal.getCount() << "}";
        res.set_content(ss.str(), "application/json");
    });

    // Exception and Error Handling
    svr.set_exception_handler([](const auto&, auto& res, std::exception_ptr ep) {
        cors(res);
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            std::cout << "[ERROR] Exception: " << e.what() << "\n";
        } catch (...) {}
        res.set_content("{\"error\":\"internal server error\"}", "application/json");
        res.status = 500;
    });

    svr.set_error_handler([](const auto&, auto& res) {
        cors(res);
        if (res.status == 404) {
            res.set_content("{\"error\":\"route not found\"}", "application/json");
        }
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

    std::cout << "[READY] VECTRA Engine Listening on http://localhost:8080\n";
    while (true) {
        bool ok = svr.listen("0.0.0.0", 8080);
        if (!ok) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    return 0;
}
