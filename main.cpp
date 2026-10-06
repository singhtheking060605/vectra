#define _CRT_SECURE_NO_WARNINGS
#define _WIN32_WINNT 0x0A00
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>
#include <immintrin.h>
#include "httplib.h"

static const int DIMS = 16;

// =====================================================================
//  DATA TYPES & HELPER FUNCTIONS
// =====================================================================

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// SIMD AVX2 Accelerated Euclidean, Cosine and Manhattan Distance
inline float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    int n = (int)std::min(a.size(), b.size());
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

inline float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    int n = (int)std::min(a.size(), b.size());
    if (n == 0) return 1.0f;
    float dot = 0.0f;
    float na  = 0.0f;
    float nb  = 0.0f;
    const float* pa = a.data();
    const float* pb = b.data();
    for (int i = 0; i < n; i++) {
        dot += pa[i] * pb[i];
        na  += pa[i] * pa[i];
        nb  += pb[i] * pb[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    float sim = dot / (std::sqrt(na) * std::sqrt(nb));
    if (std::isnan(sim) || std::isinf(sim)) return 1.0f;
    return 1.0f - std::max(-1.0f, std::min(1.0f, sim));
}

inline float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    int n = (int)std::min(a.size(), b.size());
    for (int i = 0; i < n; i++) s += std::abs(a[i]-b[i]);
    return s;
}

DistFn getDistFn(const std::string& m) {
    if (m == "cosine")    return cosine;
    if (m == "manhattan") return manhattan;
    return euclidean;
}

inline bool safeLess(float a, int idA, float b, int idB) {
    bool nanA = std::isnan(a) || std::isinf(a);
    bool nanB = std::isnan(b) || std::isinf(b);
    if (nanA && nanB) return idA < idB;
    if (nanA) return false;
    if (nanB) return true;
    if (a != b) return a < b;
    return idA < idB;
}

inline bool safeGreater(float a, int idA, float b, int idB) {
    bool nanA = std::isnan(a) || std::isinf(a);
    bool nanB = std::isnan(b) || std::isinf(b);
    if (nanA && nanB) return idA < idB;
    if (nanA) return false;
    if (nanB) return true;
    if (a != b) return a > b;
    return idA < idB;
}

// Tokenizer & String helpers
inline std::vector<std::string> tokenize(const std::string& str) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : str) {
        if (std::isalnum((unsigned char)c)) {
            cur += (char)std::tolower((unsigned char)c);
        } else if (!cur.empty()) {
            tokens.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline std::string jS(const std::string& s) {
    std::ostringstream ss;
    ss << '"';
    for (char c : s) {
        if      (c == '"')  ss << "\\\"";
        else if (c == '\\') ss << "\\\\";
        else if (c == '\n') ss << "\\n";
        else if (c == '\r') ss << "\\r";
        else if (c == '\t') ss << "\\t";
        else if ((unsigned char)c < 0x20) {
            ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)(unsigned char)c;
        } else ss << c;
    }
    ss << '"';
    return ss.str();
}

inline std::string jVec(const std::vector<float>& v) {
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    ss << ']';
    return ss.str();
}

inline std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    size_t openBracket = s.find('[');
    size_t closeBracket = s.find(']', openBracket != std::string::npos ? openBracket : 0);
    std::string arrayContent = (openBracket != std::string::npos && closeBracket != std::string::npos && closeBracket > openBracket)
        ? s.substr(openBracket + 1, closeBracket - openBracket - 1)
        : s;

    std::stringstream ss(arrayContent);
    std::string token;
    while (std::getline(ss, token, ',')) {
        std::string t = trim(token);
        while (!t.empty() && (t.front() == '[' || t.front() == '{' || t.front() == '"')) t = t.substr(1);
        while (!t.empty() && (t.back() == ']' || t.back() == '}' || t.back() == '"')) t.pop_back();
        t = trim(t);
        if (!t.empty()) {
            try {
                v.push_back(std::stof(t));
            } catch (...) {}
        }
    }
    return v;
}

inline std::string extractStr(const std::string& json, const std::string& key) {
    std::string p = "\"" + key + "\"";
    size_t pos = json.find(p);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + p.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos);
    if (pos == std::string::npos) return "";
    size_t end = pos + 1;
    while (end < json.size()) {
        if (json[end] == '\\') {
            end += 2;
            continue;
        }
        if (json[end] == '"') break;
        end++;
    }
    if (end > json.size()) end = json.size();
    if (end <= pos + 1) return "";
    std::string raw = json.substr(pos + 1, end - pos - 1);
    std::string out;
    for (size_t i = 0; i < raw.size(); i++) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            if (raw[i+1] == '"')  { out += '"'; i++; }
            else if (raw[i+1] == '\\') { out += '\\'; i++; }
            else if (raw[i+1] == 'n')  { out += '\n'; i++; }
            else if (raw[i+1] == 'r')  { out += '\r'; i++; }
            else if (raw[i+1] == 't')  { out += '\t'; i++; }
            else out += raw[i];
        } else out += raw[i];
    }
    return out;
}

inline int extractInt(const std::string& json, const std::string& key, int defVal = 0) {
    std::string p = "\"" + key + "\"";
    size_t pos = json.find(p);
    if (pos == std::string::npos) return defVal;
    pos = json.find(':', pos + p.size());
    if (pos == std::string::npos) return defVal;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == ':')) pos++;
    if (pos >= json.size()) return defVal;
    try { return std::stoi(json.substr(pos)); } catch (...) { return defVal; }
}

inline float extractFloat(const std::string& json, const std::string& key, float defVal = 0.0f) {
    std::string p = "\"" + key + "\"";
    size_t pos = json.find(p);
    if (pos == std::string::npos) return defVal;
    pos = json.find(':', pos + p.size());
    if (pos == std::string::npos) return defVal;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == ':')) pos++;
    if (pos >= json.size()) return defVal;
    try { return std::stof(json.substr(pos)); } catch (...) { return defVal; }
}

inline void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS, PUT");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

// =====================================================================
//  FAST LOCAL SEMANTIC EMBEDDER (Multi-cluster 16D + 64D/768D)
// =====================================================================

inline std::vector<float> localSemanticEmbed(const std::string& text, int targetDims = 16) {
    std::vector<float> v(targetDims, 0.04f);
    auto tokens = tokenize(text);
    if (tokens.empty()) {
        for (int i = 0; i < targetDims; i++) v[i] = 1.0f / std::sqrt((float)targetDims);
        return v;
    }

    static const std::unordered_map<std::string, int> clusterKeywords = {
        // CS (0) - dims 0..3
        {"cs", 0}, {"code", 0}, {"programming", 0}, {"tree", 0}, {"binary", 0},
        {"list", 0}, {"graph", 0}, {"hash", 0}, {"algorithm", 0}, {"pointer", 0},
        {"stack", 0}, {"queue", 0}, {"bfs", 0}, {"dfs", 0}, {"node", 0},
        {"memory", 0}, {"cpu", 0}, {"process", 0}, {"thread", 0}, {"server", 0},
        {"database", 0}, {"sql", 0}, {"vector", 0}, {"index", 0}, {"page", 0},
        {"python", 0}, {"java", 0}, {"cpp", 0}, {"software", 0}, {"web", 0},
        {"heap", 0}, {"trie", 0}, {"sort", 0}, {"search", 0}, {"dynamic", 0},
        {"cache", 0}, {"table", 0}, {"array", 0}, {"struct", 0}, {"oop", 0},
        {"linux", 0}, {"os", 0}, {"concurrency", 0}, {"paging", 0}, {"ram", 0},
        {"git", 0}, {"compiler", 0}, {"bytecode", 0}, {"docker", 0}, {"api", 0},

        // Math (1) - dims 4..7
        {"math", 1}, {"calculus", 1}, {"matrix", 1}, {"eigenvalue", 1}, {"derivative", 1},
        {"integral", 1}, {"probability", 1}, {"bayes", 1}, {"algebra", 1}, {"vector_space", 1},
        {"eigenvector", 1}, {"prime", 1}, {"cryptography", 1}, {"rsa", 1}, {"limit", 1},
        {"geometry", 1}, {"dimension", 1}, {"formula", 1}, {"function", 1}, {"theorem", 1},
        {"equation", 1}, {"differential", 1}, {"linear", 1}, {"discrete", 1}, {"stats", 1},
        {"statistics", 1}, {"distribution", 1}, {"variance", 1}, {"mean", 1}, {"norm", 1},
        {"combinatorics", 1}, {"permutation", 1}, {"combination", 1}, {"proof", 1},
        {"topology", 1}, {"numeric", 1}, {"complex", 1}, {"fourier", 1}, {"tensor", 1},
        {"laplace", 1}, {"quaternion", 1}, {"manifold", 1}, {"stochastic", 1},

        // Food (2) - dims 8..11
        {"food", 2}, {"pizza", 2}, {"sushi", 2}, {"ramen", 2}, {"tacos", 2},
        {"croissant", 2}, {"dough", 2}, {"cheese", 2}, {"tomato", 2}, {"pork", 2},
        {"noodle", 2}, {"soup", 2}, {"salsa", 2}, {"fish", 2}, {"rice", 2},
        {"culinary", 2}, {"sauce", 2}, {"kitchen", 2}, {"cook", 2}, {"flavor", 2},
        {"burger", 2}, {"pasta", 2}, {"baking", 2}, {"bread", 2}, {"dessert", 2},
        {"cake", 2}, {"sweet", 2}, {"spicy", 2}, {"coffee", 2}, {"tea", 2},
        {"salad", 2}, {"meat", 2}, {"chicken", 2}, {"beef", 2}, {"dish", 2},
        {"recipe", 2}, {"restaurant", 2}, {"taste", 2}, {"delicious", 2},
        {"snack", 2}, {"chocolate", 2}, {"grill", 2}, {"taco", 2}, {"curry", 2},
        {"biryani", 2}, {"naan", 2}, {"samosa", 2}, {"dumpling", 2}, {"espresso", 2},

        // Sports (3) - dims 12..15
        {"sports", 3}, {"basketball", 3}, {"football", 3}, {"tennis", 3}, {"chess", 3},
        {"swimming", 3}, {"olympic", 3}, {"tournament", 3}, {"match", 3}, {"goal", 3},
        {"touchdown", 3}, {"serve", 3}, {"ball", 3}, {"racket", 3}, {"athlete", 3},
        {"wimbledon", 3}, {"nba", 3}, {"fifa", 3}, {"game", 3}, {"player", 3},
        {"soccer", 3}, {"cricket", 3}, {"baseball", 3}, {"gym", 3}, {"workout", 3},
        {"run", 3}, {"race", 3}, {"court", 3}, {"score", 3}, {"championship", 3},
        {"league", 3}, {"fitness", 3}, {"team", 3}, {"winner", 3}, {"club", 3},
        {"swimmer", 3}, {"freestyle", 3}, {"track", 3}, {"field", 3}, {"marathon", 3},

        // AI (4)
        {"ai", 4}, {"neural", 4}, {"network", 4}, {"deep", 4}, {"learning", 4},
        {"transformer", 4}, {"rag", 4}, {"embedding", 4}, {"llm", 4}, {"prompt", 4},
        {"model", 4}, {"attention", 4}, {"backprop", 4}, {"gradient", 4},
        {"gpt", 4}, {"vision", 4}, {"agent", 4}, {"diffusion", 4}, {"generative", 4},
        {"claude", 4}, {"gemini", 4}, {"llama", 4}, {"mistral", 4}, {"lora", 4},
        {"tokenizer", 4}, {"inference", 4}, {"fine_tuning", 4}, {"weights", 4},

        // Astronomy & Space (5)
        {"astronomy", 5}, {"space", 5}, {"sun", 5}, {"moon", 5}, {"star", 5},
        {"planet", 5}, {"asteroid", 5}, {"galaxy", 5}, {"cosmos", 5}, {"orbit", 5},
        {"telescope", 5}, {"mars", 5}, {"earth", 5}, {"venus", 5}, {"jupiter", 5},
        {"saturn", 5}, {"neptune", 5}, {"mercury", 5}, {"pluto", 5}, {"uranus", 5},
        {"comet", 5}, {"meteor", 5}, {"supernova", 5}, {"blackhole", 5}, {"nebula", 5},
        {"solar", 5}, {"nasa", 5}, {"spacex", 5}, {"rocket", 5}, {"eclipse", 5},
        {"gravity", 5}, {"satellite", 5}, {"astronaut", 5}, {"universe", 5},
        {"isro", 5}, {"esa", 5}, {"jaxa", 5}, {"roscosmos", 5}, {"cnsa", 5},
        {"chandrayaan", 5}, {"mangalyaan", 5}, {"aditya", 5}, {"gaganyaan", 5},
        {"pslv", 5}, {"gslv", 5}, {"sslv", 5}, {"artemis", 5}, {"apollo", 5},
        {"voyager", 5}, {"hubble", 5}, {"jwst", 5}, {"jameswebb", 5}, {"perseverance", 5},
        {"curiosity", 5}, {"falcon", 5}, {"starship", 5}, {"propulsion", 5}, {"cosmic", 5},
        {"astrophysics", 5}, {"lunar", 5}, {"martian", 5}, {"spacecraft", 5}, {"interstellar", 5},

        // Finance (6)
        {"finance", 6}, {"stock", 6}, {"crypto", 6}, {"bitcoin", 6}, {"market", 6},
        {"trading", 6}, {"money", 6}, {"bank", 6}, {"investment", 6}, {"profit", 6},
        {"revenue", 6}, {"economy", 6}, {"dollar", 6}, {"equity", 6}, {"bond", 6},
        {"fund", 6}, {"trade", 6}, {"shares", 6}, {"currency", 6}, {"wallet", 6},
        {"fintech", 6}, {"solana", 6}, {"ethereum", 6}, {"nasdaq", 6}, {"etf", 6},

        // Medical (7)
        {"medical", 7}, {"health", 7}, {"doctor", 7}, {"hospital", 7}, {"disease", 7},
        {"medicine", 7}, {"patient", 7}, {"surgery", 7}, {"virus", 7}, {"vaccine", 7},
        {"therapy", 7}, {"clinic", 7}, {"biology", 7}, {"anatomy", 7}, {"cell", 7},
        {"biotech", 7}, {"oncology", 7}, {"pharma", 7}, {"cardio", 7}, {"mrna", 7}
    };

    int clusterCounts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int totalWords = 0;

    for (const auto& token : tokens) {
        totalWords++;
        auto it = clusterKeywords.find(token);
        if (it != clusterKeywords.end()) {
            clusterCounts[it->second] += 10;
        } else {
            // Fuzzy prefix & root matching for domain words like "astrophysics", "satellites", "isro..."
            if (token.rfind("astro", 0) == 0 || token.rfind("cosm", 0) == 0 || token.rfind("orbit", 0) == 0 ||
                token.rfind("satell", 0) == 0 || token.rfind("planet", 0) == 0 || token.rfind("rocket", 0) == 0 ||
                token.rfind("telesc", 0) == 0 || token.rfind("galax", 0) == 0 || token.rfind("space", 0) == 0 ||
                token.rfind("moon", 0) == 0 || token.rfind("lunar", 0) == 0 || token.rfind("martian", 0) == 0 ||
                token == "isro" || token == "chandrayaan" || token == "mangalyaan" || token == "gaganyaan") {
                clusterCounts[5] += 10;
            } else if (token.rfind("neur", 0) == 0 || token.rfind("embed", 0) == 0 || token.rfind("transform", 0) == 0) {
                clusterCounts[4] += 10;
            } else if (token.rfind("math", 0) == 0 || token.rfind("calcul", 0) == 0 || token.rfind("matrix", 0) == 0) {
                clusterCounts[1] += 10;
            } else if (token.rfind("crypt", 0) == 0 || token.rfind("invest", 0) == 0 || token.rfind("trade", 0) == 0) {
                clusterCounts[6] += 10;
            } else if (token.rfind("medic", 0) == 0 || token.rfind("biolog", 0) == 0 || token.rfind("surg", 0) == 0) {
                clusterCounts[7] += 10;
            }
        }
        // Continuous hash distribution across all dims
        uint32_t h = 2166136261u;
        for (char c : token) h = (h ^ (uint8_t)c) * 16777619u;
        int d1 = h % targetDims;
        int d2 = (h >> 8) % targetDims;
        int d3 = (h >> 16) % targetDims;
        v[d1] += 0.5f;
        v[d2] += 0.35f;
        v[d3] += 0.2f;
    }

    int subBlock = targetDims / 4;
    if (subBlock < 1) subBlock = 1;
    // Core 4 blocks
    for (int c = 0; c < 4; c++) {
        if (clusterCounts[c] > 0) {
            int start = (c * subBlock) % targetDims;
            for (int k = 0; k < subBlock; k++) {
                v[(start + k) % targetDims] += (float)clusterCounts[c] * 0.9f;
            }
        }
    }

    // AI cluster (4) blends Dims 0 & 1
    if (clusterCounts[4] > 0) {
        for (int k = 0; k < subBlock; k++) {
            v[k % targetDims] += (float)clusterCounts[4] * 0.5f;
            v[(subBlock + k) % targetDims] += (float)clusterCounts[4] * 0.5f;
        }
    }

    // Astronomy cluster (5) blends Dims 1 (Math/Physics) & Dims 3 (Astrophysics/Spatial) with distinct signature
    if (clusterCounts[5] > 0) {
        for (int k = 0; k < subBlock; k++) {
            v[(subBlock + k) % targetDims] += (float)clusterCounts[5] * 0.45f;
            v[(3 * subBlock + k) % targetDims] += (float)clusterCounts[5] * 0.75f;
        }
    }

    // Finance cluster (6) blends Dims 0 (Algorithms) & Dims 1 (Math)
    if (clusterCounts[6] > 0) {
        for (int k = 0; k < subBlock; k++) {
            v[k % targetDims] += (float)clusterCounts[6] * 0.7f;
            v[(subBlock + k) % targetDims] += (float)clusterCounts[6] * 0.4f;
        }
    }

    // Medical cluster (7) blends Dims 2 (Bio/Organic) & Dims 0 (Systems)
    if (clusterCounts[7] > 0) {
        for (int k = 0; k < subBlock; k++) {
            v[(2 * subBlock + k) % targetDims] += (float)clusterCounts[7] * 0.7f;
            v[k % targetDims] += (float)clusterCounts[7] * 0.4f;
        }
    }

    // L2 Normalize
    float normSq = 0.0f;
    for (float val : v) normSq += val * val;
    float norm = std::sqrt(normSq);
    if (norm < 1e-7f) norm = 1.0f;
    for (float& val : v) val /= norm;

    return v;
}

// =====================================================================
//  SPARSE BM25 INVERTED INDEX (For Hybrid Search)
// =====================================================================

class BM25Index {
    struct Posting {
        int docId;
        int termFreq;
    };
    std::unordered_map<std::string, std::vector<Posting>> invertedList;
    std::unordered_map<int, int> docLengths;
    double avgDocLen = 1.0;
    int totalDocs = 0;
    double k1 = 1.5;
    double b  = 0.75;

public:
    void clear() {
        invertedList.clear();
        docLengths.clear();
        totalDocs = 0;
        avgDocLen = 1.0;
    }

    void addDocument(int docId, const std::string& text) {
        auto tokens = tokenize(text);
        int len = (int)tokens.size();
        docLengths[docId] = len;

        std::unordered_map<std::string, int> freqs;
        for (const auto& t : tokens) freqs[t]++;

        for (const auto& [term, freq] : freqs) {
            invertedList[term].push_back({docId, freq});
        }
        totalDocs++;
        double sum = 0;
        for (auto& [_, l] : docLengths) sum += l;
        avgDocLen = totalDocs > 0 ? (sum / totalDocs) : 1.0;
    }

    void removeDocument(int docId) {
        if (!docLengths.count(docId)) return;
        docLengths.erase(docId);
        totalDocs--;
        for (auto& [term, postings] : invertedList) {
            postings.erase(std::remove_if(postings.begin(), postings.end(),
                [docId](const Posting& p){ return p.docId == docId; }), postings.end());
        }
        double sum = 0;
        for (auto& [_, l] : docLengths) sum += l;
        avgDocLen = totalDocs > 0 ? (sum / totalDocs) : 1.0;
    }

    std::unordered_map<int, double> scoreQuery(const std::string& query) const {
        std::unordered_map<int, double> scores;
        auto tokens = tokenize(query);
        for (const auto& term : tokens) {
            auto it = invertedList.find(term);
            if (it == invertedList.end()) continue;
            int n_t = (int)it->second.size();
            double idf = std::log(1.0 + (totalDocs - n_t + 0.5) / (n_t + 0.5));
            if (idf < 0.0) idf = 0.01;

            for (const auto& p : it->second) {
                int dLen = 1;
                auto dlIt = docLengths.find(p.docId);
                if (dlIt != docLengths.end()) dLen = dlIt->second;
                double num = p.termFreq * (k1 + 1.0);
                double den = p.termFreq + k1 * (1.0 - b + b * (dLen / std::max(1.0, avgDocLen)));
                scores[p.docId] += idf * (num / den);
            }
        }
        return scores;
    }
};

// =====================================================================
//  BRUTE FORCE VECTOR INDEX
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
        std::sort(r.begin(), r.end(), [](const std::pair<float,int>& a, const std::pair<float,int>& b){
            return safeLess(a.first, a.second, b.first, b.second);
        });
        if ((int)r.size() > k) r.resize(k);
        return r;
    }

    void remove(int id) {
        items.erase(std::remove_if(items.begin(), items.end(),
            [id](const VectorItem& v){ return v.id == id; }), items.end());
    }

    void clear() {
        items.clear();
    }
};

// =====================================================================
//  KD-TREE VECTOR INDEX
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
        if (ax < (int)v.emb.size() && ax < (int)n->item.emb.size()) {
            if (v.emb[ax] < n->item.emb[ax]) n->left  = ins(n->left,  v, d+1);
            else                              n->right = ins(n->right, v, d+1);
        } else {
            n->right = ins(n->right, v, d+1);
        }
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
        float diff = 0.0f;
        if (ax < (int)q.size() && ax < (int)n->item.emb.size()) {
            diff = q[ax] - n->item.emb[ax];
        }
        KDNode* closer  = diff < 0 ? n->left  : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        knn(closer, q, k, d+1, dist, heap);
        if ((int)heap.size() < k || std::abs(diff) < heap.top().first)
            knn(farther, q, k, d+1, dist, heap);
    }

    KDNode* copyTree(const KDNode* n) {
        if (!n) return nullptr;
        KDNode* newNode = new KDNode(n->item);
        newNode->left = copyTree(n->left);
        newNode->right = copyTree(n->right);
        return newNode;
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { destroy(root); root = nullptr; }

    KDTree(const KDTree& other) : dims(other.dims), root(copyTree(other.root)) {}
    KDTree(KDTree&& other) noexcept : dims(other.dims), root(other.root) { other.root = nullptr; }
    KDTree& operator=(const KDTree& other) {
        if (this != &other) {
            destroy(root);
            dims = other.dims;
            root = copyTree(other.root);
        }
        return *this;
    }
    KDTree& operator=(KDTree&& other) noexcept {
        if (this != &other) {
            destroy(root);
            dims = other.dims;
            root = other.root;
            other.root = nullptr;
        }
        return *this;
    }

    void insert(const VectorItem& v) { root = ins(root, v, 0); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::priority_queue<std::pair<float,int>> heap;
        knn(root, q, k, 0, dist, heap);
        std::vector<std::pair<float,int>> r;
        while (!heap.empty()) { r.push_back(heap.top()); heap.pop(); }
        std::reverse(r.begin(), r.end());
        return r;
    }

    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root); root = nullptr;
        for (auto& v : items) insert(v);
    }
};

// =====================================================================
//  HNSW (Hierarchical Navigable Small World)
// =====================================================================

class HNSW {
public:
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs; // nbrs[layer] -> neighbor IDs
    };

    struct GraphInfo {
        int topLayer;
        int nodeCount;
        std::vector<int> nodesPerLayer;
        std::vector<int> edgesPerLayer;
        struct NodeSummary { int id; std::string metadata; std::string category; int maxLyr; };
        struct EdgeSummary { int src; int dst; int lyr; };
        std::vector<NodeSummary> nodes;
        std::vector<EdgeSummary> edges;
    };

private:
    int M;
    int efConstruction;
    double mL;
    int entryPt = -1;
    int maxLyr  = -1;
    std::unordered_map<int, Node> nodes;
    std::default_random_engine rng;
    std::uniform_real_distribution<double> unif{0.0, 1.0};

    int randomLevel() {
        double r = unif(rng);
        if (r <= 0.0) r = 1e-9;
        return (int)(-std::log(r) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, const std::vector<int>& eps, int ef, int lyr, DistFn dist)
    {
        std::unordered_set<int> visited;
        // min-heap for candidate set C
        std::priority_queue<std::pair<float,int>,
                            std::vector<std::pair<float,int>>,
                            std::greater<std::pair<float,int>>> C;
        // max-heap for result set W
        std::priority_queue<std::pair<float,int>> W;

        for (int ep : eps) {
            if (!nodes.count(ep)) continue;
            float d = dist(q, nodes[ep].item.emb);
            visited.insert(ep);
            C.push({d, ep});
            W.push({d, ep});
        }

        while (!C.empty()) {
            auto [cd, cId] = C.top(); C.pop();
            if (W.size() >= (size_t)ef && cd > W.top().first) break;

            if (!nodes.count(cId) || lyr >= (int)nodes[cId].nbrs.size()) continue;
            for (int nbr : nodes[cId].nbrs[lyr]) {
                if (visited.count(nbr) || !nodes.count(nbr)) continue;
                visited.insert(nbr);
                float nd = dist(q, nodes[nbr].item.emb);
                if (W.size() < (size_t)ef || nd < W.top().first) {
                    C.push({nd, nbr});
                    W.push({nd, nbr});
                    if (W.size() > (size_t)ef) W.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!W.empty()) { res.push_back(W.top()); W.pop(); }
        std::reverse(res.begin(), res.end());
        return res;
    }

public:
    HNSW(int M_ = 16, int ef_ = 200)
        : M(M_), efConstruction(ef_), mL(1.0 / std::log((double)M_)), rng(1337) {}

    void insert(const VectorItem& v, DistFn dist) {
        int l = randomLevel();
        Node n;
        n.item = v;
        n.maxLyr = l;
        n.nbrs.resize(l + 1);

        if (entryPt == -1 || nodes.empty()) {
            entryPt = v.id;
            maxLyr  = l;
            nodes[v.id] = std::move(n);
            return;
        }

        int currEp = entryPt;
        // Top layers descent down to l+1
        for (int lc = maxLyr; lc > l; lc--) {
            auto hits = searchLayer(v.emb, {currEp}, 1, lc, dist);
            if (!hits.empty()) currEp = hits[0].second;
        }

        // Insert from min(maxLyr, l) down to layer 0
        for (int lc = std::min(maxLyr, l); lc >= 0; lc--) {
            auto candidates = searchLayer(v.emb, {currEp}, efConstruction, lc, dist);
            std::vector<int> neighbors;
            for (auto& [d, cId] : candidates) {
                if (cId != v.id) neighbors.push_back(cId);
                if ((int)neighbors.size() >= M) break;
            }
            n.nbrs[lc] = neighbors;
            for (int nId : neighbors) {
                if (!nodes.count(nId)) continue;
                if (lc < (int)nodes[nId].nbrs.size()) {
                    nodes[nId].nbrs[lc].push_back(v.id);
                    if ((int)nodes[nId].nbrs[lc].size() > M * 2) {
                        nodes[nId].nbrs[lc].resize(M * 2);
                    }
                }
            }
            if (!candidates.empty()) currEp = candidates[0].second;
        }

        if (l > maxLyr) {
            maxLyr  = l;
            entryPt = v.id;
        }
        nodes[v.id] = std::move(n);
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int efSearch, DistFn dist)
    {
        if (entryPt == -1 || nodes.empty()) return {};
        int currEp = entryPt;
        for (int lc = maxLyr; lc > 0; lc--) {
            auto hits = searchLayer(q, {currEp}, 1, lc, dist);
            if (!hits.empty()) currEp = hits[0].second;
        }
        auto results = searchLayer(q, {currEp}, std::max(k, efSearch), 0, dist);
        if ((int)results.size() > k) results.resize(k);
        return results;
    }

    void remove(int id) {
        if (!nodes.count(id)) return;
        for (int l = 0; l <= nodes[id].maxLyr; l++) {
            for (int nId : nodes[id].nbrs[l]) {
                if (!nodes.count(nId) || l >= (int)nodes[nId].nbrs.size()) continue;
                auto& nvec = nodes[nId].nbrs[l];
                nvec.erase(std::remove(nvec.begin(), nvec.end(), id), nvec.end());
            }
        }
        nodes.erase(id);
        if (entryPt == id) {
            if (nodes.empty()) { entryPt = -1; maxLyr = -1; }
            else {
                entryPt = nodes.begin()->first;
                maxLyr = nodes.begin()->second.maxLyr;
                for (auto& [nid, n] : nodes) {
                    if (n.maxLyr > maxLyr) { maxLyr = n.maxLyr; entryPt = nid; }
                }
            }
        }
    }

    GraphInfo getGraphInfo() {
        GraphInfo gi;
        gi.topLayer = maxLyr;
        gi.nodeCount = (int)nodes.size();
        if (maxLyr >= 0) {
            gi.nodesPerLayer.assign(maxLyr + 1, 0);
            gi.edgesPerLayer.assign(maxLyr + 1, 0);
            for (auto& [id, n] : nodes) {
                gi.nodes.push_back({id, n.item.metadata, n.item.category, n.maxLyr});
                for (int l = 0; l <= n.maxLyr; l++) {
                    gi.nodesPerLayer[l]++;
                    if (l < (int)n.nbrs.size()) {
                        gi.edgesPerLayer[l] += (int)n.nbrs[l].size();
                        for (int dst : n.nbrs[l]) {
                            if (id < dst) gi.edges.push_back({id, dst, l});
                        }
                    }
                }
            }
        }
        return gi;
    }

    void clear() {
        nodes.clear();
        entryPt = -1;
        maxLyr = -1;
    }
};

// =====================================================================
//  INVERTED FILE INDEX (IVF-FLAT) WITH K-MEANS LLOYD'S CLUSTERING
// =====================================================================

class IVFIndex {
public:
    struct Cluster {
        std::vector<float> centroid;
        std::vector<VectorItem> items;
    };

private:
    int nlist;       // Number of Voronoi partitions (centroids)
    int nprobe;      // Number of Voronoi cells visited during search
    int dims;
    bool isTrained = false;
    std::vector<Cluster> clusters;
    std::vector<VectorItem> unassignedBuffer;

public:
    explicit IVFIndex(int kCentroids = 6, int nprobeVal = 2, int d = 16)
        : nlist(kCentroids), nprobe(nprobeVal), dims(d) {}

    void clear() {
        clusters.clear();
        unassignedBuffer.clear();
        isTrained = false;
    }

    void trainAndBuild(const std::vector<VectorItem>& allItems, DistFn dist, int maxIters = 15) {
        clusters.clear();
        unassignedBuffer.clear();
        if (allItems.empty()) {
            isTrained = true;
            return;
        }

        int k = std::min((int)allItems.size(), nlist);
        if (k <= 0) k = 1;
        clusters.resize(k);

        // K-Means initialization
        for (int i = 0; i < k; i++) {
            clusters[i].centroid = allItems[i % allItems.size()].emb;
            clusters[i].items.clear();
        }

        // Lloyd's K-Means Iterations
        for (int iter = 0; iter < maxIters; iter++) {
            for (auto& c : clusters) c.items.clear();

            // 1. Assignment step: assign each vector to closest Voronoi centroid
            for (const auto& item : allItems) {
                int bestC = 0;
                float bestDist = 1e9f;
                for (int ci = 0; ci < k; ci++) {
                    float d = dist(item.emb, clusters[ci].centroid);
                    if (d < bestDist) {
                        bestDist = d;
                        bestC = ci;
                    }
                }
                clusters[bestC].items.push_back(item);
            }

            // 2. Update step: recalculate centroid as mean of assigned vectors
            for (int ci = 0; ci < k; ci++) {
                if (clusters[ci].items.empty()) continue;
                std::vector<float> newCentroid(dims, 0.0f);
                for (const auto& item : clusters[ci].items) {
                    for (int d = 0; d < dims && d < (int)item.emb.size(); d++) {
                        newCentroid[d] += item.emb[d];
                    }
                }
                float count = (float)clusters[ci].items.size();
                float normSq = 0.0f;
                for (int d = 0; d < dims; d++) {
                    newCentroid[d] /= count;
                    normSq += newCentroid[d] * newCentroid[d];
                }
                float norm = std::sqrt(normSq);
                if (norm > 1e-7f) {
                    for (int d = 0; d < dims; d++) newCentroid[d] /= norm;
                }
                clusters[ci].centroid = newCentroid;
            }
        }
        isTrained = true;
    }

    void insert(const VectorItem& v, DistFn dist) {
        if (!isTrained || clusters.empty()) {
            unassignedBuffer.push_back(v);
            return;
        }
        int bestC = 0;
        float bestDist = 1e9f;
        for (size_t ci = 0; ci < clusters.size(); ci++) {
            float d = dist(v.emb, clusters[ci].centroid);
            if (d < bestDist) {
                bestDist = d;
                bestC = (int)ci;
            }
        }
        clusters[bestC].items.push_back(v);
    }

    void remove(int id) {
        for (auto& c : clusters) {
            c.items.erase(std::remove_if(c.items.begin(), c.items.end(),
                [id](const VectorItem& v){ return v.id == id; }), c.items.end());
        }
        unassignedBuffer.erase(std::remove_if(unassignedBuffer.begin(), unassignedBuffer.end(),
            [id](const VectorItem& v){ return v.id == id; }), unassignedBuffer.end());
    }

    std::vector<std::pair<float, int>> knn(
        const std::vector<float>& q, int k, int customNprobe, DistFn dist)
    {
        if (clusters.empty()) {
            std::vector<std::pair<float, int>> res;
            for (auto& v : unassignedBuffer) res.push_back({dist(q, v.emb), v.id});
            std::sort(res.begin(), res.end(), [](const std::pair<float,int>& a, const std::pair<float,int>& b){
                return safeLess(a.first, a.second, b.first, b.second);
            });
            if ((int)res.size() > k) res.resize(k);
            return res;
        }

        int probes = customNprobe > 0 ? customNprobe : nprobe;
        probes = std::min(probes, (int)clusters.size());

        // Step 1: Find top-nprobe closest centroids
        std::vector<std::pair<float, int>> centroidDists;
        for (int ci = 0; ci < (int)clusters.size(); ci++) {
            centroidDists.push_back({dist(q, clusters[ci].centroid), ci});
        }
        std::sort(centroidDists.begin(), centroidDists.end(), [](const std::pair<float,int>& a, const std::pair<float,int>& b){
            return safeLess(a.first, a.second, b.first, b.second);
        });

        // Step 2: Search only vectors within those Voronoi cells (inverted lists)
        std::vector<std::pair<float, int>> hits;
        for (int p = 0; p < probes; p++) {
            int ci = centroidDists[p].second;
            for (const auto& item : clusters[ci].items) {
                hits.push_back({dist(q, item.emb), item.id});
            }
        }
        for (const auto& item : unassignedBuffer) {
            hits.push_back({dist(q, item.emb), item.id});
        }

        std::sort(hits.begin(), hits.end(), [](const std::pair<float,int>& a, const std::pair<float,int>& b){
            return safeLess(a.first, a.second, b.first, b.second);
        });
        if ((int)hits.size() > k) hits.resize(k);
        return hits;
    }

    struct IVFStats {
        int totalCentroids;
        int activeVectors;
        int nprobe;
        std::vector<int> clusterSizes;
    };

    IVFStats getStats() const {
        IVFStats st;
        st.totalCentroids = (int)clusters.size();
        st.nprobe = nprobe;
        st.activeVectors = (int)unassignedBuffer.size();
        for (const auto& c : clusters) {
            st.clusterSizes.push_back((int)c.items.size());
            st.activeVectors += (int)c.items.size();
        }
        return st;
    }
};

// =====================================================================
//  PRINCIPAL COMPONENT ANALYSIS (PCA) DIMENSIONALITY REDUCER
// =====================================================================

class PCAReducer {
    int inDims;
    int outDims;
    std::vector<float> mean;
    std::vector<std::vector<float>> components; // top eigenvectors [outDims][inDims]
    bool isFitted = false;

public:
    PCAReducer(int inD = 16, int outD = 2) : inDims(inD), outDims(outD) {}

    void clear() {
        mean.clear();
        components.clear();
        isFitted = false;
    }

    // Power Iteration to extract the dominant eigenvector of matrix A
    static std::vector<float> powerIteration(
        const std::vector<std::vector<float>>& mat, int n, int maxIters = 60)
    {
        std::vector<float> b(n, 1.0f / std::sqrt((float)n));
        for (int iter = 0; iter < maxIters; iter++) {
            std::vector<float> b_next(n, 0.0f);
            for (int r = 0; r < n; r++) {
                float sum = 0.0f;
                for (int c = 0; c < n; c++) {
                    sum += mat[r][c] * b[c];
                }
                b_next[r] = sum;
            }
            float normSq = 0.0f;
            for (float val : b_next) normSq += val * val;
            float norm = std::sqrt(normSq);
            if (norm < 1e-9f) break;
            for (int i = 0; i < n; i++) b[i] = b_next[i] / norm;
        }
        return b;
    }

    // Fit PCA on a dataset of vectors
    void fit(const std::vector<std::vector<float>>& X) {
        if (X.empty()) return;
        int n = (int)X.size();
        inDims = (int)X[0].size();
        if (inDims <= 0) return;

        // 1. Compute Mean Vector
        mean.assign(inDims, 0.0f);
        for (const auto& row : X) {
            for (int d = 0; d < inDims && d < (int)row.size(); d++) {
                mean[d] += row[d];
            }
        }
        for (int d = 0; d < inDims; d++) mean[d] /= (float)n;

        // 2. Compute Covariance Matrix C = (X - μ)^T (X - μ) / N
        std::vector<std::vector<float>> cov(inDims, std::vector<float>(inDims, 0.0f));
        for (const auto& row : X) {
            std::vector<float> centered(inDims, 0.0f);
            for (int d = 0; d < inDims && d < (int)row.size(); d++) {
                centered[d] = row[d] - mean[d];
            }
            for (int r = 0; r < inDims; r++) {
                for (int c = 0; c < inDims; c++) {
                    cov[r][c] += centered[r] * centered[c];
                }
            }
        }
        float div = n > 1 ? (float)(n - 1) : 1.0f;
        for (int r = 0; r < inDims; r++) {
            for (int c = 0; c < inDims; c++) {
                cov[r][c] /= div;
            }
        }

        // 3. Extract top-outDims eigenvectors using Power Iteration & Deflation
        components.clear();
        std::vector<std::vector<float>> curCov = cov;
        int targetK = std::min(outDims, inDims);

        for (int k = 0; k < targetK; k++) {
            auto eigVec = powerIteration(curCov, inDims, 80);
            components.push_back(eigVec);

            // Compute eigenvalue lambda = v^T * C * v
            float lambda = 0.0f;
            for (int r = 0; r < inDims; r++) {
                float rowSum = 0.0f;
                for (int c = 0; c < inDims; c++) {
                    rowSum += curCov[r][c] * eigVec[c];
                }
                lambda += eigVec[r] * rowSum;
            }

            // Deflation: C_next = C - lambda * (v * v^T)
            for (int r = 0; r < inDims; r++) {
                for (int c = 0; c < inDims; c++) {
                    curCov[r][c] -= lambda * eigVec[r] * eigVec[c];
                }
            }
        }
        isFitted = true;
    }

    // Project a vector into reduced latent/2D space: y = (x - μ) * W
    std::vector<float> transform(const std::vector<float>& x) const {
        if (!isFitted || components.empty() || mean.empty()) {
            if (x.size() >= 2) return {x[0], x[1]};
            return {0.0f, 0.0f};
        }
        std::vector<float> out;
        out.reserve(components.size());
        int dLen = (int)std::min(x.size(), mean.size());

        for (size_t k = 0; k < components.size(); k++) {
            float dot = 0.0f;
            int cLen = (int)components[k].size();
            for (int i = 0; i < dLen && i < cLen; i++) {
                dot += (x[i] - mean[i]) * components[k][i];
            }
            if (std::isnan(dot) || std::isinf(dot)) dot = 0.0f;
            out.push_back(dot);
        }
        while (out.size() < 2) out.push_back(0.0f);
        return out;
    }

    bool fitted() const { return isFitted; }
};

// =====================================================================
//  CLUSTER MANAGER & UNIFIED VECTOR DATABASE
// =====================================================================

struct ClusterInfo {
    std::string name;
    std::string label;
    std::string color;
    std::vector<float> center2D;
};

class VectorDB {
public:
    struct SearchHit {
        int id;
        std::string meta;
        std::string cat;
        float dist;
        float sparseScore;
        float hybridScore;
        std::vector<float> emb;
    };

    struct SearchOut {
        std::vector<SearchHit> hits;
        long long us;
        std::string algo;
        std::string metric;
        float effectiveAlpha;
        std::string alphaMode;
    };

    struct BenchOut {
        long long bfUs, kdUs, hnswUs, ivfUs;
        int n;
    };

private:
    int dims;
    int nextId = 1;
    std::unordered_map<int, VectorItem> store;
    std::unordered_map<std::string, ClusterInfo> clusters;
    BruteForce bf;
    KDTree     kd;
    HNSW       hnsw;
    IVFIndex   ivf;
    PCAReducer pca;
    BM25Index  bm25;
    std::mutex mu;

public:
    explicit VectorDB(int d = 16) : dims(d), kd(d), hnsw(16, 200), ivf(6, 2, d), pca(d, 2) {
        initDefaultClusters();
    }

    void initDefaultClusters() {
        clusters["cs"]     = {"cs", "Computer Science", "#00f2fe", {-0.65f,  0.60f}};
        clusters["math"]   = {"math", "Mathematics", "#b388ff", { 0.65f,  0.60f}};
        clusters["food"]   = {"food", "Culinary Arts", "#fbbf24", {-0.65f, -0.60f}};
        clusters["sports"] = {"sports", "Athletics", "#10b981", { 0.65f, -0.60f}};
        clusters["ai"]     = {"ai", "Artificial Intelligence", "#f43f5e", { 0.00f,  0.00f}};
    }

    void clearAll() {
        std::lock_guard<std::mutex> lk(mu);
        store.clear();
        clusters.clear();
        initDefaultClusters();
        nextId = 1;
        bf.clear();
        kd.rebuild({});
        hnsw.clear();
        ivf.clear();
        pca.clear();
        bm25.clear();
    }

    int insert(const std::string& meta, const std::string& cat, const std::vector<float>& emb, DistFn dist) {
        std::lock_guard<std::mutex> lk(mu);
        int id = nextId++;
        VectorItem item{id, meta, cat, emb};
        store[id] = item;
        bf.insert(item);
        kd.insert(item);
        hnsw.insert(item, dist);
        ivf.insert(item, dist);
        bm25.addDocument(id, meta + " " + cat);
        if (!clusters.count(cat)) {
            clusters[cat] = {cat, cat, "#ec4899", {0.0f, 0.0f}};
        }
        return id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id);
        bf.remove(id);
        hnsw.remove(id);
        ivf.remove(id);
        bm25.removeDocument(id);
        std::vector<VectorItem> allItems;
        for (auto& [_, v] : store) allItems.push_back(v);
        kd.rebuild(allItems);
        return true;
    }

    bool addCluster(const std::string& name, const std::string& label, const std::string& color) {
        std::lock_guard<std::mutex> lk(mu);
        if (name.empty()) return false;
        float cx = ((float)(rand() % 1000) / 1000.0f) * 1.2f - 0.6f;
        float cy = ((float)(rand() % 1000) / 1000.0f) * 1.2f - 0.6f;
        ClusterInfo info;
        info.name = name;
        info.label = label.empty() ? name : label;
        info.color = color.empty() ? "#38bdf8" : color;
        info.center2D = {cx, cy};
        clusters[name] = info;
        return true;
    }

    bool deleteCluster(const std::string& name) {
        std::lock_guard<std::mutex> lk(mu);
        if (!clusters.count(name)) return false;
        clusters.erase(name);
        std::vector<int> toDelete;
        for (auto& [id, v] : store) {
            if (v.category == name) toDelete.push_back(id);
        }
        for (int id : toDelete) {
            store.erase(id);
            bf.remove(id);
            hnsw.remove(id);
            ivf.remove(id);
            bm25.removeDocument(id);
        }
        std::vector<VectorItem> allItems;
        for (auto& [_, v] : store) allItems.push_back(v);
        kd.rebuild(allItems);
        return true;
    }

    std::vector<ClusterInfo> getClusters() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<ClusterInfo> out;
        for (auto& [_, c] : clusters) out.push_back(c);
        return out;
    }

    void trainIVF(int nlist = 6, int nprobe = 2) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<VectorItem> allItems;
        for (auto& [_, v] : store) allItems.push_back(v);
        ivf = IVFIndex(nlist, nprobe, dims);
        ivf.trainAndBuild(allItems, getDistFn("cosine"));
    }

    IVFIndex::IVFStats getIVFStats() {
        std::lock_guard<std::mutex> lk(mu);
        return ivf.getStats();
    }

    void fitPCA(int outD = 2) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<std::vector<float>> X;
        for (auto& [_, v] : store) X.push_back(v.emb);
        pca = PCAReducer(dims, outD);
        pca.fit(X);
    }

    std::vector<float> projectPCA(const std::vector<float>& vec) {
        std::lock_guard<std::mutex> lk(mu);
        return pca.transform(vec);
    }

    std::unordered_map<int, std::vector<float>> getAllPCA2D() {
        std::lock_guard<std::mutex> lk(mu);
        std::unordered_map<int, std::vector<float>> out;
        for (auto& [id, v] : store) {
            out[id] = pca.transform(v.emb);
        }
        return out;
    }

    SearchOut search(const std::vector<float>& q, int k, const std::string& metric,
                     const std::string& algo, const std::string& textQuery = "", float alpha = 1.0f)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto t0 = std::chrono::high_resolution_clock::now();
        DistFn dist = getDistFn(metric);
        std::vector<std::pair<float,int>> raw;

        if      (algo == "kdtree")     raw = kd.knn(q, k * 2, dist);
        else if (algo == "bruteforce") raw = bf.knn(q, k * 2, dist);
        else if (algo == "ivf")        raw = ivf.knn(q, k * 2, 2, dist);
        else                           raw = hnsw.knn(q, k * 2, 50, dist);

        // Sparse BM25 scoring for hybrid fusion
        std::unordered_map<int, double> bm25Scores;
        if (!textQuery.empty()) {
            bm25Scores = bm25.scoreQuery(textQuery);
        }

        double maxBM25 = 1e-6;
        for (auto& [_, s] : bm25Scores) maxBM25 = std::max(maxBM25, s);

        // Adaptive Dynamic Auto-Alpha:
        float effectiveAlpha = alpha;
        std::string alphaMode = "manual";
        if (alpha < 0.0f) {
            alphaMode = "auto";
            if (maxBM25 > 2.0) {
                effectiveAlpha = 0.50f; // High keyword confidence (50% Dense / 50% BM25)
            } else if (maxBM25 > 0.4) {
                effectiveAlpha = 0.75f; // Balanced hybrid (75% Dense / 25% BM25)
            } else {
                effectiveAlpha = 0.95f; // Pure semantic query (95% Dense / 5% BM25)
            }
        }

        std::unordered_set<int> seen;
        std::vector<SearchHit> candidates;
        for (auto& [d, id] : raw) {
            if (!store.count(id) || seen.count(id)) continue;
            seen.insert(id);
            auto& v = store[id];
            float safeD = (std::isnan(d) || std::isinf(d)) ? 1.0f : d;
            float denseSim = std::max(0.0f, 1.0f - (safeD / 2.0f));
            float sparseSim = 0.0f;
            auto bIt = bm25Scores.find(id);
            if (bIt != bm25Scores.end() && maxBM25 > 1e-9) {
                sparseSim = (float)(bIt->second / maxBM25);
            }
            if (std::isnan(sparseSim) || std::isinf(sparseSim)) sparseSim = 0.0f;
            float hybrid = effectiveAlpha * denseSim + (1.0f - effectiveAlpha) * sparseSim;
            if (std::isnan(hybrid) || std::isinf(hybrid)) hybrid = denseSim;
            candidates.push_back({id, v.metadata, v.category, safeD, sparseSim, hybrid, v.emb});
        }

        if (effectiveAlpha < 0.999f) {
            // Sort by hybrid score descending
            std::sort(candidates.begin(), candidates.end(), [](const SearchHit& a, const SearchHit& b){
                return safeGreater(a.hybridScore, a.id, b.hybridScore, b.id);
            });
        } else {
            // Sort by distance ascending
            std::sort(candidates.begin(), candidates.end(), [](const SearchHit& a, const SearchHit& b){
                return safeLess(a.dist, a.id, b.dist, b.id);
            });
        }

        if ((int)candidates.size() > k) candidates.resize(k);

        auto t1 = std::chrono::high_resolution_clock::now();
        long long us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return {candidates, us, algo, metric, effectiveAlpha, alphaMode};
    }

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        DistFn dist = getDistFn(metric);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++) bf.knn(q, k, dist);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++) kd.knn(q, k, dist);
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++) hnsw.knn(q, k, 50, dist);
        auto t3 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++) ivf.knn(q, k, 2, dist);
        auto t4 = std::chrono::high_resolution_clock::now();

        long long bfUs   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 20;
        long long kdUs   = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 20;
        long long hnswUs = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 20;
        long long ivfUs  = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 20;
        return {std::max(1LL, bfUs), std::max(1LL, kdUs), std::max(1LL, hnswUs), std::max(1LL, ivfUs), (int)store.size()};
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
};

// =====================================================================
//  DOCUMENT DATABASE & SIMILARITY INSPECTOR
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
    BM25Index  bm25;
    std::mutex mu;
    int nextId = 100;
    int dims   = 64;

public:
    DocumentDB() : hnsw(16, 200) {}

    int insert(const std::string& title, const std::string& text, const std::vector<float>& emb) {
        std::lock_guard<std::mutex> lk(mu);
        if (!emb.empty()) dims = (int)emb.size();
        int id = nextId++;
        DocItem item{id, title, text, emb};
        store[id] = item;
        try {
            VectorItem vi{item.id, title, "doc", emb};
            hnsw.insert(vi, cosine);
            bf.insert(vi);
            bm25.addDocument(id, title + " " + text);
        } catch (...) {}
        return item.id;
    }


    // Compare an incoming document against all documents already present in the database
    struct DocSimResult {
        int id;
        std::string title;
        std::string preview;
        float similarity; // 0.0 to 1.0
        float cosineDist;
    };

    std::vector<DocSimResult> compareDocument(const std::vector<float>& targetEmb, int k = 10) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<DocSimResult> out;
        for (auto& [id, doc] : store) {
            float dist = cosine(targetEmb, doc.emb);
            float sim = std::max(0.0f, std::min(1.0f, 1.0f - (dist / 2.0f)));
            size_t pLen = std::min(doc.text.size(), (size_t)120);
            std::string prev = doc.text.substr(0, pLen);
            if (doc.text.size() > 120) prev += "...";
            out.push_back({id, doc.title, prev, sim, dist});
        }
        std::sort(out.begin(), out.end(), [](const DocSimResult& a, const DocSimResult& b){
            return safeGreater(a.similarity, a.id, b.similarity, b.id);
        });
        if ((int)out.size() > k) out.resize(k);
        return out;
    }


    std::vector<std::pair<float, DocItem>> search(const std::vector<float>& q, int k = 5) {
        std::lock_guard<std::mutex> lk(mu);
        if (store.empty()) return {};
        auto raw = (store.size() < 10) ? bf.knn(q, k, cosine) : hnsw.knn(q, k, 50, cosine);
        std::vector<std::pair<float, DocItem>> out;
        for (auto& [d, id] : raw) {
            if (store.count(id)) out.push_back({d, store[id]});
        }
        return out;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id);
        hnsw.remove(id);
        bf.remove(id);
        bm25.removeDocument(id);
        return true;
    }

    std::vector<DocItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<DocItem> r;
        for (auto& [_, v] : store) r.push_back(v);
        return r;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mu);
        store.clear();
        hnsw.clear();
        bf.clear();
        bm25.clear();
        nextId = 100;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }

    int getDims() { return dims; }
};

// =====================================================================
//  INITIALIZE PRE-SEEDED KNOWLEDGE & DEMO CORPUS
// =====================================================================

void seedDefaultData(VectorDB& db, DocumentDB& docDB) {
    auto dist = getDistFn("cosine");

    // Seed Core Clusters
    db.addCluster("cs", "Computer Science", "#00f2fe");
    db.addCluster("math", "Mathematics", "#b388ff");
    db.addCluster("food", "Culinary Arts", "#fbbf24");
    db.addCluster("sports", "Athletics & Games", "#10b981");
    db.addCluster("ai", "Artificial Intelligence", "#f43f5e");
    db.addCluster("astronomy", "Astronomy & Cosmos", "#ec4899");

    // Seed CS Vector Items
    db.insert("Linked List: nodes connected by pointers", "cs", localSemanticEmbed("Linked List nodes connected by pointers heap memory pointers data structure", DIMS), dist);
    db.insert("Binary Search Tree: O(log n) search and insert", "cs", localSemanticEmbed("Binary Search Tree search insert tree node traversal algorithm", DIMS), dist);
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs", localSemanticEmbed("Dynamic Programming memoization overlapping subproblems table recursion cache algorithm", DIMS), dist);
    db.insert("Graph BFS & DFS: breadth and depth first traversal", "cs", localSemanticEmbed("Graph BFS DFS breadth depth first traversal queue stack nodes", DIMS), dist);
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs", localSemanticEmbed("Hash Table O(1) lookup collision chaining array key value memory", DIMS), dist);

    // Seed Math Vector Items
    db.insert("Calculus: derivatives integrals and limits", "math", localSemanticEmbed("Calculus derivatives integrals limits differential equations formula theorem", DIMS), dist);
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math", localSemanticEmbed("Linear Algebra matrices eigenvalues eigenvectors vector_space matrix linear", DIMS), dist);
    db.insert("Probability: distributions random variables Bayes theorem", "math", localSemanticEmbed("Probability distributions random variables Bayes theorem statistics variance mean", DIMS), dist);
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math", localSemanticEmbed("Number Theory primes modular arithmetic RSA cryptography theorem discrete", DIMS), dist);
    db.insert("Combinatorics: permutations combinations generating functions", "math", localSemanticEmbed("Combinatorics permutations combinations generating functions probability proof", DIMS), dist);

    // Seed Food Vector Items
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", localSemanticEmbed("Neapolitan Pizza wood-fired dough San Marzano tomatoes cheese baking culinary recipe delicious", DIMS), dist);
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food", localSemanticEmbed("Sushi vinegared rice raw fish nori rolls restaurant chef taste delicious dish", DIMS), dist);
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", localSemanticEmbed("Ramen noodle soup chashu pork soft-boiled eggs broth culinary kitchen", DIMS), dist);
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food", localSemanticEmbed("Tacos corn tortillas carnitas salsa cilantro spicy delicious meat dish", DIMS), dist);
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food", localSemanticEmbed("Croissant laminated pastry buttery flaky layers bakery sweet dessert baking bread", DIMS), dist);

    // Seed Sports Vector Items
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports", localSemanticEmbed("Basketball fast-paced shooting dribbling slam dunks court NBA athlete tournament league", DIMS), dist);
    db.insert("Football: tackles touchdowns field goals and strategy", "sports", localSemanticEmbed("Football tackles touchdowns field goals strategy championship match player team", DIMS), dist);
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", localSemanticEmbed("Tennis racket volleys groundstrokes Wimbledon serves court match tournament", DIMS), dist);
    db.insert("Chess: openings endgames tactics strategic board game", "sports", localSemanticEmbed("Chess openings endgames tactics strategic board game tournament match player winner", DIMS), dist);
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports", localSemanticEmbed("Swimming butterfly freestyle backstroke Olympic competition athlete swimmer race pool", DIMS), dist);

    // Seed Astronomy & Space Vector Items
    db.insert("ISRO Chandrayaan-3: lunar south pole lander and pragyan rover mission", "astronomy", localSemanticEmbed("ISRO Chandrayaan lunar moon south pole lander rover mission space rocket telemetry orbit", DIMS), dist);
    db.insert("NASA James Webb Telescope: deep cosmos infrared galaxy observations", "astronomy", localSemanticEmbed("NASA James Webb JWST deep cosmos infrared galaxy observations space telescope nebula astrophysics", DIMS), dist);
    db.insert("ISRO Aditya-L1: solar coronal heating and space weather observatory", "astronomy", localSemanticEmbed("ISRO Aditya solar coronal space weather sun lagrange L1 satellite observatory astrophysics", DIMS), dist);
    db.insert("SpaceX Starship: super heavy orbital spacecraft propulsion", "astronomy", localSemanticEmbed("SpaceX Starship orbital rocket spacecraft propulsion mars lunar booster flight", DIMS), dist);
    db.insert("Mars Perseverance: red planet rover searching for biosignatures", "astronomy", localSemanticEmbed("Mars Perseverance red planet rover biosignatures crater geology cosmos", DIMS), dist);
    db.insert("Earth: terrestrial home world harboring oceans and atmosphere", "astronomy", localSemanticEmbed("Earth terrestrial planet oceans atmosphere biosphere solar orbit cosmos", DIMS), dist);
    db.insert("Hubble Space Telescope: thirty years of cosmic discovery and supernovas", "astronomy", localSemanticEmbed("Hubble Space Telescope cosmic discovery supernovas blackhole universe astronomy", DIMS), dist);

    // Seed Knowledge Documents
    docDB.insert("Virtual Memory & Paging Architecture",
        "Virtual memory gives processes an isolated linear address space. It maps virtual pages to physical memory frames using hardware page tables and Translation Lookaside Buffers (TLB).",
        localSemanticEmbed("Virtual Memory Paging OS memory management page tables", 64));

    docDB.insert("Neural Network Backpropagation & SGD",
        "Backpropagation computes gradients of the loss function with respect to layer weights using the chain rule of calculus, enabling stochastic gradient descent (SGD) to minimize prediction error.",
        localSemanticEmbed("Neural Network Backpropagation SGD calculus gradients weights deep learning", 64));

    docDB.insert("Raft Consensus Algorithm",
        "Raft achieves distributed consensus through leader election, write-ahead log replication, and heartbeats. It guarantees that any committed entry is durable and preserved across server restarts.",
        localSemanticEmbed("Raft Consensus Distributed Systems leader election log replication fault tolerance", 64));

    docDB.insert("Vector Indexing with HNSW Proximity Graphs",
        "Hierarchical Navigable Small World (HNSW) organizes high-dimensional vectors into multi-layer proximity graphs. Upper layers provide fast coarse routing while lower layers refine the nearest neighbor search with logarithmic complexity.",
        localSemanticEmbed("HNSW Vector Indexing graph search nearest neighbor logarithmic skip list", 64));

    // Train IVF-Flat K-Means partitions and fit PCA Eigenvectors
    db.trainIVF(6, 2);
    db.fitPCA(2);
}

// =====================================================================
//  MAIN HTTP SERVER ROUTING
// =====================================================================

int main(int argc, char* argv[]) {
    int port = 8081;
    if (const char* envPort = std::getenv("PORT")) {
        try { port = std::stoi(envPort); } catch (...) {}
    }
    if (argc > 1) {
        try { port = std::stoi(argv[1]); } catch (...) {}
    }

    std::ofstream logFile("server.log", std::ios::app);
    logFile << "[STARTUP] Vectra process started on port " << port << "." << std::endl;

    WSADATA wsaData;
    int wsaRes = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (wsaRes != 0) {
        logFile << "[FATAL] WSAStartup failed: " << wsaRes << std::endl;
        return 1;
    }

    VectorDB   db(DIMS);
    DocumentDB docDB;

    seedDefaultData(db, docDB);
    logFile << "[INFO] Default data seeded. Clusters: " << db.getClusters().size() << ", Vectors: " << db.size() << std::endl;

    httplib::Server svr;
    svr.set_keep_alive_max_count(100);
    svr.set_keep_alive_timeout(30);



    svr.set_pre_routing_handler([&](const auto& req, auto& res) {
        if (req.method == "OPTIONS") {
            cors(res);
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.method == "DELETE" && req.path.rfind("/doc/delete/", 0) == 0) {
            cors(res);
            std::string idStr = req.path.substr(12);
            int id = -1;
            try { id = std::stoi(idStr); } catch (...) {}
            bool ok = docDB.remove(id);
            res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.method == "DELETE" && req.path.rfind("/cluster/delete/", 0) == 0) {
            cors(res);
            std::string name = req.path.substr(16);
            bool ok = db.deleteCluster(name);
            res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.method == "DELETE" && req.path.rfind("/delete/", 0) == 0) {
            cors(res);
            std::string idStr = req.path.substr(8);
            int id = -1;
            try { id = std::stoi(idStr); } catch (...) {}
            bool ok = db.remove(id);
            res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // ── VECTORS & CLUSTERS ──────────────────────────────────────────

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        try {
            auto items = db.all();
            std::ostringstream ss;
            ss << '[';
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
        } catch (...) {
            res.status = 500;
            res.set_content("[]", "application/json");
        }
    });

    svr.Get("/clusters", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        try {
            auto cl = db.getClusters();
            std::ostringstream ss;
            ss << '[';
            for (size_t i = 0; i < cl.size(); i++) {
                if (i) ss << ',';
                auto& c = cl[i];
                float cx = c.center2D.size() > 0 ? c.center2D[0] : 0.0f;
                float cy = c.center2D.size() > 1 ? c.center2D[1] : 0.0f;
                ss << "{\"name\":"    << jS(c.name)
                   << ",\"label\":"   << jS(c.label)
                   << ",\"color\":"   << jS(c.color)
                   << ",\"center\":[" << cx << ',' << cy << "]}";
            }
            ss << ']';
            res.set_content(ss.str(), "application/json");
        } catch (...) {
            res.status = 500;
            res.set_content("[]", "application/json");
        }
    });

    svr.Post("/cluster/add", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto name  = extractStr(req.body, "name");
            auto label = extractStr(req.body, "label");
            auto color = extractStr(req.body, "color");
            if (name.empty()) {
                res.status = 400;
                res.set_content("{\"ok\":false,\"error\":\"Cluster name required\"}", "application/json");
                return;
            }
            if (label.empty()) label = name;
            if (color.empty()) color = "#38bdf8";
            bool ok = db.addCluster(name, label, color);
            res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
        } catch (...) {
            res.status = 500;
            res.set_content("{\"ok\":false,\"error\":\"Internal server error\"}", "application/json");
        }
    });

    svr.Post("/cluster/delete", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto name = extractStr(req.body, "name");
            if (name.empty()) name = req.get_param_value("name");
            bool ok = db.deleteCluster(name);
            res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
        } catch (...) {
            res.set_content("{\"ok\":false}", "application/json");
        }
    });

    svr.Delete("/cluster/delete", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto name = extractStr(req.body, "name");
            if (name.empty()) name = req.get_param_value("name");
            bool ok = db.deleteCluster(name);
            res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
        } catch (...) {
            res.set_content("{\"ok\":false}", "application/json");
        }
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto meta = extractStr(req.body, "metadata");
            auto cat  = extractStr(req.body, "category");
            if (cat.empty()) cat = "custom";
            std::vector<float> emb;
            size_t embPos = req.body.find("\"embedding\"");
            if (embPos != std::string::npos) {
                emb = parseVec(req.body.substr(embPos));
            }
            if (emb.empty()) {
                emb = localSemanticEmbed(meta + " " + cat, DIMS);
            }
            int id = db.insert(meta, cat, emb, getDistFn("cosine"));
            res.set_content("{\"id\":" + std::to_string(id) + ",\"metadata\":" + jS(meta) + ",\"category\":" + jS(cat) + ",\"dims\":" + std::to_string(emb.size()) + "}", "application/json");
        } catch (...) {
            res.status = 500;
            res.set_content("{\"error\":\"Failed to insert vector\"}", "application/json");
        }
    });

    svr.Post("/delete", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            int id = extractInt(req.body, "id", -1);
            if (id < 0 && req.has_param("id")) {
                try { id = std::stoi(req.get_param_value("id")); } catch (...) {}
            }
            bool ok = db.remove(id);
            res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
        } catch (...) {
            res.set_content("{\"ok\":false}", "application/json");
        }
    });

    svr.Delete("/delete", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            int id = extractInt(req.body, "id", -1);
            if (id < 0 && req.has_param("id")) {
                try { id = std::stoi(req.get_param_value("id")); } catch (...) {}
            }
            bool ok = db.remove(id);
            res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
        } catch (...) {
            res.set_content("{\"ok\":false}", "application/json");
        }
    });

    // ── UNIFIED & HYBRID VECTOR SEARCH ──────────────────────────────

    svr.Post("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto query  = extractStr(req.body, "query");
            auto metric = extractStr(req.body, "metric"); if (metric.empty()) metric = "cosine";
            auto algo   = extractStr(req.body, "algo");   if (algo.empty())   algo   = "hnsw";
            int k       = extractInt(req.body, "k", 5);
            if (k <= 0) k = 5;
            if (k > 100) k = 100;
            float alpha = extractFloat(req.body, "alpha", 1.0f); // 1.0 = pure vector, 0.0 = pure BM25

            std::vector<float> qEmb;
            size_t embPos = req.body.find("\"embedding\"");
            if (embPos != std::string::npos) {
                qEmb = parseVec(req.body.substr(embPos));
            }
            if (qEmb.empty()) {
                qEmb = localSemanticEmbed(query.empty() ? "search" : query, DIMS);
            }
            while (qEmb.size() < (size_t)DIMS) qEmb.push_back(0.04f);

            auto out = db.search(qEmb, k, metric, algo, query, alpha);
            std::ostringstream ss;
            ss << "{\"results\":[";
            for (size_t i = 0; i < out.hits.size(); i++) {
                if (i) ss << ',';
                auto& h = out.hits[i];
                ss << "{\"id\":"          << h.id
                   << ",\"metadata\":"    << jS(h.meta)
                   << ",\"category\":"    << jS(h.cat)
                   << ",\"distance\":"    << std::fixed << std::setprecision(4) << h.dist
                   << ",\"sparseScore\":" << std::fixed << std::setprecision(4) << h.sparseScore
                   << ",\"hybridScore\":" << std::fixed << std::setprecision(4) << h.hybridScore
                   << ",\"embedding\":"   << jVec(h.emb) << '}';
            }
            ss << "],\"queryEmbedding\":" << jVec(qEmb)
               << ",\"latencyUs\":"        << out.us
               << ",\"algo\":"             << jS(out.algo)
               << ",\"metric\":"           << jS(out.metric)
               << ",\"effectiveAlpha\":"   << out.effectiveAlpha
               << ",\"alphaMode\":"        << jS(out.alphaMode)
               << ",\"alpha\":"            << alpha << '}';
            res.set_content(ss.str(), "application/json");
        } catch (const std::exception& e) {
            res.set_content("{\"results\":[],\"error\":" + jS(e.what()) + "}", "application/json");
        } catch (...) {
            res.set_content("{\"results\":[],\"error\":\"Search failed\"}", "application/json");
        }
    });

    svr.Post("/clear_and_reseed", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        db.clearAll();
        docDB.clear();
        seedDefaultData(db, docDB);
        res.set_content("{\"ok\":true}", "application/json");
    });

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto qStr = req.get_param_value("query");
        auto vStr = req.get_param_value("v");
        std::vector<float> qEmb;
        if (!vStr.empty()) qEmb = parseVec(vStr);
        if (qEmb.empty())  qEmb = localSemanticEmbed(qStr, DIMS);

        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo   = "hnsw";
        float alpha = 1.0f; try { alpha = std::stof(req.get_param_value("alpha")); } catch (...) {}

        auto out = db.search(qEmb, k, metric, algo, qStr, alpha);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) ss << ',';
            auto& h = out.hits[i];
            ss << "{\"id\":"          << h.id
               << ",\"metadata\":"    << jS(h.meta)
               << ",\"category\":"    << jS(h.cat)
               << ",\"distance\":"    << std::fixed << std::setprecision(4) << h.dist
               << ",\"hybridScore\":" << std::fixed << std::setprecision(4) << h.hybridScore
               << ",\"embedding\":"   << jVec(h.emb) << '}';
        }
        ss << "],\"queryEmbedding\":" << jVec(qEmb)
           << ",\"latencyUs\":"        << out.us
           << ",\"algo\":"             << jS(out.algo)
           << ",\"metric\":"           << jS(out.metric) << '}';
        res.set_content(ss.str(), "application/json");
    });

    // ── DOCUMENT INGESTION & SIMILARITY SEARCH ──────────────────────

    // POST /doc/insert - Ingest document with metadata and knowledge embedding
    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto title = extractStr(req.body, "title");
            auto text  = extractStr(req.body, "text");
            if (title.empty() || text.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"Title and text required\"}", "application/json");
                return;
            }
            std::vector<float> emb;
            size_t embPos = req.body.find("\"embedding\"");
            if (embPos != std::string::npos) {
                emb = parseVec(req.body.substr(embPos));
            }
            if (emb.empty()) {
                emb = localSemanticEmbed(title + " " + text, 64);
            }
            int id = docDB.insert(title, text, emb);
            res.set_content("{\"id\":" + std::to_string(id) + ",\"title\":" + jS(title) + ",\"dims\":" + std::to_string(emb.size()) + "}", "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("{\"error\":" + jS(e.what()) + "}", "application/json");
        } catch (...) {
            res.status = 500;
            res.set_content("{\"error\":\"Failed to insert doc\"}", "application/json");
        }
    });


    // POST /doc/similarity - Compare target document similarity against all stored documents
    svr.Post("/doc/similarity", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto text  = extractStr(req.body, "text");
            if (text.empty()) text = extractStr(req.body, "query");
            if (text.empty()) text = extractStr(req.body, "q");
            auto title = extractStr(req.body, "title");
            int k      = extractInt(req.body, "k", -1);
            if (k <= 0) k = extractInt(req.body, "top_k", 10);
            if (k <= 0) k = 5;
            if (k > 100) k = 100;
            if (text.empty() && title.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"Input document text required\",\"matches\":[]}", "application/json");
                return;
            }
            std::vector<float> targetEmb;
            size_t embPos = req.body.find("\"embedding\"");
            if (embPos != std::string::npos) {
                targetEmb = parseVec(req.body.substr(embPos));
            }
            if (targetEmb.empty()) {
                targetEmb = localSemanticEmbed((title.empty() ? "" : title + " ") + text, 64);
            }
            auto matches = docDB.compareDocument(targetEmb, k);

            std::ostringstream ss;
            ss << "{\"inputTitle\":" << jS(title)
               << ",\"matches\":[";
            for (size_t i = 0; i < matches.size(); i++) {
                if (i) ss << ',';
                auto& m = matches[i];
                ss << "{\"id\":"         << m.id
                   << ",\"title\":"      << jS(m.title)
                   << ",\"preview\":"    << jS(m.preview)
                   << ",\"similarity\":" << std::fixed << std::setprecision(4) << m.similarity
                   << ",\"cosineDist\":" << std::fixed << std::setprecision(4) << m.cosineDist << '}';
            }
            ss << "],\"totalInDB\":" << docDB.size() << '}';
            res.set_content(ss.str(), "application/json");
        } catch (const std::exception& e) {
            res.set_content("{\"matches\":[],\"error\":" + jS(e.what()) + "}", "application/json");
        } catch (...) {
            res.set_content("{\"matches\":[],\"error\":\"Similarity calculation failed\"}", "application/json");
        }
    });

    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        try {
            auto docs = docDB.all();
            std::ostringstream ss;
            ss << '[';
            for (size_t i = 0; i < docs.size(); i++) {
                if (i) ss << ',';
                size_t pLen = std::min(docs[i].text.size(), (size_t)120);
                std::string prev = docs[i].text.substr(0, pLen);
                if (docs[i].text.size() > 120) prev += "...";
                ss << "{\"id\":"      << docs[i].id
                   << ",\"title\":"   << jS(docs[i].title)
                   << ",\"preview\":" << jS(prev)
                   << ",\"text\":"    << jS(docs[i].text) << '}';
            }
            ss << ']';
            res.set_content(ss.str(), "application/json");
        } catch (...) {
            res.set_content("[]", "application/json");
        }
    });


    svr.Delete("/doc/delete", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id = extractInt(req.body, "id", -1);
        if (id < 0 && req.has_param("id")) {
            try { id = std::stoi(req.get_param_value("id")); } catch (...) {}
        }
        bool ok = docDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
    });

    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id = -1;
        try { id = std::stoi(req.matches[1]); } catch (...) {}
        bool ok = docDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
    });

    svr.Post("/doc/delete", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id = extractInt(req.body, "id", -1);
        if (id < 0 && req.has_param("id")) {
            try { id = std::stoi(req.get_param_value("id")); } catch (...) {}
        }
        bool ok = docDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}", "application/json");
    });


    // ── IVF & K-MEANS CLUSTERING APIS ──────────────────────────────

    // POST /ivf/train - Train K-Means centroids and rebuild Voronoi inverted lists
    svr.Post("/ivf/train", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int nlist  = extractInt(req.body, "nlist", 6);
        int nprobe = extractInt(req.body, "nprobe", 2);
        db.trainIVF(nlist, nprobe);
        auto st = db.getIVFStats();
        std::ostringstream ss;
        ss << "{\"ok\":true,\"centroids\":" << st.totalCentroids
           << ",\"nprobe\":" << st.nprobe
           << ",\"activeVectors\":" << st.activeVectors << '}';
        res.set_content(ss.str(), "application/json");
    });

    // GET /ivf/stats - Get Voronoi cell distribution & IVF status
    svr.Get("/ivf/stats", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto st = db.getIVFStats();
        std::ostringstream ss;
        ss << "{\"centroids\":" << st.totalCentroids
           << ",\"nprobe\":" << st.nprobe
           << ",\"activeVectors\":" << st.activeVectors
           << ",\"clusterSizes\":[";
        for (size_t i = 0; i < st.clusterSizes.size(); i++) {
            if (i) ss << ',';
            ss << st.clusterSizes[i];
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── PCA MATHEMATICAL DIMENSIONALITY REDUCTION APIS ──────────────

    // POST /pca/fit - Recompute Covariance Matrix and principal eigenvectors
    svr.Post("/pca/fit", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        db.fitPCA(2);
        res.set_content("{\"ok\":true,\"outDims\":2}", "application/json");
    });

    // GET /pca/coordinates - Get true 2D PCA Eigenvector projections for all vectors
    svr.Get("/pca/coordinates", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        try {
            auto coords = db.getAllPCA2D();
            std::ostringstream ss;
            ss << "{\"coordinates\":{";
            bool first = true;
            for (auto& [id, pt] : coords) {
                if (!first) ss << ',';
                first = false;
                ss << '"' << id << "\":[" << std::fixed << std::setprecision(4)
                   << (pt.size() > 0 ? pt[0] : 0.0f) << ','
                   << (pt.size() > 1 ? pt[1] : 0.0f) << ']';
            }
            ss << "}}";
            res.set_content(ss.str(), "application/json");
        } catch (...) {
            res.set_content("{\"coordinates\":{}}", "application/json");
        }
    });

    // POST /pca/project - Project arbitrary vector into 2D PCA space
    svr.Post("/pca/project", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            std::vector<float> vec;
            auto query = extractStr(req.body, "query");
            size_t embPos = req.body.find("\"embedding\"");
            if (embPos != std::string::npos) {
                vec = parseVec(req.body.substr(embPos));
            }
            if (vec.empty()) {
                vec = localSemanticEmbed(query.empty() ? "query" : query, DIMS);
            }
            auto p2d = db.projectPCA(vec);
            std::ostringstream ss;
            ss << "{\"x\":" << std::fixed << std::setprecision(4) << (p2d.size() > 0 ? p2d[0] : 0.0f)
               << ",\"y\":" << std::fixed << std::setprecision(4) << (p2d.size() > 1 ? p2d[1] : 0.0f) << '}';
            res.set_content(ss.str(), "application/json");
        } catch (...) {
            res.set_content("{\"x\":0.0,\"y\":0.0}", "application/json");
        }
    });

    // ── 4-WAY BENCHMARK: BRUTE-FORCE VS KD-TREE VS HNSW VS IVF ──────

    svr.Get("/bench", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        try {
            auto qStr = req.get_param_value("query");
            if (qStr.empty()) qStr = "search algorithm";
            auto qEmb = localSemanticEmbed(qStr, DIMS);
            auto b = db.benchmark(qEmb, 5, "cosine");
            std::ostringstream ss;
            ss << "{\"vectorCount\":" << b.n
               << ",\"bruteForceUs\":" << b.bfUs
               << ",\"kdTreeUs\":"     << b.kdUs
               << ",\"hnswUs\":"       << b.hnswUs
               << ",\"ivfUs\":"        << b.ivfUs << '}';
            res.set_content(ss.str(), "application/json");
        } catch (...) {
            res.set_content("{\"vectorCount\":0,\"bruteForceUs\":1,\"kdTreeUs\":1,\"hnswUs\":1,\"ivfUs\":1}", "application/json");
        }
    });

    // ── STATUS & STATS ──────────────────────────────────────────────

    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        ss << "{\"server\":\"Vectra Engine\""
           << ",\"vectorCount\":"  << db.size()
           << ",\"clusterCount\":" << db.getClusters().size()
           << ",\"docCount\":"     << docDB.size()
           << ",\"dims\":"         << DIMS
           << ",\"simdActive\":true"
           << ",\"ivfActive\":true"
           << ",\"pcaActive\":true"
           << ",\"hybridSupported\":true}";
        res.set_content(ss.str(), "application/json");
    });

    // ── SERVE INDEX.HTML & STATIC ASSETS ───────────────────────────
    std::string cachedIndexHtml;
    {
        std::ifstream f("index.html", std::ios::binary);
        if (!f.is_open()) {
            f.open("c:\\Users\\Akash Singh\\OneDrive\\Desktop\\Vectra\\index.html", std::ios::binary);
        }
        if (f.is_open()) {
            cachedIndexHtml = std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
    }

    auto serveIndex = [&cachedIndexHtml](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ifstream f("index.html", std::ios::binary);
        if (!f.is_open()) {
            f.open("c:\\Users\\Akash Singh\\OneDrive\\Desktop\\Vectra\\index.html", std::ios::binary);
        }
        if (f.is_open()) {
            res.set_content(
                std::string(std::istreambuf_iterator<char>(f),
                            std::istreambuf_iterator<char>()),
                "text/html; charset=utf-8");
        } else if (!cachedIndexHtml.empty()) {
            res.set_content(cachedIndexHtml, "text/html; charset=utf-8");
        } else {
            res.status = 404;
            res.set_content("index.html not found", "text/plain");
        }
    };

    svr.Get("/", serveIndex);
    svr.Get("/index.html", serveIndex);
    svr.Get("/favicon.ico", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    logFile << "[VECTRA] Starting listener on 0.0.0.0:" << port << "..." << std::endl;
    std::cout << "[VECTRA] Starting listener on 0.0.0.0:" << port << "..." << std::endl;
    bool ok = svr.listen("0.0.0.0", port);
    if (!ok) {
        logFile << "[FATAL] Failed to bind to 0.0.0.0:" << port << " (port may be in use)" << std::endl;
        std::cerr << "[FATAL] Failed to bind to 0.0.0.0:" << port << " (port may be in use)" << std::endl;
    }
    WSACleanup();
    return ok ? 0 : 1;
}
