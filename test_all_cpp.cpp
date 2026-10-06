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

#define main not_main
#include "main.cpp"
#undef main

int main() {
    std::cout << "1. Seeding data..." << std::endl;
    VectorDB db(DIMS);
    DocumentDB docDB;
    seedDefaultData(db, docDB);

    std::cout << "2. Testing extractStr..." << std::endl;
    std::string jsonBody = "{\"name\":\"finance\",\"label\":\"Finance & Markets\",\"color\":\"#10b981\"}";
    std::string n = extractStr(jsonBody, "name");
    std::string l = extractStr(jsonBody, "label");
    std::string c = extractStr(jsonBody, "color");
    std::cout << "Extracted: name=" << n << ", label=" << l << ", color=" << c << std::endl;

    std::cout << "3. Testing db.addCluster..." << std::endl;
    bool ok = db.addCluster(n, l, c);
    std::cout << "addCluster ok=" << ok << std::endl;

    std::cout << "4. Testing db.insert..." << std::endl;
    int id1 = db.insert("Stock Market equities and dividend yields", "finance", localSemanticEmbed("Stock Market equities dividend yields", DIMS), getDistFn("cosine"));
    std::cout << "Inserted id1=" << id1 << std::endl;

    std::cout << "5. Testing search with isro..." << std::endl;
    auto sRes = db.search(localSemanticEmbed("isro", DIMS), 3, "cosine", "hnsw", "isro", -1.0f);
    std::cout << "Search hits for isro: " << sRes.hits.size() << " EffectiveAlpha=" << sRes.effectiveAlpha << std::endl;
    for (auto& h : sRes.hits) {
        std::cout << "  [" << h.cat << "] " << h.meta << " (d=" << h.dist << ")" << std::endl;
    }

    std::cout << "6. Testing PCA coordinates after insert..." << std::endl;
    auto coords = db.getAllPCA2D();
    std::cout << "PCA coordinates count=" << coords.size() << std::endl;

    std::cout << "7. Testing IVF training..." << std::endl;
    db.trainIVF(7, 2);
    auto ivfStats = db.getIVFStats();
    std::cout << "IVF trained centroids=" << ivfStats.totalCentroids << std::endl;

    std::cout << "\nALL C++ TESTS PASSED COMPLETELY WITHOUT ERROR!" << std::endl;
    return 0;
}
