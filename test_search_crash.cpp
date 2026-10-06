#include <iostream>
#include "main.cpp"

int main() {
    std::cout << "Testing DB Initialization..." << std::endl;
    VectorDB db(16);
    DocumentDB docDB;
    seedDefaultData(db, docDB);
    std::cout << "Seeded successfully! Size: " << db.size() << std::endl;

    std::cout << "Testing search with Auto-Alpha..." << std::endl;
    auto qEmb = localSemanticEmbed("dynamic programming memoization", 16);
    auto out = db.search(qEmb, 3, "cosine", "hnsw", "dynamic programming memoization", -1.0f);
    std::cout << "Search complete! Hits: " << out.hits.size() << " EffectiveAlpha: " << out.effectiveAlpha << std::endl;
    for (auto& h : out.hits) {
        std::cout << "Hit ID: " << h.id << " Cat: " << h.cat << " Dist: " << h.dist << " Meta: " << h.meta << std::endl;
    }

    std::cout << "Testing insert into gaming..." << std::endl;
    db.addCluster("gaming", "Video Games", "#a855f7");
    auto emb1 = localSemanticEmbed("Unreal Engine: real-time ray tracing shaders", 16);
    int id1 = db.insert("Unreal Engine: real-time ray tracing shaders", "gaming", emb1, getDistFn("cosine"));
    std::cout << "Inserted id: " << id1 << std::endl;

    auto emb2 = localSemanticEmbed("Counter Strike: tactical fps competitive matchmaking", 16);
    int id2 = db.insert("Counter Strike: tactical fps competitive matchmaking", "gaming", emb2, getDistFn("cosine"));
    std::cout << "Inserted id: " << id2 << std::endl;

    auto emb3 = localSemanticEmbed("Elden Ring: open world action rpg boss mechanics", 16);
    int id3 = db.insert("Elden Ring: open world action rpg boss mechanics", "gaming", emb3, getDistFn("cosine"));
    std::cout << "Inserted id: " << id3 << std::endl;

    std::cout << "Testing search after insertion..." << std::endl;
    auto sRes = db.search(emb1, 3, "cosine", "hnsw", "fps competitive", 1.0f);
    std::cout << "Search hits: " << sRes.hits.size() << std::endl;

    std::cout << "All direct C++ tests passed!" << std::endl;
    return 0;
}
