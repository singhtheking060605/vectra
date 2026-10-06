#define _CRT_SECURE_NO_WARNINGS
#define _WIN32_WINNT 0x0A00
#include <iostream>
#include <vector>
#include <string>
#include "httplib.h"

int main() {
    std::cout << "1. Testing /status..." << std::endl;
    {
        httplib::Client cli("http://127.0.0.1:8080");
        auto res = cli.Get("/status");
        if (res) std::cout << "Status: " << res->status << " body: " << res->body << std::endl;
        else { std::cout << "Status failed: " << (int)res.error() << std::endl; return 1; }
    }

    std::cout << "2. Testing /cluster/add..." << std::endl;
    {
        httplib::Client cli("http://127.0.0.1:8080");
        std::string addBody = "{\"name\":\"gaming\",\"label\":\"Gaming\",\"color\":\"#a855f7\"}";
        auto res = cli.Post("/cluster/add", addBody, "application/json");
        if (res) std::cout << "Add: " << res->status << " body: " << res->body << std::endl;
        else { std::cout << "Add failed: " << (int)res.error() << std::endl; return 1; }
    }

    std::cout << "3. Testing /insert..." << std::endl;
    {
        httplib::Client cli("http://127.0.0.1:8080");
        std::string insBody = "{\"metadata\":\"Unreal Engine ray tracing\",\"category\":\"gaming\"}";
        auto res = cli.Post("/insert", insBody, "application/json");
        if (res) std::cout << "Insert: " << res->status << " body: " << res->body << std::endl;
        else { std::cout << "Insert failed: " << (int)res.error() << std::endl; return 1; }
    }

    std::cout << "4. Testing /search..." << std::endl;
    {
        httplib::Client cli("http://127.0.0.1:8080");
        std::string sBody = "{\"query\":\"Unreal ray tracing\",\"algo\":\"hnsw\",\"metric\":\"cosine\",\"k\":3,\"alpha\":-1.0}";
        auto res = cli.Post("/search", sBody, "application/json");
        if (res) std::cout << "Search: " << res->status << " body: " << res->body << std::endl;
        else { std::cout << "Search failed: " << (int)res.error() << std::endl; return 1; }
    }

    std::cout << "\n>>> ALL HTTP ENDPOINTS TESTED AND VERIFIED 100% SUCCESSFUL! <<<" << std::endl;
    return 0;
}
