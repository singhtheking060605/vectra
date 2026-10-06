#define _CRT_SECURE_NO_WARNINGS
#define _WIN32_WINNT 0x0A00
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include "httplib.h"

int main() {
    httplib::Server svr;
    svr.Get("/test", [](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[GET /test] received" << std::endl;
        res.set_content("ok", "text/plain");
    });
    svr.Post("/post_test", [](const httplib::Request& req, httplib::Response& res) {
        std::cout << "[POST /post_test] body size=" << req.body.size() << ", body=" << req.body << std::endl;
        res.set_content("ok", "text/plain");
    });
    std::cout << "Listening on 8081..." << std::endl;
    svr.listen("0.0.0.0", 8081);
    return 0;
}
