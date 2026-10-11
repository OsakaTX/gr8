// Pack::open_manifest against manifests whose numbers wrap: a negative offset, a shape whose element count overflows,
// a negative dimension. Each must be refused, so no tensor can point outside the image.
//   g++ -std=c++20 -O2 -pthread -I engine/loader engine/loader/tests/manifest_test.cpp engine/loader/g6pack.cpp -o t && ./t
#include "g6pack.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

static int fails = 0;

static std::string manifest(const std::string& shape, const std::string& offset, const std::string& nbytes)
{
    return "{\"format\":\"g6pack-v1\",\"rank\":0,\"align\":4096,\"pack_bytes\":16384,\"tensors\":[{\"name\":\"t\","
           "\"dtype\":\"bf16\",\"shape\":" + shape + ",\"offset\":" + offset + ",\"nbytes\":" + nbytes +
           ",\"sha256\":\"" + std::string(64, '0') + "\"}]}";
}

static void expect(const char* what, const std::string& text, const char* want)
{
    char path[] = "/tmp/g6pack-manifest-test.XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) { std::perror("mkstemp"); std::exit(2); }
    close(fd);
    std::ofstream(path) << text;
    g6::Pack p;
    std::string err;
    const bool ok = p.open_manifest(path, err);
    unlink(path);
    const bool pass = want ? !ok && err.find(want) != std::string::npos : ok;
    if (!pass) fails++;
    std::printf("  %-4s %-28s %s\n", pass ? "ok" : "FAIL", what, ok ? "accepted" : err.c_str());
}

int main()
{
    expect("a plain tensor", manifest("[2,2048]", "4096", "8192"), nullptr);
    expect("offset -4096 (wraps)", manifest("[4096]", "-4096", "8192"), "negative offset");
    expect("element count overflows", manifest("[4294967296,4294967296,2]", "0", "0"), "bad shape");
    expect("a negative dimension", manifest("[-1,2]", "0", "4"), "bad shape");
    expect("runs past pack_bytes", manifest("[8192]", "4096", "16384"), "past pack_bytes");
    std::printf("manifest_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
