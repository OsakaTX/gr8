// relay.h under random sizes and credit delays: a model ring of bytes checks that a payload is never overwritten before
// the relay has forwarded it, and that every payload arrives in order, unchanged.
//   g++ -std=c++17 -O2 -I engine/comms engine/comms/tests/relay_test.cpp -o relay_test && ./relay_test
#include "relay.h"

#include <cstdio>
#include <deque>
#include <random>
#include <vector>

using namespace g6;

static int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); fails++; } } while (0)

struct Msg { uint64_t at; uint32_t len; uint32_t tag; };

static void credits(uint64_t ring, uint32_t max_len, int msgs, unsigned seed) {
  std::mt19937 rng(seed);
  RelayOut out;
  std::vector<uint32_t> mem(ring / 4, 0);   // one tag per 4 bytes of the ring
  std::deque<Msg> wire;                     // written, not yet forwarded
  uint64_t done = 0;
  int sent = 0, forwarded = 0;
  uint32_t want = 1;
  while (forwarded < msgs) {
    if (rng() % 2 == 0 && sent < msgs) {    // origin: one payload
      const uint32_t len = 4 * (1 + rng() % (max_len / 4));
      uint64_t at;
      if (out.reserve(len, ring, done, &at)) {
        CHECK(at % ring + relay_pad(len) <= ring);
        const uint32_t tag = (uint32_t)++sent;
        for (uint64_t b = 0; b < len; b += 4) mem[(at % ring + b) / 4] = tag;
        wire.push_back({at, len, tag});
      }
    } else if (!wire.empty()) {             // relay: forward the oldest payload, then return the credit
      const Msg m = wire.front(); wire.pop_front();
      CHECK(m.tag == want++);
      for (uint64_t b = 0; b < m.len; b += 4) CHECK(mem[(m.at % ring + b) / 4] == m.tag);
      done = m.at + relay_pad(m.len);
      forwarded++;
    }
  }
  CHECK(out.pos >= done);
}

int main() {
  for (unsigned s = 1; s <= 20; s++) credits(4096, 1024, 3000, s);
  credits(4096, 4096, 3000, 77);            // a payload as long as the ring
  credits(kRelayRing, kRelayChunk, 20000, 99);
  if (fails) { fprintf(stderr, "%d failures\n", fails); return 1; }
  printf("relay_test: OK\n");
  return 0;
}
