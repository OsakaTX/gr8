// relay.h: ring routes, and the relay rings' credits under random sizes and delays. A model ring of bytes checks that a
// payload is never overwritten before its forward completes and that every payload arrives, in order, unchanged.
//   g++ -std=c++17 -O2 -I engine/comms engine/comms/tests/relay_test.cpp -o relay_test && ./relay_test
#include "relay.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace g6;

static int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); fails++; } } while (0)

static void routes() {
  // TP=4: diagonals go through the next neighbour from even ranks and the prev neighbour from odd ranks
  CHECK(ring_via(0, 2, 4) == 1 && ring_via(1, 3, 4) == 0 && ring_via(2, 0, 4) == 3 && ring_via(3, 1, 4) == 2);
  for (int me = 0; me < 4; me++)
    for (int p = 0; p < 4; p++)
      if (p != me && ring_neighbour(me, p, 4)) CHECK(ring_via(me, p, 4) == p);
  // every directed cable carries one relay leg: the first leg me -> via, the second via -> diagonal
  int legs[4][4] = {};
  for (int me = 0; me < 4; me++) {
    const int d = (me + 2) % 4, v = ring_via(me, d, 4);
    legs[me][v]++; legs[v][d]++;
  }
  for (int a = 0; a < 4; a++)
    for (int b = 0; b < 4; b++)
      if (ring_neighbour(a, b, 4)) CHECK(legs[a][b] == 1);
  // a link's rail is the same from both ends, and each TP=4 rank has one link on each rail
  for (int nr : {3, 4})
    for (int me = 0; me < nr; me++)
      for (int p = 0; p < nr; p++)
        if (p != me && ring_neighbour(me, p, nr)) CHECK(ring_rail(me, p, nr) == ring_rail(p, me, nr));
  for (int me = 0; me < 4; me++) CHECK(ring_rail(me, ring_next(me, 4), 4) != ring_rail(me, ring_prev(me, 4), 4));
  // TP=3: every peer is a neighbour, so nothing is relayed
  for (int me = 0; me < 3; me++)
    for (int p = 0; p < 3; p++)
      if (p != me) CHECK(ring_neighbour(me, p, 3) && ring_via(me, p, 3) == p);
}

static void credits(uint64_t ring, int msgs, unsigned seed) {
  std::mt19937 rng(seed);
  RelayOut out; RelayIn in;
  std::vector<uint32_t> mem(ring / 4, 0);          // the relay's data ring, one u32 tag per 4 bytes
  std::vector<RelayHdr> hdrs;                      // header ring as the relay reads it (index = seq - 1)
  std::deque<RelayHdr> wire;                       // posted, not yet read by the relay
  int sent = 0, delivered = 0; uint32_t next_tag = 1;
  std::vector<uint32_t> want_tags;
  while (delivered < msgs) {
    const int act = (int)(rng() % 3);
    if (act == 0 && sent < msgs) {                 // sender: one write, inline for a quarter of them
      const bool inl = rng() % 4 == 0;
      const uint32_t len = inl ? 8 : 8 * (1 + rng() % (uint32_t)(ring / 32));
      uint64_t pos;
      if (out.reserve(inl ? 0 : len, ring, in.hdr_done, in.data_done, &pos)) {
        RelayHdr h{}; h.dest_off = (uint64_t)sent; h.data_pos = pos; h.len = len; h.inl = inl; h.seq = out.next_hdr;
        if (!inl) for (uint64_t b = 0; b < len; b += 4) mem[((pos + b) % ring) / 4] = next_tag;
        else h.ival = next_tag;
        want_tags.push_back(next_tag++);
        wire.push_back(h); sent++;
      }
    } else if (act == 1 && !wire.empty()) {        // relay: read a header and post its forward (reads the ring now)
      const RelayHdr h = wire.front(); wire.pop_front();
      CHECK(h.seq == in.next + 1);
      in.next++;
      const uint32_t tag = h.inl ? (uint32_t)h.ival : mem[(h.data_pos % ring) / 4];
      if (!h.inl) for (uint64_t b = 0; b < h.len; b += 4) CHECK(mem[((h.data_pos + b) % ring) / 4] == tag);
      CHECK(h.inl || h.data_pos % ring + relay_pad(h.len) <= ring);
      CHECK(tag == want_tags[(size_t)h.dest_off]);
      hdrs.push_back(h);
      in.forwarded(h);
    } else if (act == 2 && !in.fly.empty()) {      // the oldest forward completes: its bytes may now be reused
      const uint64_t idx = in.fly.front().first - 1;
      const RelayHdr& h = hdrs[(size_t)idx];
      if (!h.inl) for (uint64_t b = 0; b < h.len; b += 4) CHECK(mem[((h.data_pos + b) % ring) / 4] == want_tags[(size_t)h.dest_off]);
      in.completed();
      delivered++;
    }
  }
  CHECK(in.hdr_done == (uint64_t)msgs && out.next_hdr == (uint64_t)msgs);
}

int main() {
  routes();
  for (unsigned s = 1; s <= 20; s++) credits(4096, 3000, s);
  credits(1 << 20, 20000, 99);
  if (fails) { fprintf(stderr, "%d failures\n", fails); return 1; }
  printf("relay_test: OK\n");
  return 0;
}
