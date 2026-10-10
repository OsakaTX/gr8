// Ring fabric (FabricConfig::ring): each rank has QPs to its two ring neighbours only. At TP=4 the diagonal peer has no
// cable, so a write to it goes into a relay inbox on a neighbour, whose proxy forwards the same bytes to the same offset
// on the diagonal. The output is the mesh's bit for bit. This file holds the parts that touch neither verbs nor CUDA, so
// engine/comms/tests/relay_test.cpp can drive them.
#pragma once
#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>

namespace g6 {

inline int ring_prev(int me, int nr) { return (me + nr - 1) % nr; }
inline int ring_next(int me, int nr) { return (me + 1) % nr; }
inline bool ring_neighbour(int me, int p, int nr) { return p == ring_prev(me, nr) || p == ring_next(me, nr); }

// The rail of the link to neighbour p: link i -> i + 1 is on rail i % 2, the same rail at both ends (on a switch the
// rails are separate subnets), and at TP=4 each rank has one link on each rail.
inline int ring_rail(int me, int p, int nr) { return (p == ring_next(me, nr) ? me : p) % 2; }

// The neighbour that relays my writes to peer p (p itself when it is a neighbour). Even ranks go through next and odd
// ranks through prev, so at TP=4 every directed cable carries exactly one relay leg.
inline int ring_via(int me, int p, int nr) {
  if (ring_neighbour(me, p, nr)) return p;
  return me % 2 == 0 ? ring_next(me, nr) : ring_prev(me, nr);
}

// One relayed write, as the sender posts it into the relay's header ring with an inline RDMA write. seq is written last
// and is the header index + 1, so the relay's proxy reads a header only once its seq has landed. An 8-byte write from
// a register (a bulk flag) travels in ival with inl = 1 and no data.
struct RelayHdr {
  uint64_t dest_off;   // offset from the destination's advertised address (Fabric peer addr)
  uint64_t data_pos;   // monotonic position of the payload in the relay's data ring (position % ring = offset)
  uint64_t ival;       // inl: the 8 bytes to write
  uint32_t len;        // payload bytes
  uint32_t inl;
  uint64_t seq;
};
static_assert(sizeof(RelayHdr) == 40, "RelayHdr is sent as one inline write");
constexpr int kRelayHdrs = 4096;            // header ring entries per inbox
constexpr size_t kRelaySlot = 64;           // one header per cache line, so seq lands with the rest of it
constexpr uint32_t kRelayChunk = 1u << 20;  // a longer write is relayed in chunks of this size, in order

inline uint64_t relay_pad(uint32_t len) { return (len + 7ull) & ~7ull; }

// Sender side. hdr_done and data_done are the relay's credits: headers whose forward has completed, and the data ring
// position below which every byte has been forwarded.
struct RelayOut {
  uint64_t next_hdr = 0;   // headers posted
  uint64_t pos = 0;        // data ring position after the last payload
  // Room for len bytes (0 for an inline write)? Yes: *data_pos = the payload's monotonic position, and the state
  // advances. A payload never wraps: one that would cross the end of the ring starts at the next ring start.
  bool reserve(uint32_t len, uint64_t ring, uint64_t hdr_done, uint64_t data_done, uint64_t* data_pos) {
    if (next_hdr - hdr_done >= (uint64_t)kRelayHdrs) return false;
    const uint64_t n = relay_pad(len);
    uint64_t p = pos;
    if (n && p % ring + n > ring) p += ring - p % ring;
    if (p + n - data_done > ring) return false;
    *data_pos = p;
    pos = p + n;
    next_hdr++;
    return true;
  }
};

// Relay side. Forwards complete in order (one QP to the destination), so the credits advance with the oldest one.
struct RelayIn {
  uint64_t next = 0;        // next header index to read
  uint64_t hdr_done = 0;    // credits: headers forwarded
  uint64_t data_done = 0;   //          data ring position forwarded
  std::deque<std::pair<uint64_t, uint64_t>> fly;   // forwards in flight: (header index + 1, payload end position)
  void forwarded(const RelayHdr& h) { fly.emplace_back(h.seq, h.inl ? 0 : h.data_pos + relay_pad(h.len)); }
  // the oldest forward completed
  void completed() {
    hdr_done = fly.front().first;
    if (fly.front().second) data_done = fly.front().second;
    fly.pop_front();
  }
};

}  // namespace g6
