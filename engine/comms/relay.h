// Ring fabric relay ring (fabric.cpp, FabricConfig::ring): the origin's side of the bounded buffer on the relay rank that
// a diagonal write passes through. This file touches neither verbs nor CUDA, so engine/comms/tests/relay_test.cpp can
// drive it.
#pragma once
#include <cstdint>

namespace g6 {

constexpr uint64_t kRelayRing = 32ull << 20;   // relay ring bytes per ring Fabric (at most the region's receive side)
constexpr uint32_t kRelayChunk = 1u << 20;     // a longer write crosses the relay in chunks of this size, in order

inline uint64_t relay_pad(uint64_t len) { return (len + 63) & ~63ull; }

// Positions are monotonic byte counts, and position % ring is the offset in the ring. done is the relay's credit: every
// byte below it has been forwarded, so it may be overwritten.
struct RelayOut {
  uint64_t pos = 0;   // the ring position after the last payload
  // Room for len bytes (at most ring)? Yes: *at = the payload's position, and pos advances. A payload never wraps: one
  // that would cross the end of the ring starts at the next ring start. The bytes in [done, pos) are not forwarded yet.
  bool reserve(uint32_t len, uint64_t ring, uint64_t done, uint64_t* at) {
    const uint64_t n = relay_pad(len);
    uint64_t p = pos;
    if (p % ring + n > ring) p += ring - p % ring;
    if (done != pos && p + n - done > ring) return false;
    *at = p;
    pos = p + n;
    return true;
  }
};

}  // namespace g6
