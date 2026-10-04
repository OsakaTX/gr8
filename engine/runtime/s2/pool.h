// S2 below R v1 (S2-STATE.md) run allocator over the native pool pages. Host only, no CUDA. One instance per rank;
// every call is at an all-rank event fence (service.h). Head: reserve (first-fit, lowest base). Workers: claim the
// head's exact grant. A reservation of more than `below` rows (a session past R, CP decode) runs solo at base 0: it
// is admitted only into an EMPTY pool and, while resident, holds the whole pool (free 0, nothing else admits).
// Two sizes on the CP binding: `pages` = the logical capacity (Config::pool_rows / 64; a past-R session at base 0 reads
// the table as built, CP map included), `place` = the physical pages every rank has for runs at base B (S2Model::
// pool_pages, the fewest of main and draft). Runs at or below R are placed inside [0, place) only.
#pragma once
#include "scheduler.h"
#include <vector>

namespace g6s2 {
class RunPool {
    uint32_t pages_, place_, below_rows_, version_ = 0;
    std::vector<bool> used_;
    bool exclusive_ = false;
    Run held_{};                               // the exclusive run, while resident
    uint32_t used_pages() const { uint32_t n = 0; for (bool u : used_) n += u; return n; }
    void mark(const Run& r, bool on) {
        require(r.pages && uint64_t(r.base_page) + r.pages <= pages_, "pool run outside the pool");
        for (uint32_t i = r.base_page; i < r.base_page + r.pages; ++i) {
            require(used_[i] != on, on ? "pool run overlaps a live run" : "pool release of a free page");
            used_[i] = on;
        }
    }
public:
    RunPool(uint32_t pool_pages, uint32_t place_pages, uint32_t below_rows)
        : pages_(pool_pages), place_(place_pages), below_rows_(below_rows), used_(pool_pages, false) {
        require(place_pages && place_pages <= pool_pages && below_rows && below_rows % PAGE == 0 && below_rows / PAGE <= place_pages,
                "bad S2 pool / placement / below-R size");
    }
    bool exclusive(uint64_t rows) const { return rows > below_rows_; }
    uint32_t free_pages() const { return exclusive_ ? 0 : pages_ - used_pages(); }
    uint32_t largest_run() const {
        if (exclusive_) return 0;
        if (!used_pages()) return pages_;      // empty: a past-R run takes [0, n) of the logical pool
        uint32_t best = 0, run = 0;
        for (uint32_t i = 0; i < place_; ++i) { run = used_[i] ? 0 : run + 1; best = std::max(best, run); }
        return best;
    }
    // head, before reserve: a past-R request waits for an empty pool (FIFO head blocks, as a capacity stall)
    bool admissible(uint64_t rows) const { return !exclusive_ && (!exclusive(rows) || !used_pages()); }
    Run reserve(uint64_t rows) {
        const uint32_t n = uint32_t((rows + PAGE - 1) / PAGE);
        require(rows && n <= pages_ && admissible(rows) && !exclusive_, "pool reserve without a fit");
        uint32_t base = 0, run = 0;
        const uint32_t end = exclusive(rows) ? pages_ : place_;
        for (uint32_t i = 0; i < end && run < n; ++i) {
            if (used_[i]) { run = 0; base = i + 1; } else ++run;
        }
        require(run == n, "pool reserve: no contiguous run");
        require(version_ != UINT32_MAX, "pool table version exhausted");
        Run r{base, n, ++version_};
        claim(r, rows);
        return r;
    }
    void claim(const Run& r, uint64_t rows) {
        require(r.pages == (rows + PAGE - 1) / PAGE && r.table_version >= version_ && r.table_version, "pool claim differs from the grant");
        require(!exclusive_, "pool claim while an exclusive run is resident");
        if (exclusive(rows)) require(r.base_page == 0 && !used_pages(), "exclusive run not at base 0 of an empty pool");
        else require(uint64_t(r.base_page) + r.pages <= place_, "run below R outside the physical pages");
        mark(r, true);
        version_ = r.table_version;
        if (exclusive(rows)) { exclusive_ = true; held_ = r; }
    }
    void release(const Run& r) {
        mark(r, false);
        if (exclusive_ && r.base_page == held_.base_page && r.pages == held_.pages && r.table_version == held_.table_version) {
            exclusive_ = false; held_ = {};
        }
    }
};
} // namespace g6s2
