#pragma once
#include "attn.h"

// Allocate with cudaMalloc, not pinned host or managed memory. A glue kernel
// updates the first four words before every attention graph invocation.
// Positions, block-table entries and projection inputs live at fixed device
// addresses in G6Attn and are updated in place, not baked-in scalar values.
struct G6AttnStepCell {
    int context, page_count, table_size, reuse_rows;
    unsigned valid; // written by captured validation kernel; not caller input
};
struct G6AttnReplay {
    G6Attn fixed;   // context/page_count/table_size are allocation limits
    G6AttnStepCell* cell;
    void* scratch;
    size_t scratch_bytes;
};

// Capture/replay entry: topology is fixed by rows, reuse mode and capacity.
// All per-step validation/clearing uses a captured device kernel. No allocation,
// host reads of live control words, graph update or recapture is required.
extern "C" int g6_attn_replay_run(const G6AttnReplay*, void* stream);
