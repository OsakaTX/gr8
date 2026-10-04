// host-only: the row shape of one draft step (shared by rank.cuh and the S2 CPU test doubles)
#pragma once
// A chain draft step carries one row per session: M == 1 outside S2 (solo / M4 driver), M == ns under S2, where a
// chain wave batches every live session at that depth (s2::draft_waves). nullptr = shape ok, else the die() message.
inline const char* draft_shape_error(int M, bool chain, bool s2on, int ns)
{
    if (!chain) return nullptr;
    if (!s2on) return M == 1 ? nullptr : "a chain draft step has one row";
    return M == ns && ns > 0 ? nullptr : "s2: a chain draft step has one row per session (M != sessions)";
}
