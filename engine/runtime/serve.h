// --serve DIR (g6m4 and m4sim): requests come as files, rank 0 streams the committed tokens to a file; the front-end
// shim (frontend/g6shim.py) turns that into PROTOCOL.md frames. Every rank reads the same request file and runs the
// same steps; only rank 0 writes.
//   DIR/req.<n>  (written atomically: req.<n>.tmp then rename), two lines:
//       n_gen <max_tokens> temp <float> seed <uint32> min <min_tokens> stop <id,id,..|-> [top_p <float>]
//       <prompt ids, comma-separated>
//   DIR/quit     ends the loop
//   DIR/ready    (every rank) written once the model is loaded, before the first request is read
//   DIR/resp.<n> (rank 0, one flushed line per event):
//       t <id> <id> ..                                   newly committed tokens
//       m <rounds> <draft_tokens> <accepted> <acc_pos0> ..   cumulative MTP counters since start
//       end <stop|length|error> <stop_id|-1> <completion_tokens>
#pragma once
#include <sys/stat.h>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace g6rt {

struct ServeReq
{
    std::vector<int> prompt, stop;
    int n_gen = 0, min_tokens = 0;
    float temp = 0.f, top_p = 0.f;
    uint32_t seed = 1234u;
};

inline bool serve_exists(const std::string& p) { struct stat s; return stat(p.c_str(), &s) == 0; }

// DIR/ready: the rank has loaded and waits for requests
inline void serve_ready(const std::string& dir)
{
    if (FILE* f = fopen((dir + "/ready").c_str(), "w")) fclose(f);
}

// blocks until DIR/req.<n> or DIR/quit exists; false on quit
inline bool serve_next(const std::string& dir, size_t n, ServeReq& r)
{
    const std::string f = dir + "/req." + std::to_string(n);
    for (;;)
    {
        if (serve_exists(f)) break;
        if (serve_exists(dir + "/quit")) return false;
        usleep(300);
    }
    FILE* fp = fopen(f.c_str(), "r");
    if (!fp) return false;
    r = ServeReq();
    char stop[65536] = {0};
    if (fscanf(fp, "n_gen %d temp %f seed %u min %d stop %65535s", &r.n_gen, &r.temp, &r.seed, &r.min_tokens, stop) != 5) r.n_gen = 0;
    if (fscanf(fp, " top_p %f", &r.top_p) != 1) r.top_p = 0.f;    // optional, same line (older shims: absent)
    std::string ids;
    int c;
    while ((c = fgetc(fp)) != EOF) ids.push_back((char)c);
    fclose(fp);
    auto ints = [](const std::string& s) {
        std::vector<int> v;
        const char* p = s.c_str();
        while (*p)
        {
            char* e;
            const long x = strtol(p, &e, 10);
            if (e == p) { ++p; continue; }
            v.push_back((int)x);
            p = e;
        }
        return v;
    };
    if (strcmp(stop, "-") != 0) r.stop = ints(stop);
    r.prompt = ints(ids);
    return true;
}

// G6_PF_TIMES=1 (measurement only, default off; host clock only, no extra syncs): wall-clock laps of one request on this
// process, from the request file's mtime to the first committed token; one stderr line per request:
//   # times seq <n> <item>=<ms> .. total=<ms>
struct PfTimes
{
    bool on = getenv("G6_PF_TIMES") && atoi(getenv("G6_PF_TIMES")) == 1;
    std::chrono::steady_clock::time_point t;
    std::string s;
    double tot = 0;
    bool first = false;
    // start of a request; file: the request file (its mtime -> the 'poll' item), or empty
    void start(const std::string& file)
    {
        if (!on) return;
        t = std::chrono::steady_clock::now(); s.clear(); tot = 0; first = false;
        struct stat st;
        if (!file.empty() && stat(file.c_str(), &st) == 0)
        {
            timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            const double ms = (now.tv_sec - st.st_mtim.tv_sec) * 1e3 + (now.tv_nsec - st.st_mtim.tv_nsec) * 1e-6;
            add("poll", ms);
        }
    }
    void add(const char* k, double ms) { char b[64]; snprintf(b, sizeof b, " %s=%.3f", k, ms); s += b; tot += ms; }
    void lap(const char* k)
    {
        if (!on) return;
        const auto n = std::chrono::steady_clock::now();
        add(k, std::chrono::duration<double, std::milli>(n - t).count());
        t = n;
    }
    void first_token(size_t q)                       // call when the first token is committed
    {
        if (!on || first) return;
        first = true;
        lap("to_first_token");
        fprintf(stderr, "# times seq %zu%s total=%.3f\n", q, s.c_str(), tot);
    }
};

// counts committed tokens against n_gen, stop ids and min_tokens; rank 0 writes them to DIR/resp.<n>
struct Emitter
{
    FILE* f = nullptr;
    bool on = false;
    int n_gen = 0, min_tokens = 0, emitted = 0, stop_id = -1;
    bool done = false;
    std::vector<int> stop;
    void open(const std::string& dir, size_t n, bool rank0, const ServeReq& r)
    {
        on = true; n_gen = r.n_gen; min_tokens = r.min_tokens; stop = r.stop; emitted = 0; stop_id = -1; done = false;
        f = rank0 ? fopen((dir + "/resp." + std::to_string(n)).c_str(), "w") : nullptr;
    }
    // returns true when the request is finished (a stop id was emitted or n_gen reached)
    bool push(const int* t, int k)
    {
        if (!on) return false;
        if (done) return true;
        std::string line = "t";
        for (int i = 0; i < k && !done; ++i)
        {
            line += " " + std::to_string(t[i]);
            ++emitted;
            for (int s : stop)
                if (t[i] == s && emitted > min_tokens) { stop_id = s; done = true; }
            if (emitted >= n_gen) done = true;
        }
        if (f) { fprintf(f, "%s\n", line.c_str()); fflush(f); }
        return done;
    }
    void end(const char* reason_if_error, long rounds, int K, long tokens, const long* acc)
    {
        if (!on) return;
        if (f)
        {
            if (K > 0)
            {
                fprintf(f, "m %ld %ld %ld", rounds, rounds * K, tokens - rounds);
                for (int i = 0; i < K; ++i)
                {
                    long c = 0;
                    for (int a = i + 1; a <= K; ++a) c += acc[a];
                    fprintf(f, " %ld", c);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "end %s %d %d\n", reason_if_error ? reason_if_error : stop_id >= 0 ? "stop" : "length", stop_id, emitted);
            fclose(f);
            f = nullptr;
        }
        on = false;
    }
};

}   // namespace g6rt
