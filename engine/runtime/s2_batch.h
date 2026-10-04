// S2 batched decode driver, host only (component 6).
// Runs S sessions' MTP sequences with their rows sharing engine steps. Per session the step sequence is the one
// rank.cuh mtp_sequence runs for that session alone (same rows, positions, sample keys, contexts, in the same order):
//   prompt   each session's prompt steps run as that session's own step (one session per step, scheduler rule ii),
//            each followed by its first-form draft step, as in mtp_sequence
//   chain    K-1 chain steps, all sessions' rows in one step each (one row per session, class c1)
//   rounds   verify: one step, every active session's K+1 rows (class of K+1); accept per session; emit per session
//            first-form draft: one step per row class of a+1 (A1 on drafts: c1 = sessions with a = 0, c3 = a = 1..2,
//            c5 = a >= 3), then the K-1 chain steps (all those sessions, class c1)
// A session leaves when its emit says finished (right after the verify, as mtp_sequence returns there) or at the start
// of the first round after it holds n_gen tokens (mtp_sequence runs that round's draft steps first, so this does too).
// Sessions join at round boundaries (Batch::admit) and may be cancelled there (Batch::cancel); see class Batch.
// Within a step the sessions are in ascending index and each session's rows are contiguous, in position order.
// No CUDA types: the engine adapter (m4sim / g6m4) turns a Step into its StepIn and per-session layout.
#pragma once
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace s2
{

// the k1 split classes of bind::k1_split: 1 row / 2-3 rows / 4+ rows (A1: chosen by a session's own row count)
inline int row_class(int rows) { return rows == 1 ? 1 : rows <= 3 ? 3 : 5; }

struct Row
{
    int sess, tok, pos, spos, tgt;    // spos: the position the row's sample is keyed to (SampleRow pos)
};
// one engine step. sess / row0 / rows / ctx: per session in the step (ctx = that session's last row position + 1)
struct Step
{
    enum Kind { MAIN, DRAFT, CHAIN } kind = MAIN;
    // first-step timer: what this step is, for E1/E2 grouping.
    // TAIL = the final PromptTail prompt MAIN (design E1); DECODE1 = the first Decode-kind round MAIN
    // after the tail (design E2); ROUND = every steady decode-round MAIN. DRAFT/CHAIN steps carry
    // the class of the MAIN they follow (TAIL/DECODE1/ROUND), so the readout pairs them.
    enum Event { TAIL, DECODE1, ROUND } event = ROUND;
    // first-step timer: per-session event class, parallel to sess.
    // A steady session in a mixed step keeps its own class; `event` = the strongest class in the step.
    std::vector<int> ev;
    int cls = 1;
    std::vector<int> sess, row0, rows, ctx;
    std::vector<Row> r;
    int M() const { return (int)r.size(); }
    // per-session tag of the step: kind, then (session, rows) pairs, used as the graph / binding key
    std::string key() const
    {
        std::string k = kind == MAIN ? "m" : kind == DRAFT ? "d" : "c";
        k += std::to_string(cls);
        for (size_t i = 0; i < sess.size(); ++i) k += ":" + std::to_string(sess[i]) + "x" + std::to_string(rows[i]);
        return k;
    }
};

struct Session
{
    std::vector<int> prompt;
    int first = 0;                    // prompt tokens before this index were taken by a prefill path
    int chunk = 4;                    // rows per prompt step
    int n_gen = 16;
};
struct Stats
{
    long rounds = 0, tokens = 0;
    std::vector<long> acc;            // [K+1] rounds by accepted drafts
};

// run(step) -> the sampled token of every row of the step (step.M() ids), for MAIN, DRAFT and CHAIN steps
// round(sess, round, base, a, K): per-session round line (mtp_sequence prints "round N base B accepted a of K")
// emit(sess, toks, k) -> true = the session is finished (stop id or max tokens)
// returns per session the generated tokens (as mtp_sequence returns them)
struct Hooks
{
    std::function<std::vector<int>(const Step&)> run;
    std::function<void(int, long, int, int, int)> round;
    std::function<bool(int, const int*, int)> emit;
};

// Persistent driver state for the scheduler (component 5): stable slots 0..S2N_SLOTS-1, all calls at round boundaries.
//   admit(slots, sessions)  prompt (tail) steps of each new session as its own steps (+ first-form drafts), in the order
//                           given, then one batched chain over those still live. A session finished by emit during its
//                           prompt is finished at once (finished(q), result(q)).
//   round(order)            one round over the sessions of `order` that are active (admitted, not finished, fewer than
//                           n_gen tokens), rows in that order; false = none active (nothing ran)
//   cancel(q)               the session leaves now (result(q) = what it committed); its slot can be admitted again
//   active(q) / finished(q) / result(q) / stats(q)
// Acceptance and draft state stay here; the scheduler keeps generations and pages. mtp_batch below = admit(all) and
// rounds in slot order until none is active.
constexpr int S2N_SLOTS = 8;
class Batch
{
public:
    Batch(int K, int max_rows, Hooks h) : K_(K), max_rows_(max_rows), h_(std::move(h)), st_(S2N_SLOTS), stats_(S2N_SLOTS) {}

    void admit(const std::vector<int>& qs, const std::vector<Session>& ss)
    {
        if (qs.size() != ss.size()) throw std::runtime_error("s2: admit: slots and sessions differ in count");
        std::vector<int> live;
        for (size_t k = 0; k < qs.size(); ++k)
        {
            const int q = qs[k];
            if (q < 0 || q >= S2N_SLOTS || st_[q].on) throw std::runtime_error("s2: admit: slot " + std::to_string(q) + " is out of range or in use");
            if (prompt(q, ss[k])) live.push_back(q);
        }
        { std::vector<int> evs; for (size_t k = 0; k < live.size(); ++k) evs.push_back(Step::TAIL);
          chain(live, evs); } // first-step timer: post-admit chains follow the tail steps
    }
    bool active(int q) const { const St& x = st_[q]; return x.on && !x.done && (int)x.tokens.size() - x.n < x.s.n_gen; }
    bool finished(int q) const { return st_[q].on && !active(q); }
    void cancel(int q) { St& x = st_[q]; x.done = true; x.on = false; }
    std::vector<int> result(int q) const
    {
        const St& x = st_[q];
        const size_t end = std::min(x.tokens.size(), (size_t)x.n + x.s.n_gen);
        return std::vector<int>(x.tokens.begin() + std::min((size_t)x.n, end), x.tokens.begin() + end);
    }
    const Stats& stats(int q) const { return stats_[q]; }
    // the slot can be admitted again (after finished(q) or cancel(q))
    void release(int q) { st_[q].on = false; }

    // one decode round of every active session. Sessions are grouped, in order, so a main step holds at most max_rows rows
    // (S2-STATE.md: S = 3, 4 at mtp 2 with MX = 8 run two groups per round); a group runs its main step, its first-form
    // drafts and its chain like a whole round. A session's rows and steps do not depend on the grouping (row invariance, A1).
    bool round(const std::vector<int>& order)
    {
        std::vector<int> act;
        for (int q : order) if (active(q)) act.push_back(q);
        if (act.empty()) return false;
        const int per = max_rows_ / (K_ + 1);
        if (per < 1) throw std::runtime_error("s2: max rows < one session's verify rows");
        for (size_t g = 0; g < act.size(); g += per)
            round_group(std::vector<int>(act.begin() + g, act.begin() + std::min(act.size(), g + per)));
        return true;
    }

private:
    void round_group(const std::vector<int>& act)
    {
        Step v;
        v.kind = Step::MAIN; v.cls = row_class(K_ + 1);
        // first-step timer: E1/E2 grouping. The first decode-round MAIN of a
        // session is DECODE1 (design E2); every later round MAIN is ROUND. A step's label is the
        // LOWEST class among its sessions (an unlabeled steady session never masks another's E2).
        v.event = Step::ROUND;
        for (int q : act) if (st_[q].rounds == 0) { v.event = Step::DECODE1; break; } // strongest class in the step
        std::vector<int> base(S2N_SLOTS, 0);
        for (int q : act)
        {
            St& x = st_[q];
            base[q] = (int)x.tokens.size() - 1;
            std::vector<Row> rows;
            for (int r = 0; r <= K_; ++r)
                rows.push_back(Row{q, r ? x.d[r] : x.tokens[base[q]], base[q] + r, base[q] + r + 1, -1});
            add_rows(v, q, rows, x.rounds == 0 ? Step::DECODE1 : Step::ROUND); // first-step timer: per-session E2
        }
        check(v);
        const std::vector<int> t = h_.run(v);
        std::vector<int> on, acc(S2N_SLOTS, 0);
        for (size_t i = 0; i < v.sess.size(); ++i)
        {
            const int q = v.sess[i];
            St& x = st_[q];
            const int* so = &t[v.row0[i]];
            int a = 0;
            while (a < K_ && x.d[a + 1] == so[a]) ++a;
            for (int r = 0; r <= a; ++r) x.tokens.push_back(so[r]);
            if (h_.round) h_.round(q, stats_[q].rounds, base[q], a, K_);
            ++stats_[q].rounds; stats_[q].tokens += a + 1; ++stats_[q].acc[a];
            ++st_[q].rounds;
            acc[q] = a;
            if (h_.emit && h_.emit(q, so, a + 1)) { x.done = true; continue; }
            on.push_back(q);
        }
        for (int c : {1, 3, 5})                                   // first-form draft, one step per row class (A1)
        {
            Step d;
            d.kind = Step::DRAFT; d.cls = c; d.event = v.event;
            for (size_t i = 0; i < v.sess.size(); ++i)
            {
                const int q = v.sess[i];
                if (st_[q].done || row_class(acc[q] + 1) != c) continue;
                add_rows(d, q, draft_rows(q, acc[q] + 1, base[q], &t[v.row0[i]]), v.ev[i]);
            }
            if (d.sess.empty()) continue;
            check(d);
            const std::vector<int> u = h_.run(d);
            for (size_t i = 0; i < d.sess.size(); ++i) st_[d.sess[i]].d[1] = u[d.row0[i] + d.rows[i] - 1];
        }
        { // first-step timer: chain carries each survivor's own round class
            std::vector<int> evs;
            for (int q : on) { size_t i = 0; while (i < v.sess.size() && v.sess[i] != q) ++i;
                evs.push_back(i < v.sess.size() ? v.ev[i] : int(v.event)); }
            chain(on, evs);
        }
    }

    struct St
    {
        Session s;
        std::vector<int> tokens, d;
        int n = 0;
        bool on = false, done = false;
        int rounds = 0; // first-step timer: per-session decode rounds; round 1 MAIN = DECODE1 (design E2)
    };
    int K_, max_rows_;
    Hooks h_;
    std::vector<St> st_;
    std::vector<Stats> stats_;

    static void add_rows(Step& s, int q, const std::vector<Row>& rows, int ev)
    {
        s.sess.push_back(q);
        s.row0.push_back(s.M());
        s.rows.push_back((int)rows.size());
        s.ctx.push_back(rows.back().pos + 1);
        s.r.insert(s.r.end(), rows.begin(), rows.end());
        s.ev.push_back(ev); // first-step timer: per-session event class
    }
    void check(const Step& s) const
    {
        if (s.M() > max_rows_) throw std::runtime_error("s2: step of " + std::to_string(s.M()) + " rows > max rows " + std::to_string(max_rows_));
    }
    // draft rows of session q at positions pos0.. with next tokens next[0..M-1] (rank.cuh draft_inputs)
    static std::vector<Row> draft_rows(int q, int M, int pos0, const int* next)
    {
        std::vector<Row> v;
        for (int r = 0; r < M; ++r) v.push_back(Row{q, next[r], pos0 + r, pos0 + r + 2, -1});
        return v;
    }
    // chain steps for the sessions in qs (each: base = its tokens.size() - 1; d[1] set; fills d[2..K])
    void chain(const std::vector<int>& qs, const std::vector<int>& evs)
    {
        // first-step timer: each session's chain carries THAT session's round class (evs, parallel to qs).
        for (int j = 1; j < K_ && !qs.empty(); ++j)
        {
            Step s;
            s.kind = Step::CHAIN; s.cls = 1;
            for (size_t k = 0; k < qs.size(); ++k)
            {
                const int q = qs[k];
                const int base = (int)st_[q].tokens.size() - 1;
                add_rows(s, q, draft_rows(q, 1, base - 1 + j, &st_[q].d[j]), evs[k]);
            }
            check(s);
            const std::vector<int> t = h_.run(s);
            for (size_t i = 0; i < s.sess.size(); ++i) st_[s.sess[i]].d[j + 1] = t[s.row0[i]];
        }
    }
    // the prompt steps of session q as its own steps (rank.cuh mtp_sequence, prompt loop); false = finished by emit
    bool prompt(int q, const Session& p)
    {
        St& x = st_[q];
        x = St();
        x.s = p; x.on = true;
        x.tokens = p.prompt;
        x.n = (int)p.prompt.size();
        x.d.assign(K_ + 1, 0);
        stats_[q] = Stats();
        stats_[q].acc.assign(K_ + 1, 0);
        const int n = x.n, PS = (n - p.first + p.chunk - 1) / p.chunk;
        for (int s = 0; s < PS; ++s)
        {
            const int left = n - p.first - s * p.chunk, M = left < p.chunk ? left : p.chunk;
            std::vector<Row> rows;
            for (int r = 0; r < M; ++r)
            {
                const int pos = p.first + s * p.chunk + r;
                rows.push_back(Row{q, p.prompt[pos], pos, pos + 1, pos + 1 < n ? p.prompt[pos + 1] : -1});
            }
            Step m;
            m.kind = Step::MAIN; m.cls = row_class(M); m.event = s == PS - 1 ? Step::TAIL : Step::ROUND; // single session: step event = session event
            add_rows(m, q, rows, s == PS - 1 ? Step::TAIL : Step::ROUND);
            check(m);
            const std::vector<int> t = h_.run(m);
            if (s == PS - 1)
            {
                x.tokens.push_back(t[M - 1]);
                if (h_.emit && h_.emit(q, &t[M - 1], 1)) { x.done = true; return false; }
            }
            std::vector<int> next(M);
            for (int r = 0; r < M; ++r) next[r] = x.tokens[rows[r].pos + 1];
            Step d;
            d.kind = Step::DRAFT; d.cls = row_class(M); d.event = m.event;
            add_rows(d, q, draft_rows(q, M, rows[0].pos, next.data()), m.ev.back());
            check(d);
            const std::vector<int> u = h_.run(d);
            x.d[1] = u[M - 1];
        }
        return true;
    }
};

inline std::vector<std::vector<int>> mtp_batch(const std::vector<Session>& S, int K, int max_rows, const Hooks& h,
                                               std::vector<Stats>& stats)
{
    const int NS = (int)S.size();
    if (NS > S2N_SLOTS) throw std::runtime_error("s2: more sessions than slots");
    Batch b(K, max_rows, h);
    std::vector<int> qs(NS);
    for (int q = 0; q < NS; ++q) qs[q] = q;
    b.admit(qs, S);
    while (b.round(qs)) {}
    std::vector<std::vector<int>> out(NS);
    stats.assign(NS, Stats());
    for (int q = 0; q < NS; ++q) { out[q] = b.result(q); stats[q] = b.stats(q); }
    return out;
}

}   // namespace s2
