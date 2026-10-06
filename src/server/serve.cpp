// `bnk serve`: the engine behind a line protocol on stdin/stdout (one JSON object per line).
//
//   in : {"op":"generate","id":..,"prompt":[ids],"max_tokens":N,"temperature":..,"top_p":..,"top_k":..,
//         "min_p":..,"presence_penalty":..,"repetition_penalty":..,"seed":..,"stop_ids":[ids],"draft":n,
//         "draft_min_p":p,"checkpoint":pos}  (checkpoint: where to snapshot state for the next request to resume)  (draft: tokens the drafter may propose per round; draft_min_p: it stops below this)
//        {"op":"cancel","id":..}   {"op":"stats"}   {"op":"reset"}   {"op":"quit"}
//   Up to `slots` generate requests run at once (each in a conversation slot of its own; --slots): one is admitted
//   (its prompt read) while the others keep decoding between its prompt chunks (BNK_READ_SHARE of each chunk's
//   time, 0.5 by default), then all of them decode together, one batched forward per round.
//   out: {"type":"ready",..}  {"type":"prefill","id":..}  {"type":"tokens","id":..,"ids":[..]}
//        {"type":"done","id":..,"reason":"stop|length|cancel|context"} {"type":"error",..}
//        {"type":"telemetry",..}: every 250 ms while working and every second while idle (also the answer to
//        "stats", sent at once from the last snapshot). Counters under "life" are cumulative over the process,
//        in-flight work included, so a reader gets exact rates from the differences of two snapshots.
#include "server/serve.h"

#include <sys/sysinfo.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "engine/generate.h"
#include "server/json.h"

namespace bnk {

namespace {

std::mutex g_out_mu;
void emit(const std::string & line) {
    std::lock_guard<std::mutex> lk(g_out_mu);
    fwrite(line.data(), 1, line.size(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

// After a request failed: when the GPU context is gone (an illegal access or another sticky error: every CUDA
// call fails from then on), the process exits, so the server restarts the engine rather than fail every request
// that follows. An ordinary error is reported once and cleared; a sticky one is returned again.
void exit_if_gpu_lost() {
    if (cudaDeviceSynchronize() == cudaSuccess) return;
    cudaGetLastError();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) return;
    fprintf(stderr, "bnk: the GPU context is lost (%s): exiting so the engine can be restarted\n", cudaGetErrorString(e));
    fflush(stderr);
    fflush(stdout);
    _exit(3);
}

struct Totals {
    int64_t requests = 0, gen_tokens = 0, prompt_tokens = 0, prefill_tokens = 0, reused = 0;
    int64_t rounds = 0, drafted = 0, accepted = 0;
    double gen_ms = 0, prefill_ms = 0, verify_ms = 0, draft_ms = 0, cpu_expert_ms = 0;
};

// What the request in flight has done so far (worker thread only).
struct Current {
    std::string id, phase = "idle";
    int64_t prompt_tokens = 0, reused = 0, prefill_done = 0, prefill_total = 0, gen_tokens = 0;
    double t_start = 0, prefill_ms = 0, gen_t0 = 0;
    // in-flight sums for the lifetime counters
    double gen_ms = 0, verify_ms = 0, draft_ms = 0;
    int64_t rounds = 0, drafted = 0, accepted = 0;
};

double wall_s() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

double rss_mb() {
    long pages = 0, res = 0;
    FILE * f = fopen("/proc/self/statm", "r");
    if (f) {
        if (fscanf(f, "%ld %ld", &pages, &res) != 2) res = 0;
        fclose(f);
    }
    return (double) res * sysconf(_SC_PAGESIZE) / 1048576.0;
}

template <typename V>
std::string arr(const V & v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
    return o + "]";
}

// cur: the request shown in "req"; inflight: the in-flight requests summed (for the lifetime counters)
std::string telemetry_json(Engine & eng, const GenStats & gs, const Totals & tot, const Current & cur,
                           const Current & inflight, int active, const std::string & model_name) {
    size_t vfree = 0, vtotal = 0;
    cudaMemGetInfo(&vfree, &vtotal);
    struct sysinfo si;
    sysinfo(&si);
    const auto & c = eng.cfg();
    const auto & tm = eng.times;
    const bool busy = cur.phase != "idle";
    const double now = now_ms();
    // the request in flight
    JsonOut rq;
    rq.kv("id", cur.id)
        .kv("prompt_tokens", cur.prompt_tokens)
        .kv("reused", cur.reused)
        .kv("prefill_done", cur.prefill_done)
        .kv("prefill_total", cur.prefill_total)
        .kv("prefill_ms", cur.phase == "prefill" ? now - cur.t_start : cur.prefill_ms)
        .kv("gen_tokens", cur.gen_tokens)
        .kv("gen_ms", cur.phase == "decode" ? now - cur.gen_t0 : 0.0)
        .kv("elapsed_ms", busy ? now - cur.t_start : 0.0)
        .kv("rounds", busy ? gs.rounds : 0)
        .kv("drafted", busy ? gs.drafted : 0)
        .kv("accepted", busy ? gs.accepted : 0)
        .kv("routed", busy ? tm.routed : 0)
        .kv("misses", busy ? tm.misses : 0);
    // lifetime counters, in-flight work included
    const int64_t fresh_now = cur.phase == "prefill" ? std::max<int64_t>(0, cur.prefill_done) : 0;
    const bool decoding = inflight.phase == "decode";
    int64_t routed = 0, misses = 0;
    for (size_t i = 0; i < eng.layer_routed.size(); ++i) {
        routed += eng.layer_routed[i];
        misses += eng.layer_misses[i];
    }
    JsonOut lf;
    lf.kv("requests", tot.requests)
        .kv("prompt_tokens", tot.prompt_tokens)
        .kv("prefill_tokens", tot.prefill_tokens + fresh_now)
        .kv("prefill_ms", tot.prefill_ms + (cur.phase == "prefill" ? now - cur.t_start : 0.0))
        .kv("gen_tokens", tot.gen_tokens + inflight.gen_tokens)
        .kv("gen_ms", tot.gen_ms + (decoding ? inflight.gen_ms : 0.0))
        .kv("rounds", tot.rounds + (decoding ? inflight.rounds : 0))
        .kv("drafted", tot.drafted + (decoding ? inflight.drafted : 0))
        .kv("accepted", tot.accepted + (decoding ? inflight.accepted : 0))
        .kv("verify_ms", tot.verify_ms + (decoding ? inflight.verify_ms : 0.0))
        .kv("draft_ms", tot.draft_ms + (decoding ? inflight.draft_ms : 0.0))
        .kv("cpu_expert_ms", tot.cpu_expert_ms + (busy ? tm.cpu_experts_ms : 0.0))
        .kv("routed", routed)
        .kv("misses", misses)
        .kv("swaps", (int64_t) eng.cache().swaps_done)
        .kv("parks", eng.park_stats.parks)
        .kv("restores", eng.park_stats.restores)
        .kv("park_evictions", eng.park_stats.evictions)
        .kv("park_ms", eng.park_stats.park_ms)
        .kv("restore_ms", eng.park_stats.restore_ms);
    std::vector<int> slots(c.n_layer);
    for (int il = 0; il < c.n_layer; ++il) slots[il] = eng.cache().slots(il);
    JsonOut o;
    o.kv("type", "telemetry")
        .kv("t", wall_s())
        .kv("model", model_name)
        .kv("phase", cur.phase)
        .kv("active", active)
        .kv("slots", eng.slots())
        .kv("n_ctx", eng.max_ctx())
        .kv("pos", eng.pos())
        .kv("n_layer", c.n_layer)
        .kv("n_expert", c.n_expert)
        .kv("n_expert_used", c.n_expert_used)
        .kv("vram_total_mb", (double) vtotal / 1048576.0)
        .kv("vram_used_mb", (double) (vtotal - vfree) / 1048576.0)
        .kv("ram_total_mb", (double) si.totalram * si.mem_unit / 1048576.0)
        .kv("ram_free_mb", (double) (si.freeram + si.bufferram) * si.mem_unit / 1048576.0)
        .kv("rss_mb", rss_mb())
        .kv("experts_resident", eng.cache().resident())
        .kv("experts_total", c.n_layer * c.n_expert)
        .kv("expert_cache_gb", (double) eng.cache().bytes() / 1073741824.0)
        .kv("kv_gb", (double) eng.kv_bytes_mapped() / 1073741824.0)
        .kv("parked", eng.parked())
        .kv("parked_gb", (double) eng.parked_bytes() / 1073741824.0)
        .kv("cpu_threads", eng.cpu_threads())
        .kv("mtp", eng.mtp() != nullptr)
        .raw("req", rq.done())
        .raw("life", lf.done())
        .raw("layer_slots", arr(slots))
        .raw("layer_routed", arr(eng.layer_routed))
        .raw("layer_misses", arr(eng.layer_misses));
    return o.done();
}

struct Pending {
    Json req;
};

}  // namespace

// One request in flight: its slot's generator, limits, and what it has done.
struct Active {
    std::string id;
    int slot = 0;
    std::unique_ptr<Generator> gen;
    std::vector<int32_t> stop_ids, prompt;
    int max_tokens = 0, produced = 0;
    std::vector<int32_t> out;   // tokens of the last round, not sent yet
    std::string reason = "length";
    bool stop = false;
    int64_t fresh = 0;
    double t0 = 0, t1 = 0, tg0 = 0;
    double interleave_ms = 0;   // the others' decoding run between this request's prompt chunks
    Current cur;
};

int serve_main(Engine & eng, const GenOptions & gopt, const std::string & model_name) {
    const int n_slots = eng.slots();
    Totals tot;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Json> queue;
    std::atomic<bool> quit{false};
    std::set<std::string> cancelled;   // ids asked to stop (reader thread -> worker)
    std::mutex cur_mu;
    // the last telemetry line, for "stats" answered from the reader thread
    std::mutex snap_mu;
    std::string snapshot;
    double last_tel = 0;
    std::vector<std::unique_ptr<Active>> active;
    std::vector<double> slot_used(n_slots, 0);   // when each slot last served (least recently used goes first)
    Active * shown = nullptr;                    // the request "req" describes: the one being admitted, else the oldest
    Current idle;
    GenStats no_stats;
    auto telemetry = [&](bool force) {
        const double t = now_ms();
        if (!force && t - last_tel < 250) return;
        last_tel = t;
        Current sum;
        for (auto & a : active) {
            const GenStats & g = a->gen->stats;
            sum.gen_tokens += a->produced;
            if (a->cur.phase == "decode") {
                sum.phase = "decode";
                sum.gen_ms += t - a->tg0;
                sum.rounds += g.rounds;
                sum.drafted += g.drafted;
                sum.accepted += g.accepted;
                sum.verify_ms += g.verify_ms;
                sum.draft_ms += g.draft_ms;
            }
        }
        const Active * sh = shown ? shown : (active.empty() ? nullptr : active.front().get());
        std::string line = telemetry_json(eng, sh ? sh->gen->stats : no_stats, tot, sh ? sh->cur : idle, sum,
                                          (int) active.size(), model_name);
        {
            std::lock_guard<std::mutex> lk(snap_mu);
            snapshot = line;
        }
        emit(line);
    };

    {
        JsonOut o;
        o.kv("type", "ready").kv("model", model_name).kv("n_ctx", eng.max_ctx()).kv("n_vocab", eng.cfg().n_vocab)
            .kv("mtp", eng.mtp() != nullptr).kv("eos", eng.cfg().eos_token).kv("slots", n_slots);
        emit(o.done());
    }
    telemetry(true);
    std::function<void(size_t, size_t)> between_chunks;   // set below, once the round exists
    eng.on_prefill_progress = [&](size_t done, size_t total) {
        if (shown) {
            shown->cur.prefill_done = (int64_t) done;
            shown->cur.prefill_total = (int64_t) total;
        }
        telemetry(false);
        if (between_chunks) between_chunks(done, total);
    };

    std::thread reader([&]() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            Json j;
            try {
                j = Json::parse(line);
            } catch (const std::exception & e) {
                emit(JsonOut().kv("type", "error").kv("message", std::string("bad request: ") + e.what()).done());
                continue;
            }
            const std::string op = j["op"].str();
            if (op == "stats") {
                std::lock_guard<std::mutex> lk(snap_mu);
                if (!snapshot.empty()) emit(snapshot);
                continue;
            }
            if (op == "cancel") {
                std::lock_guard<std::mutex> lk(cur_mu);
                cancelled.insert(j["id"].str());
                std::lock_guard<std::mutex> lk2(mu);
                for (auto it = queue.begin(); it != queue.end(); ++it)
                    if ((*it)["id"].str() == j["id"].str()) { queue.erase(it); break; }
                continue;
            }
            if (op == "quit") {
                quit = true;
                cv.notify_all();
                break;
            }
            std::lock_guard<std::mutex> lk(mu);
            queue.push_back(j);
            cv.notify_all();
        }
        quit = true;
        cv.notify_all();
    });

    auto is_cancelled = [&](const std::string & id) {
        std::lock_guard<std::mutex> lk(cur_mu);
        return cancelled.count(id) > 0;
    };
    auto is_stop = [](const Active & a, int32_t t) {
        return std::find(a.stop_ids.begin(), a.stop_ids.end(), t) != a.stop_ids.end();
    };
    // sends a round's tokens (up to a stop token / the budget) and decides whether the request is done
    auto deliver = [&](Active & a) {
        std::vector<int32_t> send;
        for (int32_t t : a.out) {
            if (is_stop(a, t)) { a.stop = true; a.reason = "stop"; break; }
            send.push_back(t);
            if (++a.produced >= a.max_tokens) { a.stop = true; break; }
        }
        a.out.clear();
        a.cur.gen_tokens = a.produced;
        if (!send.empty()) emit(JsonOut().kv("type", "tokens").kv("id", a.id).kv("ids", send).done());
        if (a.stop) return;
        if (is_cancelled(a.id)) { a.stop = true; a.reason = "cancel"; return; }
        eng.select(a.slot);
        if (eng.pos() + 2 >= eng.max_ctx()) { a.stop = true; a.reason = "context"; }
    };
    auto finish_request = [&](Active & a) {
        const double tg1 = now_ms();
        const auto & gs = a.gen->stats;
        const double gms = tg1 - a.tg0;
        tot.requests++;
        tot.gen_tokens += a.produced;
        tot.gen_ms += gms;
        tot.prompt_tokens += (int64_t) a.prompt.size();
        tot.reused += gs.reused_tokens;
        tot.rounds += gs.rounds;
        tot.drafted += gs.drafted;
        tot.accepted += gs.accepted;
        tot.verify_ms += gs.verify_ms;
        tot.draft_ms += gs.draft_ms;
        tot.cpu_expert_ms += eng.times.cpu_experts_ms;
        const double tps = a.produced > 1 && gms > 0 ? (a.produced - 1) / (gms / 1000) : 0;
        const double prefill_tps = a.fresh > 0 ? a.fresh / ((a.t1 - a.t0) / 1000) : 0;
        emit(JsonOut().kv("type", "done").kv("id", a.id).kv("reason", a.reason).kv("gen_tokens", a.produced)
                 .kv("gen_ms", gms).kv("tps", tps).kv("prompt_tokens", (int64_t) a.prompt.size())
                 .kv("reused", gs.reused_tokens).kv("prefill_ms", a.t1 - a.t0).kv("prefill_tps", prefill_tps)
                 .kv("rounds", gs.rounds).kv("drafted", gs.drafted).kv("accepted", gs.accepted)
                 .kv("tokens_per_round", gs.rounds ? (double) gs.emitted / gs.rounds : 0.0)
                 .kv("expert_miss_rate", eng.times.routed ? (double) eng.times.misses / eng.times.routed : 0.0)
                 .kv("verify_ms", gs.verify_ms).kv("draft_ms", gs.draft_ms)
                 .kv("cpu_expert_ms", eng.times.cpu_experts_ms).kv("batched", n_slots > 1).done());
        slot_used[a.slot] = now_ms();
        std::lock_guard<std::mutex> lk(cur_mu);
        cancelled.erase(a.id);
    };
    // the free slot to read a prompt into: the one whose conversation it continues best, else the least recently
    // used (the engine's parking still brings back a conversation parked in host RAM)
    auto pick_slot = [&](const std::vector<int32_t> & prompt) {
        int best = -1, best_r = -1;
        for (int s = 0; s < n_slots; ++s) {
            bool busy = false;
            for (auto & a : active) busy |= a->slot == s;
            if (busy) continue;
            eng.select(s);
            const int r = eng.reusable(prompt);
            if (r > best_r || (r == best_r && slot_used[s] < slot_used[best])) best = s, best_r = r;
        }
        return best;
    };
    // the slots the engine may not move off the GPU when VRAM is short: those decoding, and the one being read
    auto mark_busy = [&](int reading) {
        std::vector<int> b;
        for (auto & x : active) b.push_back(x->slot);
        if (reading >= 0) b.push_back(reading);
        eng.set_busy(b);
    };
    // reads one request's prompt into a free slot (the others wait meanwhile); false = it failed or ended at once
    auto admit = [&](Json & req) -> bool {
        const std::string id = req["id"].str();
        auto a = std::make_unique<Active>();
        a->id = id;
        try {
            for (const auto & v : req["prompt"].a) a->prompt.push_back((int32_t) v.num());
            if (a->prompt.empty()) throw std::runtime_error("empty prompt");
            if ((int) a->prompt.size() >= eng.max_ctx()) throw std::runtime_error("prompt longer than the context");
            for (const auto & v : req["stop_ids"].a) a->stop_ids.push_back((int32_t) v.num());
            if (a->stop_ids.empty()) a->stop_ids.push_back(eng.cfg().eos_token);
            SamplingParams sp;
            sp.temperature = (float) req["temperature"].num(0.0);
            sp.top_k = (int) req["top_k"].num(0);
            sp.top_p = (float) req["top_p"].num(1.0);
            sp.min_p = (float) req["min_p"].num(0.0);
            sp.presence_penalty = (float) req["presence_penalty"].num(0.0);
            sp.repetition_penalty = (float) req["repetition_penalty"].num(1.0);
            sp.seed = (uint64_t) req["seed"].num(0);
            a->max_tokens = (int) req["max_tokens"].num(1024);
            a->slot = pick_slot(a->prompt);
            mark_busy(a->slot);
            a->gen = std::make_unique<Generator>(eng, eng.mtp(), gopt, a->slot);
            a->gen->set_draft(req.has("draft") ? (int) req["draft"].num() : gopt.max_draft);
            a->gen->set_draft_min_p(req.has("draft_min_p") ? (float) req["draft_min_p"].num() : gopt.min_p);
            a->cur.id = id;
            a->cur.phase = "prefill";
            a->cur.prompt_tokens = a->cur.prefill_total = (int64_t) a->prompt.size();
            a->t0 = a->cur.t_start = now_ms();
            shown = a.get();
            telemetry(true);
            const int32_t tok = a->gen->start(a->prompt, sp, req.has("checkpoint") ? (int) req["checkpoint"].num() : -1);
            a->t1 = now_ms() - a->interleave_ms;
            const auto & gs = a->gen->stats;
            a->fresh = gs.prompt_tokens - gs.reused_tokens;
            a->cur.reused = gs.reused_tokens;
            a->cur.prefill_ms = a->t1 - a->t0;
            a->cur.prefill_done = a->cur.prefill_total = a->fresh;
            tot.prefill_tokens += a->fresh;
            tot.prefill_ms += a->t1 - a->t0;
            emit(JsonOut().kv("type", "prefill").kv("id", id).kv("prompt_tokens", (int64_t) a->prompt.size())
                     .kv("reused", gs.reused_tokens).kv("ms", a->t1 - a->t0)
                     .kv("tps", a->fresh > 0 ? a->fresh / ((a->t1 - a->t0) / 1000) : 0.0).done());
            a->cur.phase = "decode";
            a->tg0 = a->cur.gen_t0 = now_ms();
            a->out = {tok};
            shown = nullptr;
            deliver(*a);
            if (a->stop) {
                finish_request(*a);
                return false;
            }
            active.push_back(std::move(a));
            return true;
        } catch (const std::exception & e) {
            shown = nullptr;
            emit(JsonOut().kv("type", "error").kv("id", id).kv("message", e.what()).done());
            exit_if_gpu_lost();
            if (a->gen) {   // the slot may be half-processed: start it clean next time
                try { eng.select(a->slot); eng.reset(); } catch (...) {}
            }
            return false;
        }
    };

    // one round for every request in flight: their windows in one batched forward; finished requests leave
    auto run_round = [&]() {
        try {
            std::vector<Generator *> gens;
            for (auto & a : active) gens.push_back(a->gen.get());
            const std::vector<int> rows = Generator::share_rows(gens);
            std::vector<Engine::BatchWin> wins;
            for (size_t i = 0; i < active.size(); ++i) {
                const auto & w = active[i]->gen->window(rows[i]);
                wins.push_back({active[i]->slot, w.data(), (int) w.size(), w.size() == 1});
            }
            const double f0 = now_ms();
            eng.forward_batch(wins);
            const double f1 = now_ms();
            // verify and commit every request, then all their drafts in one drafter pass (BNK_DRAFT_BATCH=0:
            // each request drafts on its own)
            static const bool draft_batch = !getenv("BNK_DRAFT_BATCH") || atoi(getenv("BNK_DRAFT_BATCH")) != 0;
            for (auto & a : active) {
                a->gen->stats.verify_ms += f1 - f0;
                a->out = a->gen->finish(!draft_batch);
            }
            if (draft_batch) Generator::draft_batch(gens);
            for (auto & a : active) deliver(*a);
        } catch (const std::exception & e) {
            for (auto & a : active)
                emit(JsonOut().kv("type", "error").kv("id", a->id).kv("message", e.what()).done());
            exit_if_gpu_lost();
            for (auto & a : active) {
                try { eng.select(a->slot); eng.reset(); } catch (...) {}
            }
            active.clear();
        }
        for (size_t i = 0; i < active.size();) {
            if (active[i]->stop) {
                finish_request(*active[i]);
                active.erase(active.begin() + (long) i);
                mark_busy(-1);
            } else {
                ++i;
            }
        }
    };
    // While a request's prompt is read, the others keep decoding between its chunks: after each chunk but the last
    // (the last one leaves the first token's logits in the decode buffers), rounds for share x that chunk's time.
    static const double read_share = getenv("BNK_READ_SHARE") ? atof(getenv("BNK_READ_SHARE")) : 0.5;
    double chunk_t0 = 0;
    between_chunks = [&](size_t done, size_t total) {
        if (!shown || done >= total || active.empty() || read_share <= 0) {
            chunk_t0 = now_ms();
            return;
        }
        const double t0 = now_ms(), budget = read_share * (t0 - (chunk_t0 > shown->t0 ? chunk_t0 : shown->t0));
        while (!active.empty() && now_ms() - t0 < budget) run_round();
        eng.select(shown->slot);   // back to the prompt being read
        shown->interleave_ms += now_ms() - t0;
        chunk_t0 = now_ms();
    };
    static double last_save = now_ms();
    while (!quit) {
        // 1. admit waiting requests while slots are free (each prompt read in turn)
        Json req;
        bool have = false;
        {
            std::unique_lock<std::mutex> lk(mu);
            if (active.empty() &&
                !cv.wait_for(lk, std::chrono::milliseconds(1000), [&]() { return quit || !queue.empty(); })) {
                lk.unlock();
                telemetry(true);   // idle heartbeat
                continue;
            }
            if (quit) break;
            // a generate request while a slot is free; anything else (reset ...) once nothing is in flight
            if (!queue.empty() &&
                (active.empty() || ((int) active.size() < n_slots && queue.front()["op"].str() == "generate"))) {
                req = queue.front();
                queue.pop_front();
                have = true;
            }
        }
        if (have) {
            const std::string op = req["op"].str();
            const std::string id = req["id"].str();
            if (op == "reset") {
                if (active.empty()) {
                    for (int s = 0; s < n_slots; ++s) {
                        eng.select(s);
                        eng.reset();
                    }
                    eng.drop_parked_all();   // a clean slate: parked conversations go too
                    emit(JsonOut().kv("type", "reset").kv("id", id).done());
                } else {
                    std::lock_guard<std::mutex> lk(mu);
                    queue.push_back(req);   // after the requests in flight
                }
            } else if (op != "generate") {
                emit(JsonOut().kv("type", "error").kv("id", id).kv("message", "unknown op " + op).done());
            } else {
                admit(req);
            }
            telemetry(true);
            continue;   // admit every waiting request before the next round
        }
        if (active.empty()) continue;
        // 2. one round for every request in flight: their windows in one batched forward
        run_round();
        if (active.empty()) eng.times = StageTimes{};
        telemetry(false);
        // keep the learned routing counts on disk (the process may be killed rather than quit)
        if (now_ms() - last_save > 60000) {
            eng.save_counts();
            last_save = now_ms();
        }
    }
    eng.save_counts();
    reader.detach();
    return 0;
}

}  // namespace bnk
