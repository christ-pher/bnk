// `bnk serve`: the engine behind a line protocol on stdin/stdout (one JSON object per line).
//
//   in : {"op":"generate","id":..,"prompt":[ids],"max_tokens":N,"temperature":..,"top_p":..,"top_k":..,
//         "min_p":..,"presence_penalty":..,"repetition_penalty":..,"seed":..,"stop_ids":[ids],"draft":n,
//         "draft_min_p":p}  (draft: tokens the drafter may propose per round; draft_min_p: it stops below this)
//        {"op":"cancel","id":..}   {"op":"stats"}   {"op":"reset"}   {"op":"quit"}
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

std::string telemetry_json(Engine & eng, const Generator & gen, const Totals & tot, const Current & cur,
                           const std::string & model_name) {
    size_t vfree = 0, vtotal = 0;
    cudaMemGetInfo(&vfree, &vtotal);
    struct sysinfo si;
    sysinfo(&si);
    const auto & c = eng.cfg();
    const auto & gs = gen.stats;
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
        .kv("gen_tokens", tot.gen_tokens + (busy ? cur.gen_tokens : 0))
        .kv("gen_ms", tot.gen_ms + (cur.phase == "decode" ? now - cur.gen_t0 : 0.0))
        .kv("rounds", tot.rounds + (cur.phase == "decode" ? gs.rounds : 0))
        .kv("drafted", tot.drafted + (cur.phase == "decode" ? gs.drafted : 0))
        .kv("accepted", tot.accepted + (cur.phase == "decode" ? gs.accepted : 0))
        .kv("verify_ms", tot.verify_ms + (cur.phase == "decode" ? gs.verify_ms : 0.0))
        .kv("draft_ms", tot.draft_ms + (cur.phase == "decode" ? gs.draft_ms : 0.0))
        .kv("cpu_expert_ms", tot.cpu_expert_ms + (busy ? tm.cpu_experts_ms : 0.0))
        .kv("routed", routed)
        .kv("misses", misses)
        .kv("swaps", (int64_t) eng.cache().swaps_done);
    std::vector<int> slots(c.n_layer);
    for (int il = 0; il < c.n_layer; ++il) slots[il] = eng.cache().slots(il);
    JsonOut o;
    o.kv("type", "telemetry")
        .kv("t", wall_s())
        .kv("model", model_name)
        .kv("phase", cur.phase)
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

int serve_main(Engine & eng, const GenOptions & gopt, const std::string & model_name) {
    Generator gen(eng, eng.mtp(), gopt);
    Totals tot;
    Current cur;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Json> queue;
    std::atomic<bool> quit{false};
    std::atomic<bool> cancel{false};
    std::string current_id;
    std::mutex cur_mu;
    // the last telemetry line, for "stats" answered from the reader thread
    std::mutex snap_mu;
    std::string snapshot;
    double last_tel = 0;
    auto telemetry = [&](bool force) {
        const double t = now_ms();
        if (!force && t - last_tel < 250) return;
        last_tel = t;
        std::string line = telemetry_json(eng, gen, tot, cur, model_name);
        {
            std::lock_guard<std::mutex> lk(snap_mu);
            snapshot = line;
        }
        emit(line);
    };

    {
        JsonOut o;
        o.kv("type", "ready").kv("model", model_name).kv("n_ctx", eng.max_ctx()).kv("n_vocab", eng.cfg().n_vocab)
            .kv("mtp", eng.mtp() != nullptr).kv("eos", eng.cfg().eos_token);
        emit(o.done());
    }
    telemetry(true);
    eng.on_prefill_progress = [&](size_t done, size_t total) {
        cur.prefill_done = (int64_t) done;
        cur.prefill_total = (int64_t) total;
        telemetry(false);
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
                if (j["id"].str() == current_id) cancel = true;
                std::lock_guard<std::mutex> lk2(mu);
                for (auto it = queue.begin(); it != queue.end(); ++it)
                    if ((*it)["id"].str() == j["id"].str()) { queue.erase(it); break; }
                continue;
            }
            if (op == "quit") {
                quit = true;
                cancel = true;
                cv.notify_all();
                break;
            }
            std::lock_guard<std::mutex> lk(mu);
            queue.push_back(j);
            cv.notify_all();
        }
        quit = true;
        cancel = true;
        cv.notify_all();
    });

    while (!quit) {
        Json req;
        {
            std::unique_lock<std::mutex> lk(mu);
            if (!cv.wait_for(lk, std::chrono::milliseconds(1000), [&]() { return quit || !queue.empty(); })) {
                lk.unlock();
                telemetry(true);   // idle heartbeat
                continue;
            }
            if (quit && queue.empty()) break;
            req = queue.front();
            queue.pop_front();
        }
        const std::string op = req["op"].str();
        const std::string id = req["id"].str();
        try {
            if (op == "reset") {
                eng.reset();
                emit(JsonOut().kv("type", "reset").kv("id", id).done());
                continue;
            }
            if (op != "generate") {
                emit(JsonOut().kv("type", "error").kv("id", id).kv("message", "unknown op " + op).done());
                continue;
            }
            std::vector<int32_t> prompt;
            for (const auto & v : req["prompt"].a) prompt.push_back((int32_t) v.num());
            if (prompt.empty()) throw std::runtime_error("empty prompt");
            if ((int) prompt.size() >= eng.max_ctx()) throw std::runtime_error("prompt longer than the context");
            std::vector<int32_t> stop_ids;
            for (const auto & v : req["stop_ids"].a) stop_ids.push_back((int32_t) v.num());
            if (stop_ids.empty()) stop_ids.push_back(eng.cfg().eos_token);
            SamplingParams sp;
            sp.temperature = (float) req["temperature"].num(0.0);
            sp.top_k = (int) req["top_k"].num(0);
            sp.top_p = (float) req["top_p"].num(1.0);
            sp.min_p = (float) req["min_p"].num(0.0);
            sp.presence_penalty = (float) req["presence_penalty"].num(0.0);
            sp.repetition_penalty = (float) req["repetition_penalty"].num(1.0);
            sp.seed = (uint64_t) req["seed"].num(0);
            const int max_tokens = (int) req["max_tokens"].num(1024);
            gen.set_draft(req.has("draft") ? (int) req["draft"].num() : gopt.max_draft);
            gen.set_draft_min_p(req.has("draft_min_p") ? (float) req["draft_min_p"].num() : gopt.min_p);
            {
                std::lock_guard<std::mutex> lk(cur_mu);
                current_id = id;
                cancel = false;
            }
            gen.stats = GenStats{};
            eng.times = StageTimes{};
            cur = Current{};
            cur.id = id;
            cur.phase = "prefill";
            cur.prompt_tokens = (int64_t) prompt.size();
            cur.prefill_total = (int64_t) prompt.size();
            const double t0 = now_ms();
            cur.t_start = t0;
            telemetry(true);
            int32_t tok = gen.start(prompt, sp);
            const double t1 = now_ms();
            const int64_t fresh = gen.stats.prompt_tokens - gen.stats.reused_tokens;
            cur.reused = gen.stats.reused_tokens;
            cur.prefill_ms = t1 - t0;
            cur.prefill_done = cur.prefill_total = fresh;
            tot.prefill_tokens += fresh;
            tot.prefill_ms += t1 - t0;
            emit(JsonOut().kv("type", "prefill").kv("id", id).kv("prompt_tokens", (int64_t) prompt.size())
                     .kv("reused", gen.stats.reused_tokens).kv("ms", t1 - t0)
                     .kv("tps", fresh > 0 ? fresh / ((t1 - t0) / 1000) : 0.0).done());
            cur.phase = "decode";
            int produced = 0;
            std::string reason = "length";
            auto is_stop = [&](int32_t t) { return std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end(); };
            std::vector<int32_t> out{tok};
            bool stop = is_stop(tok);
            if (stop) reason = "stop";
            const double tg0 = now_ms();
            cur.gen_t0 = tg0;
            telemetry(true);
            while (true) {
                // emit what we have (minus a stop token)
                std::vector<int32_t> send;
                for (int32_t t : out) {
                    if (is_stop(t)) { stop = true; reason = "stop"; break; }
                    send.push_back(t);
                    if (++produced >= max_tokens) { stop = true; break; }
                }
                cur.gen_tokens = produced;
                if (!send.empty()) emit(JsonOut().kv("type", "tokens").kv("id", id).kv("ids", send).done());
                telemetry(false);
                if (stop) break;
                if (cancel) { reason = "cancel"; break; }
                if (eng.pos() + 2 >= eng.max_ctx()) { reason = "context"; break; }
                out = gen.next();
            }
            const double tg1 = now_ms();
            const auto & gs = gen.stats;
            const double gms = tg1 - tg0;
            tot.requests++;
            tot.gen_tokens += produced;
            tot.gen_ms += gms;
            tot.prompt_tokens += (int64_t) prompt.size();
            tot.reused += gs.reused_tokens;
            tot.rounds += gs.rounds;
            tot.drafted += gs.drafted;
            tot.accepted += gs.accepted;
            tot.verify_ms += gs.verify_ms;
            tot.draft_ms += gs.draft_ms;
            tot.cpu_expert_ms += eng.times.cpu_experts_ms;
            const double tps = produced > 1 && gms > 0 ? (produced - 1) / (gms / 1000) : 0;
            const double prefill_tps = fresh > 0 ? fresh / ((t1 - t0) / 1000) : 0;
            emit(JsonOut().kv("type", "done").kv("id", id).kv("reason", reason).kv("gen_tokens", produced)
                     .kv("gen_ms", gms).kv("tps", tps).kv("prompt_tokens", (int64_t) prompt.size())
                     .kv("reused", gs.reused_tokens).kv("prefill_ms", t1 - t0).kv("prefill_tps", prefill_tps)
                     .kv("rounds", gs.rounds).kv("drafted", gs.drafted).kv("accepted", gs.accepted)
                     .kv("tokens_per_round", gs.rounds ? (double) gs.emitted / gs.rounds : 0.0)
                     .kv("expert_miss_rate", eng.times.routed ? (double) eng.times.misses / eng.times.routed : 0.0)
                     .kv("verify_ms", gs.verify_ms).kv("draft_ms", gs.draft_ms)
                     .kv("cpu_expert_ms", eng.times.cpu_experts_ms).done());
        } catch (const std::exception & e) {
            emit(JsonOut().kv("type", "error").kv("id", id).kv("message", e.what()).done());
            // the state may be half-processed: start clean next time
            try { eng.reset(); } catch (...) {}
        }
        {
            std::lock_guard<std::mutex> lk(cur_mu);
            current_id.clear();
        }
        cur = Current{};
        telemetry(true);
        // keep the learned routing counts on disk (the process may be killed rather than quit)
        static double last_save = now_ms();
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
