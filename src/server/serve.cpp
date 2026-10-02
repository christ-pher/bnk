// `bnk serve`: the engine behind a line protocol on stdin/stdout (one JSON object per line).
//
//   in : {"op":"generate","id":..,"prompt":[ids],"max_tokens":N,"temperature":..,"top_p":..,"top_k":..,
//         "min_p":..,"presence_penalty":..,"repetition_penalty":..,"seed":..,"stop_ids":[ids],"draft":n}
//        {"op":"cancel","id":..}   {"op":"stats"}   {"op":"reset"}   {"op":"quit"}
//   out: {"type":"ready",..}  {"type":"prefill","id":..}  {"type":"tokens","id":..,"ids":[..]}
//        {"type":"done","id":..,"reason":"stop|length|cancel|context"} {"type":"stats",..} {"type":"error",..}
#include "server/serve.h"

#include <sys/sysinfo.h>
#include <unistd.h>

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
    int64_t requests = 0, gen_tokens = 0, prompt_tokens = 0, reused = 0;
    double gen_ms = 0, prefill_ms = 0;
    double last_tps = 0, last_prefill_tps = 0, last_accept = 0, last_tpr = 0, last_miss = 0;
};

std::string stats_json(Engine & eng, const Totals & tot, const std::string & model_name) {
    size_t vfree = 0, vtotal = 0;
    cudaMemGetInfo(&vfree, &vtotal);
    struct sysinfo si;
    sysinfo(&si);
    const auto & tm = eng.times;
    JsonOut o;
    o.kv("type", "stats")
        .kv("model", model_name)
        .kv("n_ctx", eng.max_ctx())
        .kv("pos", eng.pos())
        .kv("vram_total_mb", (double) vtotal / 1048576.0)
        .kv("vram_used_mb", (double) (vtotal - vfree) / 1048576.0)
        .kv("ram_total_mb", (double) si.totalram * si.mem_unit / 1048576.0)
        .kv("ram_free_mb", (double) (si.freeram + si.bufferram) * si.mem_unit / 1048576.0)
        .kv("experts_resident", eng.cache().resident())
        .kv("experts_total", eng.cfg().n_layer * eng.cfg().n_expert)
        .kv("expert_cache_gb", (double) eng.cache().bytes() / 1073741824.0)
        .kv("cache_swaps", (int64_t) eng.cache().swaps_done)
        .kv("routed", (int64_t) tm.routed)
        .kv("misses", (int64_t) tm.misses)
        .kv("cpu_threads", eng.cpu_threads())
        .kv("mtp", eng.mtp() != nullptr)
        .kv("requests", tot.requests)
        .kv("gen_tokens", tot.gen_tokens)
        .kv("prompt_tokens", tot.prompt_tokens)
        .kv("reused_tokens", tot.reused)
        .kv("avg_tps", tot.gen_ms > 0 ? tot.gen_tokens / (tot.gen_ms / 1000) : 0.0)
        .kv("last_tps", tot.last_tps)
        .kv("last_prefill_tps", tot.last_prefill_tps)
        .kv("last_accept", tot.last_accept)
        .kv("last_tokens_per_round", tot.last_tpr)
        .kv("last_miss_rate", tot.last_miss);
    return o.done();
}

struct Pending {
    Json req;
};

}  // namespace

int serve_main(Engine & eng, const GenOptions & gopt, const std::string & model_name) {
    Generator gen(eng, eng.mtp(), gopt);
    Totals tot;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Json> queue;
    std::atomic<bool> quit{false};
    std::atomic<bool> cancel{false};
    std::string current_id;
    std::mutex cur_mu;

    {
        JsonOut o;
        o.kv("type", "ready").kv("model", model_name).kv("n_ctx", eng.max_ctx()).kv("n_vocab", eng.cfg().n_vocab)
            .kv("mtp", eng.mtp() != nullptr).kv("eos", eng.cfg().eos_token);
        emit(o.done());
    }

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
            cv.wait(lk, [&]() { return quit || !queue.empty(); });
            if (quit && queue.empty()) break;
            req = queue.front();
            queue.pop_front();
        }
        const std::string op = req["op"].str();
        const std::string id = req["id"].str();
        try {
            if (op == "stats") {
                emit(stats_json(eng, tot, model_name));
                continue;
            }
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
            {
                std::lock_guard<std::mutex> lk(cur_mu);
                current_id = id;
                cancel = false;
            }
            gen.stats = GenStats{};
            eng.times = StageTimes{};
            const double t0 = now_ms();
            int32_t tok = gen.start(prompt, sp);
            const double t1 = now_ms();
            const int64_t fresh = gen.stats.prompt_tokens - gen.stats.reused_tokens;
            emit(JsonOut().kv("type", "prefill").kv("id", id).kv("prompt_tokens", (int64_t) prompt.size())
                     .kv("reused", gen.stats.reused_tokens).kv("ms", t1 - t0)
                     .kv("tps", fresh > 0 ? fresh / ((t1 - t0) / 1000) : 0.0).done());
            int produced = 0;
            std::string reason = "length";
            auto is_stop = [&](int32_t t) { return std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end(); };
            std::vector<int32_t> out{tok};
            bool stop = is_stop(tok);
            if (stop) reason = "stop";
            const double tg0 = now_ms();
            while (true) {
                // emit what we have (minus a stop token)
                std::vector<int32_t> send;
                for (int32_t t : out) {
                    if (is_stop(t)) { stop = true; reason = "stop"; break; }
                    send.push_back(t);
                    if (++produced >= max_tokens) { stop = true; break; }
                }
                if (!send.empty()) emit(JsonOut().kv("type", "tokens").kv("id", id).kv("ids", send).done());
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
            tot.last_tps = produced > 1 && gms > 0 ? (produced - 1) / (gms / 1000) : 0;
            tot.last_prefill_tps = fresh > 0 ? fresh / ((t1 - t0) / 1000) : 0;
            tot.last_accept = gs.drafted ? (double) gs.accepted / gs.drafted : 0;
            tot.last_tpr = gs.rounds ? (double) gs.emitted / gs.rounds : 0;
            tot.last_miss = eng.times.routed ? (double) eng.times.misses / eng.times.routed : 0;
            emit(JsonOut().kv("type", "done").kv("id", id).kv("reason", reason).kv("gen_tokens", produced)
                     .kv("gen_ms", gms).kv("tps", tot.last_tps).kv("prompt_tokens", (int64_t) prompt.size())
                     .kv("reused", gs.reused_tokens).kv("prefill_ms", t1 - t0).kv("prefill_tps", tot.last_prefill_tps)
                     .kv("rounds", gs.rounds).kv("drafted", gs.drafted).kv("accepted", gs.accepted)
                     .kv("tokens_per_round", tot.last_tpr).kv("expert_miss_rate", tot.last_miss)
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
