// What the generator, the server and the command line need from an engine: the MoE engine (engine.h) and the
// dense engine (dense/dense_engine.h) both implement it, so the layers above run either model family unchanged.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/model.h"
#include "kernels/gemv.h"

namespace bnk {

struct StageTimes {
    double total_ms = 0, cpu_experts_ms = 0, ple_ms = 0, wait_ms = 0;
    int64_t misses = 0, routed = 0;
    int calls = 0;
};

struct ParkStats { int64_t parks = 0, restores = 0, evictions = 0, forks = 0; double park_ms = 0, restore_ms = 0, fork_ms = 0; };

// one pipeline stage (GPU) as the telemetry shows it
struct StageInfo {
    int device = 0, layer_begin = 0, layer_end = 0;
    int experts_resident = 0;      // MoE: experts held in this GPU's cache
    size_t cache_bytes = 0;        // ... and their VRAM
    double gpu_ms = 0;             // GPU time in decode forwards, summed
    int64_t gpu_forwards = 0;
    double prompt_ms = 0;          // wall time in prompt chunks
    int64_t prompt_tokens = 0;
};

// The speculative drafter (the model's MTP layer): K/V per conversation slot, drafts from the main model's final
// residual rows. Cell i pairs residual row i with the token at position i+1 and predicts the token at i+2.
class Drafter {
public:
    virtual ~Drafter() = default;
    // K/V for cells cell0..cell0+n-1 from main-model residual rows R (device, [n][res_dim]) and the tokens that
    // follow them; with `draft`, the full layer on the last row: returns its greedy draft (prob in *prob)
    // and keeps its residual for step(). Returns -1 when !draft.
    virtual int run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) = 0;
    // One more draft from the last residual, pairing it with `tok` at `cell`.
    virtual int step(int32_t tok, int cell, float * prob) = 0;
    // Batched drafting: several conversations' drafter passes in one forward. run_multi is run() for every item;
    // step_multi is step() for the items given by index into the last run_multi/step_multi.
    struct RunItem { int slot; const float * R_rows; const int32_t * tokens; int n; int cell0; };
    struct StepItem { int prev; int32_t tok; int cell; };   // prev: the item's index in the previous call
    virtual void run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) = 0;
    virtual void step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) = 0;
    double ms = 0;
    int64_t calls = 0;
};

class EngineApi {
public:
    virtual ~EngineApi() = default;
    struct BatchWin { int slot; const int32_t * tokens; int T; bool commit_all; };
    static constexpr int kTopkMax = 1024;

    // ---- shared counters and hooks (the first stage's)
    StageTimes times;
    std::vector<int64_t> layer_routed, layer_misses;   // lifetime routing counters per layer (MoE)
    ParkStats park_stats;
    std::function<void(size_t, size_t)> on_prefill_progress;   // after every prefill chunk: (tokens done, total)
    double last_rebalance_ms = 0;
    int last_T = 0;                     // rows of the last forward (the logits rows)
    bool dump_all = false;              // debugging (disables graphs): residual after every layer
    std::vector<std::vector<float>> dumped_layers;   // dumped_layers[il] = [T][res_dim]

    // ---- the pipeline
    virtual std::vector<int> devices() const = 0;
    virtual int device() const = 0;
    virtual int last_device() const = 0;
    virtual int stages() const = 0;
    virtual StageInfo stage_info(int i) const = 0;
    virtual std::string model_name() const = 0;
    virtual const Config & cfg() const = 0;
    virtual int res_dim() const = 0;   // floats per residual row (hc * n_embd for the HC models, n_embd otherwise)
    virtual Drafter * mtp() = 0;
    virtual int max_ctx() const = 0;
    virtual int cpu_threads() const = 0;
    virtual int64_t graph_captures() const = 0;
    virtual size_t graph_bytes() const = 0;

    // ---- conversation slots
    virtual int slots() const = 0;
    virtual int current_slot() const = 0;
    virtual void select(int slot) = 0;
    virtual void reset() = 0;
    virtual int pos() const = 0;
    virtual const std::vector<int32_t> & history() const = 0;
    virtual void checkpoint() = 0;
    virtual int rollback(int max_pos) = 0;
    virtual void select_conversation(const std::vector<int32_t> & prompt) = 0;
    virtual int reusable(const std::vector<int32_t> & prompt) const = 0;
    // true: select_conversation can fork a prefix another conversation has processed (a shared system prompt)
    // into the current slot without disturbing that conversation; the server then prefers an empty slot for a
    // new conversation over rolling a sibling's slot back
    virtual bool can_fork() const { return false; }
    virtual int park_min() const { return 2048; }   // tokens a conversation must have to be worth keeping (parking)

    // Lookahead on a two-stage pipeline (docs/LOOKAHEAD.md): while the second stage runs a verify window, the
    // first stage runs the window that follows if every draft is accepted. false: unsupported (the calls throw).
    virtual bool lookahead_ok() const { return false; }
    // `win` through both stages; `chain` is called on a worker thread while the GPUs work and may fill the
    // tokens of the window that would follow full acceptance (returns their count, 0 = none), which the first
    // stage then runs speculatively. stage0_done: `win`'s first-stage half was last round's confirmed
    // speculation (only the second stage runs it).
    virtual void forward_lookahead(const BatchWin &, int /*lane*/, const std::function<int(std::vector<int32_t> &)> & /*chain*/,
                                   bool /*stage0_done*/) { throw std::runtime_error("lookahead: not supported by this engine"); }
    virtual bool spec_pending() const { return false; }   // a speculative window awaits the verdict
    virtual int spec_lane() const { return -1; }
    virtual void spec_confirm() { throw std::runtime_error("lookahead: not supported"); }   // it held: it is the pending window now
    virtual void spec_discard() { throw std::runtime_error("lookahead: not supported"); }   // it did not
    // a request ends while its next window's first-stage half is pending (a confirmed speculation): that half
    // is dropped, the slot's state is what the verified windows committed
    virtual void spec_abort() {}
    virtual void set_busy(const std::vector<int> & slots) = 0;
    virtual int parked() const = 0;
    virtual size_t parked_bytes() const = 0;
    virtual void drop_parked_all() = 0;
    virtual void save_counts() = 0;

    // ---- forwards
    virtual void prefill(const std::vector<int32_t> & tokens) = 0;
    virtual void forward(const int32_t * tokens, int T, bool commit_all = true) = 0;
    virtual void forward_batch(const std::vector<BatchWin> & wins) = 0;
    virtual void forward_pipe(const std::vector<BatchWin> & first, int first_lane, const std::vector<BatchWin> & second,
                              int second_lane) = 0;
    virtual int row0_of(int slot) const = 0;
    virtual void commit(int c) = 0;
    virtual const float * residual_dev() const = 0;
    virtual void set_mtp_pending(const float * R_row_dev, int cell) = 0;
    virtual const float * mtp_pending_R() const = 0;
    virtual int mtp_pending_cell() const = 0;

    // ---- logits of the last forward
    virtual int argmax(int t) = 0;
    virtual void argmax_all(int T, int32_t * out, int row0 = 0) = 0;
    virtual std::vector<float> logits_host(int t) = 0;
    virtual void logits_rows_host(int T, float * out, int row0 = 0) = 0;
    virtual void topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z = nullptr,
                                float temp = 1.f) = 0;

    // ---- telemetry
    virtual int64_t cache_swaps() const = 0;
    virtual int cache_slots(int il) const = 0;
    virtual int cache_resident() const = 0;
    virtual size_t cache_bytes() const = 0;
    virtual size_t kv_bytes_mapped() const = 0;
};

}  // namespace bnk
