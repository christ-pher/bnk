// What one exchange between the two GPUs costs on this host (no NVLink, no peer access): the residual rows of a
// decode window (T x 10240 floats) from GPU 0 to GPU 1 and back, through pinned host memory with a cross-device
// event, and through cudaMemcpyPeerAsync (the driver stages it the same way). The number a tensor-parallel layer
// split has to pay ~2 times per layer.
//   build/bench_xfer [T=4] [exchanges=100]
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

int main(int argc, char ** argv) {
    //   build/bench_xfer [T=4] [exchanges=100] [floats per row=2560] [peer=1]
    const int T = argc > 1 ? atoi(argv[1]) : 4, n = argc > 2 ? atoi(argv[2]) : 100;
    const int row = argc > 3 ? atoi(argv[3]) : 2560, want_peer = argc > 4 ? atoi(argv[4]) : 1;
    const size_t bytes = (size_t) T * row * 4;
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { fprintf(stderr, "needs two GPUs\n"); return 1; }
    int p2p = 0;
    CK(cudaDeviceCanAccessPeer(&p2p, 0, 1));
    float * d0, * d1, * h;
    cudaStream_t s0, s1;
    cudaEvent_t e0, e1, t0, t1;
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&d0, bytes));
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&e0, cudaEventDisableTiming));
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaHostAlloc((void **) &h, bytes, cudaHostAllocPortable));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&d1, bytes));
    if (p2p && want_peer) {   // direct copies over PCIe instead of the driver's staging through host memory
        CK(cudaDeviceEnablePeerAccess(0, 0));
        CK(cudaSetDevice(0));
        CK(cudaDeviceEnablePeerAccess(1, 0));
        CK(cudaSetDevice(1));
    }
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming));
    printf("T = %d rows of %d floats (%zu KB per exchange), peer access %s%s\n", T, row, bytes / 1024, p2p ? "possible" : "no", p2p && want_peer ? ", enabled" : "");

    // A: GPU 0 -> host -> GPU 1 -> host -> GPU 0, events only (no host round trips in the loop)
    auto round_host = [&]() {
        CK(cudaSetDevice(0));
        CK(cudaMemcpyAsync(h, d0, bytes, cudaMemcpyDeviceToHost, s0));
        CK(cudaEventRecord(e0, s0));
        CK(cudaSetDevice(1));
        CK(cudaStreamWaitEvent(s1, e0, 0));
        CK(cudaMemcpyAsync(d1, h, bytes, cudaMemcpyHostToDevice, s1));
        CK(cudaMemcpyAsync(h, d1, bytes, cudaMemcpyDeviceToHost, s1));
        CK(cudaEventRecord(e1, s1));
        CK(cudaSetDevice(0));
        CK(cudaStreamWaitEvent(s0, e1, 0));
        CK(cudaMemcpyAsync(d0, h, bytes, cudaMemcpyHostToDevice, s0));
    };
    // B: cudaMemcpyPeerAsync both ways
    auto round_peer = [&]() {
        CK(cudaSetDevice(0));
        CK(cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s0));
        CK(cudaEventRecord(e0, s0));
        CK(cudaSetDevice(1));
        CK(cudaStreamWaitEvent(s1, e0, 0));
        CK(cudaMemcpyPeerAsync(d0, 0, d1, 1, bytes, s1));
        CK(cudaEventRecord(e1, s1));
        CK(cudaSetDevice(0));
        CK(cudaStreamWaitEvent(s0, e1, 0));
    };
    for (int pass = 0; pass < 2; ++pass) {
        auto f = [&]() { if (pass == 0) round_host(); else round_peer(); };
        for (int i = 0; i < 10; ++i) f();
        CK(cudaSetDevice(0));
        CK(cudaStreamSynchronize(s0));
        CK(cudaEventRecord(t0, s0));
        for (int i = 0; i < n; ++i) f();
        CK(cudaSetDevice(0));
        CK(cudaEventRecord(t1, s0));
        CK(cudaEventSynchronize(t1));
        float ms;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        printf("%s: %.1f us per round trip (two exchanges), %.1f us per exchange; 96 exchanges = %.2f ms\n",
               pass == 0 ? "pinned host + events" : "cudaMemcpyPeerAsync ", ms * 1000 / n, ms * 500 / n, ms * 48 / n);
    }
    return 0;
}
