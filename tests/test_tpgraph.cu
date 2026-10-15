// Can one CUDA graph hold both GPUs' work with host-staged exchanges between them? Two streams on two devices
// are captured into one graph: per "layer", each device adds to its vector, copies it to pinned host memory,
// and after the other's event adds the other's half (an all-reduce of two partial sums, as a tensor-parallel
// layer needs). Checks the result and times 96 such layers per graph launch.
//   build/test_tpgraph [rows=4] [floats per row=2560] [layers=96]
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void add_k(float * y, const float * x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void work_k(float * y, float v, int n) {   // "the layer": y += v
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += v;
}

int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4, row = argc > 2 ? atoi(argv[2]) : 2560, L = argc > 3 ? atoi(argv[3]) : 96;
    const int n = T * row;
    const size_t bytes = (size_t) n * 4;
    float * d[2], * h[2][2], * tmp[2];   // host buffers per side and layer parity (a layer's copy out must not overwrite what the other side still reads)
    cudaStream_t s[2];
    cudaEvent_t ev[2][128];
    for (int g = 0; g < 2; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMalloc(&d[g], bytes));
        CK(cudaMalloc(&tmp[g], bytes));
        CK(cudaMemset(d[g], 0, bytes));
        CK(cudaStreamCreateWithFlags(&s[g], cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) CK(cudaHostAlloc((void **) &h[g][b], bytes, cudaHostAllocPortable));
        for (int l = 0; l < L + 1; ++l) CK(cudaEventCreateWithFlags(&ev[g][l], cudaEventDisableTiming));
    }
    // capture: fork s[1] off s[0] (one capture), L layers, join
    CK(cudaSetDevice(0));
    CK(cudaStreamBeginCapture(s[0], cudaStreamCaptureModeThreadLocal));
    CK(cudaEventRecord(ev[0][L], s[0]));
    CK(cudaSetDevice(1));
    CK(cudaStreamWaitEvent(s[1], ev[0][L], 0));
    const int blocks = (n + 255) / 256;
    for (int l = 0; l < L; ++l) {
        for (int g = 0; g < 2; ++g) {
            CK(cudaSetDevice(g));
            work_k<<<blocks, 256, 0, s[g]>>>(d[g], 1e-3f * (1 + g), n);                       // partial sum of this layer
            CK(cudaMemcpyAsync(h[g][l & 1], d[g], bytes, cudaMemcpyDeviceToHost, s[g]));    // out to the host
            CK(cudaEventRecord(ev[g][l], s[g]));
        }
        for (int g = 0; g < 2; ++g) {
            CK(cudaSetDevice(g));
            CK(cudaStreamWaitEvent(s[g], ev[1 - g][l], 0));                           // the other's partial is there
            CK(cudaMemcpyAsync(tmp[g], h[1 - g][l & 1], bytes, cudaMemcpyHostToDevice, s[g]));
            add_k<<<blocks, 256, 0, s[g]>>>(d[g], tmp[g], n);                        // all-reduced: both hold the sum
        }
    }
    CK(cudaSetDevice(1));
    CK(cudaEventRecord(ev[1][L], s[1]));
    CK(cudaSetDevice(0));
    CK(cudaStreamWaitEvent(s[0], ev[1][L], 0));
    cudaGraph_t graph;
    CK(cudaStreamEndCapture(s[0], &graph));
    cudaGraphExec_t exec;
    CK(cudaGraphInstantiate(&exec, graph, 0));
    size_t nodes = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &nodes));
    // correctness: after one launch every element on both devices = L * (1 + 2) ... but each layer adds the
    // OTHER's running total, so compute the expected sequence on the host
    double a = 0, b = 0;
    for (int l = 0; l < L; ++l) { const double na = a + 1e-3, nb = b + 2e-3; a = na + nb; b = nb + na; }
    CK(cudaGraphLaunch(exec, s[0]));
    CK(cudaStreamSynchronize(s[0]));
    std::vector<float> out(n);
    CK(cudaMemcpy(out.data(), d[0], bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; ++i) ok &= fabs(out[i] - a) < 1e-3 * fabs(a);
    CK(cudaSetDevice(1));
    CK(cudaMemcpy(out.data(), d[1], bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) ok &= fabs(out[i] - b) < 1e-3 * fabs(b);
    printf("one graph over two devices: %zu nodes, result %s\n", nodes, ok ? "correct" : "WRONG");
    // timing
    CK(cudaSetDevice(0));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(exec, s[0]));
    CK(cudaStreamSynchronize(s[0]));
    const int reps = 20;
    CK(cudaEventRecord(t0, s[0]));
    for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(exec, s[0]));
    CK(cudaEventRecord(t1, s[0]));
    CK(cudaEventSynchronize(t1));
    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    printf("%d layers x all-reduce of %d x %d floats: %.2f ms per graph launch, %.1f us per layer\n", L, T, row, ms / reps,
           ms * 1000 / reps / L);
    return ok ? 0 : 1;
}
