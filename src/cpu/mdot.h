// Multi-token AVX2 dot products (decode a weight row once, dot it with up to T activation rows).
#pragma once

namespace bnk {

bool mdot_supported(int ggml_type);
// s[t] = row . y[t] for t < T; y[t] are activation rows in the format's ggml vec_dot type (Q8_K, or Q8_0 for IQ4_NL)
void mdot(int ggml_type, int n, const void * row, const void * const * y, int T, float * s);

}  // namespace bnk
