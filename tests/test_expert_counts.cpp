// The routing-counts file's merge (ExpertCache's merge_counts) and the host store's spare blobs, without a GPU.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/expert_cache.h"

using namespace bnk;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    setenv("BNK_COUNTS_CAP", "1000", 1);
    // delta += old
    std::vector<uint32_t> d{10, 20, 30}, o{1, 2, 3};
    merge_counts(d, o);
    CHECK(d[0] == 11 && d[1] == 22 && d[2] == 33);
    // a size mismatch ignores the old file
    std::vector<uint32_t> d2{5, 5}, o2{1, 2, 3};
    merge_counts(d2, o2);
    CHECK(d2[0] == 5 && d2[1] == 5);
    // over the cap: halved until under it, the order kept
    std::vector<uint32_t> d3{800, 400, 200}, o3{800, 400, 200};
    merge_counts(d3, o3);
    uint64_t tot = 0;
    for (auto v : d3) tot += v;
    CHECK(tot <= 1000);
    CHECK(d3[0] > d3[1] && d3[1] > d3[2]);
    CHECK(d3[0] == 400 && d3[1] == 200 && d3[2] == 100);   // 3200 -> 1600 -> 800
    // the ranking from counts: hottest first, ties interleaved across layers
    std::vector<uint32_t> c{5, 1, 9, 2};   // 2 layers x 2 experts
    Ranking r = ranking_from_counts(c, 2, 2);
    CHECK(r.size() == 4 && r[0] == std::make_pair(1, 0) && r[1] == std::make_pair(0, 0));
    printf(fails ? "expert counts: FAILURES\n" : "expert counts: all OK\n");
    return fails ? 1 : 0;
}
