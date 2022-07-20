/* This source file is for generating shared libraries for different sketches.
 * Compile with `g++ -DSKETCH=CMSketch` to generate libraries for different
 * kinds of sketches by replacing "CMSketch" to other sketch names.
 */

#ifndef SKETCH
#pragma message("SKETCH undefined, using CMSketch")
#define SKETCH CMSketch
#endif

// Count Sketches use int32_t counters, while others use uint32_t
#ifndef COUNTER_TYPE
#define COUNTER_TYPE uint32_t
#define COUNTER_TYPE_UINT32
#endif

// e.g. `GLUE(CMSketch, _new)` generates `CMSketch_new`
#define GLUE_HELPER(x, y) x##y
#define GLUE(x, y) GLUE_HELPER(x, y)

// Use ShiftEncoder or 1-bit indicators
// #define UseShiftEncoder

#include <ctime>
#include "trace.h"
#include "murmur3.h"
#include "SimpleTimer.h"
#include "CMSketch.h"
#include "CountSketch.h"
#include "CUSketch.h"
#include "CMMSketch.h"
#include "CMLSketch.h"
#include "CSMSketch.h"

extern "C" {

SKETCH *GLUE(SKETCH, _new)(int read_num, int d, bool is_compsen = false) {
    SKETCH *sketch = new SKETCH(read_num, d);
    if (is_compsen) sketch->compress_algo = "Compressive Sensing (restored)";
    return sketch;
}

void GLUE(SKETCH, _delete)(SKETCH *sketch) { delete sketch; }

void init_arr(SKETCH *sketch, int d_i, COUNTER_TYPE *in_data, int w, int w0,
              uint32_t seed) {
    sketch->init_arr(d_i, in_data, w, w0, seed);
}

void copy_array(SKETCH *sketch, int d_i, COUNTER_TYPE *out_data) {
    for (int i = 0; i < sketch->arr[d_i].w; i++) {
        out_data[i] = sketch->arr[d_i].counters[i];
    }
}

void insert_dataset(SKETCH *sketch, int start, int end) {
    sketch->insert_dataset(start, end);
}

double query_dataset(SKETCH *sketch, int k, double *out_AAE, int ignore_max) {
    return sketch->query_dataset(k, out_AAE, ignore_max);
}

double compress(SKETCH *sketch, int ratio, CompAlgoType algo, int zero_thld,
                int round_thld, int num_frags, int num_threads,
                int tower_settings_id, int sbf_size, double *out_tower_secs) {
    SimpleTimer timer;
    timer.start();

    switch (algo) {
        case HOKUSAI:
            sketch->compress_hokusai(ratio);
            break;
        case ELASTIC:
            sketch->compress_elastic(ratio);
            break;
        case COMPSEN: {
            vector<pair<int, int>> tower_settings;
            switch (tower_settings_id) {
                case 0:
                    tower_settings.assign({{4, 4}, {8, 1}});
                    break;
                case 1:  // Best
                    tower_settings.assign({{4, 2}, {4, 2}, {4, 1}});
                    break;
                case 2:
                    tower_settings.assign({{8, 8}, {16, 1}});
                    break;
                case 3:
                    exit(-1);  // deprecated
                    break;
                case 4:  // zero_thld = 1024 (10 bits)
                    tower_settings.assign({{2, 4}, {4, 2}, {4, 1}});
                    break;
                case 5:  // zero_thld = 2048 (11 bits)
                    tower_settings.assign({{1, 4}, {2, 4}, {8, 1}});
                    break;
                default:
                    tower_settings.clear();
            }
            return sketch->compress_compsen(
                ratio, zero_thld, round_thld, num_frags, num_threads,
                tower_settings, sbf_size, out_tower_secs);
        }
        default:
            fprintf(stderr, "Unknown compression algorithm type %d\n", algo);
            return 0;
    }

    timer.stop();
    return timer.elapsedSeconds();
}

#ifdef SIMD_COMPRESS
double compress_4096_simd(SKETCH *sketch, int ratio) {
    double total_comp_secs = 0;
    for (int d_i = 0; d_i < sketch->d; d_i++) {
        total_comp_secs += sketch->arr[d_i].compress_4096_simd(ratio);
    }
    return total_comp_secs;
}
#endif

double cs_decompress(SKETCH *sketch, int num_threads) {
    return sketch->decompress_compsen(num_threads);
}

void copy_A_and_y_frags(SKETCH *sketch, int d_i, int *A_frags_data,
                        int *y_frags_data) {
    assert(sketch->compress_algo.substr(0, 19) == "Compressive Sensing");
    int i = 0;
    for (auto A : sketch->arr[d_i].A_frags) {
        for (auto row : A) {
            for (auto e : row) {
                A_frags_data[i++] = e;
            }
        }
    }

    i = 0;
    for (auto y : sketch->arr[d_i].y_frags) {
        for (auto e : y) {
            y_frags_data[i++] = e;
        }
    }
}

// Recover small counters from the tower
double copy_tower_small_counters(SKETCH *sketch, int d_i,
                                 COUNTER_TYPE *out_data, int ignore_levels) {
    auto &tower_settings = sketch->arr[d_i].tower_settings;
    if (tower_settings.empty()) return -1;
    auto &tower_counters = sketch->arr[d_i].tower_counters;
    auto &overflow_indicators = sketch->arr[d_i].overflow_indicators;
    auto &sbf = sketch->arr[d_i].sbf;
    auto &tower_widths = sketch->arr[d_i].tower_widths;

    int level_cnt = tower_settings.size();
    int recovered_w = sketch->arr[d_i].tower_widths[0];
    memset(out_data, 0, recovered_w * sizeof(COUNTER_TYPE));

    vector<COUNTER_TYPE *> tower_recovered(level_cnt);
    for (int i = 0; i < level_cnt; i++) {
        tower_recovered[i] = new COUNTER_TYPE[tower_widths[i]];
        memset(tower_recovered[i], 0, tower_widths[i] * sizeof(COUNTER_TYPE));
    }

    SimpleTimer timer;
    timer.start();

#ifdef COUNTER_TYPE_UINT32
    for (int j = 0; j < tower_widths[level_cnt - 1]; j++) {
        int top_bits = tower_settings[level_cnt - 1].first;
        tower_recovered[level_cnt - 1][j] =
            ACCESS32(tower_counters[level_cnt - 1], top_bits, j);
    }

    for (int i = level_cnt - 2; i >= 0; i--) {
        int bits = tower_settings[i].first;
        int cnts = tower_settings[i].second;
        for (int k = 0; k < tower_widths[i]; k += 64) {
            uint64_t bitmap = sbf->query(k);

            for (int j = k; j < k + 64 && j < tower_widths[i]; j++) {
                if (i >= ignore_levels)
                    tower_recovered[i][j] =
                        ACCESS32(tower_counters[i], bits, j);
                else
                    tower_recovered[i][j] = 0;

#ifdef UseShiftEncoder
                if ((bitmap >> (j - k)) & 1) {
#else
                if (ACCESS8(overflow_indicators[i], 1, j)) {
#endif
                    int upper_bits = tower_settings[i + 1].first;
                    uint32_t upper_val = tower_recovered[i + 1][j / cnts];
                    tower_recovered[i][j] += upper_val * (1 << bits);
                }
            }
        }
    }
#else
    for (int j = 0; j < tower_widths[level_cnt - 1]; j++) {
        int top_bits = tower_settings[level_cnt - 1].first;
        tower_recovered[level_cnt - 1][j] = U2INT(
            ACCESS32(tower_counters[level_cnt - 1], top_bits, j), top_bits);
    }

    for (int i = level_cnt - 2; i >= 0; i--) {
        int bits = tower_settings[i].first;
        int cnts = tower_settings[i].second;
        for (int j = 0; j < tower_widths[i]; j++) {
            if (i >= ignore_levels)
                tower_recovered[i][j] =
                    U2INT(ACCESS32(tower_counters[i], bits, j), bits);
            else
                tower_recovered[i][j] = 0;

            if (ACCESS8(overflow_indicators[i], 1, j)) {
                int upper_bits = tower_settings[i + 1].first;
                int32_t upper_val = tower_recovered[i + 1][j / cnts];
                tower_recovered[i][j] += upper_val * (1 << (bits - 1));
            }
        }
    }
#endif

    // Write the recovered result
    for (int j = 0; j < recovered_w; j++) {
        out_data[j] = tower_recovered[0][j];
    }

    timer.stop();
    return timer.elapsedSeconds();
}

void query_and_copy(SKETCH *sketch, int k, int *out_data) {
    // printf("inside query_and_copy %p\n", sketch);
    // printf("sketch->ans.size()=%lu\n", sketch->ans.size());
    int i = 0;
    // printf("ready to go in loop\n");
    for (auto flow : sketch->ans) {
        // printf("inside loop\n");
        string flow_id = flow.second;
        int true_val = -flow.first;
        // printf("querying %d\n", i);
        int est_val = (int)sketch->query(flow_id);
        // printf("query %d ok\n", i);
        if (est_val < 0) {
            fprintf(stderr, "est_val=%d, true_val=%d\n", est_val, true_val);
            exit(-1);
        }
        // printf("writing out_data %d\n", i);
        out_data[i] = est_val;
        // printf("written out_data %d\n", i);
        // printf("%d, est_val=%d, (true_val=%d)\n", i, est_val, true_val);
        if (++i == k) break;
    }
}

double calc_acc(int k, int *in_data, double *out_AAE) {
    auto input = load_dataset(-1);
    auto ans = groundtruth(input);

    double ARE = 0.0;
    double AAE = 0.0;

    int i = 0;
    for (auto flow : ans) {
        string flow_id = flow.second;
        int true_val = -flow.first;
        int est_val = in_data[i];
        if (est_val < 0) {
            fprintf(stderr, "est_val=%d, true_val=%d\n", est_val, true_val);
            exit(-1);
        }
        int error = est_val - true_val;
        ARE += abs(error) * 1.0 / true_val;
        AAE += abs(error);

        if (debug) {
            printf("i=%d, est_val=%d, true_val=%d, error=%d, ARE+=%f\n", i,
                   est_val, true_val, error, (abs(error) * 1.0 / true_val));
        }

        if (++i == k) {
            if (debug) printf("Top-%d flow real frequency: %d \n", k, true_val);
            break;
        }
    }
    ARE /= i;
    AAE /= i;

    if (k == -1) k = i;
    // printf("\033[1;31mONLY CALC top-%d, ARE %f, AAE %f\033[0m\n", k, ARE, AAE);

    if (out_AAE) *out_AAE = AAE;
    return ARE;
}
}
