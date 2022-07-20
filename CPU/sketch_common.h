#ifndef _SKETCH_COMMON_H_
#define _SKETCH_COMMON_H_

#include <cassert>
#include <vector>
#include <numeric>
#include <random>
#include <thread>
#include "trace.h"
#include "murmur3.h"
#include "SimpleTimer.h"
#include "omp.h"
#include "utils.h"
#include "ShiftBF.h"
#include <iostream>
#include <immintrin.h>

enum CompAlgoType { HOKUSAI = 0, ELASTIC = 1, COMPSEN = 2 };

bool debug = false;
extern "C" {
void set_debug_level(bool debug_) { debug = debug_; }
}

// Divide `total` into `pieces` as evenly as possible
vector<int> split_integer(int total, int pieces) {
    assert(total > 0 && pieces > 0 && pieces <= total);
    vector<int> indexes;
    int q = total / pieces;
    int r = total % pieces;
    int curr = 0;
    for (int i = 0; i < pieces - r; i++) {
        indexes.push_back(curr);
        curr += q;
    }
    for (int i = 0; i < r; i++) {
        indexes.push_back(curr);
        curr += q + 1;
    }
    indexes.push_back(curr);  // end
    assert(indexes.size() == pieces + 1);
    return indexes;
}

// Inner product of matrix A and vector x
vector<int> matvec_dot(const vector<vector<int>> &A, const vector<int> &x) {
    assert(A[0].size() == x.size());
    vector<int> res;
    for (const vector<int> &row : A) {
        int e = inner_product(row.begin(), row.end(), x.begin(), 0);
        res.push_back(e);
    }
    return res;
}

template <typename CounterType>
class CounterArray {
  public:
    CounterType *counters = nullptr;
    uint32_t seed;
    int w0 = -1;  // original width
    int w;  // current width (since compression will cause width to change)
    vector<vector<vector<int>>> A_frags;
    vector<vector<int>> y_frags;

    vector<pair<int, int>> tower_settings;
    vector<uint32_t *> tower_counters;
    vector<uint8_t *> overflow_indicators;
    ShiftBF *sbf;  // Overflow indicators using Shift Bloom Filter
    vector<int> tower_widths;

    uint32_t kappa[2];  // SIMD

    virtual ~CounterArray() {
        for (auto arr : tower_counters) delete[] arr;
        for (auto arr : overflow_indicators) delete[] arr;
        if (counters) delete[] counters;
        if (sbf) delete sbf;
    }

    virtual void insert(pair<string, float> pkt, uint32_t f = 1) = 0;

    virtual CounterType query(string flow) = 0;

    void init(CounterType *in_data, int w_, int w0_, uint32_t seed_) {
        seed = seed_;
        w0 = w0_ > 0 ? w0_ : w_;
        w = w_;
        kappa[0] = 8;
        sbf = new ShiftBF(w / 2, 97);

        if (counters) delete[] counters;
        counters = new CounterType[w];
        memset(counters, 0, sizeof(CounterType) * w);

        if (in_data) {
            for (int i = 0; i < w; ++i) counters[i] = in_data[i];
        }
    }

    int hash_value(string flow) {
        return MurmurHash3_x86_32(flow.data(), flow.length(), seed) % w0 % w;
    }

    double cs_compress(int ratio, int zero_thld, int round_thld, int num_frags,
                       int num_threads,
                       const vector<pair<int, int>> &tower_settings_,
                       int sbf_size, double *out_tower_secs) {
        A_frags.resize(num_frags);
        y_frags.resize(num_frags);

        int m = (w + ratio - 1) / ratio;           // ceil(w / ratio)
        int m1 = (m + num_frags - 1) / num_frags;  // ceil(m / num_frags)
        const vector<int> idxs = split_integer(w, num_frags);
        std::default_random_engine generator;
        std::binomial_distribution<int> distribution(1, 0.5);

        // Generate random matrices
        for (int i = 0; i < num_frags; i++) {
            int frag_len = idxs[i + 1] - idxs[i];
            A_frags[i].assign(m1, vector<int>(frag_len));
            for (int j = 0; j < m1; j++) {
                for (int k = 0; k < frag_len; k++) {
                    A_frags[i][j][k] = distribution(generator);
                }
            }
        }

        // Tower encoding for counters with small values
        if (!tower_settings_.empty()) {
            // assert(zero_thld == 4096);
            tower_settings = tower_settings_;
            int level_cnt = tower_settings.size();

            tower_counters.resize(level_cnt);
            overflow_indicators.resize(level_cnt - 1);
            tower_widths.resize(level_cnt);

            int counter_num = w;
            for (int i = 0; i < level_cnt; i++) {
                tower_widths[i] = counter_num;
                int bits = tower_settings[i].first;
                int cnts = tower_settings[i].second;

                int num_per_slot = 32 / bits;
                int packed_width =
                    (counter_num + num_per_slot - 1) / num_per_slot;
                tower_counters[i] = new uint32_t[packed_width];
                memset(tower_counters[i], 0, packed_width * sizeof(uint32_t));

                if (i < level_cnt - 1) {
                    int o_packed_width = (counter_num + 7) / 8;
                    overflow_indicators[i] = new uint8_t[o_packed_width];
                    memset(overflow_indicators[i], 0,
                           o_packed_width * sizeof(uint8_t));
                }

                counter_num = (counter_num + cnts - 1) / cnts;
            }

            CounterType small_counters[w];
            memset(small_counters, 0, w * sizeof(CounterType));
            for (int j = 0; j < w; j++) {
                if (counters[j] < zero_thld) small_counters[j] = counters[j];
            }

            if (sbf_size > 0) {
                delete sbf;
                sbf = new ShiftBF(sbf_size, 97);
            }

            SimpleTimer tower_timer;
            tower_timer.start();

            // Encode counters into each level of the tower
#ifdef COUNTER_TYPE_UINT32
            for (int k = 0; k < w; k += 64) {
                uint64_t bitmap = 0;

                for (int j = k; j < k + 64 && j < w; j++) {
                    uint32_t value = small_counters[j];
                    int idx = j;
                    for (int i = 0; i < level_cnt; i++) {
                        int bits = tower_settings[i].first;
                        int cnts = tower_settings[i].second;
                        uint32_t curr_val =
                            ACCESS32(tower_counters[i], bits, idx);

                        if (i == level_cnt - 1) {  // Top level of tower
                            uint32_t counter_max = (1 << bits) - 1;
                            if (value > counter_max - curr_val) {  // Overflow
                                WRITE32(tower_counters[i], bits, idx,
                                        counter_max);
                            } else {
                                WRITE32(tower_counters[i], bits, idx,
                                        (curr_val + value) % (1 << bits));
                            }
                        } else {  // Lower levels of tower
                            WRITE32(tower_counters[i], bits, idx,
                                    (curr_val + value) % (1 << bits));
                            value = (curr_val + value) / (1 << bits);
                            if (value) {  // Overflow to the next level
#ifdef UseShiftEncoder
                                bitmap |= (1ull << (j - k));
#else
                                WRITE8(overflow_indicators[i], 1, idx, 1);
#endif
                            } else {
                                break;
                            }
                            idx /= cnts;
                        }
                    }
                }
                sbf->insert(bitmap, k);
            }
#else
            for (int j = 0; j < w; j++) {
                int32_t value = small_counters[j];
                int idx = j;
                for (int i = 0; i < level_cnt; i++) {
                    int bits = tower_settings[i].first;
                    int cnts = tower_settings[i].second;
                    int32_t curr_val =
                        U2INT(ACCESS32(tower_counters[i], bits, idx), bits);

                    if (i == level_cnt - 1) {  // Top level of tower
                        int32_t counter_max = (1 << (bits - 1)) - 1;
                        if (value + curr_val > counter_max) {  // Overflow
                            WRITE32(tower_counters[i], bits, idx, counter_max);
                        } else if (value + curr_val < -counter_max) {
                            WRITE32(tower_counters[i], bits, idx,
                                    INT2U(-counter_max, bits));
                        } else {
                            WRITE32(
                                tower_counters[i], bits, idx,
                                INT2U((curr_val + value) % (1 << (bits - 1)),
                                      bits));
                        }
                    } else {  // Lower levels of tower
                        WRITE32(tower_counters[i], bits, idx,
                                INT2U((curr_val + value) % (1 << (bits - 1)),
                                      bits));
                        value = (curr_val + value) / (1 << (bits - 1));
                        if (value) {  // Overflow to the next level
                            WRITE8(overflow_indicators[i], 1, idx, 1);
                        } else {
                            break;
                        }
                        idx /= cnts;
                    }
                }
            }
#endif

            tower_timer.stop();
            if (out_tower_secs) *out_tower_secs = tower_timer.elapsedSeconds();
        }
        // Tower encoding ended

        // Sketch Sensing for counters with large values
        std::thread threads[num_threads];
        const vector<int> frag_idxs = split_integer(num_frags, num_threads);

        SimpleTimer timer;
        timer.start();

        for (int i = 0; i < w; i++) {
            CounterType value = counters[i];
            if (abs((int)value) < zero_thld) {  // Discard data
                counters[i] = 0;
            } else {  // Round data
                counters[i] = value / round_thld * round_thld;
            }
        }

        // Compress big counters with compressive sensing
        auto comp_tfunc = [&idxs, this](int start, int end) {
            for (int i = start; i < end; i++) {
                vector<int> x1(counters + idxs[i], counters + idxs[i + 1]);
                y_frags[i] = matvec_dot(A_frags[i], x1);
            }
        };

        int choice = num_threads > 1 ? 0 : 1;
        switch (choice) {
            case 1:  // Serial
                for (int i = 0; i < num_frags; i++) {
                    vector<int> x1(counters + idxs[i], counters + idxs[i + 1]);
                    y_frags[i] = matvec_dot(A_frags[i], x1);
                }
                break;

            case 2:  // Serial using thread func
                comp_tfunc(0, num_frags);
                break;

            default:  // Multithreading
                for (int t = 0; t < num_threads; t++) {
                    threads[t] =
                        std::thread(comp_tfunc, frag_idxs[t], frag_idxs[t + 1]);
                }
                for (int t = 0; t < num_threads; t++) {
                    threads[t].join();
                }
        }

        timer.stop();

        // Make counters unavailable, use A_frags and y_frags instead
        delete[] counters;
        counters = nullptr;

        return timer.elapsedSeconds();
    }

#ifdef SIMD_COMPRESS
    double compress_4096_simd(int ratio) {
        tower_counters.resize(2);
        overflow_indicators.resize(1);
        tower_widths.resize(2);
        uint32_t *small_counters = tower_counters[0];      // 8 bits
        uint32_t *medium_counters = tower_counters[1];     // 16 bits
        uint8_t *small_overflow = overflow_indicators[0];  // 1 bit flag

        uint32_t *large_counters = new uint32_t[w];
        memset(large_counters, 0, sizeof(uint32_t) * w);
        for (int i = 0; i < w; ++i)
            if (counters[i] >= 4096) large_counters[i] = counters[i];

        // use compressive sensing to compress large counters...
        int num_frags = w / 250;
        A_frags.resize(num_frags);
        y_frags.resize(num_frags);

        int m = (w + ratio - 1) / ratio;           // ceil(w / ratio)
        int m1 = (m + num_frags - 1) / num_frags;  // ceil(m / num_frags)
        const vector<int> idxs = split_integer(w, num_frags);
        std::default_random_engine generator;
        std::binomial_distribution<int> distribution(1, 0.5);
        for (int i = 0; i < num_frags; i++) {
            int frag_len = idxs[i + 1] - idxs[i];
            A_frags[i].assign(m1, vector<int>(frag_len));
            for (int j = 0; j < m1; j++) {
                for (int k = 0; k < frag_len; k++) {
                    A_frags[i][j][k] = distribution(generator);
                }
            }
        }

        int w1 = (w + 3) / 4 + 10;
        int w2 = (w + kappa[0] * 2 - 1) / (kappa[0] * 2) + 10;
        int w0 = (w + 7) / 8 + 10;

        small_counters = new uint32_t[w1];
        medium_counters = new uint32_t[w2];
        small_overflow = new uint8_t[w0];
        memset(small_counters, 0, sizeof(uint32_t) * w1);
        memset(medium_counters, 0, sizeof(uint32_t) * w2);
        memset(small_overflow, 0, sizeof(uint8_t) * w0);

        auto comp_tfunc = [&idxs, large_counters, this](int start, int end) {
            for (int i = start; i < end; i++) {
                vector<int> x1(large_counters + idxs[i],
                               large_counters + idxs[i + 1]);
                y_frags[i] = matvec_dot(A_frags[i], x1);
            }
        };

        int num_threads = 18;
        std::thread threads[num_threads];
        const vector<int> frag_idxs = split_integer(num_frags, num_threads);

        SimpleTimer timer;
        timer.start();

        // compress large counters
        for (int t = 0; t < num_threads; t++) {
            threads[t] =
                std::thread(comp_tfunc, frag_idxs[t], frag_idxs[t + 1]);
        }
        for (int t = 0; t < num_threads; t++) {
            threads[t].join();
        }

        // start SIMD
        __m256i all_ones = _mm256_set1_epi64x(-1);
        __m256i all_eights = _mm256_set1_epi32(8);
        __m256i zero_shift = _mm256_set1_epi32(12);
        __m256i simd_mask =
            _mm256_set_epi32(0x80808080, 0x40404040, 0x20202020, 0x10101010,
                             0x08080808, 0x04040404, 0x02020202, 0x01010101);

        for (int i = 0; i < w; i += 8) {
            __m256i item = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(counters + i));
            __m256i test_tmp = _mm256_srlv_epi32(item, zero_shift);
            __mmask8 valid = _mm256_test_epi32_mask(test_tmp, all_ones);

            __m256i v = _mm256_set1_epi8(~valid);
            v = _mm256_and_si256(v, simd_mask);
            v = _mm256_cmpeq_epi32(v, simd_mask);
            item = _mm256_and_si256(v, item);

            //_mm256_storeu_si256(reinterpret_cast<__m256i*>(counters+i), item);

            __m128i lowbit = _mm256_cvtepi32_epi8(item);

            uint64_t lowbit_store = _mm_cvtsi128_si64(lowbit);
            *((uint64_t *)small_counters + i / 8) = lowbit_store;

            __m256i highbit = _mm256_srlv_epi32(item, all_eights);
            __mmask8 overflows = _mm256_test_epi32_mask(highbit, all_ones);
            small_overflow[i / 8] = (uint8_t)overflows;

            __m128i low_part = _mm256_extractf128_si256(highbit, 0);
            __m128i high_part = _mm256_extractf128_si256(highbit, 1);

            __m128i x = _mm_max_epi32(low_part, high_part);
            __m128i max1 = _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 0, 3, 2));
            __m128i max2 = _mm_max_epi32(x, max1);
            __m128i max3 = _mm_shuffle_epi32(max2, _MM_SHUFFLE(0, 0, 0, 1));
            __m128i max4 = _mm_max_epi32(max2, max3);
            uint16_t max_val = _mm_cvtsi128_si32(max4);

            *((uint16_t *)medium_counters + i / 8) = max_val;
        }

        timer.stop();
        return timer.elapsedSeconds();
    }
#endif

    double cs_decompress(int num_threads = 1) {
        int num_frags = A_frags.size();
        assert(num_frags != 0 && y_frags.size() == num_frags);
        vector<vector<float>> xhat_frags(num_frags);

        auto decomp_tfunc = [&xhat_frags, this](int start, int end) {
            for (int i = start; i < end; i++) {
                int m = A_frags[i].size(), n = A_frags[i][0].size();
                assert(m == y_frags[i].size());
                float fl_A[m][n];
                float fl_y[m];
                float x_hat[n];

                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < n; k++) {
                        fl_A[j][k] = A_frags[i][j][k];
                    }
                    fl_y[j] = y_frags[i][j];
                }
                omp::decompress((float *)fl_A, fl_y, x_hat, m, n);
                for (int j = 0; j < n; j++) {
                    xhat_frags[i].push_back(x_hat[j]);
                }
            }
        };

        std::thread threads[num_threads];
        const vector<int> frag_idxs = split_integer(num_frags, num_threads);

        SimpleTimer timer;
        timer.start();

        for (int t = 0; t < num_threads; t++) {
            threads[t] =
                std::thread(decomp_tfunc, frag_idxs[t], frag_idxs[t + 1]);
        }
        for (int t = 0; t < num_threads; t++) {
            threads[t].join();
        }

        timer.stop();

        if (counters) delete[] counters;
        counters = new CounterType[w];
        int k = 0;
        for (const vector<float> &x_hat : xhat_frags) {
            for (int v : x_hat) {
                counters[k++] = v;
            }
        }

        return timer.elapsedSeconds();
    }
};

template <typename ArrayType, typename CounterType = uint32_t>
class CounterMatrix {
  public:
    vector<pair<string, float>> input;
    vector<pair<int, string>> ans;
    int d;
    ArrayType *arr;

    int stream_size;
    string sketch_name;
    string compress_algo = "Uncompressed";

    CounterMatrix(int read_num, int d_) {
        input = load_dataset(read_num);
        ans = groundtruth(input, read_num);
        stream_size = input.size();
        d = d_;
        arr = new ArrayType[d];
    }

    virtual ~CounterMatrix() { delete[] arr; }

    virtual void insert(pair<string, float> pkt) = 0;

    virtual int query(string flow) = 0;

    void init_arr(int d_i, CounterType *in_data, int w, int w0, uint32_t seed) {
        arr[d_i].init(in_data, w, w0, seed);
    }

    void insert_dataset(int start = 0, int end = -1) {
        if (end < 0) end = input.size();
        for (int i = start; i < end; i++) insert(input[i]);
    }

    // Query dataset for top-k flows, return ARE and write AAE into out_AAE
    double query_dataset(int k, double *out_AAE = nullptr, int ignore_max = 0) {
        assert(ignore_max == 0 || k == -1);
        double ARE = 0.0;
        double AAE = 0.0;

        int i = 0;
        for (auto flow : ans) {
            string flow_id = flow.second;
            int true_val = -flow.first;
            if (true_val <= ignore_max) break;  // test_acc_ignore_level

            int est_val = (int)query(flow_id);
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
                if (debug)
                    printf("Top-%d flow real frequency: %d \n", k, true_val);
                break;
            }
        }
        ARE /= i;
        AAE /= i;

        if (k == -1) k = i;
        // printf("\033[1;31m%s (%s), top-%d, ARE %f, AAE %f\033[0m\n",
        //        sketch_name.c_str(), compress_algo.c_str(), k, ARE, AAE);

        if (out_AAE) *out_AAE = AAE;
        return ARE;
    }

    void compress_hokusai(int ratio) {
        compress_algo = "Hokusai";
        for (int i = 0; i < d; i++) {
            int new_w = arr[i].w / ratio;
            CounterType *new_counters = new CounterType[new_w];
            memset(new_counters, 0, sizeof(CounterType) * new_w);
            for (int j = 0; j < arr[i].w; j++) {
                new_counters[j % new_w] += arr[i].counters[j];
            }
            delete[] arr[i].counters;
            arr[i].counters = new_counters;
            arr[i].w = new_w;
        }
    }

    void compress_elastic(int ratio) {
        compress_algo = "Elastic";
        for (int i = 0; i < d; i++) {
            int new_w = arr[i].w / ratio;
            CounterType *new_counters = new CounterType[new_w];
            memset(new_counters, 0, sizeof(CounterType) * new_w);
            for (int j = 0; j < arr[i].w; j++) {
                new_counters[j % new_w] =
                    max(new_counters[j % new_w], arr[i].counters[j]);
            }
            delete[] arr[i].counters;
            arr[i].counters = new_counters;
            arr[i].w = new_w;
        }
    }

    double compress_compsen(int ratio, int zero_thld, int round_thld,
                            int num_frags, int num_threads,
                            const vector<pair<int, int>> &tower_settings,
                            int sbf_size, double *out_tower_secs = nullptr) {
        compress_algo = "Compressive Sensing (compact)";
        double secs_used = 0.0;
        double tower_secs = 0.0;
        for (int i = 0; i < d; i++) {
            double temp_tower_secs;
            secs_used += arr[i].cs_compress(
                ratio, zero_thld, round_thld, num_frags, num_threads,
                tower_settings, sbf_size, &temp_tower_secs);
            tower_secs += temp_tower_secs;
        }
        if (out_tower_secs) *out_tower_secs = tower_secs;
        return secs_used;
    }

    double decompress_compsen(int num_threads) {
        assert(compress_algo == "Compressive Sensing (compact)");
        compress_algo = "Compressive Sensing (restored)";
        double secs_used = 0;
        for (int i = 0; i < d; i++) {
            secs_used += arr[i].cs_decompress(num_threads);
        }
        return secs_used;
    }
};

#endif
