import os
import sys
import argparse
import numpy as np
import pandas as pd
from sketch_compress import *


def reject_outliers(data, m=2.):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return list(data[s < m])


def save_csv(df, filename, outdir='./results/'):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    df.to_csv(outdir+filename, index=False, sep=',')
    print("Results saved to \033[92m{}\033[0m".format(filename))


###### Below are test cases ######

# Top-k Accuracy vs. Compression Ratio
def test_acc_mem():
    wns = [60000, 80000, 100000, 120000]  # Lines
    ratios = range(2, 11)  # x-axis
    df_ARE = {**{'ratio': ratios}, **{wn: [] for wn in wns}}

    total = len(wns) * len(ratios)
    curr = 0

    for wn in wns:
        w = [wn] * args.d
        for ratio in ratios:
            print("\033[1;34m[{} / {}]\033[0m w = {}, ratio = {}".format(curr, total, wn, ratio))

            ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, args.sep_thld,
                                      args.round_thld, ratio, wn // 250, args.k, args.num_threads)
            df_ARE[wn].append(ret['ARE'])

            print("ARE = {}".format(ret['ARE']))
            curr += 1

    save_csv(pd.DataFrame(df_ARE), 'ARE_{}_mem.csv'.format(str(args.sketch)))


# Full Accuracy vs. Tower Shape
def test_fullacc_towershape():
    wns = range(200000, 1200000, 100000)  # x-axis
    df_ARE = {**{'w': wns, 'no_compress': []}, **{tower_settings_id: [] for tower_settings_id in range(3)}}

    total = len(wns) * (3 + 1)
    curr = 0

    for wn in wns:
        print("\033[1;34m[{} / {}]\033[0m w = {}, tower shape = {}".format(curr, total, wn, "baseline"))

        w = [wn] * args.d
        ret = no_compress(args.sketch, args.read_num, args.d, w, args.seed, -1)
        df_ARE['no_compress'].append(ret['ARE'])

        print("ARE = {}".format(ret['ARE']))
        curr += 1

    for tower_settings_id in range(3):
        for wn in wns:
            print("\033[1;34m[{} / {}]\033[0m w = {}, tower shape = #{}".format(curr, total, wn, tower_settings_id+1))

            w = [wn] * args.d
            ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, 4096, args.round_thld, 4, wn // 250, -1,
                                      args.num_threads, decompress_method='python', tower_settings_id=tower_settings_id)
            df_ARE[tower_settings_id].append(ret['ARE'])

            print("ARE = {}".format(ret['ARE']))
            curr += 1

    save_csv(pd.DataFrame(df_ARE), 'ARE_{}_fullacc_towershape.csv'.format(str(args.sketch)))


# Top-k Accuracy vs. Separating Threshold
def test_acc_separating():
    sep_thlds = [1024, 2048, 4096, 8192]  # Lines
    wns = range(60000, 130000, 10000)  # x-axis
    df_ARE = {**{'w': wns}, **{sep_thld: [] for sep_thld in sep_thlds}}

    total = len(sep_thlds) * len(wns)
    curr = 0

    for sep_thld in sep_thlds:
        for wn in wns:
            print("\033[1;34m[{} / {}]\033[0m sep_thld = {}, w = {}".format(curr, total, sep_thld, wn))

            w = [wn] * args.d
            ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed,
                                      sep_thld, args.round_thld, 2, wn // 250, args.k, args.num_threads)
            df_ARE[sep_thld].append(ret['ARE'])

            print("ARE = {}".format(ret['ARE']))
            curr += 1

    save_csv(pd.DataFrame(df_ARE), 'ARE_{}_separating.csv'.format(str(args.sketch)))


# Full Accuracy vs. Separating Threshold
def test_fullacc_separating():
    sep_thlds = [1024, 2048, 4096]  # Lines
    tower_setting_ids = [4, 5, 2]
    wns = range(200000, 1200000, 100000)  # x-axis
    df_ARE = {**{'w': wns}, **{sep_thld: [] for sep_thld in sep_thlds}}

    total = len(sep_thlds) * len(wns)
    curr = 0

    for i, sep_thld in enumerate(sep_thlds):
        for wn in wns:
            print("\033[1;34m[{} / {}]\033[0m sep_thld = {}, w = {}".format(curr, total, sep_thld, wn))
            w = [wn] * args.d

            ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, sep_thld,
                                      args.round_thld, 4, wn//250, -1, args.num_threads, tower_settings_id=tower_setting_ids[i])
            df_ARE[sep_thld].append(ret['ARE'])

            print("ARE = {}".format(ret['ARE']))
            curr += 1

    save_csv(pd.DataFrame(df_ARE), 'ARE_{}_full_separating.csv'.format(str(args.sketch)))


# Top-k Accuracy vs. Rounding Parameter
def test_acc_rounding():
    round_thlds = [1, 2, 4, 8]  # Lines
    wns = range(60000, 130000, 10000)  # x-axis
    df_ARE = {**{'w': wns}, **{round_thld: [] for round_thld in round_thlds}}

    total = len(round_thlds) * len(wns)
    curr = 0

    for round_thld in round_thlds:
        for wn in wns:
            print("\033[1;34m[{} / {}]\033[0m round_thld = {}, w = {}".format(curr, total, round_thld, wn))
            w = [wn] * args.d

            ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed,
                                      args.sep_thld, round_thld, 2, wn // 250, args.k, args.num_threads)
            df_ARE[round_thld].append(ret['ARE'])

            print("ARE = {}".format(ret['ARE']))
            curr += 1

    save_csv(pd.DataFrame(df_ARE), 'ARE_{}_rounding.csv'.format(str(args.sketch)))


# Flexibility of TowerEncoding
def test_acc_ignore_level():
    ignore_levels = [2, 1, 0]  # Lines
    wns = range(200000, 1200000, 100000)  # x-axis
    df_ARE = {'w': wns, **{ignore_level: [] for ignore_level in ignore_levels}}

    total = len(ignore_levels) * len(wns)
    curr = 0

    for ignore_level in ignore_levels:
        for wn in wns:
            print("\033[1;34m[{} / {}]\033[0m ignore_level = {}, w = {}".format(curr, total, ignore_level, wn))
            w = [wn] * args.d

            ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, 4096, args.round_thld,
                                      4, wn//250, -1, args.num_threads, tower_settings_id=1, ignore_level=ignore_level)
            df_ARE[ignore_level].append(ret['ARE'])

            print("ARE = {}".format(ret['ARE']))
            curr += 1

    save_csv(pd.DataFrame(df_ARE), 'ARE_{}_ignore_level.csv'.format(str(args.sketch)))


# Efficiency of TowerEncoding
def test_time_towerencoding():
    wns = range(200000, 1200000, 100000)  # x-axis
    df_comp = {**{'w': []}, **{tower_settings_id: [] for tower_settings_id in range(4)}}
    df_decomp = {**{'w': []}, **{tower_settings_id: [] for tower_settings_id in range(4)}}

    total = len(wns) * 4
    curr = 0

    # Loop by x-axis first
    for wn in wns:
        w = [wn] * args.d
        df_comp['w'].append(wn)
        df_decomp['w'].append(wn)

        for tower_settings_id in range(4):
            print("\033[1;34m[{} / {}]\033[0m w = {}, tower shape = #{}".format(curr, total, wn, tower_settings_id+1))

            temp_comp = []
            temp_decomp = []
            for _ in range(10):
                ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, 4096, args.round_thld, 4, wn // 250, -1, args.num_threads,
                                          do_query=False, decompress_method='python', tower_settings_id=tower_settings_id)
                # Tip: `res = [] * w[i]` to speed up this test
                temp_comp.append(ret['tower_comp_secs'])
                temp_decomp.append(ret['tower_decomp_secs'])
                print("Compression time = {} s, decompression time = {} s".format(ret['tower_comp_secs'], ret['tower_decomp_secs']))

            df_comp[tower_settings_id].append(np.average(reject_outliers(temp_comp)))
            df_decomp[tower_settings_id].append(np.average(reject_outliers(temp_decomp)))
            curr += 1

    save_csv(pd.DataFrame(df_comp), 'SECS_{}_tower_comp.csv'.format(str(args.sketch)))
    save_csv(pd.DataFrame(df_decomp), 'SECS_{}_tower_decomp.csv'.format(str(args.sketch)))


# Efficiency of SketchSensing
def test_time_sketchsensing():
    ratios = [4, 6, 8, 10]  # Lines
    wns = range(60000, 130000, 10000)  # x-axis
    df_comp = {**{'w': wns}, **{ratio: [] for ratio in ratios}}
    df_decomp = {**{'w': wns}, **{ratio: [] for ratio in ratios}}

    total = len(ratios) * len(wns)
    curr = 0

    for ratio in ratios:
        for wn in wns:
            print("\033[1;34m[{} / {}]\033[0m ratio = {}, w = {}".format(curr, total, ratio, wn))
            w = [wn] * args.d

            temp_comp = []
            temp_decomp = []
            for _ in range(10):
                ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, args.sep_thld,
                                          args.round_thld, ratio, wn // 250, args.k, args.num_threads,
                                          do_query=False, decompress_method='cpp')
                temp_comp.append(ret['comp_secs'])
                temp_decomp.append(ret['decomp_secs'])
                print("Compression time = {}, decompression time = {}".format(ret['comp_secs'], ret['decomp_secs']))

            df_comp[ratio].append(np.average(reject_outliers(temp_comp)))
            df_decomp[ratio].append(np.average(reject_outliers(temp_decomp)))
            curr += 1

    save_csv(pd.DataFrame(df_comp), 'SECS_{}_sensing_comp.csv'.format(str(args.sketch)))
    save_csv(pd.DataFrame(df_decomp), 'SECS_{}_sensing_decomp.csv'.format(str(args.sketch)))


# Efficiency of Privacy-preserving Compression
def test_time_encryption():
    def homomorphic(A_frags_ds, y_frags_ds, _):
        nonlocal encrypt_secs
        nonlocal decrypt_secs
        print("Running homomorphic encryption and decryption")
        context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)

        counter_cnt = len(y_frags_ds[0]) * 0.05

        y_front_ds = []
        for i in range(args.d):
            temp = []
            for y_frag in y_frags_ds[i]:
                for j, v in enumerate(y_frag):
                    if j > counter_cnt:
                        break
                    temp.append(v)
            y_front_ds.append(temp)

        # Encrypt
        time_start = time.time()
        y_encrypted_ds = []
        for i in range(args.d):
            y_encrypted = ts.bfv_vector(context, y_front_ds[i])
            y_encrypted_ds.append(y_encrypted)
        encrypt_secs = time.time() - time_start

        # Decrypt
        time_start = time.time()
        for i in range(args.d):
            y_decrypted = y_encrypted_ds[i].decrypt()
            # assert all(??)
        decrypt_secs = time.time() - time_start
        return A_frags_ds, y_frags_ds

    wns = range(10000, 110000, 10000)  # x-axis
    df = {'w': wns, 'encrypt': [], 'decrypt': []}
    ratio = 2

    total = len(wns)
    curr = 0

    for wn in wns:
        print("\033[1;34m[{} / {}]\033[0m w = {}".format(curr, total, wn))
        w = [wn] * args.d

        temp_encrypt = []
        temp_decrypt = []
        for _ in range(10):
            encrypt_secs, decrypt_secs = 0, 0
            ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, args.sep_thld, args.round_thld, ratio,
                                      wn // 250, args.k, args.num_threads, decompress_method=None, func=homomorphic)
            temp_encrypt.append(encrypt_secs)
            temp_decrypt.append(decrypt_secs)
            print("Encryption time = {} s, decryption time = {} s".format(encrypt_secs, decrypt_secs))

        df['encrypt'].append(np.average(reject_outliers(temp_encrypt)))
        df['decrypt'].append(np.average(reject_outliers(temp_decrypt)))
        curr += 1

    save_csv(pd.DataFrame(df), 'SECS_CM_encrypt_decrypt.csv')


def test_acceleration():
    wns = range(200000, 1200000, 100000)  # x-axis
    df_comp = {'w': wns, **{'serial': [], 'multithreading': [], 'simd': []}}

    total = len(wns)
    curr = 0

    for wn in wns:
        print("\033[1;34m[{} / {}]\033[0m w = {}".format(curr, total, wn))
        w = [wn] * args.d

        ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, args.sep_thld,
                                  args.round_thld, 4, wn // 250, -1, 1, do_query=False, decompress_method=None, tower_settings_id=1)
        df_comp['serial'].append(ret['comp_secs'] + ret['tower_comp_secs'])
        print("serial: {} s".format(ret['comp_secs'] + ret['tower_comp_secs']))

        ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, args.sep_thld,
                                  args.round_thld, 4, wn // 250, -1, 18, do_query=False, decompress_method=None, tower_settings_id=1)
        df_comp['multithreading'].append(ret['comp_secs'] + ret['tower_comp_secs'])
        print("multithreading: {} s".format(ret['comp_secs'] + ret['tower_comp_secs']))

        ret = compressive_sensing_comp_simd(args.read_num, args.d, w, args.seed, 4)
        df_comp['simd'].append(ret['total_comp_secs'])
        print("SIMD: {} s".format(ret['total_comp_secs']))

    save_csv(pd.DataFrame(df_comp), 'SECS_acceleration.csv')


# Top-k Accuracy (Comparison with Prior Art)
def test_acc_algos():
    ratios = range(2, 11)  # x-axis
    df_ARE = {'ratio': ratios, 'ours': [], 'hokusai': [], 'elastic': [], 'cluster_reduce': []}

    total = len(ratios)
    curr = 0

    for ratio in ratios:
        print("\033[1;34m[{} / {}]\033[0m ratio = {}".format(curr, total, ratio))

        # no_compress(args.sketch, args.read_num, args.d, args.w, args.seed, args.k)

        ret = compressive_sensing(args.sketch, args.read_num, args.d, args.w, args.seed,
                                  args.sep_thld, args.round_thld, ratio, args.w[0] // 250, args.k, args.num_threads)
        df_ARE['ours'].append(ret['ARE'])
        print("Compressive sensing: ARE = {}".format(ret['ARE']))

        ret = hokusai(args.sketch, args.read_num, args.d, args.w, args.seed, ratio, args.k)
        df_ARE['hokusai'].append(ret['ARE'])
        print("Hokusai: ARE = {}".format(ret['ARE']))

        if args.sketch == SketchType.Count:
            df_ARE['elastic'].append(np.nan)
        else:
            ret = elastic(args.sketch, args.read_num, args.d, args.w, args.seed, ratio, args.k)
            df_ARE['elastic'].append(ret['ARE'])
            print("Elastic: ARE = {}".format(ret['ARE']))

        ret = cluster_reduce(args.sketch, args.read_num, args.d, args.w, ratio, args.k, args.cr_method, args.debug)
        df_ARE['cluster_reduce'].append(ret['ARE'])
        print("Cluster Reduce: ARE = {}".format(ret['ARE']))
        curr += 1

    save_csv(pd.DataFrame(df_ARE), 'ARE_{}_algos.csv'.format(str(args.sketch)))


# Full Accuracy (Comparison with Prior Art)
def test_fullacc_algos():
    wns = range(200000, 1200000, 100000)  # x-axis
    df_ARE = {'w': wns, 'ours': [], 'hokusai': [], 'elastic': [], 'cluster_reduce': []}

    total = len(wns)
    curr = 0

    for wn in wns:
        print("\033[1;34m[{} / {}]\033[0m w = {}".format(curr, total, wn))
        w = [wn] * args.d

        ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, 4096,
                                  10, 4, wn // 250, -1, args.num_threads, tower_settings_id=1)
        df_ARE['ours'].append(ret['ARE'])
        print("Compressive sensing: ARE = {}".format(ret['ARE']))

        ret = hokusai(args.sketch, args.read_num, args.d, w, args.seed, 2, -1)
        df_ARE['hokusai'].append(ret['ARE'])
        print("Hokusai: ARE = {}".format(ret['ARE']))

        if args.sketch == SketchType.Count:
            df_ARE['elastic'].append(np.nan)
        else:
            ret = elastic(args.sketch, args.read_num, args.d, w, args.seed, 2, -1)
            df_ARE['elastic'].append(ret['ARE'])
            print("Elastic: ARE = {}".format(ret['ARE']))

        ret = cluster_reduce(args.sketch, args.read_num, args.d, w, 2, -1, args.cr_method)
        df_ARE['cluster_reduce'].append(ret['ARE'])
        print("Cluster Reduce: ARE = {}".format(ret['ARE']))
        curr += 1

    save_csv(pd.DataFrame(df_ARE), 'ARE_{}_full_algos.csv'.format(str(args.sketch)))


# Top-k Compression Efficiency (Comparison with Prior Art)
def test_time_algos():
    wns = range(60000, 130000, 10000)  # x-axis
    df_comp = {'w': wns, 'ours': [], 'hokusai': [], 'elastic': [], 'cluster_reduce': []}

    total = len(wns)
    curr = 0

    for wn in wns:
        print("\033[1;34m[{} / {}]\033[0m w = {}".format(curr, total, wn))
        w = [wn] * args.d

        ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, args.sep_thld, args.round_thld,
                                  4, wn//250, args.k, args.num_threads, do_query=False, decompress_method=None, tower_settings_id=-1)
        df_comp['ours'].append(ret['comp_secs'])
        print("Compressive sensing: Compression time = {} s".format(ret['comp_secs']))

        ret = hokusai(args.sketch, args.read_num, args.d, w, args.seed, 4, args.k)
        df_comp['hokusai'].append(ret['comp_secs'])
        print("Hokusai: Compression time = {} s".format(ret['comp_secs']))

        ret = elastic(args.sketch, args.read_num, args.d, w, args.seed, 4, args.k)
        df_comp['elastic'].append(ret['comp_secs'])
        print("Elastic: Compression time = {} s".format(ret['comp_secs']))

        ret = cluster_reduce(args.sketch, args.read_num, args.d, w, 4, args.k, method_id=args.cr_method)
        df_comp['cluster_reduce'].append(ret['comp_secs'])
        print("Cluster Reduce: Compression time = {} s".format(ret['comp_secs']))
        curr += 1

    save_csv(pd.DataFrame(df_comp), 'SECS_CM_algos.csv')


# Full Compression Efficiency (Comparison with Prior Art)
def test_time_full_algos():
    wns = range(200000, 1200000, 100000)  # x-axis
    df_comp = {'w': wns, 'ours': [], 'hokusai': [], 'elastic': [], 'cluster_reduce': []}

    total = len(wns)
    curr = 0

    for wn in wns:
        print("\033[1;34m[{} / {}]\033[0m w = {}".format(curr, total, wn))
        w = [wn] * args.d

        # ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, 4096, args.round_thld,
        #                           4, wn // 250, -1, args.num_threads, do_query=False, decompress_method=None, tower_settings_id=-1)
        # df_comp['ours'].append(ret['comp_secs'] + ret['tower_comp_secs'])

        ret = compressive_sensing_comp_simd(args.read_num, args.d, w, args.seed, 4)
        df_comp['ours'].append(ret['total_comp_secs'])
        print("Compressive sensing: Compression time = {} s".format(ret['total_comp_secs']))

        ret = hokusai(args.sketch, args.read_num, args.d, w, args.seed, 4, -1)
        df_comp['hokusai'].append(ret['comp_secs'])
        print("Hokusai: Compression time = {} s".format(ret['comp_secs']))

        ret = elastic(args.sketch, args.read_num, args.d, w, args.seed, 4, -1)
        df_comp['elastic'].append(ret['comp_secs'])
        print("Elastic: Compression time = {} s".format(ret['comp_secs']))

        ret = cluster_reduce(args.sketch, args.read_num, args.d, w, 4, -1, method_id=args.cr_method)
        df_comp['cluster_reduce'].append(ret['comp_secs'])
        print("Cluster Reduce: Compression time = {} s".format(ret['comp_secs']))

    save_csv(pd.DataFrame(df_comp), 'SECS_CM_full_algos.csv')


def test_distributed_app():
    algos = [CompAlgoType.COMPSEN, CompAlgoType.HOKUSAI, CompAlgoType.ELASTIC, CompAlgoType.CLUSRED]
    wns = range(10000, 110000, 10000)  # x-axis
    df_ARE = {'w': [], **{algo: [] for algo in algos}}
    ratio = 6

    for wn in wns:
        df_ARE['w'].append(wn)

        for algo in algos:
            if args.sketch == SketchType.Count and algo == CompAlgoType.ELASTIC:
                df_ARE[algo].append(np.nan)
                continue

            w = [wn] * args.d
            ret = distributed_data_stream(args.sketch, w, args.k, ratio, algo, args.cr_method, -1, 200)
            print(algo, wn, ret)
            df_ARE[algo].append(ret['ARE'])
        save_csv(pd.DataFrame(df_ARE), "ARE_{}_distributed.csv".format(args.sketch))

    save_csv(pd.DataFrame(df_ARE), "ARE_{}_distributed.csv".format(args.sketch))


def test_full_distributed_app():
    algos = [CompAlgoType.COMPSEN, CompAlgoType.HOKUSAI, CompAlgoType.ELASTIC, CompAlgoType.CLUSRED]
    wns = range(200000, 1200000, 100000)  # x-axis
    df_ARE = {'w': [], **{algo: [] for algo in algos}}

    for wn in wns:
        df_ARE['w'].append(wn)

        for algo in algos:
            ratio = 6 if algo == CompAlgoType.COMPSEN else 2
            print(wn, algo, ratio)
            if args.sketch == SketchType.Count and algo == CompAlgoType.ELASTIC:
                df_ARE[algo].append(np.nan)
                continue

            w = [wn] * args.d
            ret = distributed_data_stream(args.sketch, w, 253906, ratio, algo, args.cr_method, 0, 4096)
            print(algo, wn, ret)
            df_ARE[algo].append(ret['ARE'])
        save_csv(pd.DataFrame(df_ARE), "ARE_{}_full_distributed.csv".format(args.sketch))

    save_csv(pd.DataFrame(df_ARE), "ARE_{}_full_distributed.csv".format(args.sketch))


def test_shiftbf():
    wns = [500000, 700000, 900000, 1100000]  # Lines
    sbf_sizes = np.array(range(10, 61, 10)) * 8 * 1000  # x-axis
    print(sbf_sizes)
    df = {'sbf_size': [], **{wn: [] for wn in wns}}

    for sbf_size in sbf_sizes:
        df['sbf_size'].append(sbf_size)
        for wn in wns:
            w = [wn] * args.d

            ret = compressive_sensing(args.sketch, args.read_num, args.d, w, args.seed, 4096, args.round_thld,
                                      4, wn // 250, -1, args.num_threads, tower_settings_id=0, sbf_size=sbf_size)
            df[wn].append(ret['ARE'])

    save_csv(pd.DataFrame(df), 'ARE_{}_shiftbf_verybig.csv'.format(str(args.sketch)))


def test_topk_datasets():
    df = {'compressive_sensing': [], 'elastic': [], 'cluster_reduce': [], 'hokusai': []}
    ratio = 8

    no_compress(args.sketch, args.read_num, args.d, args.w, args.seed, args.k)

    ret = compressive_sensing(args.sketch, args.read_num, args.d, args.w, args.seed, args.sep_thld,
                              args.round_thld, ratio, args.w[0] // 250, args.k, args.num_threads)
    df['compressive_sensing'].append(ret['ARE'])

    ret = elastic(args.sketch, args.read_num, args.d, args.w, args.seed, ratio, args.k)
    df['elastic'].append(ret['ARE'])

    ret = cluster_reduce(args.sketch, args.read_num, args.d, args.w, ratio, args.k, args.cr_method, args.debug)
    df['cluster_reduce'].append(ret['ARE'])

    ret = hokusai(args.sketch, args.read_num, args.d, args.w, args.seed, ratio, args.k)
    df['hokusai'].append(ret['ARE'])

    save_csv(pd.DataFrame(df), 'ARE_topk_datasets.csv')


def test_full_datasets():
    df = {'compressive_sensing': [], 'elastic': [], 'cluster_reduce': [], 'hokusai': []}
    ratio = 2

    ret = compressive_sensing(args.sketch, args.read_num, args.d, args.w, args.seed, 4096,
                              args.round_thld, ratio, args.w[0] // 250, -1, args.num_threads, tower_settings_id=0)
    df['compressive_sensing'].append(ret['ARE'])

    ret = elastic(args.sketch, args.read_num, args.d, args.w, args.seed, ratio, -1)
    df['elastic'].append(ret['ARE'])

    ret = cluster_reduce(args.sketch, args.read_num, args.d, args.w, ratio, -1, args.cr_method, args.debug)
    df['cluster_reduce'].append(ret['ARE'])

    ret = hokusai(args.sketch, args.read_num, args.d, args.w, args.seed, ratio, -1)
    df['hokusai'].append(ret['ARE'])
    
    save_csv(pd.DataFrame(df), 'ARE_full_datasets.csv')


def test_algo():
    ratios = range(2, 11)
    df = pd.DataFrame({'ratio': ratios})

    # Original sketch without compressing
    original_ARE = no_compress(args.sketch, args.read_num, args.d, args.w, args.seed, args.k)['ARE']
    # df['no_compress'] = [original_ARE for _ in ratios]
    print('------------------------------------------')

    if args.algo in ['all', 'cs']:
        ARE_cs = [compressive_sensing(args.sketch, args.read_num, args.d, args.w, args.seed, args.sep_thld, args.round_thld, r, args.num_frags,
                                      args.k, args.num_threads, decompress_method=args.decompress_method)['ARE'] for r in ratios]
        df['compressive_sensing'] = ARE_cs
        print('ARE_cs', ARE_cs)

    if args.algo in ['all', 'hokusai']:
        ARE_hokusai = [hokusai(args.sketch, args.read_num, args.d,
                               args.w, args.seed, r, args.k)['ARE'] for r in ratios]
        df['hokusai'] = ARE_hokusai
        print('ARE_hokusai', ARE_hokusai)

    if args.algo in ['all', 'elastic']:
        # Elastic is not suitable for Count Sketch
        if args.sketch != SketchType.Count:
            ARE_elastic = [elastic(args.sketch, args.read_num, args.d, args.w,
                                   args.seed, r, args.k)['ARE'] for r in ratios]
            print('ARE_elastic', ARE_elastic)
        else:
            ARE_elastic = [np.nan for _ in ratios]
        df['elastic'] = ARE_elastic

    if args.algo in ['all', 'cr']:
        ARE_cr = [cluster_reduce(args.sketch, args.read_num, args.d, args.w,
                                 r, args.k, 0, args.debug)['ARE'] for r in ratios]
        df['cluster_reduce'] = ARE_cr
        print('ARE_cr', ARE_cr)

    save_csv(df, 'ARE_{}_{}_{}_{}_{}_{}_{}.csv'.format(str(args.sketch), args.k,
             args.d, args.w[0], args.sep_thld, args.round_thld, args.num_frags))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compressive sensing test for sketches")
    parser.add_argument('--read_num', default=-1, type=int, help='number of packets read from dataset')
    parser.add_argument('--k', default=500, type=int, help='report ARE for top-k frequent flows')
    parser.add_argument('--d', default=3, type=int, help='number of arrays')
    parser.add_argument('--w', default=[65000, 65000, 65000], type=int,
                        nargs='+', help='number of counters in each array')
    parser.add_argument('--seed', default=997, type=int, help='base seed')
    parser.add_argument('--sep_thld', default=9300, type=int, help='separating threshold')
    parser.add_argument('--round_thld', default=1, type=int, help='round threshold')
    parser.add_argument('--num_frags', default=300, type=int, help='number of fragments per array')
    parser.add_argument('--sketch', default=SketchType.CM, type=SketchType.from_string,
                        choices=list(SketchType), help='sketch type (CM / Count / CU / CMM / CML / CSM)')
    parser.add_argument('--algo', default='all', type=str, help='compressing algorithm')
    parser.add_argument('--num_threads', default=8, type=int,
                        help='number of threads for compressive sensing')
    parser.add_argument('--cr_method', default=4, type=int, help='compress method for Cluster Reduce')
    parser.add_argument('--test', default='test_algo', type=str, help='which test to run')
    parser.add_argument('--debug', dest='debug', action='store_true', help='turn on debugging')
    parser.set_defaults(debug=False)

    global args
    args = parser.parse_args()
    print(args)

    init(args)
    globals()[args.test]()
