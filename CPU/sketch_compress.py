from ctypes import *
import numpy as np
import cvxpy as cvx
import math
from tqdm import tqdm
from enum import IntEnum
import tenseal as ts
import time


class SketchType(IntEnum):
    CM = 0
    Count = 1
    CU = 2
    CMM = 3
    CML = 4
    CSM = 5

    @staticmethod
    def from_string(s):
        try:
            s = 'Count' if s.upper() == 'COUNT' else s.upper()
            return SketchType[s]
        except KeyError:
            raise ValueError()

    def __str__(self):
        return self.name


class CompAlgoType(IntEnum):
    HOKUSAI = 0
    ELASTIC = 1
    COMPSEN = 2
    CLUSRED = 3


class Sketch(object):
    def __init__(self, read_num, d, sketch_type, is_compsen=False):
        self.obj = getattr(lib, '{}Sketch_new'.format(str(sketch_type)))(read_num, d, is_compsen)
        self.sketch_name = '{} Sketch'.format(str(sketch_type))
        self.sketch_type = sketch_type
        self.d = d
        self.w = [0 for _ in range(d)]

    def __del__(self):
        getattr(lib, '{}Sketch_delete'.format(str(self.sketch_type)))(self.obj)

    def init_arr(self, d_i, in_data, w, w0, seed):
        self.w[d_i] = w
        lib.init_arr(self.obj, d_i, in_data, w, w0, seed)

    def copy_array(self, d_i, out_data):
        lib.copy_array(self.obj, d_i, out_data)

    def insert_dataset(self, start=0, end=-1):
        lib.insert_dataset(self.obj, start, end)

    def query_dataset(self, k, ignore_max=0):
        c_AAE = c_double()
        ARE = lib.query_dataset(self.obj, k, c_AAE, ignore_max)
        return ARE, c_AAE.value

    def compress(self, ratio, algo, zero_thld=0, round_thld=1, num_frags=300, num_threads=1, tower_settings_id=-1, sbf_size=-1):
        c_tower_secs = c_double()
        secs = lib.compress(self.obj, ratio, algo, zero_thld, round_thld,
                            num_frags, num_threads, tower_settings_id, sbf_size, c_tower_secs)
        return secs, c_tower_secs.value

    def compress_4096_simd(self, ratio):
        lib.compress_4096_simd.restype = c_double
        lib.compress_4096_simd.argtypes = [c_void_p, c_int]
        secs = lib.compress_4096_simd(self.obj, ratio)
        return secs


def split_integer(a, b):
    assert a > 0 and b > 0 and b <= a
    quotient = int(a / b)
    remainder = a % b
    return [quotient] * (b - remainder) + [quotient + 1] * remainder


def cs_compress_array(x, ratio, num_frags):
    n = len(x)
    m = int(math.ceil(n / ratio))
    m1 = int(math.ceil(m / num_frags))
    nn = split_integer(n, num_frags)
    A = []
    yy = []
    debug_frags_in = []

    i = 0
    for n1 in nn:
        x1 = x[i: i+n1]
        debug_frags_in.append(x1)
        A1 = np.random.binomial(1, 0.5, (m1, n1))
        A.append(A1)
        y1 = np.dot(A1, x1)
        yy.append(y1)
        i += n1
    return A, yy, debug_frags_in


def cs_decompress_array(A, yy, has_neg=False):
    num_frags = len(A)
    result = []
    debug_frags_out = []
    invalid = 0

    # for i in tqdm(range(num_frags)):
    for i in range(num_frags):
        vx = cvx.Variable(A[i].shape[1])
        objective = cvx.Minimize(cvx.norm(vx, 1))
        constraints = [A[i] @ vx == yy[i]]
        if not has_neg:
            constraints.append(vx >= 0)
        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=False)
        res = [round(x) for x in vx.value]
        debug_frags_out.append(res)

        cnt = 0
        for e in res:
            if e > 0:
                cnt += 1
        if cnt * 2 >= len(yy[i]):
            res = [1000000 for _ in range(len(res))]
            invalid += 1

        result = result + res

    scc_ratio = (num_frags - invalid) / num_frags
    # print("scc ratio %f" % scc_ratio)
    return result, debug_frags_out


def copy_A_and_y_frags(sketch, d_i, ratio, num_frags):
    m = int(math.ceil(sketch.w[d_i] / ratio))
    m1 = int(math.ceil(m / num_frags))
    nn = split_integer(sketch.w[d_i], num_frags)
    A_size = 0
    for n1 in nn:
        A_size += m1 * n1
    A_data = (c_int * A_size)()
    yy_data = (c_int * (m1 * num_frags))()

    lib.copy_A_and_y_frags(sketch.obj, d_i, A_data, yy_data)

    A_arr = np.array(A_data)
    yy_arr = np.array(yy_data)

    A_i = 0
    A = []
    for n1 in nn:
        A_frag = []
        for _ in range(m1):
            row = []
            for _ in range(n1):
                row.append(A_arr[A_i])
                A_i += 1
            A_frag.append(row)
        A.append(np.array(A_frag))
    yy_i = 0
    yy = []
    for _ in range(num_frags):
        yy_frag = []
        for __ in range(m1):
            yy_frag.append(yy_arr[yy_i])
            yy_i += 1
        yy.append(np.array(yy_frag))
    return A, yy


def compressive_sensing(sketch_type, read_num, d, w, seed, zero_thld, round_thld, ratio,
                        num_frags, k, num_threads, do_query=True, decompress_method='python',
                        func=None, tower_settings_id=-1, ignore_level=None, sbf_size=-1):
    sketch = Sketch(read_num, d, sketch_type)
    for i in range(d):
        sketch.init_arr(i, None, w[i], -1, seed + i)
    sketch.insert_dataset()

    ##### debug begin #####
    # debug_large_counters_ds = [[0 for j in range(w[i])] for i in range(d)]
    # debug_small_counters_ds = [[0 for j in range(w[i])] for i in range(d)]
    # for i in range(d):
    #     temp = (c_uint * w[i])()
    #     sketch.copy_array(i, temp)
    #     temp = np.array(temp)
    #     for j in range(len(temp)):
    #         if temp[j] < 4096:
    #             debug_small_counters_ds[i][j] = temp[j]
    #         else:
    #             debug_large_counters_ds[i][j] = temp[j]
    ##### debug end #####

    # Compress
    comp_secs, tower_comp_secs = sketch.compress(ratio, CompAlgoType.COMPSEN, zero_thld, round_thld,
                                                    num_frags, num_threads, tower_settings_id, sbf_size)

    # Do something with the compressed result
    if func:
        A_frags_ds, y_frags_ds = func(A_frags_ds, y_frags_ds, counters_ds)

    # Decompress and query
    decomp_secs, tower_decomp_secs = -1, -1
    ARE, AAE = -1, -1
    if decompress_method == 'cpp':
        decomp_secs = lib.cs_decompress(sketch.obj, num_threads)
        if do_query:
            ARE, AAE = sketch.query_dataset(k)

    elif decompress_method == 'python':
        sketch_copy = Sketch(read_num, d, sketch_type, is_compsen=True)
        tower_decomp_secs = 0
        for i in range(d):
            A_frags, y_frags = copy_A_and_y_frags(sketch, i, ratio, num_frags)
            res, debug_frags_out = cs_decompress_array(A_frags, y_frags, has_neg=(sketch_type == SketchType.Count))
            # res = debug_large_counters_ds[i]
            # for j, v in enumerate(debug_small_counters_ds[i]):
            #     res[j] += v

            # Recover small counters
            if tower_settings_id >= 0:
                small_data = (c_uint * w[i])()
                tower_decomp_secs += lib.copy_tower_small_counters(sketch.obj, i, small_data,
                                                                   ignore_level if ignore_level is not None else 0)
                small_counters = np.array(small_data)
                for j in range(len(res)):
                    res[j] += small_counters[j]

            out_data = (c_int * w[i])(*res) if sketch_type == SketchType.Count else (c_uint * w[i])(*res)
            sketch_copy.init_arr(i, out_data, w[i], -1, seed+i)
        if do_query:
            ignore_max = 256 if ignore_level is not None else 0
            ARE, AAE = sketch_copy.query_dataset(k, ignore_max)

    ret = {'ARE': ARE, 'AAE': AAE, 'comp_secs': comp_secs, 'decomp_secs': decomp_secs,
           'tower_comp_secs': tower_comp_secs, 'tower_decomp_secs': tower_decomp_secs}
    return ret


def compressive_sensing_comp_simd(read_num, d, w, seed, ratio):
    sketch = Sketch(read_num, d, SketchType.CM)
    for i in range(d):
        sketch.init_arr(i, None, w[i], -1, seed + i)
    sketch.insert_dataset()

    total_comp_secs = sketch.compress_4096_simd(ratio)

    ret = {'total_comp_secs': total_comp_secs}
    return ret


def hokusai(sketch_type, read_num, d, w, seed, ratio, k):
    sketch = Sketch(read_num, d, sketch_type)
    for i in range(d):
        sketch.init_arr(i, None, w[i], -1, seed+i)
    sketch.insert_dataset()

    secs_used = sketch.compress(ratio, CompAlgoType.HOKUSAI)[0]
    ARE, AAE = sketch.query_dataset(k)

    ret = {'ARE': ARE, 'AAE': AAE, 'comp_secs': secs_used}
    return ret


def elastic(sketch_type, read_num, d, w, seed, ratio, k):
    assert sketch_type != SketchType.Count
    sketch = Sketch(read_num, d, sketch_type)
    for i in range(d):
        sketch.init_arr(i, None, w[i], -1, seed+i)
    sketch.insert_dataset()

    secs_used = sketch.compress(ratio, CompAlgoType.ELASTIC)[0]
    ARE, AAE = sketch.query_dataset(k)

    ret = {'ARE': ARE, 'AAE': AAE, 'comp_secs': secs_used}
    return ret


def cluster_reduce(sketch_type, read_num, d, w, ratio, k, method_id, debug=False):
    secs_used = c_double()
    ARE = crlib.run_test(read_num, d, sum(w), ratio, k, int(sketch_type), secs_used, method_id, debug)  # w是list
    ret = {'ARE': ARE, 'comp_secs': secs_used.value}
    return ret


def no_compress(sketch_type, read_num, d, w, seed, k):
    sketch = Sketch(read_num, d, sketch_type)
    for i in range(d):
        sketch.init_arr(i, None, w[i], -1, seed+i)
    sketch.insert_dataset()
    ARE, AAE = sketch.query_dataset(k)
    ret = {'ARE': ARE, 'AAE': AAE}
    return ret


def distributed_data_stream(sketch_type, w, k, ratio, algo, cr_method_id, tower_settings_id, zero_thld):
    if algo == CompAlgoType.CLUSRED:
        c_AAE = c_double()
        ARE = crlib.distributed_data_stream(sum(w), k, ratio, c_AAE, cr_method_id)
        AAE = c_AAE.value
    else:
        slice_idxs = [0, 3390215, 6780430, 10170645, 13560860, 16951075, 20341290, 23731505, -1]
        sketches = []
        query_result = np.zeros(k, dtype=int)

        for sk_i in range(8):
            print("Sketches {} / {}".format(sk_i, 8))
            sketch = Sketch(-1, 3, sketch_type)
            for i in range(3):
                sketch.init_arr(i, None, w[i], -1, 997+i)
            sketch.insert_dataset(slice_idxs[sk_i], slice_idxs[sk_i+1])
            ##### debug begin #####
            debug_large_counters_ds = [[0 for j in range(w[i])] for i in range(3)]
            debug_small_counters_ds = [[0 for j in range(w[i])] for i in range(3)]
            for i in range(3):
                temp = (c_uint * w[i])()
                sketch.copy_array(i, temp)
                temp = np.array(temp)
                for j in range(len(temp)):
                    if temp[j] < zero_thld:
                        debug_small_counters_ds[i][j] = temp[j]
                    else:
                        debug_large_counters_ds[i][j] = temp[j]
            ##### debug end #####

            # Compress one sketch
            sketch.compress(ratio, algo, zero_thld, 1, w[i]//250, tower_settings_id=tower_settings_id)
            sketches.append(sketch)

            # Decompress and query one sketch
            if algo == CompAlgoType.COMPSEN:
                sketch_copy = Sketch(-1, 3, sketch_type, is_compsen=True)
                for i in range(3):
                    A_frags, y_frags = copy_A_and_y_frags(sketches[sk_i], i, ratio, w[i]//250)
                    # res, debug_frags_out = cs_decompress_array(
                    #     A_frags, y_frags, has_neg=(sketch_type == SketchType.Count))
                    res = debug_large_counters_ds[i]
                    # for j, v in enumerate(debug_small_counters_ds[i]):
                    #     res[j] += v

                    if tower_settings_id >= 0:
                        small_data = (c_uint * w[i])()
                        lib.copy_tower_small_counters(sketch.obj, i, small_data, 0)
                        small_counters = np.array(small_data)
                        for j in range(len(res)):
                            res[j] += small_counters[j]

                    out_data = (c_int * w[i])(*res) if sketch_type == SketchType.Count else (c_uint * w[i])(*res)
                    sketch_copy.init_arr(i, out_data, w[i], -1, 997+i)
                query_data = (c_int * k)()
                lib.query_and_copy(sketch_copy.obj, k, query_data)
            elif algo == CompAlgoType.HOKUSAI or algo == CompAlgoType.ELASTIC:
                query_data = (c_int * k)()
                lib.query_and_copy(sketches[sk_i].obj, k, query_data)
            query_result += np.array(query_data)

        c_AAE = c_double()
        ARE = lib.calc_acc(k, (c_int * k)(*query_result), c_AAE)
        AAE = c_AAE.value

    ret = {'ARE': ARE, 'AAE': AAE}
    return ret


# Read shared libraries and initialize functions
def init(args):
    np.random.seed(0)

    global lib
    lib = cdll.LoadLibrary('./lib/{}Sketch.so'.format(str(args.sketch)))
    getattr(lib, '{}Sketch_new'.format(str(args.sketch))).restype = c_void_p
    getattr(lib, '{}Sketch_new'.format(str(args.sketch))).argtypes = [c_int32, c_int32, c_bool]
    getattr(lib, '{}Sketch_delete'.format(str(args.sketch))).restype = None
    getattr(lib, '{}Sketch_delete'.format(str(args.sketch))).argtypes = [c_void_p]

    zero_thld_modified = False
    if args.sketch == SketchType.CML:
        log_base = 1.00026
        args.zero_thld = int(math.log(1 - args.zero_thld * (1 - log_base), log_base))
        zero_thld_modified = True
    elif args.sketch == SketchType.CSM:
        args.zero_thld = int(args.zero_thld / args.d)
        zero_thld_modified = True
    if zero_thld_modified:
        print("zero_thld (separating threshold) is modified to {} due to probabilistically insertion".format(args.zero_thld))
    print("==============================================================")
    print("\033[1;32mTest case {} starting...\033[0m".format(args.test))

    lib.init_arr.restype = None
    lib.init_arr.argtypes = [c_void_p, c_int32, c_void_p, c_int32, c_int32, c_uint32]
    lib.insert_dataset.restype = None
    lib.insert_dataset.argtypes = [c_void_p, c_int, c_int]
    lib.query_dataset.restype = c_double
    lib.query_dataset.argtypes = [c_void_p, c_int32, POINTER(c_double), c_int]
    lib.copy_array.restype = None
    lib.copy_array.argtypes = [c_void_p, c_int32, c_void_p]
    lib.compress.restype = c_double
    lib.compress.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, POINTER(c_double)]
    lib.cs_decompress.restype = c_double
    lib.cs_decompress.argtypes = [c_void_p, c_int]
    lib.copy_A_and_y_frags.restype = None
    lib.copy_A_and_y_frags.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_int)]
    lib.copy_tower_small_counters.restype = c_double
    lib.copy_tower_small_counters.argtypes = [c_void_p, c_int, c_void_p, c_int]
    lib.query_and_copy.restype = None
    lib.query_and_copy.argtypes = [c_void_p, c_int, POINTER(c_int)]
    lib.calc_acc.restype = c_double
    lib.calc_acc.argtypes = [c_int, POINTER(c_int), POINTER(c_double)]
    lib.set_debug_level(args.debug)

    global crlib
    crlib = cdll.LoadLibrary('./lib/ClusterReduceWrapper.so')
    crlib.run_test.restype = c_double
    crlib.run_test.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, POINTER(c_double), c_int, c_bool]
    crlib.distributed_data_stream.restype = c_double
    crlib.distributed_data_stream.argtypes = [c_int, c_int, c_int, POINTER(c_double), c_int]
