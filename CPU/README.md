# TowerSensing on CPU

## Overview

We implement and test TowerSensing (including *TowerEncoding*, *SketchSensing* and *ShiftEncoder*) on a CPU platform (Intel i9-10980XE, 18-core 4.2 GHz CPU with 128GB 3200 MHz DDR4 memory and 24.75 MB L3 cache). We implement three existing
sketch compression algorithms: Cluster-Reduce [1], Hokusai [2],
and Elastic [3]. We apply TowerSensing and existing algorithms
on six sketches: CM [4], CU [5], Count [6], CMM [7], CML [8],
and CSM [9].

Most of the key codes (sketch data structures, insertion and query procedures, procedures of *TowerEncoding*, *SketchSensing* and *ShiftEncoder*) are implemented with C++. We wrap the main C++ codes using Python for ease of testing.   Experimental setup and test cases are written in Python for convenience.

## Dependencies

* g++ 7.5.0 (Ubuntu 7.5.0-6ubuntu2)

* [Murmur Hash](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp)

* Python 3.8.10
  * NumPy 1.22.0
  * Pandas 1.3.5
  * tenseal 0.3.11

* [Optional] AVX-256 [Intel Intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) (SIMD operations)

## File structures

`trace.h`: Dataset reader.

`sketch_common.h`: Common data structures and algorithms for different kinds of sketches. `CounterArray` is a template base class that defines a single array with inserting and querying methods. `CounterMatrix` is a template base class containing a pointer to an array of several `CounterArray`s, and the inserting, querying and compressing methods for them. `CounterMatrix` is a complete sketch.

`CMSketch.h`, `CountSketch.h`, `CUSketch.h`, `CMMSketch.h`, `CMLSketch.h`, `CSMSketch.h`: Codes for each kind of sketches, including the methods of sketch insertion and query. Classes in these source files inherit from template specializations of `CounterArray` and `CounterMatrix`.

`api.cpp`: Code for generating shared libraries for different sketches (see `Makefile`). It exposes C interfaces for Python [ctypes](https://docs.python.org/3/library/ctypes.html).

`Makefile`: Commands to compile shared libraries for different sketches.

`valid_query_rate.cpp`: A test case for approximate recovery.

`sketch_compress.py`: Encapsulates calls to functions in the shared libraries. It also includes some commonly-used procedures for test cases to call.

`tests.py`: Test cases.

## Datasets

Datasets used in CPU experiments can be downloaded from the links below. After downloading, modify the corresponding paths in `trace.h`.

* CAIDA18: [130000.dat](https://drive.google.com/file/d/1iLe7QJwzj7FQRy8iEiRjbnCW-uuuJ27m/view?usp=sharing)
* Zipf: [003.dat](https://drive.google.com/file/d/1BWVzXKc11mwvSSrHuYurXtsgPGbJBe_A/view?usp=sharing)


**Notification:** These data files are only used for testing the performance of TowerSensing and the related algorithms in this project. Commercial purposes are fully excluded. Please do not use these datasets for other purpose. 


## How to run

### Build

First, download the datasets from the above links, and put them in the `dataset` folder. 
Modify the `filename` in `trace.h` to specify which dataset to be used.

Then, build the shared libraries with the following command. 

```bash
make all [USER_DEFINES=-DSIMD_COMPRESS]
```

Shared libraries (.so files) are generated in `lib/`. Use `-j` to build in parallel. 
You can decide whether to use SIMD by whether or not defining the `SIMD_COMPRESS` macro when calling the make command.


### Test cases

Below we show some examples of running tests on CM Sketches. 

Each test case is defined with a function starting with "test_" in `tests.py`. Test results will be output to `results/` directory in csv format. The generic command to run a test is shown below:

```bash
python3 tests.py --test {test name} --sketch {sketch type} --k {top-k} --d {number of arrays} --w {number of counters in each array} --sep_thld {separating threshold} --round_thld {rounding parameter}
```

You can run the follwoing command to see the specific definition of each parameter: 

```bash
python3 tests.py -h
```

Note that not all parameters are valid in every test. For details, please refer to each test case definition.


* Top-k Accuracy vs. Compression Ratio

  ```bash
  python3 tests.py --test test_acc_mem --sketch CM --k 500
  ```

* Full Accuracy vs. Tower Shape

  ```bash
  python3 tests.py --test test_fullacc_towershape --sketch CM
  ```

* Top-k Accuracy vs. Separating Threshold

  ```bash
  python3 tests.py --test test_acc_separating --sketch CM --k 500
  ```

* Full Accuracy vs. Separating Threshold

  ```bash
  python3 tests.py --test test_fullacc_separating --sketch CM
  ```

* Top-k Accuracy vs. Rounding Parameter

  ```bash
  python3 tests.py --test test_acc_rounding --sketch CM
  ```

* Flexibility of TowerEncoding

  ```bash
  python3 tests.py --test test_acc_ignore_level --sketch CM
  ```

* Efficiency of TowerEncoding

  ```bash
  python3 tests.py --test test_time_towerencoding --sketch CM
  ```

* Efficiency of SketchSensing

  ```bash
  python3 tests.py --test test_time_sketchsensing --sketch CM
  ```

* Top-k Accuracy (Comparison with Prior Art)

  ```bash
  python3 tests.py --test test_acc_algos --sketch CM
  ```

* Top-k Compression Efficiency (Comparison with Prior Art)

  ```bash
  python3 tests.py --test test_time_algos --sketch CM
  ```


## References

[1] Yikai Zhao, Zheng Zhong, Yuanpeng Li, Yi Zhou, Yifan Zhu, Li Chen, YiWang, and Tong Yang. Cluster-reduce: Compressing sketches for distributed data streams. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 2316–2326, 2021.

[2] Sergiy Matusevych, Alex Smola, and Amr Ahmed. Hokusai-sketching streams in real time. arXiv preprint arXiv:1210.4891, 2012.

[3] Tong Yang, Jie Jiang, Peng Liu, Qun Huang, Junzhi Gong, Yang Zhou, Rui Miao, Xiaoming Li, and Steve Uhlig. Elastic sketch: Adaptive and fast network-wide measurements. In Proceedings of the 2018 Conference of the ACM Special Interest Group on Data Communication (SIGCOMM), pages 561–575, 2018.

[4] Graham Cormode and S Muthukrishnan. An improved data stream summary: the count-min sketch and its applications. Journal of Algorithms, 2005.

[5] Cristian Estan and George Varghese. New directions in traffic measurement and accounting. ACM SIGMCOMM CCR, 2002.

[6] Moses Charikar, Kevin Chen, and Martin Farach-Colton. Finding frequent items in data streams. In Automata, Languages and Programming. 2002.

[7] Fan Deng and Davood Rafiei. New estimation algorithms for streaming data: Count-min can do more. Webdocs. Cs. Ualberta. Ca, 2007.

[8] Guillaume Pitel and Geoffroy Fouquier. Count-min-log sketch: Approximately counting with approximate counters. arXiv preprint arXiv:1502.04885, 2015.

[9] Tao Li, Shigang Chen, and Yibei Ling. Per-flow traffic measurement through randomized counter sharing. IEEE/ACM Transactions on Networking, 20(5):1622–1634, 2012.

