#ifndef _TRACE_H_
#define _TRACE_H_

#include <vector>
#include <map>
#include <cstring>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;

struct {
    vector<pair<string, float>> dataset;
    vector<pair<int, string>> groundtruth;
} cache;

vector<pair<string, float>> loadCAIDA18(
    int read_num = -1,
    const char* filename = "./dataset/130000.dat") {
    FILE* pf = fopen(filename, "rb");
    if (!pf) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(-1);
    }

    vector<pair<string, float>> vec;
    double ftime = -1;
    char trace[30];
    int i = 0;
    while (fread(trace, 1, 21, pf)) {
        if (i++ == read_num) break;

        string tkey(trace, 4);  // SrcIP
        double ttime = *(double*)(trace + 13);
        if (ftime < 0) ftime = ttime;
        vec.push_back(pair<string, float>(tkey, ttime - ftime));
    }
    fclose(pf);
    return vec;
}

vector<pair<string, float>> loadCriteo(
    int read_num = -1,
    const char* filename = "./dataset/criteo_key.log") {
    vector<pair<string, float>> vec;

    ifstream logFile(filename);
    if (logFile.fail()) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(-1);
    }
    string str;

    while (getline(logFile, str)) {
        vec.push_back(pair<string, float>(str.substr(10), 1));
        str.clear();
    }

    logFile.close();
    return vec;
}

vector<pair<string, float>> loadZipf(
    int read_num = -1,
    const char* filename = "./dataset/003.dat") {
    int MAX_ITEM = INT32_MAX;
    ifstream inFile(filename, ios::binary);
    if (inFile.fail()) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(-1);
    }
    ios::sync_with_stdio(false);

    char key[13];
    vector<pair<string, float>> vec;
    // map<int, int> ip_count;
    for (int i = 0; i < MAX_ITEM; ++i) {
        inFile.read(key, 4);
        if (inFile.gcount() < 4) break;

        string str = string(key, 4);
        vec.push_back(pair<string, float>(string(key, 4), 1));
        // freq[str]++;
    }
    inFile.close();
    return vec;
}

vector<pair<int, string>> groundtruth(const vector<pair<string, float>>& input,
                                      int read_num = -1) {
    if (!cache.groundtruth.empty()) {
        return cache.groundtruth;
    }
    map<string, int> key2cnt;
    int i = 0;
    for (auto [tkey, ttime] : input) {
        ++key2cnt[tkey];
        if (++i == read_num) break;
    }
    vector<pair<int, string>> ans;
    for (auto flow : key2cnt) ans.push_back({-flow.second, flow.first});
    sort(ans.begin(), ans.end());
    printf("Dataset: %lu packets, %lu flows\n", input.size(), ans.size());

    cache.groundtruth = ans;
    return ans;
}

vector<pair<string, float>> load_dataset(int read_num = -1) {
    if (!cache.dataset.empty()) return cache.dataset;

    auto dataset = loadCAIDA18(read_num);
    cache.dataset = dataset;
    return dataset;
}

#endif  // _TRACE_H_
