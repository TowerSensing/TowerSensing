/*
 * Count-Mean-Min Sketch (CMM)
 *
 * Deng F, Rafiei D. New estimation algorithms for streaming data: Count-min
 * can do more[J]. Webdocs. Cs. Ualberta. Ca, 2007.
 */

#ifndef _CMMSKETCH_H_
#define _CMMSKETCH_H_

#include "sketch_common.h"
#include "CMSketch.h"

class CMMSketch : public CounterMatrix<CMArray> {
  public:
    CMMSketch(int read_num, int d_) : CounterMatrix<CMArray>(read_num, d_) {
        sketch_name = "CMM Sketch";
    }

    void insert(pair<string, float> pkt) {
        for (int i = 0; i < d; ++i) arr[i].insert(pkt);
    }

    int query(string flow) {
        int values[d];
        for (int i = 0; i < d; ++i) {
            uint32_t value = arr[i].query(flow);
            int noise = (stream_size - value) / (arr[i].w - 1);
            values[i] = (int)value - noise;
        }
        sort(values, values + d);
        int ret = d % 2 == 0 ? ((values[d / 2 - 1] + values[d / 2]) / 2)
                             : values[d / 2];
        return max(0, ret);
    }
};

#endif
