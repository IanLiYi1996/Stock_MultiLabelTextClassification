import pandas as pd
import numpy as np
import math

def calc_dcg(sorted_vec, at):
    '''
    sorted_vec: list[tuple], type of element is tuple,
    tuple(v0, v1), v0: predict score; v1: label score
    at: int, calculate dcg@at
    '''
    ranking = [t[1] for t in sorted_vec[0: at]]
    dcg_ = sum([(2**r - 1) / math.log(i + 2, 2) for i, r in enumerate(ranking)])
    return dcg_

def calc_ndcg(vec, at):
    '''
    vec: list[tuple], type of element is tuple,
    tuple(v0, v1), v0: predict score; v1: label score
    at: int, calculate ndcg@at
    '''
    sorted_vec = sorted(vec, key=lambda t: t[1], reverse=True)
    ideal_dcg = calc_dcg(sorted_vec, at)
    sorted_vec = sorted(vec, key=lambda t: t[0], reverse=True)
    cur_dcg = calc_dcg(sorted_vec, at)
    if ideal_dcg == 0:
        return 0
    else:
        return cur_dcg / ideal_dcg

if __name__ == "__main__":
    p = pd.read_csv('/bigd/liyi/longtest.csv', usecols=['score'])
    score_p = np.array(p).T
    score_p = score_p.tolist()[0]

    s = pd.read_csv('/bigd/liyi/result_longtetext.csv', usecols=['腾讯控股'])
    score_s = np.array(s).T
    score_s = score_s.tolist()[0]
    assert(len(score_p)==len(score_s))

    pack = zip(score_s, score_p)
    pack = list(pack)
    len(pack)