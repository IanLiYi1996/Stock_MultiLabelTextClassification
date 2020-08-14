# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 10:33
# @Author  : supinyu
# @File    : stock_relevance_1.py

"""
股票相关性V1版本---基于BM25算法
"""

import os
import math
import json
from collections import defaultdict
import sys
sys.path.append('C:/Users/liyi_intern/Documents/Codes/nlp-stock-relevance/')
from core.config import DATA_DIR
from core.util.word_segment import analyse
from core.util.log import logger

base_dir = os.path.dirname(os.path.abspath(__file__))
idf_dict = json.load(
    open(os.path.join(DATA_DIR, "long_idf.json"), encoding='utf-8'))

query_expansion_dict = {}
fd = open(os.path.join(
    DATA_DIR, "new_stock_words.csv"), encoding='utf-8')

for line in fd:
    items = line.strip().split("%")
    query_expansion_dict[items[0]] = set(items[-1].strip().split("#"))

HITWORDS = defaultdict(list)
for k, v in query_expansion_dict.items():
    for vv in v:
        vv = vv.lower()
        HITWORDS[vv].append(k)

v_title = 3
v_content = 1
bs = 0.8
avsls_title = 12
avsls_content = 1772
k1 = 100
N = 5209236


def compute_b(text, avsl):
    b = 1 - bs + bs * len(text) / avsl
    return b


def compute_tf(words, q):
    tf = len([w for w in words if w.lower() == q.lower()])
    return tf


def score(title, content, stock):
    querys = query_expansion_dict.get(stock, [stock])
    title_words = [w.value for w in title]
    content_words = [w.value for w in content]
    b_title = compute_b(title_words, avsls_title)
    b_content = compute_b(content_words, avsls_content)
    s = 0
    for q in querys:
        tf_title = compute_tf(title_words, q)
        tf_content = compute_tf(content_words, q)
        tf = v_title * tf_title / b_title + v_content * tf_content / b_content
        n = idf_dict.get(stock, 0)
        w_idf = math.log((N - n + 0.5) / (n + 0.5))
        s += tf * w_idf / (k1 + tf)
    return s


def scores(title, content):
    """
    股票相关性V1主函数，算法使用bm25算法
    Args:
        title: str
        content: str

    Returns:

    """
    t = set(
        [x.value.lower() for x in title + content])
    meanfull_words = HITWORDS.keys() & t
    if meanfull_words:
        sd = {}
        stock_candidates = []
        for word in meanfull_words:
            stock_candidates.extend(HITWORDS[word])
        for stock in set(stock_candidates):
            s = score(title, content, stock)
            sd[stock] = s
        return sd
    return {}


def scores_v1(title, content):
    title_seg = analyse(title)
    content_seg = analyse(content)
    result_v1 = scores(title_seg, content_seg)
    # logger.info(result_v1)
    # result_score_map = {}
    # for item in result_v1:
    #     if result_v1[item] >= 1.2:
    #         result_score_map[item] = 4
    #     elif 1.0 <= result_v1[item] < 1.2:
    #         result_score_map[item] = 3
    #     elif 0.8 < result_v1[item] <= 1.0:
    #         result_score_map[item] = 2
    #     elif result_v1[item] <= 0.7:
    #         result_score_map[item] = 1
    return result_v1


if __name__ == "__main__":
    title = analyse("""成本持续下降+渠道快速扩张”，中顺洁柔悄然穿越牛熊""")
    content = analyse(
        """b站自己造的星$Facebook$$中顺洁柔(SZ002511)$中顺洁柔(sz002511)发布2019年一季报，符合预期，近期木浆价格下行弹性显现，盈利水平环比显著回暖。产品结构不断优化，渠道布局持续扩张，全国性产能布局持续进行，业绩高增长可期。广发轻工赵中平认为产品渠道双轮驱动构筑公司核心竞争力，高毛利产品占比提升调整产品结构，各生产基地产能投产计划打开增长瓶颈。公司未来渠道扩张优势及新品研发核心壁垒有望持续助力公司业绩增长""")
    t = scores(title, content)
    print(t)
    logger.info(t)
