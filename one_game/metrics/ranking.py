# coding=utf-8

import numpy as np
import ctypes
import os

def dcg_score(element_list):

    score = 0.0
    for order, rel in enumerate(element_list):
        score += (np.power(2.0, rel) - 1.0) / np.log(order + 2)
    return score


def ndcg_score(reference, hypothesis):
    return dcg_score(hypothesis)/dcg_score(reference)


def load_c_ndcg_score():

    lib_name = "libranking.dll" if os.name == "nt" else "libranking.so"
    lib_path = os.path.join(os.path.dirname(__file__), lib_name)
    lib = ctypes.CDLL(lib_path)
    c_ndcg_score = lib.ndcg_score
    c_ndcg_score.restype = ctypes.c_double

    def ndcg_score(reference, hypothesis):

        reference = list(reference)
        hypothesis = list(hypothesis)

        num_rels = len(reference)
        rel_type = ctypes.c_int * num_rels

        return c_ndcg_score(
            rel_type(*reference),
            rel_type(*hypothesis),
            ctypes.c_int(num_rels)
        )
    return ndcg_score

try:
    ndcg_score = load_c_ndcg_score()
except:
    print("cannot load c_ndcg_score, use Python version instead")
    pass



def precision_score(reference, hypothesis):
    result = np.sum(hypothesis, dtype=np.float32)/len(hypothesis)
    return result


def recall_score(reference, hypothesis):
    result = np.sum(hypothesis, dtype=np.float32) / len(reference)
    return result
