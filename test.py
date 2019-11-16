import numpy as np
import pandas as pd
import py_common_subseq

from time import time


def list_rindex(list, item):
    for i in reversed(range(len(list))):
        if list[i] == item:
            return i


def get_indices(sequence, subsequence):
    indices = []
    for i in reversed(range(len(subsequence))):
        list_index = list_rindex(sequence, subsequence[i])
        indices.append(list_index + 1)
        sequence = sequence[:list_index]

    return list(reversed(indices))


start_time = time()

s1 = [1000000, 2000000, 3000000, 4000000, 4000000, 1000000, 2000000, 3000000, 4000000, 1000000, 2000000]
seq_1 = list(map(str, s1))
s2 = [3000000, 1000000, 2000000, 4000000, 4000000, 1000000, 3000000, 1000000, 2000000, 3000000, 1000000]
seq_2 = list(map(str, s2))

subsequences = py_common_subseq.find_common_subsequences(seq_1, seq_2, sep=',')

lcs = list(map(int, max(subsequences, key=len).split(',')[1:]))

print(lcs)
print(get_indices(s1, lcs))
print(get_indices(s2, lcs))

print(time() - start_time)
