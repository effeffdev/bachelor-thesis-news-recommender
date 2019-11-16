import numpy as np
import pandas as pd

class Metrics:
    """
    Metrics(length=20)

    Calculates MRR (Mean Reciprocal Rank) and HR (Hit Rate) considering the top N results

    Parameters
    --------
    length: int
        N considered results
    """

    def __init__(self, length: int = 20):
        self.length = length

        self.num = 0
        self.pos = 0
        self.hit = 0

    def reset(self):
        self.num = 0
        self.pos = 0
        self.hit = 0

    def add(self, result: pd.Series, item: int):
        """
        Update the metrics for the given result set and item.

        :param result:
        :param item:
        """
        self.num += 1

        res = result[:self.length]

        if item in res.index:
            self.hit += 1
            rank = res.index.get_loc(item) + 1
            self.pos += (1.0 / rank)

    def result(self):
        return [
            ("HR@" + str(self.length), (self.hit/self.num)),
            ("MRR@" + str(self.length), (self.pos/self.num))
        ]
