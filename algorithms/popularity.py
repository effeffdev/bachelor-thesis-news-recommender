import numpy as np
import pandas as pd
from typing import List


class Popularity:
    """
    Popularity(keep_n=100)

    Predictor that scores items based on their popularity in the training set.

    Parameters
    --------
    keep_n : int
        Keep top scored N items. All other items get score 0.
    """

    def __init__(self, keep_n: int = 100):
        self.keep_n = keep_n
        self.popularity_list = pd.Series([])

    def train(self, training_data: pd.DataFrame):
        """
        Trains the Predictor with the given training data

        :param training_data: The data to train the predictor with
        """
        content_count = training_data.groupby('SID_Content').size()
        content_count = content_count / (content_count + 1)
        content_count.sort_values(ascending=False, inplace=True)
        self.popularity_list = content_count.head(self.keep_n)

    def predict(self, items_to_score: List[int]) -> pd.Series:
        """
        computes an item score for all given items

        :param items_to_score: the ids of all items to score
        :return: an item score for all given items
        """
        predictions = np.zeros(len(items_to_score))
        mask = np.isin(items_to_score, self.popularity_list.index)
        predictions[mask] = self.popularity_list[items_to_score[mask]]
        return pd.Series(data=predictions, index=items_to_score)
