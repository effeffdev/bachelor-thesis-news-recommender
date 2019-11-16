from typing import List

import numpy as np
import pandas as pd
from math import sqrt


class SSKNN:
    """
    SSKNN(k=100, sample_size=500)

    Parameters
    --------
    k: int
        Use k nearest neighbors to calculate the score for each item.
    sample_size: int
        Length of the sample used for calculating the nearest neighbors.
    """

    def __init__(self, k: int = 100, sample_size: int = 500):
        self.k = k
        self.sample_size = sample_size

        self.session = -1
        self.items = []
        self.considered_sessions = set()
        self.session_to_items = dict()
        self.item_to_sessions = dict()

    def train(self, training_data: pd.DataFrame):
        """
        Trains the Predictor with the given training data

        :param training_data: The data to train the predictor with
        """
        session_pos = training_data.columns.get_loc('Session')
        item_pos = training_data.columns.get_loc('SID_Content')

        session = -1
        items_in_session = set()

        for row in training_data.itertuples(index=False):
            if row[session_pos] != session:
                if len(items_in_session) > 0:
                    self.session_to_items.update({session: items_in_session})

                session = row[session_pos]
                items_in_session = set()

            items_in_session.add(row[item_pos])

            sessions_for_item = self.item_to_sessions.get(row[item_pos])
            if sessions_for_item is None:
                sessions_for_item = set()
                self.item_to_sessions.update({row[item_pos]: sessions_for_item})

            sessions_for_item.add(row[session_pos])

        self.session_to_items.update({session: items_in_session})

    def predict(self, curr_session_id: int, last_item_id: int, items_to_score: List[int]) -> pd.Series:
        """
        Calculate prediction scores for each given item.

        :param curr_session_id: id of the current session
        :param last_item_id: id of the current item
        :param items_to_score: the ids of all items to score
        :return: an item score for all given items
        """

        if self.session != curr_session_id:
            self.session = curr_session_id
            self.items = list()
            self.considered_sessions = set()

        self.items.append(last_item_id)

        neighbors_with_similarity = self.get_neighbors(set(self.items), last_item_id)
        scores = self.score_items(neighbors_with_similarity, self.items)

        predictions = np.zeros(len(items_to_score))
        mask = np.isin(items_to_score, list(scores.keys()))

        scored_items = items_to_score[mask]
        values = [scores[x] for x in scored_items]
        predictions[mask] = values
        return pd.Series(data=predictions, index=items_to_score)

    def get_neighbors(self, items, last_item_id):
        # get neighbor candidates
        sessions_for_item = self.item_to_sessions.get(last_item_id)
        if sessions_for_item is None:
            sessions_for_item = set()

        self.considered_sessions = self.considered_sessions | sessions_for_item

        neighbor_candidates = set()
        if self.sample_size == 0:
            neighbor_candidates = self.considered_sessions

        else:
            if len(self.considered_sessions) > self.sample_size:
                sorted_sessions = sorted(self.considered_sessions, reverse=True)
                cnt = 0
                for session in sorted_sessions:
                    cnt = cnt + 1
                    if cnt > self.sample_size:
                        break
                    neighbor_candidates.add(session)

            else:
                neighbor_candidates = self.considered_sessions

        # calculate similarity
        neighbors_with_similarity = []
        cnt = 0
        for neighbor in neighbor_candidates:
            cnt = cnt + 1
            neighbor_items = self.session_to_items.get(neighbor)

            sim = self.cosine(neighbor_items, items)
            if sim > 0:
                neighbors_with_similarity.append((neighbor, sim))

        sorted_neighbors = sorted(neighbors_with_similarity, reverse=True, key=lambda x: x[1])

        return sorted_neighbors[:self.k]

    def score_items(self, neighbors_with_similarity, items):
        scores = dict()

        for session_with_similarity in neighbors_with_similarity:
            session_items = self.session_to_items.get(session_with_similarity[0])
            step = 1

            for item in reversed(items):
                if item in session_items:
                    decay = 1/step
                    break
                step += 1

            for item in session_items:
                old_score = scores.get(item)
                sim = session_with_similarity[1]

                if old_score is None:
                    scores.update({item: (sim * decay)})
                else:
                    new_score = old_score + (sim * decay)
                    scores.update(({item: new_score}))

        return scores

    @staticmethod
    def cosine(first, second):
        li = len(first & second)
        la = len(first)
        lb = len(second)
        return li / sqrt(la) * sqrt(lb)
