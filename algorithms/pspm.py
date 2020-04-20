import numpy as np
import pandas as pd
from math import isnan
from prefixspan import PrefixSpan
from typing import List, Set
import py_common_subseq as cs


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


def get_longest_common_subsequence(test_seq, user_seq):
    subsequences = cs.find_common_subsequences(list(map(str, test_seq)), list(map(str, user_seq)), sep=',')

    if len(subsequences) == 1:
        return []

    lcs = list(map(int, max(subsequences, key=len).split(',')[1:]))
    return lcs


class PSPM:
    def __init__(self, delta: float = 0.25):
        self.delta = delta

        self.db = []
        self.item_to_sessions = dict()
        self.session = -1
        self.items = list()
        self.competence = []
        self.last = 0
        self.bcs_sigma = 0
        self.bcs_a = 0
        self.relevant_sessions_indices = set()
        self.items_to_scores = dict()

    def train(self, training_data: pd.DataFrame):
        session_loc = training_data.columns.get_loc('Session')
        item_loc = training_data.columns.get_loc('SID_Content')
        last_session = -1
        sequence = []
        counter = 0

        for row in training_data.itertuples(index=False):
            item = row[item_loc]
            session = row[session_loc]

            if session != last_session:
                if last_session != -1:
                    self.db.append(sequence)
                    sequence = []
                    counter += 1

                last_session = session

            sequence.append(item)

            sessions_for_item = self.item_to_sessions.get(item)
            if sessions_for_item is None:
                sessions_for_item = set()
                self.item_to_sessions.update({item: sessions_for_item})

            sessions_for_item.add(counter)

        # add last session
        self.db.append(sequence)

    def predict(self, curr_session_id: int, last_item_id: int, items_to_score: List[int]) -> pd.Series:
        if self.session != curr_session_id:
            self.session = curr_session_id
            self.items = list()
            self.relevant_sessions_indices = set()

        self.items.append(last_item_id)
        self.relevant_sessions_indices = self.relevant_sessions_indices.union(self.item_to_sessions.get(last_item_id))

        return self.score_session(self.items, items_to_score, self.relevant_sessions_indices)

    def score_session(self, items: List[int], items_to_score: List[int], relevant_sessions_indices: Set[int]):
        scores = self.items_to_scores.get(str(items))
        if scores is None:
            self.competence = []

            self.last = len(items)
            self.bcs_sigma = (self.last - 1) / (2 * np.sqrt(2 * np.log(2)))
            if self.bcs_sigma == 0:
                self.bcs_sigma = 0.1
            self.bcs_a = 1 / (self.bcs_sigma * np.sqrt(2 * np.pi))
            total_bcs_weight = sum([self.bcs_weight(i + 1) for i, x in enumerate(items)])

            relevant_sessions = [self.db[i] for i in relevant_sessions_indices]

            for session in relevant_sessions:
                lcs = get_longest_common_subsequence(session, items)
                lcs_indices = get_indices(items, lcs)

                bcs = sum([self.bcs_weight(x) for x in lcs_indices]) / total_bcs_weight

                fes_last = len(session)
                self.lcs_last = get_indices(session, lcs)[-1]
                self.fes_sigma = (fes_last - self.lcs_last) / (2 * np.sqrt(2 * np.log(2)))
                if self.fes_sigma == 0:
                    self.fes_sigma = 0.1
                self.fes_a = 1 / (self.fes_sigma * np.sqrt(2 * np.pi))
                cni = session[self.lcs_last:]
                unique_cni = set(cni)
                fes = sum([self.fes_weight(cni.index(x) + 1) for x in unique_cni]) / len(items)

                self.competence.append(0 if bcs == 0 or fes == 0 else (bcs * fes) / (1 / 2 * (bcs + fes)))

            # mine patterns
            self.total_weight = sum(self.competence)

            ps = PrefixSpan(relevant_sessions)

            patterns = ps.frequent(self.delta, key=self.pattern_key, bound=self.pattern_key)

            scores = self.score_items(patterns)

            self.items_to_scores.update({str(items): scores})
        predictions = np.zeros(len(items_to_score))
        mask = np.isin(items_to_score, list(scores.keys()))
        scored_items = items_to_score[mask]
        values = [scores[x] for x in scored_items]
        predictions[mask] = values
        return pd.Series(data=predictions, index=items_to_score)

    def predict_session(self, session_items: List[int], items_to_score: List[int]) -> pd.Series:
        relevant_sessions_indices = set.union(*[self.item_to_sessions.get(item_id) for item_id in session_items])

        return self.score_session(session_items, items_to_score, relevant_sessions_indices)

    def score_items(self, patterns):
        scores = dict()

        for pattern in patterns:
            support, items = pattern

            for item in items:
                old_score = scores.get(item)

                if old_score is None:
                    scores.update({item: support})
                else:
                    new_score = old_score + support
                    scores.update({item: new_score})

        return scores

    def pattern_key(self, pattern, matches):
        if self.total_weight == 0:
            return 0
        else:
            return sum([self.competence[i] for i, _ in matches]) / self.total_weight

    def bcs_weight(self, current):
        return self.bcs_a * np.exp(- ((current - self.last) ** 2 / 2 * self.bcs_sigma ** 2))

    def fes_weight(self, current):
        return self.fes_a * np.exp(- ((current - self.lcs_last) ** 2 / 2 * self.fes_sigma ** 2))
