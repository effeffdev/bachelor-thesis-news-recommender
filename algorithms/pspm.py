import numpy as np
import pandas as pd
from math import isnan
from prefixspan import PrefixSpan
from typing import List

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

        print(self.items)

        scores = self.items_to_scores.get(str(self.items))
        if scores is None:
            self.competence = []

            self.last = len(self.items)
            self.bcs_sigma = (self.last - 1) / (2 * np.sqrt(2 * np.log(2)))
            if self.bcs_sigma == 0:
                self.bcs_sigma = 0.1
            self.bcs_a = 1 / (self.bcs_sigma * np.sqrt(2 * np.pi))
            total_bcs_weight = sum([self.bcs_weight(i + 1) for i, x in enumerate(self.items)])

            relevant_sessions = [self.db[i] for i in self.relevant_sessions_indices]

            for session in relevant_sessions:
                lcs = self.find_longest_common_subsequence(session, self.items)
                bcs = sum([self.bcs_weight(x) for x in lcs]) / total_bcs_weight

                # TODO the lcs_last index might not be correct all the time!
                fes_last = len(session)
                self.lcs_last = fes_last - session[::-1].index(self.items[lcs[-1] - 1])
                self.fes_sigma = (fes_last - self.lcs_last) / (2 * np.sqrt(2 * np.log(2)))
                if self.fes_sigma == 0:
                    self.fes_sigma = 0.1
                self.fes_a = 1 / (self.fes_sigma * np.sqrt(2 * np.pi))
                cni = session[self.lcs_last:]
                unique_cni = set(cni)
                fes = sum([self.fes_weight(cni.index(x) + 1) for x in unique_cni]) / len(self.items)

                # TODO: why can there be invalid value in double_scalars warning???
                if (bcs == 0 and fes == 0) or isnan(bcs) or isnan(fes):
                    print('Gotcha!')
                self.competence.append((bcs * fes) / (1/2 * (bcs + fes)))

            # mine patterns
            self.total_weight = sum(self.competence)

            ps = PrefixSpan(relevant_sessions)

            patterns = ps.frequent(self.delta, key=self.pattern_key, bound=self.pattern_key)

            scores = self.score_items(patterns)

            self.items_to_scores.update({str(self.items): scores})

        predictions = np.zeros(len(items_to_score))
        mask = np.isin(items_to_score, list(scores.keys()))

        scored_items = items_to_score[mask]
        values = [scores[x] for x in scored_items]
        predictions[mask] = values
        return pd.Series(data=predictions, index=items_to_score)

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

    # TODO bottleneck! needs to be more efficient in time and space (=> MemoryError for 11 item session)
    def find_longest_common_subsequence(self, test_seq, user_seq):
        if len(test_seq) < len(user_seq):
            seq1 = user_seq
            seq2 = test_seq
            is_user_short = False
        else:
            seq1 = test_seq
            seq2 = user_seq
            is_user_short = True

        subseq_last_row = [{''}] * (len(seq2) + 1)
        subseq_curr_row = [{''}] + [set()] * (len(seq2))

        for row in range(1, len(seq1) + 1):
            for col in range(1, len(seq2) + 1):
                if seq1[row-1] == seq2[col-1]:
                    diagonal_cell_value = subseq_last_row[col-1]
                    position = col if is_user_short else row
                    new_cell_value = self.add_matched_element(position, diagonal_cell_value, ',')
                else:
                    above_set = subseq_last_row[col]
                    left_set = subseq_curr_row[col-1]
                    new_cell_value = above_set.union(left_set)

                subseq_curr_row[col] = new_cell_value

            subseq_last_row = subseq_curr_row
            subseq_curr_row = [{''}] + [set()] * len(seq2)

        subsequences = list(map(lambda x: [] if x == '' else x.split(','), subseq_last_row[len(seq2)]))
        return list(map(int, max(subsequences, key=len)))

    def add_matched_element(self, element, target_set, seperator):
        new_elements = map(lambda x: x + seperator + str(element) if x != '' else str(element), target_set)
        return target_set.union(new_elements)
