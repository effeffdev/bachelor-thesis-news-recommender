import os
import random
import re
from time import time
from typing import List

import math
import numpy as np
import pandas as pd


class Fossil:
    """

    """

    def __init__(self, k=100, alpha=0.2, epochs=10):
        self.k = k
        self.alpha = alpha
        self.epochs = epochs

        self.order = 1
        self.training_data = None
        self.validation_data = None
        self.item_num = 0
        self.session_num = 0
        self.learning_rate = 0.05
        self.annealing_rate = 1.
        self.reg = 0.0025
        self.current_session = None

    def train(self, training_data: pd.DataFrame, max_time: int = np.inf, model_dir: str = 'models/',
              load_model: str = None):
        """

        :param load_model:
        :param training_data:
        :param max_time:
        :param model_dir:
        :return:
        """

        max_iterations = (len(training_data) - training_data['Session'].nunique()) * self.epochs
        min_iterations = len(training_data) - training_data['Session'].nunique()
        progress = len(training_data) - training_data['Session'].nunique()

        # prepare model
        self.training_data = training_data
        self.validation_data = self.training_data[
            np.isin(training_data['Session'], training_data['Session'].unique()[-1000:])]
        self.item_num = training_data['SID_Content'].nunique()
        self.session_num = training_data['Session'].nunique()

        # train
        self.sessions = np.zeros((self.session_num, 2), dtype=np.int32)
        self.items = np.zeros(len(training_data), dtype=np.int32)

        session_loc = training_data.columns.get_loc('Session')
        item_loc = training_data.columns.get_loc('SID_Content')

        session_list = []

        self.session_map = {}
        self.session_count = 0

        self.item_map = {}
        self.item_list = []
        self.item_count = 0

        last_session = -1

        cursor = 0

        for row in training_data.itertuples(index=False):
            item = row[item_loc]
            session = row[session_loc]

            if session not in self.session_map:
                self.session_map[session] = self.session_count
                self.session_count += 1

            if item not in self.item_map:
                self.item_map[item] = self.item_count
                self.item_list.append(item)
                self.item_count += 1

            if last_session != session:
                if last_session > 0:
                    session_start_count = self.session_map[last_session]  # TODO: is this right?
                    self.sessions[session_start_count, :] = [cursor, len(session_list)]
                    self.items[cursor:cursor + len(session_list)] = session_list
                    cursor += len(session_list)

                session_list = []

            last_session = session
            session_list.append(self.item_map[item])

        session_start_count = self.session_map[last_session]  # TODO: is this right?
        self.sessions[session_start_count, :] = [cursor, len(session_list)]
        self.items[cursor:cursor + len(session_list)] = session_list
        cursor += len(session_list)

        iterations = 0
        epoch_offset = 0
        if load_model is not None:
            filename = model_dir + load_model
            file = np.load(filename)
            self.V = file['V']
            self.H = file['H']
            self.bias = file['bias']
            self.eta = file['eta']
            self.eta_bias = file['eta_bias']

            epoch_offset = float(re.search('_ne([0-9]+(\.[0-9]+)?)_', filename).group(1))
        if epoch_offset == 0:
            self.V = 1 * np.random.randn(self.item_num, self.k).astype(np.float32)
            self.H = 1 * np.random.randn(self.item_num, self.k).astype(np.float32)
            self.eta = 1 * np.random.randn(self.session_num, self.order).astype(np.float32)
            self.eta_bias = np.zeros(self.order).astype(np.float32)
            self.bias = np.zeros(self.item_num).astype(np.float32)

        start_time = time()
        next_save = int(progress)
        train_costs = []
        current_train_cost = []
        epochs = []

        while time() - start_time < max_time and iterations < max_iterations:
            cost = self.training_step()

            current_train_cost.append(cost)

            if iterations % len(training_data) == 0:
                self.learning_rate *= self.annealing_rate

            iterations += 1

            if iterations >= next_save:
                if iterations >= min_iterations:
                    # save current epoch
                    epochs.append(epoch_offset + iterations / len(training_data))

                    # average train cost
                    train_costs.append(np.mean(current_train_cost))
                    current_train_cost = []

                    # print progress
                    print(iterations, " batch, ", epochs[-1], " epochs in ", time() - start_time, "s")
                    print("Last train cost: ", train_costs[-1])

                    # save model
                    filename = model_dir + "fossil_ne" + str(epochs[-1]) + "_k" + str(self.k)
                    print('Save model in ' + filename)
                    if not os.path.exists(os.path.dirname(filename)):
                        os.makedirs(os.path.dirname(filename))
                    np.savez(filename, V=self.V, H=self.H, bias=self.bias, eta=self.eta, eta_bias=self.eta_bias)

                # compute next checkpoint
                if isinstance(progress, int):
                    next_save += progress
                else:
                    next_save += next_save * (progress - 1)

    def training_step(self):
        session_id = random.randrange(self.session_num)
        while self.sessions[session_id, 1] < 2:
            session_id = random.randrange(self.session_num)
        session_items = self.items[
                        self.sessions[session_id, 0]:
                        self.sessions[session_id, 0] + self.sessions[session_id, 1]
                        ]

        index = random.randrange(1, len(session_items))

        false_item = random.randrange(self.item_num)
        while false_item in session_items[:index + 1]:
            false_item = random.randrange(self.item_num)

        true_item = session_items[index]
        past_items = session_items[:index]

        long_term = np.power(len(past_items), -self.alpha) * self.V[past_items, :].sum(axis=0)
        short_term = np.dot(
            (self.eta_bias + self.eta[session_id, :])[:self.order], self.V[past_items[:-self.order - 1:-1], :]
        )

        # Compute error
        x_true = self.score_item(session_id, past_items, true_item)
        x_false = self.score_item(session_id, past_items, false_item)
        delta = 1 / (1 + math.exp(-min(10, max(-10, x_false - x_true))))

        # Compute update
        V_update = self.learning_rate * (delta * np.power(len(past_items), -self.alpha) * (self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.V[past_items, :])
        V_update2 = self.learning_rate * delta * np.outer((self.eta_bias + self.eta[session_id, :])[:self.order],self.H[true_item, :] - self.H[false_item, :])
        H_true_up = self.learning_rate * (delta * (long_term + short_term) - self.reg * self.H[true_item, :])
        H_false_up = self.learning_rate * (-delta * (long_term + short_term) - self.reg * self.H[true_item, :])
        bias_true_up = self.learning_rate * (delta - self.reg * self.bias[true_item])
        bias_false_up = self.learning_rate * (-delta - self.reg * self.bias[false_item])
        eta_bias_up = self.learning_rate * (delta * np.dot(self.V[past_items[:-self.order - 1:-1], :], self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.eta_bias[:self.order])
        eta_up = self.learning_rate * (delta * np.dot(self.V[past_items[:-self.order - 1:-1], :], self.H[true_item, :] - self.H[false_item, :]) - self.reg * self.eta[session_id, :self.order])

        # Update
        self.V[past_items, :] += V_update
        self.V[past_items[:-self.order - 1:-1], :] += V_update2
        self.H[true_item, :] += H_true_up
        self.H[false_item, :] += H_false_up
        self.bias[true_item] += bias_true_up
        self.bias[false_item] += bias_false_up
        self.eta_bias[:self.order] += eta_bias_up
        self.eta[session_id, :self.order] += eta_up

        return delta

    def score_item(self, session_id, past_items, item=None):
        long_term = np.power(len(past_items), -self.alpha) * self.V[past_items, :].sum(axis=0)

        if session_id is None:
            short_term = np.dot((self.eta_bias + self.eta.mean(axis=0))[:self.order],
                                self.V[past_items[:-self.order - 1:-1], :])
        else:
            short_term = np.dot((self.eta_bias + self.eta[session_id, :])[:self.order],
                                self.V[past_items[:-self.order - 1:-1], :])

        if item is not None:
            return self.bias[item] + np.dot(long_term + short_term, self.H[item, :])
        else:
            return self.bias + np.dot(long_term + short_term, self.H.T)

    def predict(self, curr_session_id: int, last_item_id: int) -> pd.Series:
        item_index = self.item_map[last_item_id]

        if self.current_session is None or self.current_session != curr_session_id:
            self.current_session = curr_session_id
            self.session = [item_index]
        else:
            self.session.append(item_index)

        scores = self.score_item(session_id=None, past_items=self.session)

        return pd.Series(data=scores, index=self.item_list)
