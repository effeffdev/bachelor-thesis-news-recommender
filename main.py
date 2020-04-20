import numpy as np
import pandas as pd

from algorithms.popularity import Popularity
from algorithms.pspm import PSPM
from algorithms.ssknn import SSKNN
from algorithms.fossil import Fossil
from metrics import Metrics

from time import time

day1 = pd.read_csv('../data/data_6232_30sec_min_noduplicates.csv')
day2 = pd.read_csv('../data/data_6233_30sec_min_noduplicates.csv')
training_data = day1.append(day2)
test_data = pd.read_csv('../data/data_6234_30sec_min_noduplicates.csv')

# start_time = time()
# print('started training Popularity')
# pop = Popularity()
# pop.train(training_data)
# print('finished training Popularity in ' + str(time() - start_time) + 's')

start_time = time()
print('started training SSKNN')
ssknn = SSKNN()
ssknn.train(training_data)
print('finished training SSKNN in ' + str(time() - start_time) + 's')

# start_time = time()
# print('started training Fossil')
# fossil = Fossil(epochs=0)
# fossil.train(training_data, load_model='fossil_ne6.277346591004779_k100.npz')
# print('finished training Fossil in ' + str(time() - start_time) + 's')

# start_time = time()
# print('started training PSPM')
# pspm = PSPM()
# pspm.train(training_data)
# print('finished training PSPM in ' + str(time() - start_time) + 's')

met20 = Metrics(20)
met10 = Metrics(10)
met5 = Metrics(5)

# sort test data if necessary
# test_data.sort_values(['ID_Day', 'ID_Visit', 'PiNr'], inplace=True)

# floating window
# test_length = len(test_data)
#
# all_items = training_data['SID_Content'].unique()
# last_session = -1
# last_item = -1
#
# start_time = time()
# for i in range(test_length):
#
#     if i % 1000 == 0:
#         print('process: ' + str(i) + ' of ' + str(test_length) + ' -> ' + str(i / test_length * 100.0) + '% in ' + str(time() - start_time) + 's')
#
#     session = test_data['Session'].values[i]
#     item = test_data['SID_Content'].values[i]
#
#     if item not in all_items:
#         continue
#
#     # new session
#     if last_session != session:
#         last_session = session
#     else:
#         # predictions = pop.predict(all_items)
#         # predictions = ssknn.predict(session, last_item, all_items)
#         # predictions = fossil.predict(session, last_item)
#         predictions = pspm.predict(session, last_item, all_items)
#
#         # TODO: cleanup necessary?
#         predictions[np.isnan(predictions)] = 0
#         predictions.sort_values(ascending=False, inplace=True)
#
#         met20.add(predictions, item)
#         met10.add(predictions, item)
#         met5.add(predictions, item)
#
#     last_item = item

# only check last item in session
test_length = len(test_data)

all_items = training_data['SID_Content'].unique()
last_session = -1
last_item = -1
session_items = list()

start_time = time()
for i in range(test_length):
    if i % 1000 == 0:
        print('process: ' + str(i) + ' of ' + str(test_length) + ' -> ' + str(i / test_length * 100.0) + '% in ' + str(time() - start_time) + 's')

    session = test_data['Session'].values[i]
    item = test_data['SID_Content'].values[i]

    # not for ssknn!
    if item not in all_items:
        continue

    if last_session != session:
        if len(session_items) > 0:
            try:
                # predictions = pop.predict(all_items)
                predictions = ssknn.predict_session(session_items, all_items)
                # predictions = fossil.predict_session(session_items)
                # predictions = pspm.predict_session(session_items, all_items)

                # TODO: cleanup necessary?
                predictions[np.isnan(predictions)] = 0
                predictions.sort_values(ascending=False, inplace=True)

                met20.add(predictions, last_item)
                met10.add(predictions, last_item)
                met5.add(predictions, last_item)
            except MemoryError as error:
                print(error)
                print('session id: ', last_session)
                print(session_items)

        last_session = session
        session_items = list()
    else:
        session_items.append(last_item)

    last_item = item

result20 = met20.result()
result10 = met10.result()
result5 = met5.result()

print(result20[0][0] + ': ' + str(result20[0][1]) + ' - ' + result20[1][0] + ': ' + str(result20[1][1]))
print(result10[0][0] + ': ' + str(result10[0][1]) + ' - ' + result10[1][0] + ': ' + str(result10[1][1]))
print(result5[0][0] + ': ' + str(result5[0][1]) + ' - ' + result5[1][0] + ': ' + str(result5[1][1]))
