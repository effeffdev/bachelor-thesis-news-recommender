import numpy as np
import pandas as pd

from algorithms.popularity import Popularity
from algorithms.pspm import PSPM
from algorithms.ssknn import SSKNN
from algorithms.fossil import Fossil
from metrics import Metrics

from time import time

day1 = pd.read_csv('../data/data_6232_10sec_min.csv')
day2 = pd.read_csv('../data/data_6233_10sec_min.csv')
training_data = day1.append(day2)
test_data = pd.read_csv('../data/data_6234_10sec_min.csv').head(100000)

start_time = time()
print('started training Popularity')
pop = Popularity()
pop.train(training_data)
print('finished training Popularity in ' + str(time() - start_time) + 's')

# print('started training SSKNN')
# ssknn = SSKNN()
# ssknn.train(training_data)
# print('finished training SSKNN')

# print('started training Fossil')
# fossil = Fossil(epochs=0)
# fossil.train(training_data, load_model='fossil_ne0.775604523714502_k100.npz')
# print('finished training Fossil')

# start_time = time()
# print('started training PSPM')
# pspm = PSPM()
# pspm.train(training_data)
# print('finished training PSPM in ' + str(time() - start_time) + 's')

met20 = Metrics(20)
# met10 = Metrics(10)
# met5 = Metrics(5)

# sort test data if necessary
# test_data.sort_values(['ID_Day', 'ID_Visit', 'PiNr'], inplace=True)

test_length = len(test_data)

all_items = training_data['SID_Content'].unique()
last_session = -1
last_item = -1

start_time = time()
for i in range(test_length):

    if i % 1000 == 0:
        print('process: ' + str(i) + ' of ' + str(test_length) + ' -> ' + str(i / test_length * 100.0) + '% in ' + str(time() - start_time) + 's')

    session = test_data['Session'].values[i]
    item = test_data['SID_Content'].values[i]

    if item not in all_items:
        continue

    # new session
    if last_session != session:
        last_session = session
    else:
        predictions = pop.predict(all_items)
        # predictions = ssknn.predict(session, last_item, all_items)
        # predictions = fossil.predict(session, last_item)
        # predictions = pspm.predict(session, last_item, all_items)

        # TODO: cleanup necessary?
        predictions[np.isnan(predictions)] = 0
        predictions.sort_values(ascending=False, inplace=True)

        met20.add(predictions, item)
        # met10.add(predictions, item)
        # met5.add(predictions, item)

    last_item = item

result20 = met20.result()
# result10 = met10.result()
# result5 = met5.result()

print(result20[0][0] + ': ' + str(result20[0][1]) + ' - ' + result20[1][0] + ': ' + str(result20[1][1]))
# print(result10[0][0] + ': ' + str(result10[0][1]) + ' - ' + result10[1][0] + ': ' + str(result10[1][1]))
# print(result5[0][0] + ': ' + str(result5[0][1]) + ' - ' + result5[1][0] + ': ' + str(result5[1][1]))
