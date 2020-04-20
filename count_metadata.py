import pandas as pd

day1 = pd.read_csv('../data/data_6232_30sec_min_noduplicates_nouser.csv')
day2 = pd.read_csv('../data/data_6233_30sec_min_noduplicates_nouser.csv')
day3 = pd.read_csv('../data/data_6234_30sec_min_noduplicates_nouser.csv')
all_days = day1.append(day2).append(day3)

num_entries = len(all_days)
num_sessions = all_days['Session'].nunique()
num_items = all_days['SID_Content'].nunique()

print('entries:', num_entries)
print('sessions:', num_sessions)
print('content:', num_items)
print('entries/day:', num_entries/3)
print('sessions/day:', num_sessions/3)
print('entries/session:', num_entries/num_sessions)
