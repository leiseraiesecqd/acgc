import pickle
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


start_time = time.time()


# Load CSV

def load_data():
    train_f = np.loadtxt("./stock_train_data_20170910.csv", dtype=np.float, skiprows=1, delimiter=",")
    test_f = np.loadtxt("./stock_test_data_20170910.csv", dtype=np.float32, skiprows=1, delimiter=",")

    return train_f, test_f


try:
    print('Loding data...')
    train_f, test_f = load_data()
except Exception as e:
    print('Unable to read data: ', e)
    raise

train_x = train_f[:, 1:90]
train_y = train_f[:, 90]
test_x = train_f


# Shuffle and split dataset

print('Shuffling and splitting data...')

ss_train = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
train_idx, valid_idx = next(ss_train.split(train_x, train_y))

valid_x, valid_y = train_x[valid_idx], train_y[valid_idx]
train_x, train_y = train_x[train_idx], train_y[train_idx]

train_w = train_x[:, -1]
train_x = train_x[:, :-1]
valid_w = valid_x[:, -1]
valid_x = valid_x[:, :-1]


# Save data

print('Saving data...')

with open('train_x.p', 'wb') as f:
    pickle.dump(train_x, f)

with open('train_y.p', 'wb') as f:
    pickle.dump(train_y, f)

with open('train_w.p', 'wb') as f:
    pickle.dump(train_w, f)

with open('valid_x.p', 'wb') as f:
    pickle.dump(valid_x, f)

with open('valid_y.p', 'wb') as f:
    pickle.dump(valid_y, f)

with open('valid_w.p', 'wb') as f:
    pickle.dump(valid_w, f)

with open('test_x.p', 'wb') as f:
    pickle.dump(test_x, f)

end_time = time.time()
total_time = end_time - start_time

print('Done!')
print('Using {:.3}s'.format(total_time))
