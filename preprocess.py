import pickle
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


train_csv_path = './inputs/stock_train_data_20170910.csv'
test_csv_path = './inputs/stock_test_data_20170910.csv'
preprocessed_path = './preprocessed_data/'


class DataPreProcess:

    def __init__(self, train_path, test_path, prepro_path):

        self.train_path = train_path
        self.test_path = test_path
        self.prepro_path = prepro_path

    # Load CSV files
    def load_csv(self):

        train_f = np.loadtxt(self.train_path , dtype=np.float, skiprows=1, delimiter=",")
        test_f = np.loadtxt(self.test_path, dtype=np.float32, skiprows=1, delimiter=",")

        return train_f, test_f

    # Load Data
    def load_data(self):

        try:
            print('Loading data...')
            train_f, test_f = self.load_csv()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        train_x = train_f[:, 1:90]
        train_y = train_f[:, 90]
        test_x = train_f

        return train_x, train_y, test_x

    # Shuffle and split dataset
    def spit_data(self, train_x, train_y):

        print('Shuffling and splitting data...')

        ss_train = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
        train_idx, valid_idx = next(ss_train.split(train_x, train_y))

        valid_x, valid_y = train_x[valid_idx], train_y[valid_idx]
        train_x, train_y = train_x[train_idx], train_y[train_idx]

        train_w = train_x[:, -1]
        train_x = train_x[:, :-1]
        valid_w = valid_x[:, -1]
        valid_x = valid_x[:, :-1]

        return train_x, train_y, train_w, valid_x, valid_y, valid_w

    # Save Data
    def save_data(self, train_x, train_y, train_w, valid_x, valid_y, valid_w, test_x):

        print('Saving data...')

        with open(self.prepro_path + 'train_x.p', 'wb') as f:
            pickle.dump(train_x, f)
        with open(self.prepro_path + 'train_y.p', 'wb') as f:
            pickle.dump(train_y, f)
        with open(self.prepro_path + 'train_w.p', 'wb') as f:
            pickle.dump(train_w, f)
        with open(self.prepro_path + 'valid_x.p', 'wb') as f:
            pickle.dump(valid_x, f)
        with open(self.prepro_path + 'valid_y.p', 'wb') as f:
            pickle.dump(valid_y, f)
        with open(self.prepro_path + 'valid_w.p', 'wb') as f:
            pickle.dump(valid_w, f)
        with open(self.prepro_path + 'test_x.p', 'wb') as f:
            pickle.dump(test_x, f)

    # Preprocessing
    def preprocess(self):

        start_time = time.time()

        train_data_x, train_data_y, test_data_x = self.load_data()
        train_x, train_y, train_w, valid_x, valid_y, valid_w = self.spit_data(train_data_x, train_data_y)
        self.save_data(train_x, train_y, train_w, valid_x, valid_y, valid_w, test_data_x)

        end_time = time.time()
        total_time = end_time - start_time

        print('Done!')
        print('Using {:.3}s'.format(total_time))


if __name__ == "__main__":

    DPP = DataPreProcess(train_csv_path, test_csv_path, preprocessed_path)
    DPP.preprocess()










