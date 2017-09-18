import pickle
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


train_csv_path = './inputs/stock_train_data_20170910.csv'
test_csv_path = './inputs/stock_test_data_20170910.csv'
preprocessed_path = './preprocessed_data/'


class DataPreProcess:

    train_x = np.array([])
    train_y = np.array([])
    train_w = np.array([])
    test_x = np.array([])

    def __init__(self, train_path, test_path, prepro_path):

        self.train_path = train_path
        self.test_path = test_path
        self.prepro_path = prepro_path

    # Load CSV files
    def load_csv(self):

        train_f = np.loadtxt(self.train_path, dtype=np.float64, skiprows=1, delimiter=",")
        test_f = np.loadtxt(self.test_path, dtype=np.float64, skiprows=1, delimiter=",")

        return train_f, test_f

    # Load Data
    def load_data(self):

        try:
            print('Loading data...')
            train_f, test_f = self.load_csv()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        self.train_x = train_f[:, 1:]                  # feature + weight + label + group + era
        self.train_w = train_f[:, 89]                  # weight
        np.delete(self.train_x, [88, 89, 90], axis=1)  # feature + era
        self.train_y = train_f[:, 90]                  # label
        self.test_x = test_f                           # feature + group

    # Shuffle and split dataset
    def random_spit_data(self):

        print('Shuffling and splitting data...')

        ss_train = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
        train_idx, valid_idx = next(ss_train.split(self.train_x, self.train_y))

        valid_x, valid_y = self.train_x[valid_idx], self.train_y[valid_idx]
        train_x, train_y = self.train_x[train_idx], self.train_y[train_idx]

        train_w = train_x[:, -2]
        train_x = train_x[:, :-2]
        valid_w = valid_x[:, -2]
        valid_x = valid_x[:, :-2]

        return train_x, train_y, train_w, valid_x, valid_y, valid_w

    # Save Data
    def save_data(self):

        print('Saving data...')

        with open(self.prepro_path + 'train_x.p', 'wb') as f:
            pickle.dump(self.train_x, f)
        with open(self.prepro_path + 'train_y.p', 'wb') as f:
            pickle.dump(self.train_y, f)
        with open(self.prepro_path + 'train_w.p', 'wb') as f:
            pickle.dump(self.train_w, f)
        with open(self.prepro_path + 'test_x.p', 'wb') as f:
            pickle.dump(self.test_x, f)

    # Preprocessing
    def preprocess(self):

        start_time = time.time()

        self.load_data()

        # train_x, train_y, train_w, valid_x, valid_y, valid_w = self.random_spit_data(train_data_x, train_data_y)

        self.save_data()

        end_time = time.time()
        total_time = end_time - start_time

        print('Done!')
        print('Using {:.3}s'.format(total_time))


if __name__ == "__main__":

    DPP = DataPreProcess(train_csv_path, test_csv_path, preprocessed_path)
    DPP.preprocess()










