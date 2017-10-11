import time
import utils
import os
import main
import numpy as np
import pandas as pd
from os.path import isdir
from sklearn.model_selection import StratifiedShuffleSplit


train_csv_path = './inputs/stock_train_data_20171006.csv'
test_csv_path = './inputs/stock_test_data_20171006.csv'
preprocessed_path = './data/preprocessed_data/'
negative_era_list = [1, 2, 3, 4, 7, 8, 9, 15, 17]
positive_era_list = [5, 6, 10, 11, 12, 13, 14, 16, 18, 19, 20]


class DataPreProcess:

    def __init__(self, train_path, test_path, preprocess_path):

        self.train_path = train_path
        self.test_path = test_path
        self.preprocess_path = preprocess_path
        self.x_train = pd.DataFrame()
        self.x_g_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.w_train = pd.DataFrame()
        self.g_train = pd.DataFrame()
        self.e_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.x_g_test = pd.DataFrame()
        self.g_test = pd.DataFrame()
        self.id_test = pd.DataFrame()

        # Positive Data Set
        self.x_train_p = pd.DataFrame()
        self.x_g_train_p = pd.DataFrame()
        self.y_train_p = pd.DataFrame()
        self.w_train_p = pd.DataFrame()
        self.e_train_p = pd.DataFrame()
        self.id_test_p = pd.DataFrame()

        # Negative Data Set
        self.x_train_n = pd.DataFrame()
        self.x_g_train_n = pd.DataFrame()
        self.y_train_n = pd.DataFrame()
        self.w_train_n = pd.DataFrame()
        self.e_train_n = pd.DataFrame()
        self.id_test_n = pd.DataFrame()

    # Load CSV Files
    def load_csv_np(self):

        train_f = np.loadtxt(self.train_path, dtype=np.float64, skiprows=1, delimiter=",")
        test_f = np.loadtxt(self.test_path, dtype=np.float64, skiprows=1, delimiter=",")

        return train_f, test_f

    # Load Data
    def load_data_np(self):

        try:
            print('Loading data...')
            train_f, test_f = self.load_csv_np()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        self.x_train = train_f[:, 1:]                  # feature + weight + label + group + era
        self.w_train = train_f[:, 89]                  # weight
        np.delete(self.x_train, [88, 89, 90], axis=1)  # feature + era
        self.y_train = train_f[:, 90]                  # label
        self.x_test = test_f                           # feature + group

    # Load CSV Files Using Pandas
    def load_csv_pd(self):

        train_f = pd.read_csv(self.train_path, header=0, dtype=np.float64)
        test_f = pd.read_csv(self.test_path, header=0, dtype=np.float64)

        return train_f, test_f

    # Load Data Using Pandas
    def load_data_pd(self):

        try:
            print('Loading data...')
            train_f, test_f = self.load_csv_pd()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        # Drop Unnecessary Columns
        self.x_train = train_f.drop(['id', 'weight', 'label', 'group', 'era', 'feature77'], axis=1)
        self.y_train = train_f['label']
        self.w_train = train_f['weight']
        self.g_train = train_f['group']
        self.e_train = train_f['era']
        self.x_test = test_f.drop(['id', 'group', 'feature77'], axis=1)
        self.id_test = test_f['id']
        self.g_test = test_f['group']

    # Drop Outlier of a Feature by Quantile
    def drop_outliers_by_quantile(self, feature, upper_quantile_train=None, lower_quantile_train=None,
                                  upper_quantile_test=None, lower_quantile_test=None):

        # Drop upper outliers in self.x_train
        if upper_quantile_train is not None:
            upper_train = self.x_train[feature].quantile(upper_quantile_train)
            self.x_train[feature].loc[self.x_train[feature] > upper_train] = upper_train

        # Drop lower outlines in self.x_train
        if lower_quantile_train is not None:
            lower_train = self.x_train[feature].quantile(lower_quantile_train)
            self.x_train[feature].loc[self.x_train[feature] < lower_train] = lower_train

        # Drop upper outlines in self.x_test
        if upper_quantile_test is not None:
            upper_test = self.x_test[feature].quantile(upper_quantile_test)
            self.x_test[feature].loc[self.x_test[feature] > upper_test] = upper_test

        # Drop lower outlines in self.x_test
        if lower_quantile_test is not None:
            lower_test = self.x_test[feature].quantile(lower_quantile_test)
            self.x_test[feature].loc[self.x_test[feature] < lower_test] = lower_test

    # Drop Outlier of a Feature by Value
    def drop_outliers_by_value(self, feature, upper_train=None, lower_train=None):
        
        # Drop upper outliers in self.x_train
        if upper_train is not None:
            self.x_train[feature].loc[self.x_train[feature] > upper_train] = upper_train

        # Drop lower outlines in self.x_train
        if lower_train is not None:
            self.x_train[feature].loc[self.x_train[feature] < lower_train] = lower_train

    # Dropping Outliers
    def drop_outliers(self):

        print('Dropping outliers...')

        # for i in range(88):
        #     if i != 77:
        #         self.drop_outliers_by_quantile('feature' + str(i), 0.9995, 0.0005, 0.9995, 0.0005)

        # feature | upper_quantile_train | lower_quantile_train | upper_quantile_test | lower_quantile_test

        self.drop_outliers_by_value('feature0', None, None)
        self.drop_outliers_by_value('feature1', 4.42, -13.89)
        self.drop_outliers_by_value('feature2', 15, None)
        self.drop_outliers_by_value('feature3', None, None)
        self.drop_outliers_by_value('feature4', 13.83, None)
        self.drop_outliers_by_value('feature5', None, None)
        self.drop_outliers_by_value('feature6', None, None)
        self.drop_outliers_by_value('feature7', 8, None)
        self.drop_outliers_by_value('feature8', 8, None)
        self.drop_outliers_by_value('feature9', None, None)
        self.drop_outliers_by_value('feature10', 14.68, None)
        self.drop_outliers_by_value('feature11', None, None)
        self.drop_outliers_by_value('feature12', 6, None)
        self.drop_outliers_by_value('feature13', None, None)
        self.drop_outliers_by_value('feature14', 30, None)
        self.drop_outliers_by_value('feature15', None, None)
        self.drop_outliers_by_value('feature16', 5.7, None)
        self.drop_outliers_by_value('feature17', None, None)
        self.drop_outliers_by_value('feature18', 20, None)
        self.drop_outliers_by_value('feature19', 7.4, None)
        self.drop_outliers_by_value('feature20', None, None)
        self.drop_outliers_by_value('feature21', None, None)
        self.drop_outliers_by_value('feature22', None, None)
        self.drop_outliers_by_value('feature23', 21.2, None)
        self.drop_outliers_by_value('feature24', 7.41, None)
        self.drop_outliers_by_value('feature25', None, None)
        self.drop_outliers_by_value('feature26', 20.96, None)
        self.drop_outliers_by_value('feature27', None, None)
        self.drop_outliers_by_value('feature28', 7.3, None)
        self.drop_outliers_by_value('feature29', 15, -10)
        self.drop_outliers_by_value('feature30', None, None)
        self.drop_outliers_by_value('feature31', 6.7, None)
        self.drop_outliers_by_value('feature32', None, None)
        self.drop_outliers_by_value('feature33', 6.8, None)
        self.drop_outliers_by_value('feature34', 4.69, None)
        self.drop_outliers_by_value('feature35', None, None)
        self.drop_outliers_by_value('feature36', 29.51, None)
        self.drop_outliers_by_value('feature37', 6.18, None)
        self.drop_outliers_by_value('feature38', 28.28, None)
        self.drop_outliers_by_value('feature39', 14.09, None)
        self.drop_outliers_by_value('feature40', 18.63, None)
        self.drop_outliers_by_value('feature41', None, None)
        self.drop_outliers_by_value('feature42', None, None)
        self.drop_outliers_by_value('feature43', 22.58, None)
        self.drop_outliers_by_value('feature44', 8, None)
        self.drop_outliers_by_value('feature45', 14.23, None)
        self.drop_outliers_by_value('feature46', None, -7.41)
        self.drop_outliers_by_value('feature47', None, None)
        self.drop_outliers_by_value('feature48', None, None)
        self.drop_outliers_by_value('feature49', None, None)
        self.drop_outliers_by_value('feature50', None, None)
        self.drop_outliers_by_value('feature51', None, None)
        self.drop_outliers_by_value('feature52', None, None)
        self.drop_outliers_by_value('feature53', 20, None)
        self.drop_outliers_by_value('feature54', 14.32, None)
        self.drop_outliers_by_value('feature55', None, None)
        self.drop_outliers_by_value('feature56', 27.45, None)
        self.drop_outliers_by_value('feature57', None, None)
        self.drop_outliers_by_value('feature58', 16.93, None)
        self.drop_outliers_by_value('feature59', 8.297, None)
        self.drop_outliers_by_value('feature60', 11, None)
        self.drop_outliers_by_value('feature61', None, None)
        self.drop_outliers_by_value('feature62', None, None)
        self.drop_outliers_by_value('feature63', 10.2, None)
        self.drop_outliers_by_value('feature64', None, None)
        self.drop_outliers_by_value('feature65', 7.787, None)
        self.drop_outliers_by_value('feature66', 18.93, None)
        self.drop_outliers_by_value('feature67', 17.74, None)
        self.drop_outliers_by_value('feature68', 20, None)
        self.drop_outliers_by_value('feature69', 20.32, None)
        self.drop_outliers_by_value('feature70', None, None)
        self.drop_outliers_by_value('feature71', None, -13.41)
        self.drop_outliers_by_value('feature72', 20, None)
        self.drop_outliers_by_value('feature73', None, None)
        self.drop_outliers_by_value('feature74', None, None)
        self.drop_outliers_by_value('feature75', None, None)
        self.drop_outliers_by_value('feature76', 6.5, None)
        # self.drop_outliers_by_value('feature77', None, None)
        self.drop_outliers_by_value('feature78', 10.5, -30)
        self.drop_outliers_by_value('feature79', 15, None)
        self.drop_outliers_by_value('feature80', None, None)
        self.drop_outliers_by_value('feature81', None, None)
        self.drop_outliers_by_value('feature82', None, None)
        self.drop_outliers_by_value('feature83', None, None)
        self.drop_outliers_by_value('feature84', None, None)
        self.drop_outliers_by_value('feature85', 8.18, None)
        self.drop_outliers_by_value('feature86', 20, None)
        self.drop_outliers_by_value('feature87', None, None)

    # Standard Scale
    def standard_scale(self):

        print('Standard Scaling Data...')

        mean = np.zeros(len(self.x_train.columns), dtype=np.float64)
        std = np.zeros(len(self.x_train.columns), dtype=np.float64)

        for i, each in enumerate(self.x_train.columns):
            mean[i], std[i] = self.x_train[each].mean(), self.x_train[each].std()
            self.x_train.loc[:, each] = (self.x_train[each] - mean[i])/std[i]

        for i, each in enumerate(self.x_test.columns):
            self.x_test.loc[:, each] = (self.x_test[each] - mean[i])/std[i]

    # Min Max scale
    def min_max_scale(self):

        for each in self.x_train.columns:
            x_max, x_min = self.x_train[each].max(),  self.x_train[each].min()
            self.x_train.loc[:, each] = (self.x_train[each] - x_min)/(x_max - x_min)

        for each in self.x_test.columns:
            x_max, x_min = self.x_test[each].max(), self.x_test[each].min()
            self.x_test.loc[:, each] = (self.x_test[each] - x_min)/(x_max - x_min)

    # Convert Column 'group' to Dummies
    def convert_group_to_dummies(self):

        print('Converting Groups to Dummies...')

        group_train_dummies = pd.get_dummies(self.g_train, prefix='group')
        self.x_g_train = self.x_train.join(self.g_train)
        self.x_train = self.x_train.join(group_train_dummies)

        group_test_dummies = pd.get_dummies(self.g_test, prefix='group')
        self.x_g_test = self.x_test.join(self.g_test)
        self.x_test = self.x_test.join(group_test_dummies)

        # print('Shape of x_train with group dummies: {}'.format(self.x_train.shape))
        # print('Shape of x_test with group dummies: {}'.format(self.x_test.shape))

    # Split Positive and Negative Era Set
    def split_data_by_era_distribution(self, negative_era):

        print('Splitting Positive and Negative Era Set...')
        print('Negative Eras: ', negative_era)

        positive_index = []
        negative_index = []

        for i, ele in enumerate(self.e_train):

            if int(ele) in negative_era:
                negative_index.append(i)
            else:
                positive_index.append(i)

        # Positive Data
        self.x_train_p = self.x_train.ix[positive_index]
        self.x_g_train_p = self.x_g_train.ix[positive_index]
        self.y_train_p = self.y_train.ix[positive_index]
        self.w_train_p = self.w_train.ix[positive_index]
        self.e_train_p = self.e_train.ix[positive_index]

        # Negative Data
        self.x_train_n = self.x_train.ix[negative_index]
        self.x_g_train_n = self.x_g_train.ix[negative_index]
        self.y_train_n = self.y_train.ix[negative_index]
        self.w_train_n = self.w_train.ix[negative_index]
        self.e_train_n = self.e_train.ix[negative_index]

    # Shuffle and Split Data Set
    def random_spit_data(self):

        print('Shuffling and splitting data...')

        ss_train = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
        train_idx, valid_idx = next(ss_train.split(self.x_train, self.y_train))

        x_valid, y_valid = self.x_train[valid_idx], self.y_train[valid_idx]
        x_train, y_train = self.x_train[train_idx], self.y_train[train_idx]

        w_train = x_train[:, -2]
        x_train = x_train[:, :-2]
        w_valid = x_valid[:, -2]
        x_valid = x_valid[:, :-2]

        return x_train, y_train, w_train, x_valid, y_valid, w_valid

    # Save Data
    def save_data_np(self):

        if not isdir(self.preprocess_path):
            os.mkdir(self.preprocess_path)

        print('Saving data...')

        utils.save_np_to_pkl(self.x_train, self.preprocess_path + 'x_train.p')
        utils.save_np_to_pkl(self.x_g_train, self.preprocess_path + 'x_g_train.p')
        utils.save_np_to_pkl(self.y_train, self.preprocess_path + 'y_train.p')
        utils.save_np_to_pkl(self.w_train, self.preprocess_path + 'w_train.p')
        utils.save_np_to_pkl(self.x_test, self.preprocess_path + 'x_test.p')
        utils.save_np_to_pkl(self.x_g_test, self.preprocess_path + 'x_g_test.p')

    # Save Data
    def save_data_pd(self):

        if not isdir(self.preprocess_path):
            os.mkdir(self.preprocess_path)

        print('Saving data...')

        self.x_train.to_pickle(self.preprocess_path + 'x_train.p')
        self.x_g_train.to_pickle(self.preprocess_path + 'x_g_train.p')
        self.y_train.to_pickle(self.preprocess_path + 'y_train.p')
        self.w_train.to_pickle(self.preprocess_path + 'w_train.p')
        self.e_train.to_pickle(self.preprocess_path + 'e_train.p')
        self.x_test.to_pickle(self.preprocess_path + 'x_test.p')
        self.x_g_test.to_pickle(self.preprocess_path + 'x_g_test.p')
        self.id_test.to_pickle(self.preprocess_path + 'id_test.p')

    # Save Data Split by Era Distribution
    def save_data_by_era_distribution_pd(self):

        if not isdir(self.preprocess_path):
            os.mkdir(self.preprocess_path)

        # Positive Data
        print('Saving Positive Data...')
        self.x_train_p.to_pickle(self.preprocess_path + 'x_train_p.p')
        self.x_g_train_p.to_pickle(self.preprocess_path + 'x_g_train_p.p')
        self.y_train_p.to_pickle(self.preprocess_path + 'y_train_p.p')
        self.w_train_p.to_pickle(self.preprocess_path + 'w_train_p.p')
        self.e_train_p.to_pickle(self.preprocess_path + 'e_train_p.p')

        # Negative Data
        print('Saving Negative Data...')
        self.x_train_n.to_pickle(self.preprocess_path + 'x_train_n.p')
        self.x_g_train_n.to_pickle(self.preprocess_path + 'x_g_train_n.p')
        self.y_train_n.to_pickle(self.preprocess_path + 'y_train_n.p')
        self.w_train_n.to_pickle(self.preprocess_path + 'w_train_n.p')
        self.e_train_n.to_pickle(self.preprocess_path + 'e_train_n.p')

    # Preprocessing
    def preprocess_np(self):

        start_time = time.time()

        self.load_data_np()

        # x_train, y_train, w_train, x_valid, y_valid, w_valid = self.random_spit_data(train_data_x, train_data_y)

        self.save_data_np()

        end_time = time.time()
        total_time = end_time - start_time

        print('Done!')
        print('Using {:.3}s'.format(total_time))

    # Preprocess
    def preprocess_pd(self):

        start_time = time.time()

        # Load original data
        self.load_data_pd()

        # Drop outliers
        self.drop_outliers()

        # Scale features
        # self.standard_scale()
        # self.min_max_scale()

        # Convert column 'group' to dummies
        self.convert_group_to_dummies()

        # Save Data to pickle files
        self.save_data_pd()

        # Split Positive and Negative Era Set
        negative_era = negative_era_list
        self.split_data_by_era_distribution(negative_era)

        # Save Data Split by Era Distribution
        self.save_data_by_era_distribution_pd()

        end_time = time.time()

        print('Done!')
        print('Using {:.3}s'.format(end_time - start_time))


if __name__ == '__main__':

    utils.check_dir(['./data/', preprocessed_path])
    DPP = DataPreProcess(train_csv_path, test_csv_path, preprocessed_path)
    DPP.preprocess_pd()
