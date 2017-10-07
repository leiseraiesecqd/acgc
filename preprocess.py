import time
import utils
import os
import numpy as np
import pandas as pd
from os.path import isdir
from sklearn.model_selection import StratifiedShuffleSplit


train_csv_path = './inputs/stock_train_data_20170929.csv'
test_csv_path = './inputs/stock_test_data_20170929.csv'
preprocessed_path = './preprocessed_data/'
negative_era_list = [1, 3, 4, 5, 8, 10, 12, 16]
positive_era_list = [2, 6, 7, 9, 11, 13, 14, 15, 17, 18, 19, 20]


class DataPreProcess:

    def __init__(self, train_path, test_path, prepro_path):

        self.train_path = train_path
        self.test_path = test_path
        self.prepro_path = prepro_path
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
        self.x_train = train_f.drop(['id', 'weight', 'label', 'group', 'era'], axis=1)
        self.y_train = train_f['label']
        self.w_train = train_f['weight']
        self.g_train = train_f['group']
        self.e_train = train_f['era']
        self.x_test = test_f.drop(['id', 'group'], axis=1)
        self.id_test = test_f['id']
        self.g_test = test_f['group']

    # Drop Outlier of a Feature
    def drop_outliers_of_feature(self, feature, upper_quantile_train, lower_quantile_train,
                                 upper_quantile_test, lower_quantile_test):

        # Drop upper outliers in self.x_train
        upper_train = self.x_train[feature].quantile(upper_quantile_train)
        self.x_train[feature].loc[self.x_train[feature] > upper_train] = upper_train

        # Drop lower outlines in self.x_train
        lower_train = self.x_train[feature].quantile(lower_quantile_train)
        self.x_train[feature].loc[self.x_train[feature] < lower_train] = lower_train

        # Drop upper outlines in self.x_test
        upper_test = self.x_test[feature].quantile(upper_quantile_test)
        self.x_test[feature].loc[self.x_test[feature] > upper_test] = upper_test

        lower_test = self.x_test[feature].quantile(lower_quantile_test)
        self.x_test[feature].loc[self.x_test[feature] < lower_test] = lower_test

    # Dropping Outliers
    def drop_outliers(self):

        print('Dropping outliers...')

        for i in range(88):
            self.drop_outliers_of_feature('feature' + str(i), 0.9995, 0.0005, 0.9995, 0.0005)

        # feature | upper_quantile_train | lower_quantile_train | upper_quantile_test | lower_quantile_test

        # self.drop_outliers_of_feature('feature0', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature1', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature2', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature3', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature4', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature5', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature6', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature7', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature8', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature9', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature10', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature11', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature12', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature13', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature14', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature15', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature16', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature17', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature18', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature19', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature20', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature21', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature22', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature23', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature24', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature25', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature26', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature27', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature28', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature29', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature30', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature31', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature32', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature33', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature34', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature35', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature36', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature37', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature38', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature39', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature40', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature41', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature42', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature43', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature44', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature45', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature46', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature47', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature48', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature49', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature50', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature51', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature52', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature53', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature54', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature55', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature56', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature57', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature58', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature59', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature60', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature61', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature62', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature63', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature64', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature65', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature66', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature67', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature68', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature69', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature70', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature71', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature72', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature73', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature74', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature75', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature76', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature77', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature78', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature79', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature80', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature81', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature82', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature83', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature84', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature85', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature86', 0.9995, 0.0005, 0.9995, 0.0005)
        # self.drop_outliers_of_feature('feature87', 0.9995, 0.0005, 0.9995, 0.0005)

    # Standard Scale
    def standard_scale(self):

        print('Standard Scaling Data...')

        for each in self.x_train.columns:
            mean, std = self.x_train[each].mean(), self.x_train[each].std()
            self.x_train.loc[:, each] = (self.x_train[each] - mean)/std

        for each in self.x_test.columns:
            mean, std = self.x_test[each].mean(), self.x_test[each].std()
            self.x_test.loc[:, each] = (self.x_test[each] - mean)/std

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

        if not isdir(self.prepro_path):
            os.mkdir(self.prepro_path)

        print('Saving data...')

        utils.save_np_to_pkl(self.x_train, self.prepro_path + 'x_train.p')
        utils.save_np_to_pkl(self.x_g_train, self.prepro_path + 'x_g_train.p')
        utils.save_np_to_pkl(self.y_train, self.prepro_path + 'y_train.p')
        utils.save_np_to_pkl(self.w_train, self.prepro_path + 'w_train.p')
        utils.save_np_to_pkl(self.x_test, self.prepro_path + 'x_test.p')
        utils.save_np_to_pkl(self.x_g_test, self.prepro_path + 'x_g_test.p')

    # Save Data
    def save_data_pd(self):

        if not isdir(self.prepro_path):
            os.mkdir(self.prepro_path)

        print('Saving data...')

        self.x_train.to_pickle(self.prepro_path + 'x_train.p')
        self.x_g_train.to_pickle(self.prepro_path + 'x_g_train.p')
        self.y_train.to_pickle(self.prepro_path + 'y_train.p')
        self.w_train.to_pickle(self.prepro_path + 'w_train.p')
        self.e_train.to_pickle(self.prepro_path + 'e_train.p')
        self.x_test.to_pickle(self.prepro_path + 'x_test.p')
        self.x_g_test.to_pickle(self.prepro_path + 'x_g_test.p')
        self.id_test.to_pickle(self.prepro_path + 'id_test.p')

    # Save Data Split by Era Distribution
    def save_data_by_era_distribution_pd(self):

        if not isdir(self.prepro_path):
            os.mkdir(self.prepro_path)

        print('Saving data...')

        # Positive Data

        self.x_train_p.to_pickle(self.prepro_path + 'x_train_p.p')
        self.x_g_train_p.to_pickle(self.prepro_path + 'x_g_train_p.p')
        self.y_train_p.to_pickle(self.prepro_path + 'y_train_p.p')
        self.w_train_p.to_pickle(self.prepro_path + 'w_train_p.p')
        self.e_train_p.to_pickle(self.prepro_path + 'e_train_p.p')

        # Negative Data

        self.x_train_n.to_pickle(self.prepro_path + 'x_train_n.p')
        self.x_g_train_n.to_pickle(self.prepro_path + 'x_g_train_n.p')
        self.y_train_n.to_pickle(self.prepro_path + 'y_train_n.p')
        self.w_train_n.to_pickle(self.prepro_path + 'w_train_n.p')
        self.e_train_n.to_pickle(self.prepro_path + 'e_train_n.p')

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
        self.standard_scale()
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

    DPP = DataPreProcess(train_csv_path, test_csv_path, preprocessed_path)
    DPP.preprocess_pd()
