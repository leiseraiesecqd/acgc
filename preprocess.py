import time
import numpy as np
import pandas as pd
from models import utils
from generate_adversarial_validation import GenerateValidation
from sklearn.preprocessing import PolynomialFeatures

train_csv_path = './inputs/stock_train_data_20171103.csv'
test_csv_path = './inputs/stock_test_data_20171103.csv'
preprocessed_path = './data/preprocessed_data/'
gan_prob_path = './data/gan_outputs/'
negative_era_list = [2, 3, 4, 7, 9, 11, 15]
positive_era_list = [0, 1, 5, 6, 8, 10, 12, 13, 14, 16, 17, 18, 19]
merge_era_range_list = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 34), (35, 41),  (42, 48),
                        (49, 55), (56, 62), (63, 69), (70, 76), (77, 83), (84, 90), (91, 97),
                        (98, 104), (105, 110), (111, 116), (117, 122), (123, 128), (129, 134)]
drop_feature_list = []


class DataPreProcess:

    def __init__(self, train_path, test_path, preprocess_path, use_multi_group=False):

        self.use_multi_group = use_multi_group
        if use_multi_group:
            print('------------------------------------------------------')
            print('[W] Using Multi Groups...')
            self.g1_train = pd.DataFrame()
            self.g2_train = pd.DataFrame()
            self.g1_test = pd.DataFrame()
            self.g2_test = pd.DataFrame()
        else:
            print('------------------------------------------------------')
            print('[W] Using Single Group...')
            self.g_train = pd.DataFrame()
            self.g_test = pd.DataFrame()

        self.train_path = train_path
        self.test_path = test_path
        self.preprocess_path = preprocess_path
        self.x_train = pd.DataFrame()
        self.x_g_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.w_train = pd.DataFrame()
        self.code_id_train = pd.DataFrame()
        self.e_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.x_g_test = pd.DataFrame()
        self.code_id_test = pd.DataFrame()
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

        # Validation Set
        self.x_valid = np.array([])
        self.x_g_valid = np.array([])
        self.y_valid = np.array([])

        self.drop_feature_list = ['feature' + str(f_num) for f_num in drop_feature_list]
        self.merge_era_range_list = merge_era_range_list

    # Load CSV Files Using Pandas
    def load_csv(self):

        train_f = pd.read_csv(self.train_path, header=0, dtype=np.float64)
        test_f = pd.read_csv(self.test_path, header=0, dtype=np.float64)

        return train_f, test_f

    # Load Data Using Pandas
    def load_data(self):

        try:
            print('------------------------------------------------------')
            print('Loading data...')
            train_f, test_f = self.load_csv()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        print('------------------------------------------------------')
        print('Dropping Features:\n\t', self.drop_feature_list)

        # Drop Unnecessary Columns
        # self.x_train = train_f.drop(['id', 'weight', 'label', 'group', 'era'], axis=1)
        self.x_train = train_f.drop(['id', 'weight', 'label', 'group1', 'group2', 'era', 'code_id', *self.drop_feature_list], axis=1)
        self.y_train = train_f['label']
        self.w_train = train_f['weight']
        self.code_id_train = train_f['code_id']
        self.e_train = train_f['era']
        self.x_test = test_f.drop(['id', 'group1', 'group2', 'code_id', *self.drop_feature_list], axis=1)
        self.id_test = test_f['id']
        self.code_id_test = test_f['code_id']

        if self.use_multi_group:
            self.g1_train = train_f['group1']
            self.g2_train = train_f['group2']
            self.g1_test = test_f['group1']
            self.g2_test = test_f['group2']
        else:
            self.g_train = train_f['group1']
            self.g_test = test_f['group1']

    # Merge Eras
    def merge_eras(self):

        print('------------------------------------------------------')
        print('[W] Merging Eras...')
        print('Merge Era List:\n', self.merge_era_range_list)

        self.e_train = np.array(self.e_train, dtype=int)
        e_train_merged = np.zeros_like(self.e_train, dtype=int)

        merge_counter = 0

        for i, era in enumerate(self.e_train):

            era_start, era_end = self.merge_era_range_list[merge_counter]

            if era in range(era_start, era_end+1):
                e_train_merged[i] = merge_counter
            else:
                merge_counter += 1
                era_start, era_end = self.merge_era_range_list[merge_counter]
                if era in range(era_start, era_end+1):
                    e_train_merged[i] = merge_counter
                else:
                    raise ValueError('Merge Era List Must Be Continuous!')

        self.e_train = e_train_merged

    # Drop Outlier of a Feature by Quantile
    def drop_feature_outliers_by_quantile(self, feature, upper_quantile_train=None, lower_quantile_train=None):

        if feature in self.drop_feature_list:
            pass
        else:

            # Drop upper outliers in self.x_train
            if upper_quantile_train is not None:
                upper_train = self.x_train[feature].quantile(upper_quantile_train)
                self.x_train[feature].loc[self.x_train[feature] > upper_train] = upper_train

            # Drop lower outlines in self.x_train
            if lower_quantile_train is not None:
                lower_train = self.x_train[feature].quantile(lower_quantile_train)
                self.x_train[feature].loc[self.x_train[feature] < lower_train] = lower_train

    # Drop Outlier of a Feature by Value
    def drop_feature_outliers_by_value(self, feature, upper_train=None, lower_train=None):

        if feature in self.drop_feature_list:
            pass
        else:

            # Drop upper outliers in self.x_train
            if upper_train is not None:
                self.x_train[feature].loc[self.x_train[feature] > upper_train] = upper_train

            # Drop lower outlines in self.x_train
            if lower_train is not None:
                self.x_train[feature].loc[self.x_train[feature] < lower_train] = lower_train

    # Dropping Outliers by Value
    def drop_outliers_by_value(self):

        print('------------------------------------------------------')
        print('Dropping Outliers by Value...')

        self.drop_feature_outliers_by_value('feature0', 6.22, None)
        self.drop_feature_outliers_by_value('feature1', None, None)
        self.drop_feature_outliers_by_value('feature2', 12.78, None)
        self.drop_feature_outliers_by_value('feature3', None, None)
        self.drop_feature_outliers_by_value('feature4', None, None)
        self.drop_feature_outliers_by_value('feature5', 13.36, None)
        self.drop_feature_outliers_by_value('feature6', None, None)
        self.drop_feature_outliers_by_value('feature7', None, None)
        self.drop_feature_outliers_by_value('feature8', None, None)
        self.drop_feature_outliers_by_value('feature9', 5.25, None)
        self.drop_feature_outliers_by_value('feature10', 8.36, None)
        self.drop_feature_outliers_by_value('feature11', 15.41, None)
        self.drop_feature_outliers_by_value('feature12', 12.05, None)
        self.drop_feature_outliers_by_value('feature13', 3.88, None)
        self.drop_feature_outliers_by_value('feature14', None, None)
        self.drop_feature_outliers_by_value('feature15', None, -7.35)
        self.drop_feature_outliers_by_value('feature16', None, -5.38)
        self.drop_feature_outliers_by_value('feature17', None, None)
        self.drop_feature_outliers_by_value('feature18', 5.57, None)
        self.drop_feature_outliers_by_value('feature19', None, -4.24)
        self.drop_feature_outliers_by_value('feature20', None, None)
        self.drop_feature_outliers_by_value('feature21', None, None)
        self.drop_feature_outliers_by_value('feature22', 15.73, None)
        self.drop_feature_outliers_by_value('feature23', None, None)
        self.drop_feature_outliers_by_value('feature24', None, None)
        self.drop_feature_outliers_by_value('feature25', 4.34, None)
        self.drop_feature_outliers_by_value('feature26', None, None)
        self.drop_feature_outliers_by_value('feature27', None, None)
        self.drop_feature_outliers_by_value('feature28', 19.99, None)
        self.drop_feature_outliers_by_value('feature29', 10.11, None)
        self.drop_feature_outliers_by_value('feature30', 11.57, -29.46)
        self.drop_feature_outliers_by_value('feature31', 14.02, None)
        self.drop_feature_outliers_by_value('feature32', None, None)
        self.drop_feature_outliers_by_value('feature33', 8.71, None)
        self.drop_feature_outliers_by_value('feature34', None, None)
        self.drop_feature_outliers_by_value('feature35', None, None)
        self.drop_feature_outliers_by_value('feature36', None, -7.73)
        self.drop_feature_outliers_by_value('feature37', 21, None)
        self.drop_feature_outliers_by_value('feature38', None, None)
        self.drop_feature_outliers_by_value('feature39', None, -15.63)
        self.drop_feature_outliers_by_value('feature40', 4.86, None)
        self.drop_feature_outliers_by_value('feature41', None, None)
        self.drop_feature_outliers_by_value('feature42', 4.39, None)
        self.drop_feature_outliers_by_value('feature43', 12.28, None)
        self.drop_feature_outliers_by_value('feature44', None, None)
        self.drop_feature_outliers_by_value('feature45', 23.25, None)
        self.drop_feature_outliers_by_value('feature46', 32.78, None)
        self.drop_feature_outliers_by_value('feature47', 5.63, None)
        self.drop_feature_outliers_by_value('feature48', None, None)
        self.drop_feature_outliers_by_value('feature49', None, None)
        self.drop_feature_outliers_by_value('feature50', 10.99, None)
        self.drop_feature_outliers_by_value('feature51', None, None)
        self.drop_feature_outliers_by_value('feature52', 6.44, None)
        self.drop_feature_outliers_by_value('feature53', 5.61, None)
        self.drop_feature_outliers_by_value('feature54', None, None)
        self.drop_feature_outliers_by_value('feature55', None, None)
        self.drop_feature_outliers_by_value('feature56', 6.75, None)
        self.drop_feature_outliers_by_value('feature57', 22.74, None)
        self.drop_feature_outliers_by_value('feature58', 11.39, None)
        self.drop_feature_outliers_by_value('feature59', 20.66, None)
        self.drop_feature_outliers_by_value('feature60', 31.25, None)
        self.drop_feature_outliers_by_value('feature61', 28.75, None)
        self.drop_feature_outliers_by_value('feature62', 14.06, None)
        self.drop_feature_outliers_by_value('feature63', None, None)
        self.drop_feature_outliers_by_value('feature64', 22.9, None)
        self.drop_feature_outliers_by_value('feature65', 7.01, None)
        self.drop_feature_outliers_by_value('feature66', None, None)
        self.drop_feature_outliers_by_value('feature67', 4.73, None)
        self.drop_feature_outliers_by_value('feature68', 4.23, None)
        self.drop_feature_outliers_by_value('feature69', None, None)
        self.drop_feature_outliers_by_value('feature70', 4.76, None)
        self.drop_feature_outliers_by_value('feature71', 14.09, -11.58)
        self.drop_feature_outliers_by_value('feature72', None, None)
        self.drop_feature_outliers_by_value('feature73', 10.53, None)
        self.drop_feature_outliers_by_value('feature74', None, None)
        self.drop_feature_outliers_by_value('feature75', 5.83, None)
        self.drop_feature_outliers_by_value('feature76', None, None)
        self.drop_feature_outliers_by_value('feature77', None, None)
        self.drop_feature_outliers_by_value('feature78', 7.66, None)
        self.drop_feature_outliers_by_value('feature79', 22.3, None)
        self.drop_feature_outliers_by_value('feature80', None, None)
        self.drop_feature_outliers_by_value('feature81', 25.32, None)
        self.drop_feature_outliers_by_value('feature82', 13, None)
        self.drop_feature_outliers_by_value('feature83', None, None)
        self.drop_feature_outliers_by_value('feature84', 4.48, None)
        self.drop_feature_outliers_by_value('feature85', 16.17, None)
        self.drop_feature_outliers_by_value('feature86', 20.76, None)
        self.drop_feature_outliers_by_value('feature87', None, None)

    # Drop Outliers by Quantile
    def drop_outliers_by_quantile(self):

        print('------------------------------------------------------')
        print('Dropping Outliers by Quantile...')

        # for i in range(self.x_train.shape[1]):
        #     if i != 77:
        #         self.drop_outliers_by_quantile('feature' + str(i), 0.9995, 0.0005, 0.9995, 0.0005)

        self.drop_feature_outliers_by_quantile('feature0', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature1', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature2', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature3', None, None)
        self.drop_feature_outliers_by_quantile('feature4', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature5', None, None)
        self.drop_feature_outliers_by_quantile('feature6', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature7', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature8', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature9', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature10', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature11', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature12', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature13', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature14', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature15', None, None)
        self.drop_feature_outliers_by_quantile('feature16', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature17', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature18', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature19', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature20', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature21', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature22', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature23', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature24', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature25', None, None)
        self.drop_feature_outliers_by_quantile('feature26', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature27', None, None)
        self.drop_feature_outliers_by_quantile('feature28', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature29', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature30', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature31', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature32', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature33', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature34', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature35', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature36', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature37', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature38', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature39', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature40', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature41', None, None)
        self.drop_feature_outliers_by_quantile('feature42', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature43', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature44', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature45', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature46', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature47', None, None)
        self.drop_feature_outliers_by_quantile('feature48', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature49', None, None)
        self.drop_feature_outliers_by_quantile('feature50', None, None)
        self.drop_feature_outliers_by_quantile('feature51', None, None)
        self.drop_feature_outliers_by_quantile('feature52', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature53', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature54', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature55', None, None)
        self.drop_feature_outliers_by_quantile('feature56', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature57', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature58', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature59', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature60', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature61', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature62', None, 0.0001)
        self.drop_feature_outliers_by_quantile('feature63', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature64', None, None)
        self.drop_feature_outliers_by_quantile('feature65', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature66', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature67', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature68', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature69', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature70', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature71', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature72', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature73', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature74', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature75', None, None)
        self.drop_feature_outliers_by_quantile('feature76', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature77', None, None)
        self.drop_feature_outliers_by_quantile('feature78', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature79', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature80', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature81', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature82', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature83', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature84', 0.9999, 0.0001)
        self.drop_feature_outliers_by_quantile('feature85', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature86', 0.9999, None)
        self.drop_feature_outliers_by_quantile('feature87', 0.9999, None)

    # Standard Scale
    def standard_scale(self):

        print('------------------------------------------------------')
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

        print('------------------------------------------------------')
        print('Min-Max Scaling Data...')

        for each in self.x_train.columns:
            x_max, x_min = self.x_train[each].max(),  self.x_train[each].min()
            self.x_train.loc[:, each] = (self.x_train[each] - x_min)/(x_max - x_min)

        for each in self.x_test.columns:
            x_max, x_min = self.x_test[each].max(), self.x_test[each].min()
            self.x_test.loc[:, each] = (self.x_test[each] - x_min)/(x_max - x_min)

    # Convert pandas DataFrames to numpy arrays
    def convert_pd_to_np(self):

        print('------------------------------------------------------')
        print('Converting pandas DataFrames to numpy arrays...')

        self.x_train = np.array(self.x_train, dtype=np.float64)
        self.y_train = np.array(self.y_train, dtype=np.float64)
        self.w_train = np.array(self.w_train, dtype=np.float64)
        self.e_train = np.array(self.e_train, dtype=int)
        self.code_id_train = np.array(self.code_id_train, dtype=int)
        self.x_test = np.array(self.x_test, dtype=np.float64)
        self.id_test = np.array(self.id_test, dtype=int)
        self.code_id_test = np.array(self.code_id_test, dtype=int)

    # Add Polynomial Features
    def add_polynomial_features(self):

        print('------------------------------------------------------')
        print('Adding Polynomial Features...')

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

        self.x_train = poly.fit_transform(self.x_train)
        self.x_test = poly.fit_transform(self.x_test)

    # Convert Column 'group' to Dummies
    def convert_group_to_dummies(self):

        print('------------------------------------------------------')
        print('Converting Groups of Train Set to Dummies...')

        if self.use_multi_group:
            group1_train_dummies = np.array(pd.get_dummies(self.g1_train, prefix='group1'))
            group2_train_dummies = np.array(pd.get_dummies(self.g2_train, prefix='group2'))
            group_train_dummies = np.concatenate((group1_train_dummies, group2_train_dummies), axis=1)

            # TODO: Add zero group
            group_train_dummies = np.column_stack((group_train_dummies, np.zeros_like(self.g1_train)))

            self.x_g_train = np.column_stack((self.x_train, np.array(self.g1_train)))
            self.x_g_train = np.column_stack((self.x_g_train, np.array(self.g2_train)))
        else:
            group_train_dummies = np.array(pd.get_dummies(self.g_train, prefix='group'))
            self.x_g_train = np.column_stack((self.x_train, np.array(self.g_train)))

        if self.x_train.shape[1] > 500:
            print('So Many Features!')
            self.x_train = list(self.x_train)
            for i in range(len(self.x_train)):
                self.x_train[i] = np.concatenate((self.x_train[i], group_train_dummies[i]))
            self.x_train = np.array(self.x_train, dtype=np.float64)
        else:
            self.x_train = np.concatenate((self.x_train, group_train_dummies), axis=1)

        print('------------------------------------------------------')
        print('Converting Groups of Test Set to Dummies...')

        if self.use_multi_group:
            group1_test_dummies = np.array(pd.get_dummies(self.g1_test, prefix='group1'))
            group2_test_dummies = np.array(pd.get_dummies(self.g2_test, prefix='group2'))
            group_test_dummies = np.concatenate((group1_test_dummies, group2_test_dummies), axis=1)
            self.x_g_test = np.column_stack((self.x_test, np.array(self.g1_test)))
            self.x_g_test = np.column_stack((self.x_g_test, np.array(self.g2_test)))
        else:
            group_test_dummies = np.array(pd.get_dummies(self.g_test, prefix='group'))
            self.x_g_test = np.column_stack((self.x_test, np.array(self.g_test)))

        if self.x_test.shape[1] > 500:
            print('So Many Features!')
            self.x_test = list(self.x_test)
            for i in range(len(self.x_test)):
                self.x_test[i] = np.concatenate((self.x_test[i], group_test_dummies[i]))
            self.x_test = np.array(self.x_test, dtype=np.float64)
        else:
            self.x_test = np.concatenate((self.x_test, group_test_dummies), axis=1)

    # Split Adversarial Validation Set by GAN
    def split_data_by_gan(self, load_pickle=True, sample_ratio=None, sample_by_era=True, generate_mode='valid'):

        print('------------------------------------------------------')
        print('Splitting Adversarial Validation Set by GAN...')

        if load_pickle:
            similarity_prob = utils.load_pkl_to_data(gan_prob_path + 'similarity_prob.p')
        else:
            similarity_prob = \
                GenerateValidation.train(train_path=train_csv_path, test_path=test_csv_path, global_epochs=1,
                                         similarity_prob_path=gan_prob_path, return_similarity_prob=True,
                                         load_preprocessed_data=True)

        valid_idx = []
        train_idx = []

        if sample_by_era:

            similarity_prob_e = []
            index_e = []
            similarity_prob_all = []
            index_all = []
            era_tag = 0
            era_all = [era_tag]

            for idx, era in enumerate(self.e_train):

                if idx == len(self.e_train) - 1:
                    similarity_prob_e.append(similarity_prob[idx])
                    index_e.append(idx)
                    similarity_prob_all.append(similarity_prob_e)
                    index_all.append(index_e)
                elif era_tag == era:
                    similarity_prob_e.append(similarity_prob[idx])
                    index_e.append(idx)
                else:
                    era_tag = era
                    era_all.append(era)
                    similarity_prob_all.append(similarity_prob_e)
                    index_all.append(index_e)
                    similarity_prob_e = [similarity_prob[idx]]
                    index_e = [idx]

            for e, similarity_prob_e in enumerate(similarity_prob_all):

                n_sample_e = int(len(similarity_prob_e) * sample_ratio)
                most_similar_idx_e = np.argsort(similarity_prob_e)[:, :-(n_sample_e+1):-1]
                least_similar_idx_e = np.argsort(similarity_prob_e)[:, :len(similarity_prob_e)-n_sample_e]

                if generate_mode == 'valid':
                    valid_idx += list(index_all[e][most_similar_idx_e])
                    train_idx += list(index_all[e][least_similar_idx_e])
                elif generate_mode == 'train':
                    train_idx += list(index_all[e][most_similar_idx_e])
                    valid_idx += list(index_all[e][least_similar_idx_e])
                else:
                    raise ValueError("Wrong 'generate_mode'!")
        else:

            n_sample = int(len(similarity_prob) * sample_ratio)
            most_similar_idx = np.argsort(similarity_prob)[:, :-(n_sample + 1):-1]
            least_similar_idx = np.argsort(similarity_prob)[:, :len(similarity_prob) - n_sample]

            if generate_mode == 'valid':
                valid_idx = most_similar_idx
                train_idx = least_similar_idx
            elif generate_mode == 'train':
                train_idx = least_similar_idx
                valid_idx = most_similar_idx
            else:
                raise ValueError("Wrong 'generate_mode'!")

        # Generate Validation Set
        self.x_valid = self.x_train[valid_idx]
        self.x_g_valid = self.x_g_train[valid_idx]
        self.y_valid = self.x_train[valid_idx]

        # Generate Training Set
        self.x_train = self.x_train[train_idx]
        self.x_g_train = self.x_g_train[train_idx]
        self.y_train = self.y_train[train_idx]
        self.w_train = self.w_train[train_idx]
        self.e_train = self.e_train[train_idx]

        # Save Adversarial Validation Set
        print('Saving Adversarial Validation Set...')
        utils.save_data_to_pkl(self.x_valid, self.preprocess_path + 'x_valid.p')
        utils.save_data_to_pkl(self.x_g_valid, self.preprocess_path + 'x_g_valid.p')
        utils.save_data_to_pkl(self.y_valid, self.preprocess_path + 'y_valid.p')

    # Split Positive and Negative Era Set
    def split_data_by_era_distribution(self):

        print('------------------------------------------------------')
        print('Splitting Positive and Negative Era Set...')
        print('Negative Eras: ', negative_era_list)

        positive_index = []
        negative_index = []

        for i, ele in enumerate(self.e_train):

            if int(ele) in negative_era_list:
                negative_index.append(i)
            else:
                positive_index.append(i)

        # Positive Data
        self.x_train_p = self.x_train[positive_index]
        self.x_g_train_p = self.x_g_train[positive_index]
        self.y_train_p = self.y_train[positive_index]
        self.w_train_p = self.w_train[positive_index]
        self.e_train_p = self.e_train[positive_index]

        # Negative Data
        self.x_train_n = self.x_train[negative_index]
        self.x_g_train_n = self.x_g_train[negative_index]
        self.y_train_n = self.y_train[negative_index]
        self.w_train_n = self.w_train[negative_index]
        self.e_train_n = self.e_train[negative_index]

    # Save Data
    def save_data(self):

        print('------------------------------------------------------')
        print('Saving Preprocessed Data...')
        utils.save_data_to_pkl(self.x_train, self.preprocess_path + 'x_train.p')
        utils.save_data_to_pkl(self.x_g_train, self.preprocess_path + 'x_g_train.p')
        utils.save_data_to_pkl(self.y_train, self.preprocess_path + 'y_train.p')
        utils.save_data_to_pkl(self.w_train, self.preprocess_path + 'w_train.p')
        utils.save_data_to_pkl(self.e_train, self.preprocess_path + 'e_train.p')
        utils.save_data_to_pkl(self.code_id_train, self.preprocess_path + 'code_id_train.p')
        utils.save_data_to_pkl(self.x_test, self.preprocess_path + 'x_test.p')
        utils.save_data_to_pkl(self.x_g_test, self.preprocess_path + 'x_g_test.p')
        utils.save_data_to_pkl(self.id_test, self.preprocess_path + 'id_test.p')
        utils.save_data_to_pkl(self.code_id_test, self.preprocess_path + 'code_id_test.p')

    # Save Data Split by Era Distribution
    def save_data_by_era_distribution_pd(self):

        print('------------------------------------------------------')
        print('Saving Preprocessed Data Split by Era Distribution...')

        # Positive Data
        print('Saving Positive Data...')
        utils.save_data_to_pkl(self.x_train_p, self.preprocess_path + 'x_train_p.p')
        utils.save_data_to_pkl(self.x_g_train_p, self.preprocess_path + 'x_g_train_p.p')
        utils.save_data_to_pkl(self.y_train_p, self.preprocess_path + 'y_train_p.p')
        utils.save_data_to_pkl(self.w_train_p, self.preprocess_path + 'w_train_p.p')
        utils.save_data_to_pkl(self.e_train_p, self.preprocess_path + 'e_train_p.p')

        # Negative Data
        print('Saving Negative Data...')
        utils.save_data_to_pkl(self.x_train_n, self.preprocess_path + 'x_train_n.p')
        utils.save_data_to_pkl(self.x_g_train_n, self.preprocess_path + 'x_g_train_n.p')
        utils.save_data_to_pkl(self.y_train_n, self.preprocess_path + 'y_train_n.p')
        utils.save_data_to_pkl(self.w_train_n, self.preprocess_path + 'w_train_n.p')
        utils.save_data_to_pkl(self.e_train_n, self.preprocess_path + 'e_train_n.p')

    # Preprocess
    def preprocess(self):

        print('======================================================')
        print('Start Preprocessing...')

        start_time = time.time()

        # Load original data
        self.load_data()

        # Merge Eras
        self.merge_eras()

        # Drop outliers
        # self.drop_outliers_by_value()
        # self.drop_outliers_by_quantile()

        # Scale features
        # self.standard_scale()
        # self.min_max_scale()

        # Convert pandas DataFrames to numpy arrays
        self.convert_pd_to_np()

        # Add Polynomial Features
        # self.add_polynomial_features()

        # Convert column 'group' to dummies
        self.convert_group_to_dummies()

        # Split Adversarial Validation Set by GAN
        # self.split_data_by_gan(sample_ratio=0.2, sample_by_era=True, generate_mode='valid')

        # Split Positive and Negative Era Set
        # self.split_data_by_era_distribution()

        # Save Data to pickle files
        self.save_data()

        # Save Data Split by Era Distribution
        # self.save_data_by_era_distribution_pd()

        end_time = time.time()

        print('======================================================')
        print('Done!')
        print('Using {:.3}s'.format(end_time - start_time))
        print('======================================================')


if __name__ == '__main__':

    utils.check_dir(['./data/', preprocessed_path])
    DPP = DataPreProcess(train_csv_path, test_csv_path, preprocessed_path, use_multi_group=True)
    DPP.preprocess()
