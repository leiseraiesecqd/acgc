import time
import utils
import os
import numpy as np
import pandas as pd
from os.path import isdir
from sklearn.model_selection import StratifiedShuffleSplit


train_csv_path = './inputs/stock_train_data_20170916.csv'
test_csv_path = './inputs/stock_test_data_20170916.csv'
preprocessed_path = './preprocessed_data/'


class DataPreProcess:

    x_train = np.array([])
    y_train = np.array([])
    w_train = np.array([])
    g_train = np.array([])
    e_train = np.array([])
    x_test = np.array([])
    g_test = np.array([])
    id_test = np.array([])

    def __init__(self, train_path, test_path, prepro_path):

        self.train_path = train_path
        self.test_path = test_path
        self.prepro_path = prepro_path

    # Load CSV files
    def load_csv_np(self):

        train_f = np.loadtxt(self.train_path, dtype=np.float64, skiprows=1, delimiter=",")
        test_f = np.loadtxt(self.test_path, dtype=np.float64, skiprows=1, delimiter=",")

        return train_f, test_f

    # Load data
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

    # Load CSV files Using pandas
    def load_csv_pd(self):

        train_f = pd.read_csv(self.train_path, header=0)
        test_f = pd.read_csv(self.test_path, header=0)

        for feat in train_f.columns:
            train_f[feat] = train_f[feat].map(lambda x: round(x, 6))

        for feat in test_f.columns:
            test_f[feat] = test_f[feat].map(lambda x: round(x, 6))

        return train_f, test_f

    # Load data using pandas
    def load_data_pd(self):

        try:
            print('Loading data...')
            train_f, test_f = self.load_csv_pd()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        # Drop unnecessary columns
        self.x_train = train_f.drop(['id', 'weight', 'label', 'group', 'era'], axis=1)
        self.y_train = train_f['label']
        self.w_train = train_f['weight']
        self.g_train = train_f['group']
        self.e_train = train_f['era']
        self.x_test = test_f.drop(['id', 'group'], axis=1)
        self.id_test = test_f['id']
        self.g_test = test_f['group']

    # Dropping outliers
    def drop_outliers(self):
        
        print('Dropping outliers...')
        
        # Drop upper outlines in self.x_train
        upper = self.x_train.feature0.quantile(0.9999)
        self.x_train['feature0'].loc[self.x_train['feature0'] > upper] = upper
        upper = self.x_train.feature1.quantile(0.9999)
        self.x_train['feature1'].loc[self.x_train['feature1'] > upper] = upper
        upper = self.x_train.feature2.quantile(0.9999)
        self.x_train['feature2'].loc[self.x_train['feature2'] > upper] = upper
        upper = self.x_train.feature3.quantile(0.9999)
        self.x_train['feature3'].loc[self.x_train['feature3'] > upper] = upper
        upper = self.x_train.feature4.quantile(0.9999)
        self.x_train['feature4'].loc[self.x_train['feature4'] > upper] = upper
        upper = self.x_train.feature5.quantile(0.9999)
        self.x_train['feature5'].loc[self.x_train['feature5'] > upper] = upper
        upper = self.x_train.feature6.quantile(0.9999)
        self.x_train['feature6'].loc[self.x_train['feature6'] > upper] = upper
        upper = self.x_train.feature7.quantile(0.9999)
        self.x_train['feature7'].loc[self.x_train['feature7'] > upper] = upper
        upper = self.x_train.feature8.quantile(0.9999)
        self.x_train['feature8'].loc[self.x_train['feature8'] > upper] = upper
        upper = self.x_train.feature9.quantile(0.9999)
        self.x_train['feature9'].loc[self.x_train['feature9'] > upper] = upper
        upper = self.x_train.feature10.quantile(0.9999)
        self.x_train['feature10'].loc[self.x_train['feature10'] > upper] = upper
        upper = self.x_train.feature11.quantile(0.9999)
        self.x_train['feature11'].loc[self.x_train['feature11'] > upper] = upper
        upper = self.x_train.feature12.quantile(0.9999)
        self.x_train['feature12'].loc[self.x_train['feature12'] > upper] = upper
        upper = self.x_train.feature13.quantile(0.9999)
        self.x_train['feature13'].loc[self.x_train['feature13'] > upper] = upper
        upper = self.x_train.feature14.quantile(0.9999)
        self.x_train['feature14'].loc[self.x_train['feature14'] > upper] = upper
        upper = self.x_train.feature15.quantile(0.9999)
        self.x_train['feature15'].loc[self.x_train['feature15'] > upper] = upper
        upper = self.x_train.feature16.quantile(0.9999)
        self.x_train['feature16'].loc[self.x_train['feature16'] > upper] = upper
        upper = self.x_train.feature17.quantile(0.9999)
        self.x_train['feature17'].loc[self.x_train['feature17'] > upper] = upper
        upper = self.x_train.feature18.quantile(0.9999)
        self.x_train['feature18'].loc[self.x_train['feature18'] > upper] = upper
        upper = self.x_train.feature19.quantile(0.9999)
        self.x_train['feature19'].loc[self.x_train['feature19'] > upper] = upper
        upper = self.x_train.feature20.quantile(0.9999)
        self.x_train['feature20'].loc[self.x_train['feature20'] > upper] = upper
        upper = self.x_train.feature21.quantile(0.9999)
        self.x_train['feature21'].loc[self.x_train['feature21'] > upper] = upper
        upper = self.x_train.feature22.quantile(0.9999)
        self.x_train['feature22'].loc[self.x_train['feature22'] > upper] = upper
        upper = self.x_train.feature23.quantile(0.9999)
        self.x_train['feature23'].loc[self.x_train['feature23'] > upper] = upper
        upper = self.x_train.feature24.quantile(0.9999)
        self.x_train['feature24'].loc[self.x_train['feature24'] > upper] = upper
        upper = self.x_train.feature25.quantile(0.9999)
        self.x_train['feature25'].loc[self.x_train['feature25'] > upper] = upper
        upper = self.x_train.feature26.quantile(0.9999)
        self.x_train['feature26'].loc[self.x_train['feature26'] > upper] = upper
        upper = self.x_train.feature27.quantile(0.9999)
        self.x_train['feature27'].loc[self.x_train['feature27'] > upper] = upper
        upper = self.x_train.feature28.quantile(0.9999)
        self.x_train['feature28'].loc[self.x_train['feature28'] > upper] = upper
        upper = self.x_train.feature29.quantile(0.9999)
        self.x_train['feature29'].loc[self.x_train['feature29'] > upper] = upper
        upper = self.x_train.feature30.quantile(0.9999)
        self.x_train['feature30'].loc[self.x_train['feature30'] > upper] = upper
        upper = self.x_train.feature31.quantile(0.9999)
        self.x_train['feature31'].loc[self.x_train['feature31'] > upper] = upper
        upper = self.x_train.feature32.quantile(0.9999)
        self.x_train['feature32'].loc[self.x_train['feature32'] > upper] = upper
        upper = self.x_train.feature33.quantile(0.9999)
        self.x_train['feature33'].loc[self.x_train['feature33'] > upper] = upper
        upper = self.x_train.feature34.quantile(0.9999)
        self.x_train['feature34'].loc[self.x_train['feature34'] > upper] = upper
        upper = self.x_train.feature35.quantile(0.9999)
        self.x_train['feature35'].loc[self.x_train['feature35'] > upper] = upper
        upper = self.x_train.feature36.quantile(0.9999)
        self.x_train['feature36'].loc[self.x_train['feature36'] > upper] = upper
        upper = self.x_train.feature37.quantile(0.9999)
        self.x_train['feature37'].loc[self.x_train['feature37'] > upper] = upper
        upper = self.x_train.feature38.quantile(0.9999)
        self.x_train['feature38'].loc[self.x_train['feature38'] > upper] = upper
        upper = self.x_train.feature39.quantile(0.9999)
        self.x_train['feature39'].loc[self.x_train['feature39'] > upper] = upper
        upper = self.x_train.feature40.quantile(0.9999)
        self.x_train['feature40'].loc[self.x_train['feature40'] > upper] = upper
        upper = self.x_train.feature41.quantile(0.9999)
        self.x_train['feature41'].loc[self.x_train['feature41'] > upper] = upper
        upper = self.x_train.feature42.quantile(0.9999)
        self.x_train['feature42'].loc[self.x_train['feature42'] > upper] = upper
        upper = self.x_train.feature43.quantile(0.9999)
        self.x_train['feature43'].loc[self.x_train['feature43'] > upper] = upper
        upper = self.x_train.feature44.quantile(0.9999)
        self.x_train['feature44'].loc[self.x_train['feature44'] > upper] = upper
        upper = self.x_train.feature45.quantile(0.9999)
        self.x_train['feature45'].loc[self.x_train['feature45'] > upper] = upper
        upper = self.x_train.feature46.quantile(0.9999)
        self.x_train['feature46'].loc[self.x_train['feature46'] > upper] = upper
        upper = self.x_train.feature47.quantile(0.9999)
        self.x_train['feature47'].loc[self.x_train['feature47'] > upper] = upper
        upper = self.x_train.feature48.quantile(0.9999)
        self.x_train['feature48'].loc[self.x_train['feature48'] > upper] = upper
        upper = self.x_train.feature49.quantile(0.9999)
        self.x_train['feature49'].loc[self.x_train['feature49'] > upper] = upper
        upper = self.x_train.feature50.quantile(0.9999)
        self.x_train['feature50'].loc[self.x_train['feature50'] > upper] = upper
        upper = self.x_train.feature51.quantile(0.9999)
        self.x_train['feature51'].loc[self.x_train['feature51'] > upper] = upper
        upper = self.x_train.feature52.quantile(0.9999)
        self.x_train['feature52'].loc[self.x_train['feature52'] > upper] = upper
        upper = self.x_train.feature53.quantile(0.9999)
        self.x_train['feature53'].loc[self.x_train['feature53'] > upper] = upper
        upper = self.x_train.feature54.quantile(0.9999)
        self.x_train['feature54'].loc[self.x_train['feature54'] > upper] = upper
        upper = self.x_train.feature55.quantile(0.9999)
        self.x_train['feature55'].loc[self.x_train['feature55'] > upper] = upper
        upper = self.x_train.feature56.quantile(0.9999)
        self.x_train['feature56'].loc[self.x_train['feature56'] > upper] = upper
        upper = self.x_train.feature57.quantile(0.9999)
        self.x_train['feature57'].loc[self.x_train['feature57'] > upper] = upper
        upper = self.x_train.feature58.quantile(0.9999)
        self.x_train['feature58'].loc[self.x_train['feature58'] > upper] = upper
        upper = self.x_train.feature59.quantile(0.9999)
        self.x_train['feature59'].loc[self.x_train['feature59'] > upper] = upper
        upper = self.x_train.feature60.quantile(0.9999)
        self.x_train['feature60'].loc[self.x_train['feature60'] > upper] = upper
        upper = self.x_train.feature61.quantile(0.9999)
        self.x_train['feature61'].loc[self.x_train['feature61'] > upper] = upper
        upper = self.x_train.feature62.quantile(0.9999)
        self.x_train['feature62'].loc[self.x_train['feature62'] > upper] = upper
        upper = self.x_train.feature63.quantile(0.9999)
        self.x_train['feature63'].loc[self.x_train['feature63'] > upper] = upper
        upper = self.x_train.feature64.quantile(0.9999)
        self.x_train['feature64'].loc[self.x_train['feature64'] > upper] = upper
        upper = self.x_train.feature65.quantile(0.9999)
        self.x_train['feature65'].loc[self.x_train['feature65'] > upper] = upper
        upper = self.x_train.feature66.quantile(0.9999)
        self.x_train['feature66'].loc[self.x_train['feature66'] > upper] = upper
        upper = self.x_train.feature67.quantile(0.9999)
        self.x_train['feature67'].loc[self.x_train['feature67'] > upper] = upper
        upper = self.x_train.feature68.quantile(0.9999)
        self.x_train['feature68'].loc[self.x_train['feature68'] > upper] = upper
        upper = self.x_train.feature69.quantile(0.9999)
        self.x_train['feature69'].loc[self.x_train['feature69'] > upper] = upper
        upper = self.x_train.feature70.quantile(0.9999)
        self.x_train['feature70'].loc[self.x_train['feature70'] > upper] = upper
        upper = self.x_train.feature71.quantile(0.9999)
        self.x_train['feature71'].loc[self.x_train['feature71'] > upper] = upper
        upper = self.x_train.feature72.quantile(0.9999)
        self.x_train['feature72'].loc[self.x_train['feature72'] > upper] = upper
        upper = self.x_train.feature73.quantile(0.9999)
        self.x_train['feature73'].loc[self.x_train['feature73'] > upper] = upper
        upper = self.x_train.feature74.quantile(0.9999)
        self.x_train['feature74'].loc[self.x_train['feature74'] > upper] = upper
        upper = self.x_train.feature75.quantile(0.9999)
        self.x_train['feature75'].loc[self.x_train['feature75'] > upper] = upper
        upper = self.x_train.feature76.quantile(0.9999)
        self.x_train['feature76'].loc[self.x_train['feature76'] > upper] = upper
        upper = self.x_train.feature77.quantile(0.9999)
        self.x_train['feature77'].loc[self.x_train['feature77'] > upper] = upper
        upper = self.x_train.feature78.quantile(0.9999)
        self.x_train['feature78'].loc[self.x_train['feature78'] > upper] = upper
        upper = self.x_train.feature79.quantile(0.9999)
        self.x_train['feature79'].loc[self.x_train['feature79'] > upper] = upper
        upper = self.x_train.feature80.quantile(0.9999)
        self.x_train['feature80'].loc[self.x_train['feature80'] > upper] = upper
        upper = self.x_train.feature81.quantile(0.9999)
        self.x_train['feature81'].loc[self.x_train['feature81'] > upper] = upper
        upper = self.x_train.feature82.quantile(0.9999)
        self.x_train['feature82'].loc[self.x_train['feature82'] > upper] = upper
        upper = self.x_train.feature83.quantile(0.9999)
        self.x_train['feature83'].loc[self.x_train['feature83'] > upper] = upper
        upper = self.x_train.feature84.quantile(0.9999)
        self.x_train['feature84'].loc[self.x_train['feature84'] > upper] = upper
        upper = self.x_train.feature85.quantile(0.9999)
        self.x_train['feature85'].loc[self.x_train['feature85'] > upper] = upper
        upper = self.x_train.feature86.quantile(0.9999)
        self.x_train['feature86'].loc[self.x_train['feature86'] > upper] = upper
        upper = self.x_train.feature87.quantile(0.9999)
        self.x_train['feature87'].loc[self.x_train['feature87'] > upper] = upper

        # Drop lower outlines in self.x_train
        lower = self.x_train.feature0.quantile(0.0001)
        self.x_train['feature0'].loc[self.x_train['feature0'] < lower] = lower
        lower = self.x_train.feature1.quantile(0.0001)
        self.x_train['feature1'].loc[self.x_train['feature1'] < lower] = lower
        lower = self.x_train.feature2.quantile(0.0001)
        self.x_train['feature2'].loc[self.x_train['feature2'] < lower] = lower
        lower = self.x_train.feature3.quantile(0.0001)
        self.x_train['feature3'].loc[self.x_train['feature3'] < lower] = lower
        lower = self.x_train.feature4.quantile(0.0001)
        self.x_train['feature4'].loc[self.x_train['feature4'] < lower] = lower
        lower = self.x_train.feature5.quantile(0.0001)
        self.x_train['feature5'].loc[self.x_train['feature5'] < lower] = lower
        lower = self.x_train.feature6.quantile(0.0001)
        self.x_train['feature6'].loc[self.x_train['feature6'] < lower] = lower
        lower = self.x_train.feature7.quantile(0.0001)
        self.x_train['feature7'].loc[self.x_train['feature7'] < lower] = lower
        lower = self.x_train.feature8.quantile(0.0001)
        self.x_train['feature8'].loc[self.x_train['feature8'] < lower] = lower
        lower = self.x_train.feature9.quantile(0.0001)
        self.x_train['feature9'].loc[self.x_train['feature9'] < lower] = lower
        lower = self.x_train.feature10.quantile(0.0001)
        self.x_train['feature10'].loc[self.x_train['feature10'] < lower] = lower
        lower = self.x_train.feature11.quantile(0.0001)
        self.x_train['feature11'].loc[self.x_train['feature11'] < lower] = lower
        lower = self.x_train.feature12.quantile(0.0001)
        self.x_train['feature12'].loc[self.x_train['feature12'] < lower] = lower
        lower = self.x_train.feature13.quantile(0.0001)
        self.x_train['feature13'].loc[self.x_train['feature13'] < lower] = lower
        lower = self.x_train.feature14.quantile(0.0001)
        self.x_train['feature14'].loc[self.x_train['feature14'] < lower] = lower
        lower = self.x_train.feature15.quantile(0.0001)
        self.x_train['feature15'].loc[self.x_train['feature15'] < lower] = lower
        lower = self.x_train.feature16.quantile(0.0001)
        self.x_train['feature16'].loc[self.x_train['feature16'] < lower] = lower
        lower = self.x_train.feature17.quantile(0.0001)
        self.x_train['feature17'].loc[self.x_train['feature17'] < lower] = lower
        lower = self.x_train.feature18.quantile(0.0001)
        self.x_train['feature18'].loc[self.x_train['feature18'] < lower] = lower
        lower = self.x_train.feature19.quantile(0.0001)
        self.x_train['feature19'].loc[self.x_train['feature19'] < lower] = lower
        lower = self.x_train.feature20.quantile(0.0001)
        self.x_train['feature20'].loc[self.x_train['feature20'] < lower] = lower
        lower = self.x_train.feature21.quantile(0.0001)
        self.x_train['feature21'].loc[self.x_train['feature21'] < lower] = lower
        lower = self.x_train.feature22.quantile(0.0001)
        self.x_train['feature22'].loc[self.x_train['feature22'] < lower] = lower
        lower = self.x_train.feature23.quantile(0.0001)
        self.x_train['feature23'].loc[self.x_train['feature23'] < lower] = lower
        lower = self.x_train.feature24.quantile(0.0001)
        self.x_train['feature24'].loc[self.x_train['feature24'] < lower] = lower
        lower = self.x_train.feature25.quantile(0.0001)
        self.x_train['feature25'].loc[self.x_train['feature25'] < lower] = lower
        lower = self.x_train.feature26.quantile(0.0001)
        self.x_train['feature26'].loc[self.x_train['feature26'] < lower] = lower
        lower = self.x_train.feature27.quantile(0.0001)
        self.x_train['feature27'].loc[self.x_train['feature27'] < lower] = lower
        lower = self.x_train.feature28.quantile(0.0001)
        self.x_train['feature28'].loc[self.x_train['feature28'] < lower] = lower
        lower = self.x_train.feature29.quantile(0.0001)
        self.x_train['feature29'].loc[self.x_train['feature29'] < lower] = lower
        lower = self.x_train.feature30.quantile(0.0001)
        self.x_train['feature30'].loc[self.x_train['feature30'] < lower] = lower
        lower = self.x_train.feature31.quantile(0.0001)
        self.x_train['feature31'].loc[self.x_train['feature31'] < lower] = lower
        lower = self.x_train.feature32.quantile(0.0001)
        self.x_train['feature32'].loc[self.x_train['feature32'] < lower] = lower
        lower = self.x_train.feature33.quantile(0.0001)
        self.x_train['feature33'].loc[self.x_train['feature33'] < lower] = lower
        lower = self.x_train.feature34.quantile(0.0001)
        self.x_train['feature34'].loc[self.x_train['feature34'] < lower] = lower
        lower = self.x_train.feature35.quantile(0.0001)
        self.x_train['feature35'].loc[self.x_train['feature35'] < lower] = lower
        lower = self.x_train.feature36.quantile(0.0001)
        self.x_train['feature36'].loc[self.x_train['feature36'] < lower] = lower
        lower = self.x_train.feature37.quantile(0.0001)
        self.x_train['feature37'].loc[self.x_train['feature37'] < lower] = lower
        lower = self.x_train.feature38.quantile(0.0001)
        self.x_train['feature38'].loc[self.x_train['feature38'] < lower] = lower
        lower = self.x_train.feature39.quantile(0.0001)
        self.x_train['feature39'].loc[self.x_train['feature39'] < lower] = lower
        lower = self.x_train.feature40.quantile(0.0001)
        self.x_train['feature40'].loc[self.x_train['feature40'] < lower] = lower
        lower = self.x_train.feature41.quantile(0.0001)
        self.x_train['feature41'].loc[self.x_train['feature41'] < lower] = lower
        lower = self.x_train.feature42.quantile(0.0001)
        self.x_train['feature42'].loc[self.x_train['feature42'] < lower] = lower
        lower = self.x_train.feature43.quantile(0.0001)
        self.x_train['feature43'].loc[self.x_train['feature43'] < lower] = lower
        lower = self.x_train.feature44.quantile(0.0001)
        self.x_train['feature44'].loc[self.x_train['feature44'] < lower] = lower
        lower = self.x_train.feature45.quantile(0.0001)
        self.x_train['feature45'].loc[self.x_train['feature45'] < lower] = lower
        lower = self.x_train.feature46.quantile(0.0001)
        self.x_train['feature46'].loc[self.x_train['feature46'] < lower] = lower
        lower = self.x_train.feature47.quantile(0.0001)
        self.x_train['feature47'].loc[self.x_train['feature47'] < lower] = lower
        lower = self.x_train.feature48.quantile(0.0001)
        self.x_train['feature48'].loc[self.x_train['feature48'] < lower] = lower
        lower = self.x_train.feature49.quantile(0.0001)
        self.x_train['feature49'].loc[self.x_train['feature49'] < lower] = lower
        lower = self.x_train.feature50.quantile(0.0001)
        self.x_train['feature50'].loc[self.x_train['feature50'] < lower] = lower
        lower = self.x_train.feature51.quantile(0.0001)
        self.x_train['feature51'].loc[self.x_train['feature51'] < lower] = lower
        lower = self.x_train.feature52.quantile(0.0001)
        self.x_train['feature52'].loc[self.x_train['feature52'] < lower] = lower
        lower = self.x_train.feature53.quantile(0.0001)
        self.x_train['feature53'].loc[self.x_train['feature53'] < lower] = lower
        lower = self.x_train.feature54.quantile(0.0001)
        self.x_train['feature54'].loc[self.x_train['feature54'] < lower] = lower
        lower = self.x_train.feature55.quantile(0.0001)
        self.x_train['feature55'].loc[self.x_train['feature55'] < lower] = lower
        lower = self.x_train.feature56.quantile(0.0001)
        self.x_train['feature56'].loc[self.x_train['feature56'] < lower] = lower
        lower = self.x_train.feature57.quantile(0.0001)
        self.x_train['feature57'].loc[self.x_train['feature57'] < lower] = lower
        lower = self.x_train.feature58.quantile(0.0001)
        self.x_train['feature58'].loc[self.x_train['feature58'] < lower] = lower
        lower = self.x_train.feature59.quantile(0.0001)
        self.x_train['feature59'].loc[self.x_train['feature59'] < lower] = lower
        lower = self.x_train.feature60.quantile(0.0001)
        self.x_train['feature60'].loc[self.x_train['feature60'] < lower] = lower
        lower = self.x_train.feature61.quantile(0.0001)
        self.x_train['feature61'].loc[self.x_train['feature61'] < lower] = lower
        lower = self.x_train.feature62.quantile(0.0001)
        self.x_train['feature62'].loc[self.x_train['feature62'] < lower] = lower
        lower = self.x_train.feature63.quantile(0.0001)
        self.x_train['feature63'].loc[self.x_train['feature63'] < lower] = lower
        lower = self.x_train.feature64.quantile(0.0001)
        self.x_train['feature64'].loc[self.x_train['feature64'] < lower] = lower
        lower = self.x_train.feature65.quantile(0.0001)
        self.x_train['feature65'].loc[self.x_train['feature65'] < lower] = lower
        lower = self.x_train.feature66.quantile(0.0001)
        self.x_train['feature66'].loc[self.x_train['feature66'] < lower] = lower
        lower = self.x_train.feature67.quantile(0.0001)
        self.x_train['feature67'].loc[self.x_train['feature67'] < lower] = lower
        lower = self.x_train.feature68.quantile(0.0001)
        self.x_train['feature68'].loc[self.x_train['feature68'] < lower] = lower
        lower = self.x_train.feature69.quantile(0.0001)
        self.x_train['feature69'].loc[self.x_train['feature69'] < lower] = lower
        lower = self.x_train.feature70.quantile(0.0001)
        self.x_train['feature70'].loc[self.x_train['feature70'] < lower] = lower
        lower = self.x_train.feature71.quantile(0.0001)
        self.x_train['feature71'].loc[self.x_train['feature71'] < lower] = lower
        lower = self.x_train.feature72.quantile(0.0001)
        self.x_train['feature72'].loc[self.x_train['feature72'] < lower] = lower
        lower = self.x_train.feature73.quantile(0.0001)
        self.x_train['feature73'].loc[self.x_train['feature73'] < lower] = lower
        lower = self.x_train.feature74.quantile(0.0001)
        self.x_train['feature74'].loc[self.x_train['feature74'] < lower] = lower
        lower = self.x_train.feature75.quantile(0.0001)
        self.x_train['feature75'].loc[self.x_train['feature75'] < lower] = lower
        lower = self.x_train.feature76.quantile(0.0001)
        self.x_train['feature76'].loc[self.x_train['feature76'] < lower] = lower
        lower = self.x_train.feature77.quantile(0.0001)
        self.x_train['feature77'].loc[self.x_train['feature77'] < lower] = lower
        lower = self.x_train.feature78.quantile(0.0001)
        self.x_train['feature78'].loc[self.x_train['feature78'] < lower] = lower
        lower = self.x_train.feature79.quantile(0.0001)
        self.x_train['feature79'].loc[self.x_train['feature79'] < lower] = lower
        lower = self.x_train.feature80.quantile(0.0001)
        self.x_train['feature80'].loc[self.x_train['feature80'] < lower] = lower
        lower = self.x_train.feature81.quantile(0.0001)
        self.x_train['feature81'].loc[self.x_train['feature81'] < lower] = lower
        lower = self.x_train.feature82.quantile(0.0001)
        self.x_train['feature82'].loc[self.x_train['feature82'] < lower] = lower
        lower = self.x_train.feature83.quantile(0.0001)
        self.x_train['feature83'].loc[self.x_train['feature83'] < lower] = lower
        lower = self.x_train.feature84.quantile(0.0001)
        self.x_train['feature84'].loc[self.x_train['feature84'] < lower] = lower
        lower = self.x_train.feature85.quantile(0.0001)
        self.x_train['feature85'].loc[self.x_train['feature85'] < lower] = lower
        lower = self.x_train.feature86.quantile(0.0001)
        self.x_train['feature86'].loc[self.x_train['feature86'] < lower] = lower
        lower = self.x_train.feature87.quantile(0.0001)
        self.x_train['feature87'].loc[self.x_train['feature87'] < lower] = lower

        # Drop upper outlines in self.x_test
        upper = self.x_test.feature0.quantile(0.9999)
        self.x_test['feature0'].loc[self.x_test['feature0'] > upper] = upper
        upper = self.x_test.feature1.quantile(0.9999)
        self.x_test['feature1'].loc[self.x_test['feature1'] > upper] = upper
        upper = self.x_test.feature2.quantile(0.9999)
        self.x_test['feature2'].loc[self.x_test['feature2'] > upper] = upper
        upper = self.x_test.feature3.quantile(0.9999)
        self.x_test['feature3'].loc[self.x_test['feature3'] > upper] = upper
        upper = self.x_test.feature4.quantile(0.9999)
        self.x_test['feature4'].loc[self.x_test['feature4'] > upper] = upper
        upper = self.x_test.feature5.quantile(0.9999)
        self.x_test['feature5'].loc[self.x_test['feature5'] > upper] = upper
        upper = self.x_test.feature6.quantile(0.9999)
        self.x_test['feature6'].loc[self.x_test['feature6'] > upper] = upper
        upper = self.x_test.feature7.quantile(0.9999)
        self.x_test['feature7'].loc[self.x_test['feature7'] > upper] = upper
        upper = self.x_test.feature8.quantile(0.9999)
        self.x_test['feature8'].loc[self.x_test['feature8'] > upper] = upper
        upper = self.x_test.feature9.quantile(0.9999)
        self.x_test['feature9'].loc[self.x_test['feature9'] > upper] = upper
        upper = self.x_test.feature10.quantile(0.9999)
        self.x_test['feature10'].loc[self.x_test['feature10'] > upper] = upper
        upper = self.x_test.feature11.quantile(0.9999)
        self.x_test['feature11'].loc[self.x_test['feature11'] > upper] = upper
        upper = self.x_test.feature12.quantile(0.9999)
        self.x_test['feature12'].loc[self.x_test['feature12'] > upper] = upper
        upper = self.x_test.feature13.quantile(0.9999)
        self.x_test['feature13'].loc[self.x_test['feature13'] > upper] = upper
        upper = self.x_test.feature14.quantile(0.9999)
        self.x_test['feature14'].loc[self.x_test['feature14'] > upper] = upper
        upper = self.x_test.feature15.quantile(0.9999)
        self.x_test['feature15'].loc[self.x_test['feature15'] > upper] = upper
        upper = self.x_test.feature16.quantile(0.9999)
        self.x_test['feature16'].loc[self.x_test['feature16'] > upper] = upper
        upper = self.x_test.feature17.quantile(0.9999)
        self.x_test['feature17'].loc[self.x_test['feature17'] > upper] = upper
        upper = self.x_test.feature18.quantile(0.9999)
        self.x_test['feature18'].loc[self.x_test['feature18'] > upper] = upper
        upper = self.x_test.feature19.quantile(0.9999)
        self.x_test['feature19'].loc[self.x_test['feature19'] > upper] = upper
        upper = self.x_test.feature20.quantile(0.9999)
        self.x_test['feature20'].loc[self.x_test['feature20'] > upper] = upper
        upper = self.x_test.feature21.quantile(0.9999)
        self.x_test['feature21'].loc[self.x_test['feature21'] > upper] = upper
        upper = self.x_test.feature22.quantile(0.9999)
        self.x_test['feature22'].loc[self.x_test['feature22'] > upper] = upper
        upper = self.x_test.feature23.quantile(0.9999)
        self.x_test['feature23'].loc[self.x_test['feature23'] > upper] = upper
        upper = self.x_test.feature24.quantile(0.9999)
        self.x_test['feature24'].loc[self.x_test['feature24'] > upper] = upper
        upper = self.x_test.feature25.quantile(0.9999)
        self.x_test['feature25'].loc[self.x_test['feature25'] > upper] = upper
        upper = self.x_test.feature26.quantile(0.9999)
        self.x_test['feature26'].loc[self.x_test['feature26'] > upper] = upper
        upper = self.x_test.feature27.quantile(0.9999)
        self.x_test['feature27'].loc[self.x_test['feature27'] > upper] = upper
        upper = self.x_test.feature28.quantile(0.9999)
        self.x_test['feature28'].loc[self.x_test['feature28'] > upper] = upper
        upper = self.x_test.feature29.quantile(0.9999)
        self.x_test['feature29'].loc[self.x_test['feature29'] > upper] = upper
        upper = self.x_test.feature30.quantile(0.9999)
        self.x_test['feature30'].loc[self.x_test['feature30'] > upper] = upper
        upper = self.x_test.feature31.quantile(0.9999)
        self.x_test['feature31'].loc[self.x_test['feature31'] > upper] = upper
        upper = self.x_test.feature32.quantile(0.9999)
        self.x_test['feature32'].loc[self.x_test['feature32'] > upper] = upper
        upper = self.x_test.feature33.quantile(0.9999)
        self.x_test['feature33'].loc[self.x_test['feature33'] > upper] = upper
        upper = self.x_test.feature34.quantile(0.9999)
        self.x_test['feature34'].loc[self.x_test['feature34'] > upper] = upper
        upper = self.x_test.feature35.quantile(0.9999)
        self.x_test['feature35'].loc[self.x_test['feature35'] > upper] = upper
        upper = self.x_test.feature36.quantile(0.9999)
        self.x_test['feature36'].loc[self.x_test['feature36'] > upper] = upper
        upper = self.x_test.feature37.quantile(0.9999)
        self.x_test['feature37'].loc[self.x_test['feature37'] > upper] = upper
        upper = self.x_test.feature38.quantile(0.9999)
        self.x_test['feature38'].loc[self.x_test['feature38'] > upper] = upper
        upper = self.x_test.feature39.quantile(0.9999)
        self.x_test['feature39'].loc[self.x_test['feature39'] > upper] = upper
        upper = self.x_test.feature40.quantile(0.9999)
        self.x_test['feature40'].loc[self.x_test['feature40'] > upper] = upper
        upper = self.x_test.feature41.quantile(0.9999)
        self.x_test['feature41'].loc[self.x_test['feature41'] > upper] = upper
        upper = self.x_test.feature42.quantile(0.9999)
        self.x_test['feature42'].loc[self.x_test['feature42'] > upper] = upper
        upper = self.x_test.feature43.quantile(0.9999)
        self.x_test['feature43'].loc[self.x_test['feature43'] > upper] = upper
        upper = self.x_test.feature44.quantile(0.9999)
        self.x_test['feature44'].loc[self.x_test['feature44'] > upper] = upper
        upper = self.x_test.feature45.quantile(0.9999)
        self.x_test['feature45'].loc[self.x_test['feature45'] > upper] = upper
        upper = self.x_test.feature46.quantile(0.9999)
        self.x_test['feature46'].loc[self.x_test['feature46'] > upper] = upper
        upper = self.x_test.feature47.quantile(0.9999)
        self.x_test['feature47'].loc[self.x_test['feature47'] > upper] = upper
        upper = self.x_test.feature48.quantile(0.9999)
        self.x_test['feature48'].loc[self.x_test['feature48'] > upper] = upper
        upper = self.x_test.feature49.quantile(0.9999)
        self.x_test['feature49'].loc[self.x_test['feature49'] > upper] = upper
        upper = self.x_test.feature50.quantile(0.9999)
        self.x_test['feature50'].loc[self.x_test['feature50'] > upper] = upper
        upper = self.x_test.feature51.quantile(0.9999)
        self.x_test['feature51'].loc[self.x_test['feature51'] > upper] = upper
        upper = self.x_test.feature52.quantile(0.9999)
        self.x_test['feature52'].loc[self.x_test['feature52'] > upper] = upper
        upper = self.x_test.feature53.quantile(0.9999)
        self.x_test['feature53'].loc[self.x_test['feature53'] > upper] = upper
        upper = self.x_test.feature54.quantile(0.9999)
        self.x_test['feature54'].loc[self.x_test['feature54'] > upper] = upper
        upper = self.x_test.feature55.quantile(0.9999)
        self.x_test['feature55'].loc[self.x_test['feature55'] > upper] = upper
        upper = self.x_test.feature56.quantile(0.9999)
        self.x_test['feature56'].loc[self.x_test['feature56'] > upper] = upper
        upper = self.x_test.feature57.quantile(0.9999)
        self.x_test['feature57'].loc[self.x_test['feature57'] > upper] = upper
        upper = self.x_test.feature58.quantile(0.9999)
        self.x_test['feature58'].loc[self.x_test['feature58'] > upper] = upper
        upper = self.x_test.feature59.quantile(0.9999)
        self.x_test['feature59'].loc[self.x_test['feature59'] > upper] = upper
        upper = self.x_test.feature60.quantile(0.9999)
        self.x_test['feature60'].loc[self.x_test['feature60'] > upper] = upper
        upper = self.x_test.feature61.quantile(0.9999)
        self.x_test['feature61'].loc[self.x_test['feature61'] > upper] = upper
        upper = self.x_test.feature62.quantile(0.9999)
        self.x_test['feature62'].loc[self.x_test['feature62'] > upper] = upper
        upper = self.x_test.feature63.quantile(0.9999)
        self.x_test['feature63'].loc[self.x_test['feature63'] > upper] = upper
        upper = self.x_test.feature64.quantile(0.9999)
        self.x_test['feature64'].loc[self.x_test['feature64'] > upper] = upper
        upper = self.x_test.feature65.quantile(0.9999)
        self.x_test['feature65'].loc[self.x_test['feature65'] > upper] = upper
        upper = self.x_test.feature66.quantile(0.9999)
        self.x_test['feature66'].loc[self.x_test['feature66'] > upper] = upper
        upper = self.x_test.feature67.quantile(0.9999)
        self.x_test['feature67'].loc[self.x_test['feature67'] > upper] = upper
        upper = self.x_test.feature68.quantile(0.9999)
        self.x_test['feature68'].loc[self.x_test['feature68'] > upper] = upper
        upper = self.x_test.feature69.quantile(0.9999)
        self.x_test['feature69'].loc[self.x_test['feature69'] > upper] = upper
        upper = self.x_test.feature70.quantile(0.9999)
        self.x_test['feature70'].loc[self.x_test['feature70'] > upper] = upper
        upper = self.x_test.feature71.quantile(0.9999)
        self.x_test['feature71'].loc[self.x_test['feature71'] > upper] = upper
        upper = self.x_test.feature72.quantile(0.9999)
        self.x_test['feature72'].loc[self.x_test['feature72'] > upper] = upper
        upper = self.x_test.feature73.quantile(0.9999)
        self.x_test['feature73'].loc[self.x_test['feature73'] > upper] = upper
        upper = self.x_test.feature74.quantile(0.9999)
        self.x_test['feature74'].loc[self.x_test['feature74'] > upper] = upper
        upper = self.x_test.feature75.quantile(0.9999)
        self.x_test['feature75'].loc[self.x_test['feature75'] > upper] = upper
        upper = self.x_test.feature76.quantile(0.9999)
        self.x_test['feature76'].loc[self.x_test['feature76'] > upper] = upper
        upper = self.x_test.feature77.quantile(0.9999)
        self.x_test['feature77'].loc[self.x_test['feature77'] > upper] = upper
        upper = self.x_test.feature78.quantile(0.9999)
        self.x_test['feature78'].loc[self.x_test['feature78'] > upper] = upper
        upper = self.x_test.feature79.quantile(0.9999)
        self.x_test['feature79'].loc[self.x_test['feature79'] > upper] = upper
        upper = self.x_test.feature80.quantile(0.9999)
        self.x_test['feature80'].loc[self.x_test['feature80'] > upper] = upper
        upper = self.x_test.feature81.quantile(0.9999)
        self.x_test['feature81'].loc[self.x_test['feature81'] > upper] = upper
        upper = self.x_test.feature82.quantile(0.9999)
        self.x_test['feature82'].loc[self.x_test['feature82'] > upper] = upper
        upper = self.x_test.feature83.quantile(0.9999)
        self.x_test['feature83'].loc[self.x_test['feature83'] > upper] = upper
        upper = self.x_test.feature84.quantile(0.9999)
        self.x_test['feature84'].loc[self.x_test['feature84'] > upper] = upper
        upper = self.x_test.feature85.quantile(0.9999)
        self.x_test['feature85'].loc[self.x_test['feature85'] > upper] = upper
        upper = self.x_test.feature86.quantile(0.9999)
        self.x_test['feature86'].loc[self.x_test['feature86'] > upper] = upper
        upper = self.x_test.feature87.quantile(0.9999)
        self.x_test['feature87'].loc[self.x_test['feature87'] > upper] = upper

        # Drop lower outlines in self.x_test
        lower = self.x_test.feature0.quantile(0.0001)
        self.x_test['feature0'].loc[self.x_test['feature0'] < lower] = lower
        lower = self.x_test.feature1.quantile(0.0001)
        self.x_test['feature1'].loc[self.x_test['feature1'] < lower] = lower
        lower = self.x_test.feature2.quantile(0.0001)
        self.x_test['feature2'].loc[self.x_test['feature2'] < lower] = lower
        lower = self.x_test.feature3.quantile(0.0001)
        self.x_test['feature3'].loc[self.x_test['feature3'] < lower] = lower
        lower = self.x_test.feature4.quantile(0.0001)
        self.x_test['feature4'].loc[self.x_test['feature4'] < lower] = lower
        lower = self.x_test.feature5.quantile(0.0001)
        self.x_test['feature5'].loc[self.x_test['feature5'] < lower] = lower
        lower = self.x_test.feature6.quantile(0.0001)
        self.x_test['feature6'].loc[self.x_test['feature6'] < lower] = lower
        lower = self.x_test.feature7.quantile(0.0001)
        self.x_test['feature7'].loc[self.x_test['feature7'] < lower] = lower
        lower = self.x_test.feature8.quantile(0.0001)
        self.x_test['feature8'].loc[self.x_test['feature8'] < lower] = lower
        lower = self.x_test.feature9.quantile(0.0001)
        self.x_test['feature9'].loc[self.x_test['feature9'] < lower] = lower
        lower = self.x_test.feature10.quantile(0.0001)
        self.x_test['feature10'].loc[self.x_test['feature10'] < lower] = lower
        lower = self.x_test.feature11.quantile(0.0001)
        self.x_test['feature11'].loc[self.x_test['feature11'] < lower] = lower
        lower = self.x_test.feature12.quantile(0.0001)
        self.x_test['feature12'].loc[self.x_test['feature12'] < lower] = lower
        lower = self.x_test.feature13.quantile(0.0001)
        self.x_test['feature13'].loc[self.x_test['feature13'] < lower] = lower
        lower = self.x_test.feature14.quantile(0.0001)
        self.x_test['feature14'].loc[self.x_test['feature14'] < lower] = lower
        lower = self.x_test.feature15.quantile(0.0001)
        self.x_test['feature15'].loc[self.x_test['feature15'] < lower] = lower
        lower = self.x_test.feature16.quantile(0.0001)
        self.x_test['feature16'].loc[self.x_test['feature16'] < lower] = lower
        lower = self.x_test.feature17.quantile(0.0001)
        self.x_test['feature17'].loc[self.x_test['feature17'] < lower] = lower
        lower = self.x_test.feature18.quantile(0.0001)
        self.x_test['feature18'].loc[self.x_test['feature18'] < lower] = lower
        lower = self.x_test.feature19.quantile(0.0001)
        self.x_test['feature19'].loc[self.x_test['feature19'] < lower] = lower
        lower = self.x_test.feature20.quantile(0.0001)
        self.x_test['feature20'].loc[self.x_test['feature20'] < lower] = lower
        lower = self.x_test.feature21.quantile(0.0001)
        self.x_test['feature21'].loc[self.x_test['feature21'] < lower] = lower
        lower = self.x_test.feature22.quantile(0.0001)
        self.x_test['feature22'].loc[self.x_test['feature22'] < lower] = lower
        lower = self.x_test.feature23.quantile(0.0001)
        self.x_test['feature23'].loc[self.x_test['feature23'] < lower] = lower
        lower = self.x_test.feature24.quantile(0.0001)
        self.x_test['feature24'].loc[self.x_test['feature24'] < lower] = lower
        lower = self.x_test.feature25.quantile(0.0001)
        self.x_test['feature25'].loc[self.x_test['feature25'] < lower] = lower
        lower = self.x_test.feature26.quantile(0.0001)
        self.x_test['feature26'].loc[self.x_test['feature26'] < lower] = lower
        lower = self.x_test.feature27.quantile(0.0001)
        self.x_test['feature27'].loc[self.x_test['feature27'] < lower] = lower
        lower = self.x_test.feature28.quantile(0.0001)
        self.x_test['feature28'].loc[self.x_test['feature28'] < lower] = lower
        lower = self.x_test.feature29.quantile(0.0001)
        self.x_test['feature29'].loc[self.x_test['feature29'] < lower] = lower
        lower = self.x_test.feature30.quantile(0.0001)
        self.x_test['feature30'].loc[self.x_test['feature30'] < lower] = lower
        lower = self.x_test.feature31.quantile(0.0001)
        self.x_test['feature31'].loc[self.x_test['feature31'] < lower] = lower
        lower = self.x_test.feature32.quantile(0.0001)
        self.x_test['feature32'].loc[self.x_test['feature32'] < lower] = lower
        lower = self.x_test.feature33.quantile(0.0001)
        self.x_test['feature33'].loc[self.x_test['feature33'] < lower] = lower
        lower = self.x_test.feature34.quantile(0.0001)
        self.x_test['feature34'].loc[self.x_test['feature34'] < lower] = lower
        lower = self.x_test.feature35.quantile(0.0001)
        self.x_test['feature35'].loc[self.x_test['feature35'] < lower] = lower
        lower = self.x_test.feature36.quantile(0.0001)
        self.x_test['feature36'].loc[self.x_test['feature36'] < lower] = lower
        lower = self.x_test.feature37.quantile(0.0001)
        self.x_test['feature37'].loc[self.x_test['feature37'] < lower] = lower
        lower = self.x_test.feature38.quantile(0.0001)
        self.x_test['feature38'].loc[self.x_test['feature38'] < lower] = lower
        lower = self.x_test.feature39.quantile(0.0001)
        self.x_test['feature39'].loc[self.x_test['feature39'] < lower] = lower
        lower = self.x_test.feature40.quantile(0.0001)
        self.x_test['feature40'].loc[self.x_test['feature40'] < lower] = lower
        lower = self.x_test.feature41.quantile(0.0001)
        self.x_test['feature41'].loc[self.x_test['feature41'] < lower] = lower
        lower = self.x_test.feature42.quantile(0.0001)
        self.x_test['feature42'].loc[self.x_test['feature42'] < lower] = lower
        lower = self.x_test.feature43.quantile(0.0001)
        self.x_test['feature43'].loc[self.x_test['feature43'] < lower] = lower
        lower = self.x_test.feature44.quantile(0.0001)
        self.x_test['feature44'].loc[self.x_test['feature44'] < lower] = lower
        lower = self.x_test.feature45.quantile(0.0001)
        self.x_test['feature45'].loc[self.x_test['feature45'] < lower] = lower
        lower = self.x_test.feature46.quantile(0.0001)
        self.x_test['feature46'].loc[self.x_test['feature46'] < lower] = lower
        lower = self.x_test.feature47.quantile(0.0001)
        self.x_test['feature47'].loc[self.x_test['feature47'] < lower] = lower
        lower = self.x_test.feature48.quantile(0.0001)
        self.x_test['feature48'].loc[self.x_test['feature48'] < lower] = lower
        lower = self.x_test.feature49.quantile(0.0001)
        self.x_test['feature49'].loc[self.x_test['feature49'] < lower] = lower
        lower = self.x_test.feature50.quantile(0.0001)
        self.x_test['feature50'].loc[self.x_test['feature50'] < lower] = lower
        lower = self.x_test.feature51.quantile(0.0001)
        self.x_test['feature51'].loc[self.x_test['feature51'] < lower] = lower
        lower = self.x_test.feature52.quantile(0.0001)
        self.x_test['feature52'].loc[self.x_test['feature52'] < lower] = lower
        lower = self.x_test.feature53.quantile(0.0001)
        self.x_test['feature53'].loc[self.x_test['feature53'] < lower] = lower
        lower = self.x_test.feature54.quantile(0.0001)
        self.x_test['feature54'].loc[self.x_test['feature54'] < lower] = lower
        lower = self.x_test.feature55.quantile(0.0001)
        self.x_test['feature55'].loc[self.x_test['feature55'] < lower] = lower
        lower = self.x_test.feature56.quantile(0.0001)
        self.x_test['feature56'].loc[self.x_test['feature56'] < lower] = lower
        lower = self.x_test.feature57.quantile(0.0001)
        self.x_test['feature57'].loc[self.x_test['feature57'] < lower] = lower
        lower = self.x_test.feature58.quantile(0.0001)
        self.x_test['feature58'].loc[self.x_test['feature58'] < lower] = lower
        lower = self.x_test.feature59.quantile(0.0001)
        self.x_test['feature59'].loc[self.x_test['feature59'] < lower] = lower
        lower = self.x_test.feature60.quantile(0.0001)
        self.x_test['feature60'].loc[self.x_test['feature60'] < lower] = lower
        lower = self.x_test.feature61.quantile(0.0001)
        self.x_test['feature61'].loc[self.x_test['feature61'] < lower] = lower
        lower = self.x_test.feature62.quantile(0.0001)
        self.x_test['feature62'].loc[self.x_test['feature62'] < lower] = lower
        lower = self.x_test.feature63.quantile(0.0001)
        self.x_test['feature63'].loc[self.x_test['feature63'] < lower] = lower
        lower = self.x_test.feature64.quantile(0.0001)
        self.x_test['feature64'].loc[self.x_test['feature64'] < lower] = lower
        lower = self.x_test.feature65.quantile(0.0001)
        self.x_test['feature65'].loc[self.x_test['feature65'] < lower] = lower
        lower = self.x_test.feature66.quantile(0.0001)
        self.x_test['feature66'].loc[self.x_test['feature66'] < lower] = lower
        lower = self.x_test.feature67.quantile(0.0001)
        self.x_test['feature67'].loc[self.x_test['feature67'] < lower] = lower
        lower = self.x_test.feature68.quantile(0.0001)
        self.x_test['feature68'].loc[self.x_test['feature68'] < lower] = lower
        lower = self.x_test.feature69.quantile(0.0001)
        self.x_test['feature69'].loc[self.x_test['feature69'] < lower] = lower
        lower = self.x_test.feature70.quantile(0.0001)
        self.x_test['feature70'].loc[self.x_test['feature70'] < lower] = lower
        lower = self.x_test.feature71.quantile(0.0001)
        self.x_test['feature71'].loc[self.x_test['feature71'] < lower] = lower
        lower = self.x_test.feature72.quantile(0.0001)
        self.x_test['feature72'].loc[self.x_test['feature72'] < lower] = lower
        lower = self.x_test.feature73.quantile(0.0001)
        self.x_test['feature73'].loc[self.x_test['feature73'] < lower] = lower
        lower = self.x_test.feature74.quantile(0.0001)
        self.x_test['feature74'].loc[self.x_test['feature74'] < lower] = lower
        lower = self.x_test.feature75.quantile(0.0001)
        self.x_test['feature75'].loc[self.x_test['feature75'] < lower] = lower
        lower = self.x_test.feature76.quantile(0.0001)
        self.x_test['feature76'].loc[self.x_test['feature76'] < lower] = lower
        lower = self.x_test.feature77.quantile(0.0001)
        self.x_test['feature77'].loc[self.x_test['feature77'] < lower] = lower
        lower = self.x_test.feature78.quantile(0.0001)
        self.x_test['feature78'].loc[self.x_test['feature78'] < lower] = lower
        lower = self.x_test.feature79.quantile(0.0001)
        self.x_test['feature79'].loc[self.x_test['feature79'] < lower] = lower
        lower = self.x_test.feature80.quantile(0.0001)
        self.x_test['feature80'].loc[self.x_test['feature80'] < lower] = lower
        lower = self.x_test.feature81.quantile(0.0001)
        self.x_test['feature81'].loc[self.x_test['feature81'] < lower] = lower
        lower = self.x_test.feature82.quantile(0.0001)
        self.x_test['feature82'].loc[self.x_test['feature82'] < lower] = lower
        lower = self.x_test.feature83.quantile(0.0001)
        self.x_test['feature83'].loc[self.x_test['feature83'] < lower] = lower
        lower = self.x_test.feature84.quantile(0.0001)
        self.x_test['feature84'].loc[self.x_test['feature84'] < lower] = lower
        lower = self.x_test.feature85.quantile(0.0001)
        self.x_test['feature85'].loc[self.x_test['feature85'] < lower] = lower
        lower = self.x_test.feature86.quantile(0.0001)
        self.x_test['feature86'].loc[self.x_test['feature86'] < lower] = lower
        lower = self.x_test.feature87.quantile(0.0001)
        self.x_test['feature87'].loc[self.x_test['feature87'] < lower] = lower

    # Scaling data
    def scale(self):

        print('Scaling data...')

        for each in self.x_train.columns:
            mean, std = self.x_train[each].mean(), self.x_train[each].std()
            self.x_train.loc[:, each] = (self.x_train[each] - mean)/std

        for each in self.x_test.columns:
            mean, std = self.x_test[each].mean(), self.x_test[each].std()
            self.x_test.loc[:, each] = (self.x_test[each] - mean)/std
            
    # Convert column 'group' to dummies
    def convert_group(self):
        
        print('Converting groups to dummies...')

        group_train_dummies = pd.get_dummies(self.g_train, prefix='group')
        self.x_train = self.x_train.join(group_train_dummies)

        group_test_dummies = pd.get_dummies(self.g_test, prefix='group')
        self.x_test = self.x_test.join(group_test_dummies)

        print('Shape of x_train with group dummies: {}'.format(self.x_train.shape))
        print('Shape of x_test with group dummies: {}'.format(self.x_test.shape))

    # Shuffle and split dataset
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
        utils.save_np_to_pkl(self.y_train, self.prepro_path + 'y_train.p')
        utils.save_np_to_pkl(self.w_train, self.prepro_path + 'w_train.p')
        utils.save_np_to_pkl(self.x_test, self.prepro_path + 'x_test.p')

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

    # Save Data
    def save_data_pd(self):

        if not isdir(self.prepro_path):
            os.mkdir(self.prepro_path)

        print('Saving data...')

        self.x_train.to_pickle(self.prepro_path + 'x_train.p')
        self.y_train.to_pickle(self.prepro_path + 'y_train.p')
        self.w_train.to_pickle(self.prepro_path + 'w_train.p')
        self.e_train.to_pickle(self.prepro_path + 'e_train.p')
        self.x_test.to_pickle(self.prepro_path + 'x_test.p')
        self.id_test.to_pickle(self.prepro_path + 'id_test.p')

    # Preprocessing
    def preprocess_pd(self):

        start_time = time.time()

        # Load original data
        self.load_data_pd()

        # Drop outliers
        # self.drop_outliers()

        # Scale features
        # self.scale()

        # Convert column 'group' to dummies
        self.convert_group()

        # Save Data to pickle files
        self.save_data_pd()

        end_time = time.time()

        print('Done!')
        print('Using {:.3}s'.format(end_time - start_time))


if __name__ == "__main__":

    DPP = DataPreProcess(train_csv_path, test_csv_path, preprocessed_path)
    DPP.preprocess_pd()
