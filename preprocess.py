import pickle
import time
import utils
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


train_csv_path = './inputs/stock_train_data_20170910.csv'
test_csv_path = './inputs/stock_test_data_20170910.csv'
preprocessed_path = './preprocessed_data/'


class DataPreProcess:

    train_x = np.array([])
    train_y = np.array([])
    train_w = np.array([])
    train_g = np.array([])
    train_e = np.array([])
    test_x = np.array([])
    test_g = np.array([])

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

        self.train_x = train_f[:, 1:]                  # feature + weight + label + group + era
        self.train_w = train_f[:, 89]                  # weight
        np.delete(self.train_x, [88, 89, 90], axis=1)  # feature + era
        self.train_y = train_f[:, 90]                  # label
        self.test_x = test_f                           # feature + group

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
        self.train_x = train_f.drop(['id', 'weight', 'label', 'group', 'era'], axis=1)
        self.train_y = train_f['label']
        self.train_w = train_f['weight']
        self.train_g = train_f['group']
        self.train_e = train_f['era']
        self.test_x = test_f.drop(['id', 'group'], axis=1)
        self.test_g = test_f['group']

    # Dropping outlines
    def drop_outlines(self):
        
        print('Dropping outlines...')
        
        # Drop upper outlines in self.train_x
        upper = self.train_x.feature0.quantile(0.999)
        self.train_x['feature0'].loc[self.train_x['feature0'] > upper] = upper
        upper = self.train_x.feature1.quantile(0.999)
        self.train_x['feature1'].loc[self.train_x['feature1'] > upper] = upper
        upper = self.train_x.feature2.quantile(0.999)
        self.train_x['feature2'].loc[self.train_x['feature2'] > upper] = upper
        upper = self.train_x.feature3.quantile(0.999)
        self.train_x['feature3'].loc[self.train_x['feature3'] > upper] = upper
        upper = self.train_x.feature4.quantile(0.999)
        self.train_x['feature4'].loc[self.train_x['feature4'] > upper] = upper
        upper = self.train_x.feature5.quantile(0.999)
        self.train_x['feature5'].loc[self.train_x['feature5'] > upper] = upper
        upper = self.train_x.feature6.quantile(0.999)
        self.train_x['feature6'].loc[self.train_x['feature6'] > upper] = upper
        upper = self.train_x.feature7.quantile(0.999)
        self.train_x['feature7'].loc[self.train_x['feature7'] > upper] = upper
        upper = self.train_x.feature8.quantile(0.999)
        self.train_x['feature8'].loc[self.train_x['feature8'] > upper] = upper
        upper = self.train_x.feature9.quantile(0.999)
        self.train_x['feature9'].loc[self.train_x['feature9'] > upper] = upper
        upper = self.train_x.feature10.quantile(0.999)
        self.train_x['feature10'].loc[self.train_x['feature10'] > upper] = upper
        upper = self.train_x.feature11.quantile(0.999)
        self.train_x['feature11'].loc[self.train_x['feature11'] > upper] = upper
        upper = self.train_x.feature12.quantile(0.999)
        self.train_x['feature12'].loc[self.train_x['feature12'] > upper] = upper
        upper = self.train_x.feature13.quantile(0.999)
        self.train_x['feature13'].loc[self.train_x['feature13'] > upper] = upper
        upper = self.train_x.feature14.quantile(0.999)
        self.train_x['feature14'].loc[self.train_x['feature14'] > upper] = upper
        upper = self.train_x.feature15.quantile(0.999)
        self.train_x['feature15'].loc[self.train_x['feature15'] > upper] = upper
        upper = self.train_x.feature16.quantile(0.999)
        self.train_x['feature16'].loc[self.train_x['feature16'] > upper] = upper
        upper = self.train_x.feature17.quantile(0.999)
        self.train_x['feature17'].loc[self.train_x['feature17'] > upper] = upper
        upper = self.train_x.feature18.quantile(0.999)
        self.train_x['feature18'].loc[self.train_x['feature18'] > upper] = upper
        upper = self.train_x.feature19.quantile(0.999)
        self.train_x['feature19'].loc[self.train_x['feature19'] > upper] = upper
        upper = self.train_x.feature20.quantile(0.999)
        self.train_x['feature20'].loc[self.train_x['feature20'] > upper] = upper
        upper = self.train_x.feature21.quantile(0.999)
        self.train_x['feature21'].loc[self.train_x['feature21'] > upper] = upper
        upper = self.train_x.feature22.quantile(0.999)
        self.train_x['feature22'].loc[self.train_x['feature22'] > upper] = upper
        upper = self.train_x.feature23.quantile(0.999)
        self.train_x['feature23'].loc[self.train_x['feature23'] > upper] = upper
        upper = self.train_x.feature24.quantile(0.999)
        self.train_x['feature24'].loc[self.train_x['feature24'] > upper] = upper
        upper = self.train_x.feature25.quantile(0.999)
        self.train_x['feature25'].loc[self.train_x['feature25'] > upper] = upper
        upper = self.train_x.feature26.quantile(0.999)
        self.train_x['feature26'].loc[self.train_x['feature26'] > upper] = upper
        upper = self.train_x.feature27.quantile(0.999)
        self.train_x['feature27'].loc[self.train_x['feature27'] > upper] = upper
        upper = self.train_x.feature28.quantile(0.999)
        self.train_x['feature28'].loc[self.train_x['feature28'] > upper] = upper
        upper = self.train_x.feature29.quantile(0.999)
        self.train_x['feature29'].loc[self.train_x['feature29'] > upper] = upper
        upper = self.train_x.feature30.quantile(0.999)
        self.train_x['feature30'].loc[self.train_x['feature30'] > upper] = upper
        upper = self.train_x.feature31.quantile(0.999)
        self.train_x['feature31'].loc[self.train_x['feature31'] > upper] = upper
        upper = self.train_x.feature32.quantile(0.999)
        self.train_x['feature32'].loc[self.train_x['feature32'] > upper] = upper
        upper = self.train_x.feature33.quantile(0.999)
        self.train_x['feature33'].loc[self.train_x['feature33'] > upper] = upper
        upper = self.train_x.feature34.quantile(0.999)
        self.train_x['feature34'].loc[self.train_x['feature34'] > upper] = upper
        upper = self.train_x.feature35.quantile(0.999)
        self.train_x['feature35'].loc[self.train_x['feature35'] > upper] = upper
        upper = self.train_x.feature36.quantile(0.999)
        self.train_x['feature36'].loc[self.train_x['feature36'] > upper] = upper
        upper = self.train_x.feature37.quantile(0.999)
        self.train_x['feature37'].loc[self.train_x['feature37'] > upper] = upper
        upper = self.train_x.feature38.quantile(0.999)
        self.train_x['feature38'].loc[self.train_x['feature38'] > upper] = upper
        upper = self.train_x.feature39.quantile(0.999)
        self.train_x['feature39'].loc[self.train_x['feature39'] > upper] = upper
        upper = self.train_x.feature40.quantile(0.999)
        self.train_x['feature40'].loc[self.train_x['feature40'] > upper] = upper
        upper = self.train_x.feature41.quantile(0.999)
        self.train_x['feature41'].loc[self.train_x['feature41'] > upper] = upper
        upper = self.train_x.feature42.quantile(0.999)
        self.train_x['feature42'].loc[self.train_x['feature42'] > upper] = upper
        upper = self.train_x.feature43.quantile(0.999)
        self.train_x['feature43'].loc[self.train_x['feature43'] > upper] = upper
        upper = self.train_x.feature44.quantile(0.999)
        self.train_x['feature44'].loc[self.train_x['feature44'] > upper] = upper
        upper = self.train_x.feature45.quantile(0.999)
        self.train_x['feature45'].loc[self.train_x['feature45'] > upper] = upper
        upper = self.train_x.feature46.quantile(0.999)
        self.train_x['feature46'].loc[self.train_x['feature46'] > upper] = upper
        upper = self.train_x.feature47.quantile(0.999)
        self.train_x['feature47'].loc[self.train_x['feature47'] > upper] = upper
        upper = self.train_x.feature48.quantile(0.999)
        self.train_x['feature48'].loc[self.train_x['feature48'] > upper] = upper
        upper = self.train_x.feature49.quantile(0.999)
        self.train_x['feature49'].loc[self.train_x['feature49'] > upper] = upper
        upper = self.train_x.feature50.quantile(0.999)
        self.train_x['feature50'].loc[self.train_x['feature50'] > upper] = upper
        upper = self.train_x.feature51.quantile(0.999)
        self.train_x['feature51'].loc[self.train_x['feature51'] > upper] = upper
        upper = self.train_x.feature52.quantile(0.999)
        self.train_x['feature52'].loc[self.train_x['feature52'] > upper] = upper
        upper = self.train_x.feature53.quantile(0.999)
        self.train_x['feature53'].loc[self.train_x['feature53'] > upper] = upper
        upper = self.train_x.feature54.quantile(0.999)
        self.train_x['feature54'].loc[self.train_x['feature54'] > upper] = upper
        upper = self.train_x.feature55.quantile(0.999)
        self.train_x['feature55'].loc[self.train_x['feature55'] > upper] = upper
        upper = self.train_x.feature56.quantile(0.999)
        self.train_x['feature56'].loc[self.train_x['feature56'] > upper] = upper
        upper = self.train_x.feature57.quantile(0.999)
        self.train_x['feature57'].loc[self.train_x['feature57'] > upper] = upper
        upper = self.train_x.feature58.quantile(0.999)
        self.train_x['feature58'].loc[self.train_x['feature58'] > upper] = upper
        upper = self.train_x.feature59.quantile(0.999)
        self.train_x['feature59'].loc[self.train_x['feature59'] > upper] = upper
        upper = self.train_x.feature60.quantile(0.999)
        self.train_x['feature60'].loc[self.train_x['feature60'] > upper] = upper
        upper = self.train_x.feature61.quantile(0.999)
        self.train_x['feature61'].loc[self.train_x['feature61'] > upper] = upper
        upper = self.train_x.feature62.quantile(0.999)
        self.train_x['feature62'].loc[self.train_x['feature62'] > upper] = upper
        upper = self.train_x.feature63.quantile(0.999)
        self.train_x['feature63'].loc[self.train_x['feature63'] > upper] = upper
        upper = self.train_x.feature64.quantile(0.999)
        self.train_x['feature64'].loc[self.train_x['feature64'] > upper] = upper
        upper = self.train_x.feature65.quantile(0.999)
        self.train_x['feature65'].loc[self.train_x['feature65'] > upper] = upper
        upper = self.train_x.feature66.quantile(0.999)
        self.train_x['feature66'].loc[self.train_x['feature66'] > upper] = upper
        upper = self.train_x.feature67.quantile(0.999)
        self.train_x['feature67'].loc[self.train_x['feature67'] > upper] = upper
        upper = self.train_x.feature68.quantile(0.999)
        self.train_x['feature68'].loc[self.train_x['feature68'] > upper] = upper
        upper = self.train_x.feature69.quantile(0.999)
        self.train_x['feature69'].loc[self.train_x['feature69'] > upper] = upper
        upper = self.train_x.feature70.quantile(0.999)
        self.train_x['feature70'].loc[self.train_x['feature70'] > upper] = upper
        upper = self.train_x.feature70.quantile(0.999)
        self.train_x['feature71'].loc[self.train_x['feature71'] > upper] = upper
        upper = self.train_x.feature71.quantile(0.999)
        self.train_x['feature72'].loc[self.train_x['feature72'] > upper] = upper
        upper = self.train_x.feature73.quantile(0.999)
        self.train_x['feature73'].loc[self.train_x['feature73'] > upper] = upper
        upper = self.train_x.feature74.quantile(0.999)
        self.train_x['feature74'].loc[self.train_x['feature74'] > upper] = upper
        upper = self.train_x.feature75.quantile(0.999)
        self.train_x['feature75'].loc[self.train_x['feature75'] > upper] = upper
        upper = self.train_x.feature76.quantile(0.999)
        self.train_x['feature76'].loc[self.train_x['feature76'] > upper] = upper
        upper = self.train_x.feature77.quantile(0.999)
        self.train_x['feature77'].loc[self.train_x['feature77'] > upper] = upper
        upper = self.train_x.feature78.quantile(0.999)
        self.train_x['feature78'].loc[self.train_x['feature78'] > upper] = upper
        upper = self.train_x.feature79.quantile(0.999)
        self.train_x['feature79'].loc[self.train_x['feature79'] > upper] = upper
        upper = self.train_x.feature80.quantile(0.999)
        self.train_x['feature80'].loc[self.train_x['feature80'] > upper] = upper
        upper = self.train_x.feature81.quantile(0.999)
        self.train_x['feature81'].loc[self.train_x['feature81'] > upper] = upper
        upper = self.train_x.feature81.quantile(0.999)
        self.train_x['feature81'].loc[self.train_x['feature81'] > upper] = upper
        upper = self.train_x.feature82.quantile(0.999)
        self.train_x['feature82'].loc[self.train_x['feature82'] > upper] = upper
        upper = self.train_x.feature83.quantile(0.999)
        self.train_x['feature83'].loc[self.train_x['feature83'] > upper] = upper
        upper = self.train_x.feature84.quantile(0.999)
        self.train_x['feature84'].loc[self.train_x['feature84'] > upper] = upper
        upper = self.train_x.feature85.quantile(0.999)
        self.train_x['feature85'].loc[self.train_x['feature85'] > upper] = upper
        upper = self.train_x.feature86.quantile(0.999)
        self.train_x['feature86'].loc[self.train_x['feature86'] > upper] = upper
        upper = self.train_x.feature87.quantile(0.999)
        self.train_x['feature87'].loc[self.train_x['feature87'] > upper] = upper

        # Drop lower outlines in self.train_x
        lower = self.train_x.feature0.quantile(0.001)
        self.train_x['feature0'].loc[self.train_x['feature0'] < lower] = lower
        lower = self.train_x.feature1.quantile(0.001)
        self.train_x['feature1'].loc[self.train_x['feature1'] < lower] = lower
        lower = self.train_x.feature2.quantile(0.001)
        self.train_x['feature2'].loc[self.train_x['feature2'] < lower] = lower
        lower = self.train_x.feature3.quantile(0.001)
        self.train_x['feature3'].loc[self.train_x['feature3'] < lower] = lower
        lower = self.train_x.feature4.quantile(0.001)
        self.train_x['feature4'].loc[self.train_x['feature4'] < lower] = lower
        lower = self.train_x.feature5.quantile(0.001)
        self.train_x['feature5'].loc[self.train_x['feature5'] < lower] = lower
        lower = self.train_x.feature6.quantile(0.001)
        self.train_x['feature6'].loc[self.train_x['feature6'] < lower] = lower
        lower = self.train_x.feature7.quantile(0.001)
        self.train_x['feature7'].loc[self.train_x['feature7'] < lower] = lower
        lower = self.train_x.feature8.quantile(0.001)
        self.train_x['feature8'].loc[self.train_x['feature8'] < lower] = lower
        lower = self.train_x.feature9.quantile(0.001)
        self.train_x['feature9'].loc[self.train_x['feature9'] < lower] = lower
        lower = self.train_x.feature10.quantile(0.001)
        self.train_x['feature10'].loc[self.train_x['feature10'] < lower] = lower
        lower = self.train_x.feature11.quantile(0.001)
        self.train_x['feature11'].loc[self.train_x['feature11'] < lower] = lower
        lower = self.train_x.feature12.quantile(0.001)
        self.train_x['feature12'].loc[self.train_x['feature12'] < lower] = lower
        lower = self.train_x.feature13.quantile(0.001)
        self.train_x['feature13'].loc[self.train_x['feature13'] < lower] = lower
        lower = self.train_x.feature14.quantile(0.001)
        self.train_x['feature14'].loc[self.train_x['feature14'] < lower] = lower
        lower = self.train_x.feature15.quantile(0.001)
        self.train_x['feature15'].loc[self.train_x['feature15'] < lower] = lower
        lower = self.train_x.feature16.quantile(0.001)
        self.train_x['feature16'].loc[self.train_x['feature16'] < lower] = lower
        lower = self.train_x.feature17.quantile(0.001)
        self.train_x['feature17'].loc[self.train_x['feature17'] < lower] = lower
        lower = self.train_x.feature18.quantile(0.001)
        self.train_x['feature18'].loc[self.train_x['feature18'] < lower] = lower
        lower = self.train_x.feature19.quantile(0.001)
        self.train_x['feature19'].loc[self.train_x['feature19'] < lower] = lower
        lower = self.train_x.feature20.quantile(0.001)
        self.train_x['feature20'].loc[self.train_x['feature20'] < lower] = lower
        lower = self.train_x.feature21.quantile(0.001)
        self.train_x['feature21'].loc[self.train_x['feature21'] < lower] = lower
        lower = self.train_x.feature22.quantile(0.001)
        self.train_x['feature22'].loc[self.train_x['feature22'] < lower] = lower
        lower = self.train_x.feature23.quantile(0.001)
        self.train_x['feature23'].loc[self.train_x['feature23'] < lower] = lower
        lower = self.train_x.feature24.quantile(0.001)
        self.train_x['feature24'].loc[self.train_x['feature24'] < lower] = lower
        lower = self.train_x.feature25.quantile(0.001)
        self.train_x['feature25'].loc[self.train_x['feature25'] < lower] = lower
        lower = self.train_x.feature26.quantile(0.001)
        self.train_x['feature26'].loc[self.train_x['feature26'] < lower] = lower
        lower = self.train_x.feature27.quantile(0.001)
        self.train_x['feature27'].loc[self.train_x['feature27'] < lower] = lower
        lower = self.train_x.feature28.quantile(0.001)
        self.train_x['feature28'].loc[self.train_x['feature28'] < lower] = lower
        lower = self.train_x.feature29.quantile(0.001)
        self.train_x['feature29'].loc[self.train_x['feature29'] < lower] = lower
        lower = self.train_x.feature30.quantile(0.001)
        self.train_x['feature30'].loc[self.train_x['feature30'] < lower] = lower
        lower = self.train_x.feature31.quantile(0.001)
        self.train_x['feature31'].loc[self.train_x['feature31'] < lower] = lower
        lower = self.train_x.feature32.quantile(0.001)
        self.train_x['feature32'].loc[self.train_x['feature32'] < lower] = lower
        lower = self.train_x.feature33.quantile(0.001)
        self.train_x['feature33'].loc[self.train_x['feature33'] < lower] = lower
        lower = self.train_x.feature34.quantile(0.001)
        self.train_x['feature34'].loc[self.train_x['feature34'] < lower] = lower
        lower = self.train_x.feature35.quantile(0.001)
        self.train_x['feature35'].loc[self.train_x['feature35'] < lower] = lower
        lower = self.train_x.feature36.quantile(0.001)
        self.train_x['feature36'].loc[self.train_x['feature36'] < lower] = lower
        lower = self.train_x.feature37.quantile(0.001)
        self.train_x['feature37'].loc[self.train_x['feature37'] < lower] = lower
        lower = self.train_x.feature38.quantile(0.001)
        self.train_x['feature38'].loc[self.train_x['feature38'] < lower] = lower
        lower = self.train_x.feature39.quantile(0.001)
        self.train_x['feature39'].loc[self.train_x['feature39'] < lower] = lower
        lower = self.train_x.feature40.quantile(0.001)
        self.train_x['feature40'].loc[self.train_x['feature40'] < lower] = lower
        lower = self.train_x.feature41.quantile(0.001)
        self.train_x['feature41'].loc[self.train_x['feature41'] < lower] = lower
        lower = self.train_x.feature42.quantile(0.001)
        self.train_x['feature42'].loc[self.train_x['feature42'] < lower] = lower
        lower = self.train_x.feature43.quantile(0.001)
        self.train_x['feature43'].loc[self.train_x['feature43'] < lower] = lower
        lower = self.train_x.feature44.quantile(0.001)
        self.train_x['feature44'].loc[self.train_x['feature44'] < lower] = lower
        lower = self.train_x.feature45.quantile(0.001)
        self.train_x['feature45'].loc[self.train_x['feature45'] < lower] = lower
        lower = self.train_x.feature46.quantile(0.001)
        self.train_x['feature46'].loc[self.train_x['feature46'] < lower] = lower
        lower = self.train_x.feature47.quantile(0.001)
        self.train_x['feature47'].loc[self.train_x['feature47'] < lower] = lower
        lower = self.train_x.feature48.quantile(0.001)
        self.train_x['feature48'].loc[self.train_x['feature48'] < lower] = lower
        lower = self.train_x.feature49.quantile(0.001)
        self.train_x['feature49'].loc[self.train_x['feature49'] < lower] = lower
        lower = self.train_x.feature50.quantile(0.001)
        self.train_x['feature50'].loc[self.train_x['feature50'] < lower] = lower
        lower = self.train_x.feature51.quantile(0.001)
        self.train_x['feature51'].loc[self.train_x['feature51'] < lower] = lower
        lower = self.train_x.feature52.quantile(0.001)
        self.train_x['feature52'].loc[self.train_x['feature52'] < lower] = lower
        lower = self.train_x.feature53.quantile(0.001)
        self.train_x['feature53'].loc[self.train_x['feature53'] < lower] = lower
        lower = self.train_x.feature54.quantile(0.001)
        self.train_x['feature54'].loc[self.train_x['feature54'] < lower] = lower
        lower = self.train_x.feature55.quantile(0.001)
        self.train_x['feature55'].loc[self.train_x['feature55'] < lower] = lower
        lower = self.train_x.feature56.quantile(0.001)
        self.train_x['feature56'].loc[self.train_x['feature56'] < lower] = lower
        lower = self.train_x.feature57.quantile(0.001)
        self.train_x['feature57'].loc[self.train_x['feature57'] < lower] = lower
        lower = self.train_x.feature58.quantile(0.001)
        self.train_x['feature58'].loc[self.train_x['feature58'] < lower] = lower
        lower = self.train_x.feature59.quantile(0.001)
        self.train_x['feature59'].loc[self.train_x['feature59'] < lower] = lower
        lower = self.train_x.feature60.quantile(0.001)
        self.train_x['feature60'].loc[self.train_x['feature60'] < lower] = lower
        lower = self.train_x.feature61.quantile(0.001)
        self.train_x['feature61'].loc[self.train_x['feature61'] < lower] = lower
        lower = self.train_x.feature62.quantile(0.001)
        self.train_x['feature62'].loc[self.train_x['feature62'] < lower] = lower
        lower = self.train_x.feature63.quantile(0.001)
        self.train_x['feature63'].loc[self.train_x['feature63'] < lower] = lower
        lower = self.train_x.feature64.quantile(0.001)
        self.train_x['feature64'].loc[self.train_x['feature64'] < lower] = lower
        lower = self.train_x.feature65.quantile(0.001)
        self.train_x['feature65'].loc[self.train_x['feature65'] < lower] = lower
        lower = self.train_x.feature66.quantile(0.001)
        self.train_x['feature66'].loc[self.train_x['feature66'] < lower] = lower
        lower = self.train_x.feature67.quantile(0.001)
        self.train_x['feature67'].loc[self.train_x['feature67'] < lower] = lower
        lower = self.train_x.feature68.quantile(0.001)
        self.train_x['feature68'].loc[self.train_x['feature68'] < lower] = lower
        lower = self.train_x.feature69.quantile(0.001)
        self.train_x['feature69'].loc[self.train_x['feature69'] < lower] = lower
        lower = self.train_x.feature70.quantile(0.001)
        self.train_x['feature70'].loc[self.train_x['feature70'] < lower] = lower
        lower = self.train_x.feature70.quantile(0.001)
        self.train_x['feature71'].loc[self.train_x['feature71'] < lower] = lower
        lower = self.train_x.feature71.quantile(0.001)
        self.train_x['feature72'].loc[self.train_x['feature72'] < lower] = lower
        lower = self.train_x.feature73.quantile(0.001)
        self.train_x['feature73'].loc[self.train_x['feature73'] < lower] = lower
        lower = self.train_x.feature74.quantile(0.001)
        self.train_x['feature74'].loc[self.train_x['feature74'] < lower] = lower
        lower = self.train_x.feature75.quantile(0.001)
        self.train_x['feature75'].loc[self.train_x['feature75'] < lower] = lower
        lower = self.train_x.feature76.quantile(0.001)
        self.train_x['feature76'].loc[self.train_x['feature76'] < lower] = lower
        lower = self.train_x.feature77.quantile(0.001)
        self.train_x['feature77'].loc[self.train_x['feature77'] < lower] = lower
        lower = self.train_x.feature78.quantile(0.001)
        self.train_x['feature78'].loc[self.train_x['feature78'] < lower] = lower
        lower = self.train_x.feature79.quantile(0.001)
        self.train_x['feature79'].loc[self.train_x['feature79'] < lower] = lower
        lower = self.train_x.feature80.quantile(0.001)
        self.train_x['feature80'].loc[self.train_x['feature80'] < lower] = lower
        lower = self.train_x.feature81.quantile(0.001)
        self.train_x['feature81'].loc[self.train_x['feature81'] < lower] = lower
        lower = self.train_x.feature81.quantile(0.001)
        self.train_x['feature81'].loc[self.train_x['feature81'] < lower] = lower
        lower = self.train_x.feature82.quantile(0.001)
        self.train_x['feature82'].loc[self.train_x['feature82'] < lower] = lower
        lower = self.train_x.feature83.quantile(0.001)
        self.train_x['feature83'].loc[self.train_x['feature83'] < lower] = lower
        lower = self.train_x.feature84.quantile(0.001)
        self.train_x['feature84'].loc[self.train_x['feature84'] < lower] = lower
        lower = self.train_x.feature85.quantile(0.001)
        self.train_x['feature85'].loc[self.train_x['feature85'] < lower] = lower
        lower = self.train_x.feature86.quantile(0.001)
        self.train_x['feature86'].loc[self.train_x['feature86'] < lower] = lower
        lower = self.train_x.feature87.quantile(0.001)
        self.train_x['feature87'].loc[self.train_x['feature87'] < lower] = lower

        # Drop upper outlines in self.test_x
        upper = self.test_x.feature0.quantile(0.999)
        self.test_x['feature0'].loc[self.test_x['feature0'] > upper] = upper
        upper = self.test_x.feature1.quantile(0.999)
        self.test_x['feature1'].loc[self.test_x['feature1'] > upper] = upper
        upper = self.test_x.feature2.quantile(0.999)
        self.test_x['feature2'].loc[self.test_x['feature2'] > upper] = upper
        upper = self.test_x.feature3.quantile(0.999)
        self.test_x['feature3'].loc[self.test_x['feature3'] > upper] = upper
        upper = self.test_x.feature4.quantile(0.999)
        self.test_x['feature4'].loc[self.test_x['feature4'] > upper] = upper
        upper = self.test_x.feature5.quantile(0.999)
        self.test_x['feature5'].loc[self.test_x['feature5'] > upper] = upper
        upper = self.test_x.feature6.quantile(0.999)
        self.test_x['feature6'].loc[self.test_x['feature6'] > upper] = upper
        upper = self.test_x.feature7.quantile(0.999)
        self.test_x['feature7'].loc[self.test_x['feature7'] > upper] = upper
        upper = self.test_x.feature8.quantile(0.999)
        self.test_x['feature8'].loc[self.test_x['feature8'] > upper] = upper
        upper = self.test_x.feature9.quantile(0.999)
        self.test_x['feature9'].loc[self.test_x['feature9'] > upper] = upper
        upper = self.test_x.feature10.quantile(0.999)
        self.test_x['feature10'].loc[self.test_x['feature10'] > upper] = upper
        upper = self.test_x.feature11.quantile(0.999)
        self.test_x['feature11'].loc[self.test_x['feature11'] > upper] = upper
        upper = self.test_x.feature12.quantile(0.999)
        self.test_x['feature12'].loc[self.test_x['feature12'] > upper] = upper
        upper = self.test_x.feature13.quantile(0.999)
        self.test_x['feature13'].loc[self.test_x['feature13'] > upper] = upper
        upper = self.test_x.feature14.quantile(0.999)
        self.test_x['feature14'].loc[self.test_x['feature14'] > upper] = upper
        upper = self.test_x.feature15.quantile(0.999)
        self.test_x['feature15'].loc[self.test_x['feature15'] > upper] = upper
        upper = self.test_x.feature16.quantile(0.999)
        self.test_x['feature16'].loc[self.test_x['feature16'] > upper] = upper
        upper = self.test_x.feature17.quantile(0.999)
        self.test_x['feature17'].loc[self.test_x['feature17'] > upper] = upper
        upper = self.test_x.feature18.quantile(0.999)
        self.test_x['feature18'].loc[self.test_x['feature18'] > upper] = upper
        upper = self.test_x.feature19.quantile(0.999)
        self.test_x['feature19'].loc[self.test_x['feature19'] > upper] = upper
        upper = self.test_x.feature20.quantile(0.999)
        self.test_x['feature20'].loc[self.test_x['feature20'] > upper] = upper
        upper = self.test_x.feature21.quantile(0.999)
        self.test_x['feature21'].loc[self.test_x['feature21'] > upper] = upper
        upper = self.test_x.feature22.quantile(0.999)
        self.test_x['feature22'].loc[self.test_x['feature22'] > upper] = upper
        upper = self.test_x.feature23.quantile(0.999)
        self.test_x['feature23'].loc[self.test_x['feature23'] > upper] = upper
        upper = self.test_x.feature24.quantile(0.999)
        self.test_x['feature24'].loc[self.test_x['feature24'] > upper] = upper
        upper = self.test_x.feature25.quantile(0.999)
        self.test_x['feature25'].loc[self.test_x['feature25'] > upper] = upper
        upper = self.test_x.feature26.quantile(0.999)
        self.test_x['feature26'].loc[self.test_x['feature26'] > upper] = upper
        upper = self.test_x.feature27.quantile(0.999)
        self.test_x['feature27'].loc[self.test_x['feature27'] > upper] = upper
        upper = self.test_x.feature28.quantile(0.999)
        self.test_x['feature28'].loc[self.test_x['feature28'] > upper] = upper
        upper = self.test_x.feature29.quantile(0.999)
        self.test_x['feature29'].loc[self.test_x['feature29'] > upper] = upper
        upper = self.test_x.feature30.quantile(0.999)
        self.test_x['feature30'].loc[self.test_x['feature30'] > upper] = upper
        upper = self.test_x.feature31.quantile(0.999)
        self.test_x['feature31'].loc[self.test_x['feature31'] > upper] = upper
        upper = self.test_x.feature32.quantile(0.999)
        self.test_x['feature32'].loc[self.test_x['feature32'] > upper] = upper
        upper = self.test_x.feature33.quantile(0.999)
        self.test_x['feature33'].loc[self.test_x['feature33'] > upper] = upper
        upper = self.test_x.feature34.quantile(0.999)
        self.test_x['feature34'].loc[self.test_x['feature34'] > upper] = upper
        upper = self.test_x.feature35.quantile(0.999)
        self.test_x['feature35'].loc[self.test_x['feature35'] > upper] = upper
        upper = self.test_x.feature36.quantile(0.999)
        self.test_x['feature36'].loc[self.test_x['feature36'] > upper] = upper
        upper = self.test_x.feature37.quantile(0.999)
        self.test_x['feature37'].loc[self.test_x['feature37'] > upper] = upper
        upper = self.test_x.feature38.quantile(0.999)
        self.test_x['feature38'].loc[self.test_x['feature38'] > upper] = upper
        upper = self.test_x.feature39.quantile(0.999)
        self.test_x['feature39'].loc[self.test_x['feature39'] > upper] = upper
        upper = self.test_x.feature40.quantile(0.999)
        self.test_x['feature40'].loc[self.test_x['feature40'] > upper] = upper
        upper = self.test_x.feature41.quantile(0.999)
        self.test_x['feature41'].loc[self.test_x['feature41'] > upper] = upper
        upper = self.test_x.feature42.quantile(0.999)
        self.test_x['feature42'].loc[self.test_x['feature42'] > upper] = upper
        upper = self.test_x.feature43.quantile(0.999)
        self.test_x['feature43'].loc[self.test_x['feature43'] > upper] = upper
        upper = self.test_x.feature44.quantile(0.999)
        self.test_x['feature44'].loc[self.test_x['feature44'] > upper] = upper
        upper = self.test_x.feature45.quantile(0.999)
        self.test_x['feature45'].loc[self.test_x['feature45'] > upper] = upper
        upper = self.test_x.feature46.quantile(0.999)
        self.test_x['feature46'].loc[self.test_x['feature46'] > upper] = upper
        upper = self.test_x.feature47.quantile(0.999)
        self.test_x['feature47'].loc[self.test_x['feature47'] > upper] = upper
        upper = self.test_x.feature48.quantile(0.999)
        self.test_x['feature48'].loc[self.test_x['feature48'] > upper] = upper
        upper = self.test_x.feature49.quantile(0.999)
        self.test_x['feature49'].loc[self.test_x['feature49'] > upper] = upper
        upper = self.test_x.feature50.quantile(0.999)
        self.test_x['feature50'].loc[self.test_x['feature50'] > upper] = upper
        upper = self.test_x.feature51.quantile(0.999)
        self.test_x['feature51'].loc[self.test_x['feature51'] > upper] = upper
        upper = self.test_x.feature52.quantile(0.999)
        self.test_x['feature52'].loc[self.test_x['feature52'] > upper] = upper
        upper = self.test_x.feature53.quantile(0.999)
        self.test_x['feature53'].loc[self.test_x['feature53'] > upper] = upper
        upper = self.test_x.feature54.quantile(0.999)
        self.test_x['feature54'].loc[self.test_x['feature54'] > upper] = upper
        upper = self.test_x.feature55.quantile(0.999)
        self.test_x['feature55'].loc[self.test_x['feature55'] > upper] = upper
        upper = self.test_x.feature56.quantile(0.999)
        self.test_x['feature56'].loc[self.test_x['feature56'] > upper] = upper
        upper = self.test_x.feature57.quantile(0.999)
        self.test_x['feature57'].loc[self.test_x['feature57'] > upper] = upper
        upper = self.test_x.feature58.quantile(0.999)
        self.test_x['feature58'].loc[self.test_x['feature58'] > upper] = upper
        upper = self.test_x.feature59.quantile(0.999)
        self.test_x['feature59'].loc[self.test_x['feature59'] > upper] = upper
        upper = self.test_x.feature60.quantile(0.999)
        self.test_x['feature60'].loc[self.test_x['feature60'] > upper] = upper
        upper = self.test_x.feature61.quantile(0.999)
        self.test_x['feature61'].loc[self.test_x['feature61'] > upper] = upper
        upper = self.test_x.feature62.quantile(0.999)
        self.test_x['feature62'].loc[self.test_x['feature62'] > upper] = upper
        upper = self.test_x.feature63.quantile(0.999)
        self.test_x['feature63'].loc[self.test_x['feature63'] > upper] = upper
        upper = self.test_x.feature64.quantile(0.999)
        self.test_x['feature64'].loc[self.test_x['feature64'] > upper] = upper
        upper = self.test_x.feature65.quantile(0.999)
        self.test_x['feature65'].loc[self.test_x['feature65'] > upper] = upper
        upper = self.test_x.feature66.quantile(0.999)
        self.test_x['feature66'].loc[self.test_x['feature66'] > upper] = upper
        upper = self.test_x.feature67.quantile(0.999)
        self.test_x['feature67'].loc[self.test_x['feature67'] > upper] = upper
        upper = self.test_x.feature68.quantile(0.999)
        self.test_x['feature68'].loc[self.test_x['feature68'] > upper] = upper
        upper = self.test_x.feature69.quantile(0.999)
        self.test_x['feature69'].loc[self.test_x['feature69'] > upper] = upper
        upper = self.test_x.feature70.quantile(0.999)
        self.test_x['feature70'].loc[self.test_x['feature70'] > upper] = upper
        upper = self.test_x.feature70.quantile(0.999)
        self.test_x['feature71'].loc[self.test_x['feature71'] > upper] = upper
        upper = self.test_x.feature71.quantile(0.999)
        self.test_x['feature72'].loc[self.test_x['feature72'] > upper] = upper
        upper = self.test_x.feature73.quantile(0.999)
        self.test_x['feature73'].loc[self.test_x['feature73'] > upper] = upper
        upper = self.test_x.feature74.quantile(0.999)
        self.test_x['feature74'].loc[self.test_x['feature74'] > upper] = upper
        upper = self.test_x.feature75.quantile(0.999)
        self.test_x['feature75'].loc[self.test_x['feature75'] > upper] = upper
        upper = self.test_x.feature76.quantile(0.999)
        self.test_x['feature76'].loc[self.test_x['feature76'] > upper] = upper
        upper = self.test_x.feature77.quantile(0.999)
        self.test_x['feature77'].loc[self.test_x['feature77'] > upper] = upper
        upper = self.test_x.feature78.quantile(0.999)
        self.test_x['feature78'].loc[self.test_x['feature78'] > upper] = upper
        upper = self.test_x.feature79.quantile(0.999)
        self.test_x['feature79'].loc[self.test_x['feature79'] > upper] = upper
        upper = self.test_x.feature80.quantile(0.999)
        self.test_x['feature80'].loc[self.test_x['feature80'] > upper] = upper
        upper = self.test_x.feature81.quantile(0.999)
        self.test_x['feature81'].loc[self.test_x['feature81'] > upper] = upper
        upper = self.test_x.feature81.quantile(0.999)
        self.test_x['feature81'].loc[self.test_x['feature81'] > upper] = upper
        upper = self.test_x.feature82.quantile(0.999)
        self.test_x['feature82'].loc[self.test_x['feature82'] > upper] = upper
        upper = self.test_x.feature83.quantile(0.999)
        self.test_x['feature83'].loc[self.test_x['feature83'] > upper] = upper
        upper = self.test_x.feature84.quantile(0.999)
        self.test_x['feature84'].loc[self.test_x['feature84'] > upper] = upper
        upper = self.test_x.feature85.quantile(0.999)
        self.test_x['feature85'].loc[self.test_x['feature85'] > upper] = upper
        upper = self.test_x.feature86.quantile(0.999)
        self.test_x['feature86'].loc[self.test_x['feature86'] > upper] = upper
        upper = self.test_x.feature87.quantile(0.999)
        self.test_x['feature87'].loc[self.test_x['feature87'] > upper] = upper

        # Drop lower outlines in self.test_x
        lower = self.test_x.feature0.quantile(0.001)
        self.test_x['feature0'].loc[self.test_x['feature0'] < lower] = lower
        lower = self.test_x.feature1.quantile(0.001)
        self.test_x['feature1'].loc[self.test_x['feature1'] < lower] = lower
        lower = self.test_x.feature2.quantile(0.001)
        self.test_x['feature2'].loc[self.test_x['feature2'] < lower] = lower
        lower = self.test_x.feature3.quantile(0.001)
        self.test_x['feature3'].loc[self.test_x['feature3'] < lower] = lower
        lower = self.test_x.feature4.quantile(0.001)
        self.test_x['feature4'].loc[self.test_x['feature4'] < lower] = lower
        lower = self.test_x.feature5.quantile(0.001)
        self.test_x['feature5'].loc[self.test_x['feature5'] < lower] = lower
        lower = self.test_x.feature6.quantile(0.001)
        self.test_x['feature6'].loc[self.test_x['feature6'] < lower] = lower
        lower = self.test_x.feature7.quantile(0.001)
        self.test_x['feature7'].loc[self.test_x['feature7'] < lower] = lower
        lower = self.test_x.feature8.quantile(0.001)
        self.test_x['feature8'].loc[self.test_x['feature8'] < lower] = lower
        lower = self.test_x.feature9.quantile(0.001)
        self.test_x['feature9'].loc[self.test_x['feature9'] < lower] = lower
        lower = self.test_x.feature10.quantile(0.001)
        self.test_x['feature10'].loc[self.test_x['feature10'] < lower] = lower
        lower = self.test_x.feature11.quantile(0.001)
        self.test_x['feature11'].loc[self.test_x['feature11'] < lower] = lower
        lower = self.test_x.feature12.quantile(0.001)
        self.test_x['feature12'].loc[self.test_x['feature12'] < lower] = lower
        lower = self.test_x.feature13.quantile(0.001)
        self.test_x['feature13'].loc[self.test_x['feature13'] < lower] = lower
        lower = self.test_x.feature14.quantile(0.001)
        self.test_x['feature14'].loc[self.test_x['feature14'] < lower] = lower
        lower = self.test_x.feature15.quantile(0.001)
        self.test_x['feature15'].loc[self.test_x['feature15'] < lower] = lower
        lower = self.test_x.feature16.quantile(0.001)
        self.test_x['feature16'].loc[self.test_x['feature16'] < lower] = lower
        lower = self.test_x.feature17.quantile(0.001)
        self.test_x['feature17'].loc[self.test_x['feature17'] < lower] = lower
        lower = self.test_x.feature18.quantile(0.001)
        self.test_x['feature18'].loc[self.test_x['feature18'] < lower] = lower
        lower = self.test_x.feature19.quantile(0.001)
        self.test_x['feature19'].loc[self.test_x['feature19'] < lower] = lower
        lower = self.test_x.feature20.quantile(0.001)
        self.test_x['feature20'].loc[self.test_x['feature20'] < lower] = lower
        lower = self.test_x.feature21.quantile(0.001)
        self.test_x['feature21'].loc[self.test_x['feature21'] < lower] = lower
        lower = self.test_x.feature22.quantile(0.001)
        self.test_x['feature22'].loc[self.test_x['feature22'] < lower] = lower
        lower = self.test_x.feature23.quantile(0.001)
        self.test_x['feature23'].loc[self.test_x['feature23'] < lower] = lower
        lower = self.test_x.feature24.quantile(0.001)
        self.test_x['feature24'].loc[self.test_x['feature24'] < lower] = lower
        lower = self.test_x.feature25.quantile(0.001)
        self.test_x['feature25'].loc[self.test_x['feature25'] < lower] = lower
        lower = self.test_x.feature26.quantile(0.001)
        self.test_x['feature26'].loc[self.test_x['feature26'] < lower] = lower
        lower = self.test_x.feature27.quantile(0.001)
        self.test_x['feature27'].loc[self.test_x['feature27'] < lower] = lower
        lower = self.test_x.feature28.quantile(0.001)
        self.test_x['feature28'].loc[self.test_x['feature28'] < lower] = lower
        lower = self.test_x.feature29.quantile(0.001)
        self.test_x['feature29'].loc[self.test_x['feature29'] < lower] = lower
        lower = self.test_x.feature30.quantile(0.001)
        self.test_x['feature30'].loc[self.test_x['feature30'] < lower] = lower
        lower = self.test_x.feature31.quantile(0.001)
        self.test_x['feature31'].loc[self.test_x['feature31'] < lower] = lower
        lower = self.test_x.feature32.quantile(0.001)
        self.test_x['feature32'].loc[self.test_x['feature32'] < lower] = lower
        lower = self.test_x.feature33.quantile(0.001)
        self.test_x['feature33'].loc[self.test_x['feature33'] < lower] = lower
        lower = self.test_x.feature34.quantile(0.001)
        self.test_x['feature34'].loc[self.test_x['feature34'] < lower] = lower
        lower = self.test_x.feature35.quantile(0.001)
        self.test_x['feature35'].loc[self.test_x['feature35'] < lower] = lower
        lower = self.test_x.feature36.quantile(0.001)
        self.test_x['feature36'].loc[self.test_x['feature36'] < lower] = lower
        lower = self.test_x.feature37.quantile(0.001)
        self.test_x['feature37'].loc[self.test_x['feature37'] < lower] = lower
        lower = self.test_x.feature38.quantile(0.001)
        self.test_x['feature38'].loc[self.test_x['feature38'] < lower] = lower
        lower = self.test_x.feature39.quantile(0.001)
        self.test_x['feature39'].loc[self.test_x['feature39'] < lower] = lower
        lower = self.test_x.feature40.quantile(0.001)
        self.test_x['feature40'].loc[self.test_x['feature40'] < lower] = lower
        lower = self.test_x.feature41.quantile(0.001)
        self.test_x['feature41'].loc[self.test_x['feature41'] < lower] = lower
        lower = self.test_x.feature42.quantile(0.001)
        self.test_x['feature42'].loc[self.test_x['feature42'] < lower] = lower
        lower = self.test_x.feature43.quantile(0.001)
        self.test_x['feature43'].loc[self.test_x['feature43'] < lower] = lower
        lower = self.test_x.feature44.quantile(0.001)
        self.test_x['feature44'].loc[self.test_x['feature44'] < lower] = lower
        lower = self.test_x.feature45.quantile(0.001)
        self.test_x['feature45'].loc[self.test_x['feature45'] < lower] = lower
        lower = self.test_x.feature46.quantile(0.001)
        self.test_x['feature46'].loc[self.test_x['feature46'] < lower] = lower
        lower = self.test_x.feature47.quantile(0.001)
        self.test_x['feature47'].loc[self.test_x['feature47'] < lower] = lower
        lower = self.test_x.feature48.quantile(0.001)
        self.test_x['feature48'].loc[self.test_x['feature48'] < lower] = lower
        lower = self.test_x.feature49.quantile(0.001)
        self.test_x['feature49'].loc[self.test_x['feature49'] < lower] = lower
        lower = self.test_x.feature50.quantile(0.001)
        self.test_x['feature50'].loc[self.test_x['feature50'] < lower] = lower
        lower = self.test_x.feature51.quantile(0.001)
        self.test_x['feature51'].loc[self.test_x['feature51'] < lower] = lower
        lower = self.test_x.feature52.quantile(0.001)
        self.test_x['feature52'].loc[self.test_x['feature52'] < lower] = lower
        lower = self.test_x.feature53.quantile(0.001)
        self.test_x['feature53'].loc[self.test_x['feature53'] < lower] = lower
        lower = self.test_x.feature54.quantile(0.001)
        self.test_x['feature54'].loc[self.test_x['feature54'] < lower] = lower
        lower = self.test_x.feature55.quantile(0.001)
        self.test_x['feature55'].loc[self.test_x['feature55'] < lower] = lower
        lower = self.test_x.feature56.quantile(0.001)
        self.test_x['feature56'].loc[self.test_x['feature56'] < lower] = lower
        lower = self.test_x.feature57.quantile(0.001)
        self.test_x['feature57'].loc[self.test_x['feature57'] < lower] = lower
        lower = self.test_x.feature58.quantile(0.001)
        self.test_x['feature58'].loc[self.test_x['feature58'] < lower] = lower
        lower = self.test_x.feature59.quantile(0.001)
        self.test_x['feature59'].loc[self.test_x['feature59'] < lower] = lower
        lower = self.test_x.feature60.quantile(0.001)
        self.test_x['feature60'].loc[self.test_x['feature60'] < lower] = lower
        lower = self.test_x.feature61.quantile(0.001)
        self.test_x['feature61'].loc[self.test_x['feature61'] < lower] = lower
        lower = self.test_x.feature62.quantile(0.001)
        self.test_x['feature62'].loc[self.test_x['feature62'] < lower] = lower
        lower = self.test_x.feature63.quantile(0.001)
        self.test_x['feature63'].loc[self.test_x['feature63'] < lower] = lower
        lower = self.test_x.feature64.quantile(0.001)
        self.test_x['feature64'].loc[self.test_x['feature64'] < lower] = lower
        lower = self.test_x.feature65.quantile(0.001)
        self.test_x['feature65'].loc[self.test_x['feature65'] < lower] = lower
        lower = self.test_x.feature66.quantile(0.001)
        self.test_x['feature66'].loc[self.test_x['feature66'] < lower] = lower
        lower = self.test_x.feature67.quantile(0.001)
        self.test_x['feature67'].loc[self.test_x['feature67'] < lower] = lower
        lower = self.test_x.feature68.quantile(0.001)
        self.test_x['feature68'].loc[self.test_x['feature68'] < lower] = lower
        lower = self.test_x.feature69.quantile(0.001)
        self.test_x['feature69'].loc[self.test_x['feature69'] < lower] = lower
        lower = self.test_x.feature70.quantile(0.001)
        self.test_x['feature70'].loc[self.test_x['feature70'] < lower] = lower
        lower = self.test_x.feature70.quantile(0.001)
        self.test_x['feature71'].loc[self.test_x['feature71'] < lower] = lower
        lower = self.test_x.feature71.quantile(0.001)
        self.test_x['feature72'].loc[self.test_x['feature72'] < lower] = lower
        lower = self.test_x.feature73.quantile(0.001)
        self.test_x['feature73'].loc[self.test_x['feature73'] < lower] = lower
        lower = self.test_x.feature74.quantile(0.001)
        self.test_x['feature74'].loc[self.test_x['feature74'] < lower] = lower
        lower = self.test_x.feature75.quantile(0.001)
        self.test_x['feature75'].loc[self.test_x['feature75'] < lower] = lower
        lower = self.test_x.feature76.quantile(0.001)
        self.test_x['feature76'].loc[self.test_x['feature76'] < lower] = lower
        lower = self.test_x.feature77.quantile(0.001)
        self.test_x['feature77'].loc[self.test_x['feature77'] < lower] = lower
        lower = self.test_x.feature78.quantile(0.001)
        self.test_x['feature78'].loc[self.test_x['feature78'] < lower] = lower
        lower = self.test_x.feature79.quantile(0.001)
        self.test_x['feature79'].loc[self.test_x['feature79'] < lower] = lower
        lower = self.test_x.feature80.quantile(0.001)
        self.test_x['feature80'].loc[self.test_x['feature80'] < lower] = lower
        lower = self.test_x.feature81.quantile(0.001)
        self.test_x['feature81'].loc[self.test_x['feature81'] < lower] = lower
        lower = self.test_x.feature81.quantile(0.001)
        self.test_x['feature81'].loc[self.test_x['feature81'] < lower] = lower
        lower = self.test_x.feature82.quantile(0.001)
        self.test_x['feature82'].loc[self.test_x['feature82'] < lower] = lower
        lower = self.test_x.feature83.quantile(0.001)
        self.test_x['feature83'].loc[self.test_x['feature83'] < lower] = lower
        lower = self.test_x.feature84.quantile(0.001)
        self.test_x['feature84'].loc[self.test_x['feature84'] < lower] = lower
        lower = self.test_x.feature85.quantile(0.001)
        self.test_x['feature85'].loc[self.test_x['feature85'] < lower] = lower
        lower = self.test_x.feature86.quantile(0.001)
        self.test_x['feature86'].loc[self.test_x['feature86'] < lower] = lower
        lower = self.test_x.feature87.quantile(0.001)
        self.test_x['feature87'].loc[self.test_x['feature87'] < lower] = lower

    # Scaling data
    def scale(self):

        print('Scaling data...')

        for each in self.train_x.columns:
            mean, std = self.train_x[each].mean(), self.train_x[each].std()
            self.train_x.loc[:, each] = (self.train_x[each] - mean)/std

        for each in self.test_x.columns:
            mean, std = self.test_x[each].mean(), self.test_x[each].std()
            self.test_x.loc[:, each] = (self.test_x[each] - mean)/std
            
    # Convert column 'group' to dummies
    def convert_group(self):
        
        print('Converting groups to dummies')

        group_train_dummies = pd.get_dummies(self.train_g, prefix='group')
        self.train_x = self.train_x.join(group_train_dummies)

        group_test_dummies = pd.get_dummies(self.test_g, prefix='group')
        self.test_x = self.test_x.join(group_test_dummies)

        print('Shape of train_x with group dummies: {}'.format(self.train_x.shape))
        print('Shape of test_x with group dummies: {}'.format(self.test_x.shape))

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
    def save_data_np(self):

        print('Saving data...')

        utils.save_np_to_pkl(self.train_x, self.prepro_path + 'train_x.p')
        utils.save_np_to_pkl(self.train_y, self.prepro_path + 'train_y.p')
        utils.save_np_to_pkl(self.train_w, self.prepro_path + 'train_w.p')
        utils.save_np_to_pkl(self.test_x, self.prepro_path + 'test_x.p')

    # Preprocessing
    def preprocess_np(self):

        start_time = time.time()

        self.load_data_np()

        # train_x, train_y, train_w, valid_x, valid_y, valid_w = self.random_spit_data(train_data_x, train_data_y)

        self.save_data_np()

        end_time = time.time()
        total_time = end_time - start_time

        print('Done!')
        print('Using {:.3}s'.format(total_time))

    # Save Data
    def save_data_pd(self):

        print('Saving data...')

        self.train_x.to_pickle(self.prepro_path + 'train_x.p')
        self.train_y.to_pickle(self.prepro_path + 'train_y.p')
        self.train_w.to_pickle(self.prepro_path + 'train_w.p')
        self.train_e.to_pickle(self.prepro_path + 'train_e.p')
        self.test_x.to_pickle(self.prepro_path + 'test_x.p')

    # Preprocessing
    def preprocess_pd(self):

        start_time = time.time()

        # Load original data
        self.load_data_pd()

        # Drop outlines
        self.drop_outlines()

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