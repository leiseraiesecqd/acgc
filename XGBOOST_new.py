import utils
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

#values to floats
localtime = time.asctime(time.localtime())
print('Starting Time:' + str(localtime) + '\n')

print('Loading Data...')

train_data = pd.read_csv(utils.file_train,header=0)
test_data = pd.read_csv(utils.file_test,header=0)
for feat in train_data.columns:
    train_data[feat] = train_data[feat].map(lambda x: round(x, 6))
for feat in test_data.columns:
    test_data[feat] = test_data[feat].map(lambda x: round(x, 6))

print('Shape train_data: {}\nShape test_data: {}\n'.format(train_data.shape, test_data.shape))

#preprocessing
print('Preprocessing Data...')
##drop unnecessary columns
X_train = train_data.drop(['id','weight','group','era'], axis=1)
y_label = X_train.pop('label')
X_train_group = train_data['group']
X_test = test_data.drop(['id','group'], axis=1)
X_test_group = test_data['group']

##drop outliners
print('dropping outliners...')
###drop upper outlines in X_train
upper = X_train.feature0.quantile(0.999)
X_train['feature0'].loc[X_train['feature0']>upper] = upper
upper = X_train.feature1.quantile(0.999)
X_train['feature1'].loc[X_train['feature1']>upper] = upper
upper = X_train.feature2.quantile(0.999)
X_train['feature2'].loc[X_train['feature2']>upper] = upper
upper = X_train.feature3.quantile(0.999)
X_train['feature3'].loc[X_train['feature3']>upper] = upper
upper = X_train.feature4.quantile(0.999)
X_train['feature4'].loc[X_train['feature4']>upper] = upper
upper = X_train.feature5.quantile(0.999)
X_train['feature5'].loc[X_train['feature5']>upper] = upper
upper = X_train.feature6.quantile(0.999)
X_train['feature6'].loc[X_train['feature6']>upper] = upper
upper = X_train.feature7.quantile(0.999)
X_train['feature7'].loc[X_train['feature7']>upper] = upper
upper = X_train.feature8.quantile(0.999)
X_train['feature8'].loc[X_train['feature8']>upper] = upper
upper = X_train.feature9.quantile(0.999)
X_train['feature9'].loc[X_train['feature9']>upper] = upper
upper = X_train.feature10.quantile(0.999)
X_train['feature10'].loc[X_train['feature10']>upper] = upper
upper = X_train.feature11.quantile(0.999)
X_train['feature11'].loc[X_train['feature11']>upper] = upper
upper = X_train.feature12.quantile(0.999)
X_train['feature12'].loc[X_train['feature12']>upper] = upper
upper = X_train.feature13.quantile(0.999)
X_train['feature13'].loc[X_train['feature13']>upper] = upper
upper = X_train.feature14.quantile(0.999)
X_train['feature14'].loc[X_train['feature14']>upper] = upper
upper = X_train.feature15.quantile(0.999)
X_train['feature15'].loc[X_train['feature15']>upper] = upper
upper = X_train.feature16.quantile(0.999)
X_train['feature16'].loc[X_train['feature16']>upper] = upper
upper = X_train.feature17.quantile(0.999)
X_train['feature17'].loc[X_train['feature17']>upper] = upper
upper = X_train.feature18.quantile(0.999)
X_train['feature18'].loc[X_train['feature18']>upper] = upper
upper = X_train.feature19.quantile(0.999)
X_train['feature19'].loc[X_train['feature19']>upper] = upper
upper = X_train.feature20.quantile(0.999)
X_train['feature20'].loc[X_train['feature20']>upper] = upper
upper = X_train.feature21.quantile(0.999)
X_train['feature21'].loc[X_train['feature21']>upper] = upper
upper = X_train.feature22.quantile(0.999)
X_train['feature22'].loc[X_train['feature22']>upper] = upper
upper = X_train.feature23.quantile(0.999)
X_train['feature23'].loc[X_train['feature23']>upper] = upper
upper = X_train.feature24.quantile(0.999)
X_train['feature24'].loc[X_train['feature24']>upper] = upper
upper = X_train.feature25.quantile(0.999)
X_train['feature25'].loc[X_train['feature25']>upper] = upper
upper = X_train.feature26.quantile(0.999)
X_train['feature26'].loc[X_train['feature26']>upper] = upper
upper = X_train.feature27.quantile(0.999)
X_train['feature27'].loc[X_train['feature27']>upper] = upper
upper = X_train.feature28.quantile(0.999)
X_train['feature28'].loc[X_train['feature28']>upper] = upper
upper = X_train.feature29.quantile(0.999)
X_train['feature29'].loc[X_train['feature29']>upper] = upper
upper = X_train.feature30.quantile(0.999)
X_train['feature30'].loc[X_train['feature30']>upper] = upper
upper = X_train.feature31.quantile(0.999)
X_train['feature31'].loc[X_train['feature31']>upper] = upper
upper = X_train.feature32.quantile(0.999)
X_train['feature32'].loc[X_train['feature32']>upper] = upper
upper = X_train.feature33.quantile(0.999)
X_train['feature33'].loc[X_train['feature33']>upper] = upper
upper = X_train.feature34.quantile(0.999)
X_train['feature34'].loc[X_train['feature34']>upper] = upper
upper = X_train.feature35.quantile(0.999)
X_train['feature35'].loc[X_train['feature35']>upper] = upper
upper = X_train.feature36.quantile(0.999)
X_train['feature36'].loc[X_train['feature36']>upper] = upper
upper = X_train.feature37.quantile(0.999)
X_train['feature37'].loc[X_train['feature37']>upper] = upper
upper = X_train.feature38.quantile(0.999)
X_train['feature38'].loc[X_train['feature38']>upper] = upper
upper = X_train.feature39.quantile(0.999)
X_train['feature39'].loc[X_train['feature39']>upper] = upper
upper = X_train.feature40.quantile(0.999)
X_train['feature40'].loc[X_train['feature40']>upper] = upper
upper = X_train.feature41.quantile(0.999)
X_train['feature41'].loc[X_train['feature41']>upper] = upper
upper = X_train.feature42.quantile(0.999)
X_train['feature42'].loc[X_train['feature42']>upper] = upper
upper = X_train.feature43.quantile(0.999)
X_train['feature43'].loc[X_train['feature43']>upper] = upper
upper = X_train.feature44.quantile(0.999)
X_train['feature44'].loc[X_train['feature44']>upper] = upper
upper = X_train.feature45.quantile(0.999)
X_train['feature45'].loc[X_train['feature45']>upper] = upper
upper = X_train.feature46.quantile(0.999)
X_train['feature46'].loc[X_train['feature46']>upper] = upper
upper = X_train.feature47.quantile(0.999)
X_train['feature47'].loc[X_train['feature47']>upper] = upper
upper = X_train.feature48.quantile(0.999)
X_train['feature48'].loc[X_train['feature48']>upper] = upper
upper = X_train.feature49.quantile(0.999)
X_train['feature49'].loc[X_train['feature49']>upper] = upper
upper = X_train.feature50.quantile(0.999)
X_train['feature50'].loc[X_train['feature50']>upper] = upper
upper = X_train.feature51.quantile(0.999)
X_train['feature51'].loc[X_train['feature51']>upper] = upper
upper = X_train.feature52.quantile(0.999)
X_train['feature52'].loc[X_train['feature52']>upper] = upper
upper = X_train.feature53.quantile(0.999)
X_train['feature53'].loc[X_train['feature53']>upper] = upper
upper = X_train.feature54.quantile(0.999)
X_train['feature54'].loc[X_train['feature54']>upper] = upper
upper = X_train.feature55.quantile(0.999)
X_train['feature55'].loc[X_train['feature55']>upper] = upper
upper = X_train.feature56.quantile(0.999)
X_train['feature56'].loc[X_train['feature56']>upper] = upper
upper = X_train.feature57.quantile(0.999)
X_train['feature57'].loc[X_train['feature57']>upper] = upper
upper = X_train.feature58.quantile(0.999)
X_train['feature58'].loc[X_train['feature58']>upper] = upper
upper = X_train.feature59.quantile(0.999)
X_train['feature59'].loc[X_train['feature59']>upper] = upper
upper = X_train.feature60.quantile(0.999)
X_train['feature60'].loc[X_train['feature60']>upper] = upper
upper = X_train.feature61.quantile(0.999)
X_train['feature61'].loc[X_train['feature61']>upper] = upper
upper = X_train.feature62.quantile(0.999)
X_train['feature62'].loc[X_train['feature62']>upper] = upper
upper = X_train.feature63.quantile(0.999)
X_train['feature63'].loc[X_train['feature63']>upper] = upper
upper = X_train.feature64.quantile(0.999)
X_train['feature64'].loc[X_train['feature64']>upper] = upper
upper = X_train.feature65.quantile(0.999)
X_train['feature65'].loc[X_train['feature65']>upper] = upper
upper = X_train.feature66.quantile(0.999)
X_train['feature66'].loc[X_train['feature66']>upper] = upper
upper = X_train.feature67.quantile(0.999)
X_train['feature67'].loc[X_train['feature67']>upper] = upper
upper = X_train.feature68.quantile(0.999)
X_train['feature68'].loc[X_train['feature68']>upper] = upper
upper = X_train.feature69.quantile(0.999)
X_train['feature69'].loc[X_train['feature69']>upper] = upper
upper = X_train.feature70.quantile(0.999)
X_train['feature70'].loc[X_train['feature70']>upper] = upper
upper = X_train.feature70.quantile(0.999)
X_train['feature71'].loc[X_train['feature71']>upper] = upper
upper = X_train.feature71.quantile(0.999)
X_train['feature72'].loc[X_train['feature72']>upper] = upper
upper = X_train.feature73.quantile(0.999)
X_train['feature73'].loc[X_train['feature73']>upper] = upper
upper = X_train.feature74.quantile(0.999)
X_train['feature74'].loc[X_train['feature74']>upper] = upper
upper = X_train.feature75.quantile(0.999)
X_train['feature75'].loc[X_train['feature75']>upper] = upper
upper = X_train.feature76.quantile(0.999)
X_train['feature76'].loc[X_train['feature76']>upper] = upper
upper = X_train.feature77.quantile(0.999)
X_train['feature77'].loc[X_train['feature77']>upper] = upper
upper = X_train.feature78.quantile(0.999)
X_train['feature78'].loc[X_train['feature78']>upper] = upper
upper = X_train.feature79.quantile(0.999)
X_train['feature79'].loc[X_train['feature79']>upper] = upper
upper = X_train.feature80.quantile(0.999)
X_train['feature80'].loc[X_train['feature80']>upper] = upper
upper = X_train.feature81.quantile(0.999)
X_train['feature81'].loc[X_train['feature81']>upper] = upper
upper = X_train.feature81.quantile(0.999)
X_train['feature81'].loc[X_train['feature81']>upper] = upper
upper = X_train.feature82.quantile(0.999)
X_train['feature82'].loc[X_train['feature82']>upper] = upper
upper = X_train.feature83.quantile(0.999)
X_train['feature83'].loc[X_train['feature83']>upper] = upper
upper = X_train.feature84.quantile(0.999)
X_train['feature84'].loc[X_train['feature84']>upper] = upper
upper = X_train.feature85.quantile(0.999)
X_train['feature85'].loc[X_train['feature85']>upper] = upper
upper = X_train.feature86.quantile(0.999)
X_train['feature86'].loc[X_train['feature86']>upper] = upper
upper = X_train.feature87.quantile(0.999)
X_train['feature87'].loc[X_train['feature87']>upper] = upper

###drop lower outlines in X_train
lower = X_train.feature0.quantile(0.001)
X_train['feature0'].loc[X_train['feature0']<lower] = lower
lower = X_train.feature1.quantile(0.001)
X_train['feature1'].loc[X_train['feature1']<lower] = lower
lower = X_train.feature2.quantile(0.001)
X_train['feature2'].loc[X_train['feature2']<lower] = lower
lower = X_train.feature3.quantile(0.001)
X_train['feature3'].loc[X_train['feature3']<lower] = lower
lower = X_train.feature4.quantile(0.001)
X_train['feature4'].loc[X_train['feature4']<lower] = lower
lower = X_train.feature5.quantile(0.001)
X_train['feature5'].loc[X_train['feature5']<lower] = lower
lower = X_train.feature6.quantile(0.001)
X_train['feature6'].loc[X_train['feature6']<lower] = lower
lower = X_train.feature7.quantile(0.001)
X_train['feature7'].loc[X_train['feature7']<lower] = lower
lower = X_train.feature8.quantile(0.001)
X_train['feature8'].loc[X_train['feature8']<lower] = lower
lower = X_train.feature9.quantile(0.001)
X_train['feature9'].loc[X_train['feature9']<lower] = lower
lower = X_train.feature10.quantile(0.001)
X_train['feature10'].loc[X_train['feature10']<lower] = lower
lower = X_train.feature11.quantile(0.001)
X_train['feature11'].loc[X_train['feature11']<lower] = lower
lower = X_train.feature12.quantile(0.001)
X_train['feature12'].loc[X_train['feature12']<lower] = lower
lower = X_train.feature13.quantile(0.001)
X_train['feature13'].loc[X_train['feature13']<lower] = lower
lower = X_train.feature14.quantile(0.001)
X_train['feature14'].loc[X_train['feature14']<lower] = lower
lower = X_train.feature15.quantile(0.001)
X_train['feature15'].loc[X_train['feature15']<lower] = lower
lower = X_train.feature16.quantile(0.001)
X_train['feature16'].loc[X_train['feature16']<lower] = lower
lower = X_train.feature17.quantile(0.001)
X_train['feature17'].loc[X_train['feature17']<lower] = lower
lower = X_train.feature18.quantile(0.001)
X_train['feature18'].loc[X_train['feature18']<lower] = lower
lower = X_train.feature19.quantile(0.001)
X_train['feature19'].loc[X_train['feature19']<lower] = lower
lower = X_train.feature20.quantile(0.001)
X_train['feature20'].loc[X_train['feature20']<lower] = lower
lower = X_train.feature21.quantile(0.001)
X_train['feature21'].loc[X_train['feature21']<lower] = lower
lower = X_train.feature22.quantile(0.001)
X_train['feature22'].loc[X_train['feature22']<lower] = lower
lower = X_train.feature23.quantile(0.001)
X_train['feature23'].loc[X_train['feature23']<lower] = lower
lower = X_train.feature24.quantile(0.001)
X_train['feature24'].loc[X_train['feature24']<lower] = lower
lower = X_train.feature25.quantile(0.001)
X_train['feature25'].loc[X_train['feature25']<lower] = lower
lower = X_train.feature26.quantile(0.001)
X_train['feature26'].loc[X_train['feature26']<lower] = lower
lower = X_train.feature27.quantile(0.001)
X_train['feature27'].loc[X_train['feature27']<lower] = lower
lower = X_train.feature28.quantile(0.001)
X_train['feature28'].loc[X_train['feature28']<lower] = lower
lower = X_train.feature29.quantile(0.001)
X_train['feature29'].loc[X_train['feature29']<lower] = lower
lower = X_train.feature30.quantile(0.001)
X_train['feature30'].loc[X_train['feature30']<lower] = lower
lower = X_train.feature31.quantile(0.001)
X_train['feature31'].loc[X_train['feature31']<lower] = lower
lower = X_train.feature32.quantile(0.001)
X_train['feature32'].loc[X_train['feature32']<lower] = lower
lower = X_train.feature33.quantile(0.001)
X_train['feature33'].loc[X_train['feature33']<lower] = lower
lower = X_train.feature34.quantile(0.001)
X_train['feature34'].loc[X_train['feature34']<lower] = lower
lower = X_train.feature35.quantile(0.001)
X_train['feature35'].loc[X_train['feature35']<lower] = lower
lower = X_train.feature36.quantile(0.001)
X_train['feature36'].loc[X_train['feature36']<lower] = lower
lower = X_train.feature37.quantile(0.001)
X_train['feature37'].loc[X_train['feature37']<lower] = lower
lower = X_train.feature38.quantile(0.001)
X_train['feature38'].loc[X_train['feature38']<lower] = lower
lower = X_train.feature39.quantile(0.001)
X_train['feature39'].loc[X_train['feature39']<lower] = lower
lower = X_train.feature40.quantile(0.001)
X_train['feature40'].loc[X_train['feature40']<lower] = lower
lower = X_train.feature41.quantile(0.001)
X_train['feature41'].loc[X_train['feature41']<lower] = lower
lower = X_train.feature42.quantile(0.001)
X_train['feature42'].loc[X_train['feature42']<lower] = lower
lower = X_train.feature43.quantile(0.001)
X_train['feature43'].loc[X_train['feature43']<lower] = lower
lower = X_train.feature44.quantile(0.001)
X_train['feature44'].loc[X_train['feature44']<lower] = lower
lower = X_train.feature45.quantile(0.001)
X_train['feature45'].loc[X_train['feature45']<lower] = lower
lower = X_train.feature46.quantile(0.001)
X_train['feature46'].loc[X_train['feature46']<lower] = lower
lower = X_train.feature47.quantile(0.001)
X_train['feature47'].loc[X_train['feature47']<lower] = lower
lower = X_train.feature48.quantile(0.001)
X_train['feature48'].loc[X_train['feature48']<lower] = lower
lower = X_train.feature49.quantile(0.001)
X_train['feature49'].loc[X_train['feature49']<lower] = lower
lower = X_train.feature50.quantile(0.001)
X_train['feature50'].loc[X_train['feature50']<lower] = lower
lower = X_train.feature51.quantile(0.001)
X_train['feature51'].loc[X_train['feature51']<lower] = lower
lower = X_train.feature52.quantile(0.001)
X_train['feature52'].loc[X_train['feature52']<lower] = lower
lower = X_train.feature53.quantile(0.001)
X_train['feature53'].loc[X_train['feature53']<lower] = lower
lower = X_train.feature54.quantile(0.001)
X_train['feature54'].loc[X_train['feature54']<lower] = lower
lower = X_train.feature55.quantile(0.001)
X_train['feature55'].loc[X_train['feature55']<lower] = lower
lower = X_train.feature56.quantile(0.001)
X_train['feature56'].loc[X_train['feature56']<lower] = lower
lower = X_train.feature57.quantile(0.001)
X_train['feature57'].loc[X_train['feature57']<lower] = lower
lower = X_train.feature58.quantile(0.001)
X_train['feature58'].loc[X_train['feature58']<lower] = lower
lower = X_train.feature59.quantile(0.001)
X_train['feature59'].loc[X_train['feature59']<lower] = lower
lower = X_train.feature60.quantile(0.001)
X_train['feature60'].loc[X_train['feature60']<lower] = lower
lower = X_train.feature61.quantile(0.001)
X_train['feature61'].loc[X_train['feature61']<lower] = lower
lower = X_train.feature62.quantile(0.001)
X_train['feature62'].loc[X_train['feature62']<lower] = lower
lower = X_train.feature63.quantile(0.001)
X_train['feature63'].loc[X_train['feature63']<lower] = lower
lower = X_train.feature64.quantile(0.001)
X_train['feature64'].loc[X_train['feature64']<lower] = lower
lower = X_train.feature65.quantile(0.001)
X_train['feature65'].loc[X_train['feature65']<lower] = lower
lower = X_train.feature66.quantile(0.001)
X_train['feature66'].loc[X_train['feature66']<lower] = lower
lower = X_train.feature67.quantile(0.001)
X_train['feature67'].loc[X_train['feature67']<lower] = lower
lower = X_train.feature68.quantile(0.001)
X_train['feature68'].loc[X_train['feature68']<lower] = lower
lower = X_train.feature69.quantile(0.001)
X_train['feature69'].loc[X_train['feature69']<lower] = lower
lower = X_train.feature70.quantile(0.001)
X_train['feature70'].loc[X_train['feature70']<lower] = lower
lower = X_train.feature70.quantile(0.001)
X_train['feature71'].loc[X_train['feature71']<lower] = lower
lower = X_train.feature71.quantile(0.001)
X_train['feature72'].loc[X_train['feature72']<lower] = lower
lower = X_train.feature73.quantile(0.001)
X_train['feature73'].loc[X_train['feature73']<lower] = lower
lower = X_train.feature74.quantile(0.001)
X_train['feature74'].loc[X_train['feature74']<lower] = lower
lower = X_train.feature75.quantile(0.001)
X_train['feature75'].loc[X_train['feature75']<lower] = lower
lower = X_train.feature76.quantile(0.001)
X_train['feature76'].loc[X_train['feature76']<lower] = lower
lower = X_train.feature77.quantile(0.001)
X_train['feature77'].loc[X_train['feature77']<lower] = lower
lower = X_train.feature78.quantile(0.001)
X_train['feature78'].loc[X_train['feature78']<lower] = lower
lower = X_train.feature79.quantile(0.001)
X_train['feature79'].loc[X_train['feature79']<lower] = lower
lower = X_train.feature80.quantile(0.001)
X_train['feature80'].loc[X_train['feature80']<lower] = lower
lower = X_train.feature81.quantile(0.001)
X_train['feature81'].loc[X_train['feature81']<lower] = lower
lower = X_train.feature81.quantile(0.001)
X_train['feature81'].loc[X_train['feature81']<lower] = lower
lower = X_train.feature82.quantile(0.001)
X_train['feature82'].loc[X_train['feature82']<lower] = lower
lower = X_train.feature83.quantile(0.001)
X_train['feature83'].loc[X_train['feature83']<lower] = lower
lower = X_train.feature84.quantile(0.001)
X_train['feature84'].loc[X_train['feature84']<lower] = lower
lower = X_train.feature85.quantile(0.001)
X_train['feature85'].loc[X_train['feature85']<lower] = lower
lower = X_train.feature86.quantile(0.001)
X_train['feature86'].loc[X_train['feature86']<lower] = lower
lower = X_train.feature87.quantile(0.001)
X_train['feature87'].loc[X_train['feature87']<lower] = lower

###drop upper outlines in X_test
upper = X_test.feature0.quantile(0.999)
X_test['feature0'].loc[X_test['feature0']>upper] = upper
upper = X_test.feature1.quantile(0.999)
X_test['feature1'].loc[X_test['feature1']>upper] = upper
upper = X_test.feature2.quantile(0.999)
X_test['feature2'].loc[X_test['feature2']>upper] = upper
upper = X_test.feature3.quantile(0.999)
X_test['feature3'].loc[X_test['feature3']>upper] = upper
upper = X_test.feature4.quantile(0.999)
X_test['feature4'].loc[X_test['feature4']>upper] = upper
upper = X_test.feature5.quantile(0.999)
X_test['feature5'].loc[X_test['feature5']>upper] = upper
upper = X_test.feature6.quantile(0.999)
X_test['feature6'].loc[X_test['feature6']>upper] = upper
upper = X_test.feature7.quantile(0.999)
X_test['feature7'].loc[X_test['feature7']>upper] = upper
upper = X_test.feature8.quantile(0.999)
X_test['feature8'].loc[X_test['feature8']>upper] = upper
upper = X_test.feature9.quantile(0.999)
X_test['feature9'].loc[X_test['feature9']>upper] = upper
upper = X_test.feature10.quantile(0.999)
X_test['feature10'].loc[X_test['feature10']>upper] = upper
upper = X_test.feature11.quantile(0.999)
X_test['feature11'].loc[X_test['feature11']>upper] = upper
upper = X_test.feature12.quantile(0.999)
X_test['feature12'].loc[X_test['feature12']>upper] = upper
upper = X_test.feature13.quantile(0.999)
X_test['feature13'].loc[X_test['feature13']>upper] = upper
upper = X_test.feature14.quantile(0.999)
X_test['feature14'].loc[X_test['feature14']>upper] = upper
upper = X_test.feature15.quantile(0.999)
X_test['feature15'].loc[X_test['feature15']>upper] = upper
upper = X_test.feature16.quantile(0.999)
X_test['feature16'].loc[X_test['feature16']>upper] = upper
upper = X_test.feature17.quantile(0.999)
X_test['feature17'].loc[X_test['feature17']>upper] = upper
upper = X_test.feature18.quantile(0.999)
X_test['feature18'].loc[X_test['feature18']>upper] = upper
upper = X_test.feature19.quantile(0.999)
X_test['feature19'].loc[X_test['feature19']>upper] = upper
upper = X_test.feature20.quantile(0.999)
X_test['feature20'].loc[X_test['feature20']>upper] = upper
upper = X_test.feature21.quantile(0.999)
X_test['feature21'].loc[X_test['feature21']>upper] = upper
upper = X_test.feature22.quantile(0.999)
X_test['feature22'].loc[X_test['feature22']>upper] = upper
upper = X_test.feature23.quantile(0.999)
X_test['feature23'].loc[X_test['feature23']>upper] = upper
upper = X_test.feature24.quantile(0.999)
X_test['feature24'].loc[X_test['feature24']>upper] = upper
upper = X_test.feature25.quantile(0.999)
X_test['feature25'].loc[X_test['feature25']>upper] = upper
upper = X_test.feature26.quantile(0.999)
X_test['feature26'].loc[X_test['feature26']>upper] = upper
upper = X_test.feature27.quantile(0.999)
X_test['feature27'].loc[X_test['feature27']>upper] = upper
upper = X_test.feature28.quantile(0.999)
X_test['feature28'].loc[X_test['feature28']>upper] = upper
upper = X_test.feature29.quantile(0.999)
X_test['feature29'].loc[X_test['feature29']>upper] = upper
upper = X_test.feature30.quantile(0.999)
X_test['feature30'].loc[X_test['feature30']>upper] = upper
upper = X_test.feature31.quantile(0.999)
X_test['feature31'].loc[X_test['feature31']>upper] = upper
upper = X_test.feature32.quantile(0.999)
X_test['feature32'].loc[X_test['feature32']>upper] = upper
upper = X_test.feature33.quantile(0.999)
X_test['feature33'].loc[X_test['feature33']>upper] = upper
upper = X_test.feature34.quantile(0.999)
X_test['feature34'].loc[X_test['feature34']>upper] = upper
upper = X_test.feature35.quantile(0.999)
X_test['feature35'].loc[X_test['feature35']>upper] = upper
upper = X_test.feature36.quantile(0.999)
X_test['feature36'].loc[X_test['feature36']>upper] = upper
upper = X_test.feature37.quantile(0.999)
X_test['feature37'].loc[X_test['feature37']>upper] = upper
upper = X_test.feature38.quantile(0.999)
X_test['feature38'].loc[X_test['feature38']>upper] = upper
upper = X_test.feature39.quantile(0.999)
X_test['feature39'].loc[X_test['feature39']>upper] = upper
upper = X_test.feature40.quantile(0.999)
X_test['feature40'].loc[X_test['feature40']>upper] = upper
upper = X_test.feature41.quantile(0.999)
X_test['feature41'].loc[X_test['feature41']>upper] = upper
upper = X_test.feature42.quantile(0.999)
X_test['feature42'].loc[X_test['feature42']>upper] = upper
upper = X_test.feature43.quantile(0.999)
X_test['feature43'].loc[X_test['feature43']>upper] = upper
upper = X_test.feature44.quantile(0.999)
X_test['feature44'].loc[X_test['feature44']>upper] = upper
upper = X_test.feature45.quantile(0.999)
X_test['feature45'].loc[X_test['feature45']>upper] = upper
upper = X_test.feature46.quantile(0.999)
X_test['feature46'].loc[X_test['feature46']>upper] = upper
upper = X_test.feature47.quantile(0.999)
X_test['feature47'].loc[X_test['feature47']>upper] = upper
upper = X_test.feature48.quantile(0.999)
X_test['feature48'].loc[X_test['feature48']>upper] = upper
upper = X_test.feature49.quantile(0.999)
X_test['feature49'].loc[X_test['feature49']>upper] = upper
upper = X_test.feature50.quantile(0.999)
X_test['feature50'].loc[X_test['feature50']>upper] = upper
upper = X_test.feature51.quantile(0.999)
X_test['feature51'].loc[X_test['feature51']>upper] = upper
upper = X_test.feature52.quantile(0.999)
X_test['feature52'].loc[X_test['feature52']>upper] = upper
upper = X_test.feature53.quantile(0.999)
X_test['feature53'].loc[X_test['feature53']>upper] = upper
upper = X_test.feature54.quantile(0.999)
X_test['feature54'].loc[X_test['feature54']>upper] = upper
upper = X_test.feature55.quantile(0.999)
X_test['feature55'].loc[X_test['feature55']>upper] = upper
upper = X_test.feature56.quantile(0.999)
X_test['feature56'].loc[X_test['feature56']>upper] = upper
upper = X_test.feature57.quantile(0.999)
X_test['feature57'].loc[X_test['feature57']>upper] = upper
upper = X_test.feature58.quantile(0.999)
X_test['feature58'].loc[X_test['feature58']>upper] = upper
upper = X_test.feature59.quantile(0.999)
X_test['feature59'].loc[X_test['feature59']>upper] = upper
upper = X_test.feature60.quantile(0.999)
X_test['feature60'].loc[X_test['feature60']>upper] = upper
upper = X_test.feature61.quantile(0.999)
X_test['feature61'].loc[X_test['feature61']>upper] = upper
upper = X_test.feature62.quantile(0.999)
X_test['feature62'].loc[X_test['feature62']>upper] = upper
upper = X_test.feature63.quantile(0.999)
X_test['feature63'].loc[X_test['feature63']>upper] = upper
upper = X_test.feature64.quantile(0.999)
X_test['feature64'].loc[X_test['feature64']>upper] = upper
upper = X_test.feature65.quantile(0.999)
X_test['feature65'].loc[X_test['feature65']>upper] = upper
upper = X_test.feature66.quantile(0.999)
X_test['feature66'].loc[X_test['feature66']>upper] = upper
upper = X_test.feature67.quantile(0.999)
X_test['feature67'].loc[X_test['feature67']>upper] = upper
upper = X_test.feature68.quantile(0.999)
X_test['feature68'].loc[X_test['feature68']>upper] = upper
upper = X_test.feature69.quantile(0.999)
X_test['feature69'].loc[X_test['feature69']>upper] = upper
upper = X_test.feature70.quantile(0.999)
X_test['feature70'].loc[X_test['feature70']>upper] = upper
upper = X_test.feature70.quantile(0.999)
X_test['feature71'].loc[X_test['feature71']>upper] = upper
upper = X_test.feature71.quantile(0.999)
X_test['feature72'].loc[X_test['feature72']>upper] = upper
upper = X_test.feature73.quantile(0.999)
X_test['feature73'].loc[X_test['feature73']>upper] = upper
upper = X_test.feature74.quantile(0.999)
X_test['feature74'].loc[X_test['feature74']>upper] = upper
upper = X_test.feature75.quantile(0.999)
X_test['feature75'].loc[X_test['feature75']>upper] = upper
upper = X_test.feature76.quantile(0.999)
X_test['feature76'].loc[X_test['feature76']>upper] = upper
upper = X_test.feature77.quantile(0.999)
X_test['feature77'].loc[X_test['feature77']>upper] = upper
upper = X_test.feature78.quantile(0.999)
X_test['feature78'].loc[X_test['feature78']>upper] = upper
upper = X_test.feature79.quantile(0.999)
X_test['feature79'].loc[X_test['feature79']>upper] = upper
upper = X_test.feature80.quantile(0.999)
X_test['feature80'].loc[X_test['feature80']>upper] = upper
upper = X_test.feature81.quantile(0.999)
X_test['feature81'].loc[X_test['feature81']>upper] = upper
upper = X_test.feature81.quantile(0.999)
X_test['feature81'].loc[X_test['feature81']>upper] = upper
upper = X_test.feature82.quantile(0.999)
X_test['feature82'].loc[X_test['feature82']>upper] = upper
upper = X_test.feature83.quantile(0.999)
X_test['feature83'].loc[X_test['feature83']>upper] = upper
upper = X_test.feature84.quantile(0.999)
X_test['feature84'].loc[X_test['feature84']>upper] = upper
upper = X_test.feature85.quantile(0.999)
X_test['feature85'].loc[X_test['feature85']>upper] = upper
upper = X_test.feature86.quantile(0.999)
X_test['feature86'].loc[X_test['feature86']>upper] = upper
upper = X_test.feature87.quantile(0.999)
X_test['feature87'].loc[X_test['feature87']>upper] = upper

###drop lower outlines in X_test
lower = X_test.feature0.quantile(0.001)
X_test['feature0'].loc[X_test['feature0']<lower] = lower
lower = X_test.feature1.quantile(0.001)
X_test['feature1'].loc[X_test['feature1']<lower] = lower
lower = X_test.feature2.quantile(0.001)
X_test['feature2'].loc[X_test['feature2']<lower] = lower
lower = X_test.feature3.quantile(0.001)
X_test['feature3'].loc[X_test['feature3']<lower] = lower
lower = X_test.feature4.quantile(0.001)
X_test['feature4'].loc[X_test['feature4']<lower] = lower
lower = X_test.feature5.quantile(0.001)
X_test['feature5'].loc[X_test['feature5']<lower] = lower
lower = X_test.feature6.quantile(0.001)
X_test['feature6'].loc[X_test['feature6']<lower] = lower
lower = X_test.feature7.quantile(0.001)
X_test['feature7'].loc[X_test['feature7']<lower] = lower
lower = X_test.feature8.quantile(0.001)
X_test['feature8'].loc[X_test['feature8']<lower] = lower
lower = X_test.feature9.quantile(0.001)
X_test['feature9'].loc[X_test['feature9']<lower] = lower
lower = X_test.feature10.quantile(0.001)
X_test['feature10'].loc[X_test['feature10']<lower] = lower
lower = X_test.feature11.quantile(0.001)
X_test['feature11'].loc[X_test['feature11']<lower] = lower
lower = X_test.feature12.quantile(0.001)
X_test['feature12'].loc[X_test['feature12']<lower] = lower
lower = X_test.feature13.quantile(0.001)
X_test['feature13'].loc[X_test['feature13']<lower] = lower
lower = X_test.feature14.quantile(0.001)
X_test['feature14'].loc[X_test['feature14']<lower] = lower
lower = X_test.feature15.quantile(0.001)
X_test['feature15'].loc[X_test['feature15']<lower] = lower
lower = X_test.feature16.quantile(0.001)
X_test['feature16'].loc[X_test['feature16']<lower] = lower
lower = X_test.feature17.quantile(0.001)
X_test['feature17'].loc[X_test['feature17']<lower] = lower
lower = X_test.feature18.quantile(0.001)
X_test['feature18'].loc[X_test['feature18']<lower] = lower
lower = X_test.feature19.quantile(0.001)
X_test['feature19'].loc[X_test['feature19']<lower] = lower
lower = X_test.feature20.quantile(0.001)
X_test['feature20'].loc[X_test['feature20']<lower] = lower
lower = X_test.feature21.quantile(0.001)
X_test['feature21'].loc[X_test['feature21']<lower] = lower
lower = X_test.feature22.quantile(0.001)
X_test['feature22'].loc[X_test['feature22']<lower] = lower
lower = X_test.feature23.quantile(0.001)
X_test['feature23'].loc[X_test['feature23']<lower] = lower
lower = X_test.feature24.quantile(0.001)
X_test['feature24'].loc[X_test['feature24']<lower] = lower
lower = X_test.feature25.quantile(0.001)
X_test['feature25'].loc[X_test['feature25']<lower] = lower
lower = X_test.feature26.quantile(0.001)
X_test['feature26'].loc[X_test['feature26']<lower] = lower
lower = X_test.feature27.quantile(0.001)
X_test['feature27'].loc[X_test['feature27']<lower] = lower
lower = X_test.feature28.quantile(0.001)
X_test['feature28'].loc[X_test['feature28']<lower] = lower
lower = X_test.feature29.quantile(0.001)
X_test['feature29'].loc[X_test['feature29']<lower] = lower
lower = X_test.feature30.quantile(0.001)
X_test['feature30'].loc[X_test['feature30']<lower] = lower
lower = X_test.feature31.quantile(0.001)
X_test['feature31'].loc[X_test['feature31']<lower] = lower
lower = X_test.feature32.quantile(0.001)
X_test['feature32'].loc[X_test['feature32']<lower] = lower
lower = X_test.feature33.quantile(0.001)
X_test['feature33'].loc[X_test['feature33']<lower] = lower
lower = X_test.feature34.quantile(0.001)
X_test['feature34'].loc[X_test['feature34']<lower] = lower
lower = X_test.feature35.quantile(0.001)
X_test['feature35'].loc[X_test['feature35']<lower] = lower
lower = X_test.feature36.quantile(0.001)
X_test['feature36'].loc[X_test['feature36']<lower] = lower
lower = X_test.feature37.quantile(0.001)
X_test['feature37'].loc[X_test['feature37']<lower] = lower
lower = X_test.feature38.quantile(0.001)
X_test['feature38'].loc[X_test['feature38']<lower] = lower
lower = X_test.feature39.quantile(0.001)
X_test['feature39'].loc[X_test['feature39']<lower] = lower
lower = X_test.feature40.quantile(0.001)
X_test['feature40'].loc[X_test['feature40']<lower] = lower
lower = X_test.feature41.quantile(0.001)
X_test['feature41'].loc[X_test['feature41']<lower] = lower
lower = X_test.feature42.quantile(0.001)
X_test['feature42'].loc[X_test['feature42']<lower] = lower
lower = X_test.feature43.quantile(0.001)
X_test['feature43'].loc[X_test['feature43']<lower] = lower
lower = X_test.feature44.quantile(0.001)
X_test['feature44'].loc[X_test['feature44']<lower] = lower
lower = X_test.feature45.quantile(0.001)
X_test['feature45'].loc[X_test['feature45']<lower] = lower
lower = X_test.feature46.quantile(0.001)
X_test['feature46'].loc[X_test['feature46']<lower] = lower
lower = X_test.feature47.quantile(0.001)
X_test['feature47'].loc[X_test['feature47']<lower] = lower
lower = X_test.feature48.quantile(0.001)
X_test['feature48'].loc[X_test['feature48']<lower] = lower
lower = X_test.feature49.quantile(0.001)
X_test['feature49'].loc[X_test['feature49']<lower] = lower
lower = X_test.feature50.quantile(0.001)
X_test['feature50'].loc[X_test['feature50']<lower] = lower
lower = X_test.feature51.quantile(0.001)
X_test['feature51'].loc[X_test['feature51']<lower] = lower
lower = X_test.feature52.quantile(0.001)
X_test['feature52'].loc[X_test['feature52']<lower] = lower
lower = X_test.feature53.quantile(0.001)
X_test['feature53'].loc[X_test['feature53']<lower] = lower
lower = X_test.feature54.quantile(0.001)
X_test['feature54'].loc[X_test['feature54']<lower] = lower
lower = X_test.feature55.quantile(0.001)
X_test['feature55'].loc[X_test['feature55']<lower] = lower
lower = X_test.feature56.quantile(0.001)
X_test['feature56'].loc[X_test['feature56']<lower] = lower
lower = X_test.feature57.quantile(0.001)
X_test['feature57'].loc[X_test['feature57']<lower] = lower
lower = X_test.feature58.quantile(0.001)
X_test['feature58'].loc[X_test['feature58']<lower] = lower
lower = X_test.feature59.quantile(0.001)
X_test['feature59'].loc[X_test['feature59']<lower] = lower
lower = X_test.feature60.quantile(0.001)
X_test['feature60'].loc[X_test['feature60']<lower] = lower
lower = X_test.feature61.quantile(0.001)
X_test['feature61'].loc[X_test['feature61']<lower] = lower
lower = X_test.feature62.quantile(0.001)
X_test['feature62'].loc[X_test['feature62']<lower] = lower
lower = X_test.feature63.quantile(0.001)
X_test['feature63'].loc[X_test['feature63']<lower] = lower
lower = X_test.feature64.quantile(0.001)
X_test['feature64'].loc[X_test['feature64']<lower] = lower
lower = X_test.feature65.quantile(0.001)
X_test['feature65'].loc[X_test['feature65']<lower] = lower
lower = X_test.feature66.quantile(0.001)
X_test['feature66'].loc[X_test['feature66']<lower] = lower
lower = X_test.feature67.quantile(0.001)
X_test['feature67'].loc[X_test['feature67']<lower] = lower
lower = X_test.feature68.quantile(0.001)
X_test['feature68'].loc[X_test['feature68']<lower] = lower
lower = X_test.feature69.quantile(0.001)
X_test['feature69'].loc[X_test['feature69']<lower] = lower
lower = X_test.feature70.quantile(0.001)
X_test['feature70'].loc[X_test['feature70']<lower] = lower
lower = X_test.feature70.quantile(0.001)
X_test['feature71'].loc[X_test['feature71']<lower] = lower
lower = X_test.feature71.quantile(0.001)
X_test['feature72'].loc[X_test['feature72']<lower] = lower
lower = X_test.feature73.quantile(0.001)
X_test['feature73'].loc[X_test['feature73']<lower] = lower
lower = X_test.feature74.quantile(0.001)
X_test['feature74'].loc[X_test['feature74']<lower] = lower
lower = X_test.feature75.quantile(0.001)
X_test['feature75'].loc[X_test['feature75']<lower] = lower
lower = X_test.feature76.quantile(0.001)
X_test['feature76'].loc[X_test['feature76']<lower] = lower
lower = X_test.feature77.quantile(0.001)
X_test['feature77'].loc[X_test['feature77']<lower] = lower
lower = X_test.feature78.quantile(0.001)
X_test['feature78'].loc[X_test['feature78']<lower] = lower
lower = X_test.feature79.quantile(0.001)
X_test['feature79'].loc[X_test['feature79']<lower] = lower
lower = X_test.feature80.quantile(0.001)
X_test['feature80'].loc[X_test['feature80']<lower] = lower
lower = X_test.feature81.quantile(0.001)
X_test['feature81'].loc[X_test['feature81']<lower] = lower
lower = X_test.feature81.quantile(0.001)
X_test['feature81'].loc[X_test['feature81']<lower] = lower
lower = X_test.feature82.quantile(0.001)
X_test['feature82'].loc[X_test['feature82']<lower] = lower
lower = X_test.feature83.quantile(0.001)
X_test['feature83'].loc[X_test['feature83']<lower] = lower
lower = X_test.feature84.quantile(0.001)
X_test['feature84'].loc[X_test['feature84']<lower] = lower
lower = X_test.feature85.quantile(0.001)
X_test['feature85'].loc[X_test['feature85']<lower] = lower
lower = X_test.feature86.quantile(0.001)
X_test['feature86'].loc[X_test['feature86']<lower] = lower
lower = X_test.feature87.quantile(0.001)
X_test['feature87'].loc[X_test['feature87']<lower] = lower

##scaleing data
print('scaling data...')
for each in X_train.columns:
    mean, std = X_train[each].mean(), X_train[each].std()
    X_train.loc[:, each] = (X_train[each] - mean)/std

for each in X_test.columns:
    mean, std = X_test[each].mean(), X_test[each].std()
    X_test.loc[:, each] = (X_test[each] - mean)/std

X_train = X_train.join(X_train_group)
X_test = X_test.join(X_test_group)

##convert columm 'group' to dummies
print('adding dummies...')

group_train_dummies = pd.get_dummies(X_train['group'],prefix='group')
X_train.drop(['group'],axis=1,inplace=True)
X_train=X_train.join(group_train_dummies)

group_test_dummies = pd.get_dummies(X_test['group'],prefix='group')
X_test.drop(['group'],axis=1,inplace=True)
X_test=X_test.join(group_test_dummies)

print('Shape X_train with group dummies: {}\nShape X_test with group dummies: {}'.format(X_train.shape, X_test.shape))

##feature selection
###Pearson 
print('feature engineering...')
###remove negative related features
X_train = X_train.drop(['feature78','feature18','feature26','feature6','feature13','feature16','feature43','feature69','feature40','feature20','feature83','feature33','feature28','feature81','feature47','feature45','feature9','feature75','feature44','feature62','feature57','feature49'], axis=1)
X_test = X_test.drop(['feature78','feature18','feature26','feature6','feature13','feature16','feature43','feature69','feature40','feature20','feature83','feature33','feature28','feature81','feature47','feature45','feature9','feature75','feature44','feature62','feature57','feature49'], axis=1)

###remove relations below 0.01 both positive and negative 
#X_train = X_train.drop(['feature55','feature82','feature4','feature41','feature32','feature2','feature79','feature23','feature3','feature78','feature18','feature26','feature6'], axis=1)
#X_test = X_test.drop(['feature55','feature82','feature4','feature41','feature32','feature2','feature79','feature23','feature3','feature78','feature18','feature26','feature6'], axis=1)

###XGBOOST feature importance
#X_train = X_train.drop(['feature74','feature84','feature17','feature12','feature64','feature47'], axis=1)
#X_test = X_test.drop(['feature74','feature84','feature17','feature12','feature64','feature47'], axis=1)

print('Shape X_train after feature selection: {}\nShape X_test after feature selection: {}\n'.format(X_train.shape, X_test.shape))

# model training
print("Start Modeling...")
X_train, valid_set, y_train, y_valid = train_test_split(X_train, y_label, test_size=0.22, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(valid_set, label=y_valid)
param = {'learning_rate': 0.05, 'n_estimators': 1000, 'max_depth': 10,
         'min_child_weight': 5, 'gamma': 0, 'silent': 1, 'objective': 'binary:logistic',
         'early_stopping_rounds': 50, 'subsample': 0.8, 'colsample_bytree': 0.8}
watchList = [(dtest, 'eval'), (dtrain, 'train')]
plst = list(param.items()) + [('eval_metric', 'logloss')]
bst = xgb.train(plst, dtrain, 30, watchList)

print(X_train.shape)
print(X_test.shape)
proba_test = bst.predict(xgb.DMatrix(X_test))
print('Modeling Ends!')
print('Shape train: {}\nShape test: {}\n'.format(X_train.shape, X_test.shape))

# submission
print('Writing predictions to csv file...\n')
df = pd.DataFrame({"id": test_data["id"].values, "proba": proba_test})
df.to_csv(utils.file_submission, index=False, float_format='%.6f')

localtime = time.asctime(time.localtime())
print('Ending Time:' + str(localtime))
