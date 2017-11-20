import preprocess
import tqdm
import time
import numpy as np
import pandas as pd
from models import utils

preprocess_path = preprocess.preprocessed_path
test_path = preprocess.test_csv_path
feature_list = [13, 16, 49]


class SearchSameID(object):

    def __init__(self):

        self.x_test, self.id_test = utils.load_preprocessed_data(preprocess_path)[-2:]
        self.code_id_test = utils.load_preprocessed_code_id(preprocess_path)[-1]

        self.x_test = self.x_test[:, feature_list]

    @staticmethod
    def compare(x1, x2):
        is_same = True
        for i_x1, i_x2 in zip(x1, x2):
            if i_x1 != i_x2:
                is_same = False
        return is_same

    def get_same_id_list(self, x_test, id_test, test_idx):

        same_id_list_c = []
        same_idx_list_c = []

        for i, (x, id_, idx_) in enumerate(zip(x_test, id_test, test_idx)):

            if i != len(id_test)-1:

                same_id = [id_]
                same_idx = [idx_]

                for x_row, id_row, idx_row in zip(x_test[i+1:], id_test[i+1:], test_idx[i+1:]):

                    if self.compare(x, x_row):
                        same_id.append(id_row)
                        same_idx.append(idx_row)

                if same_id != [id_]:
                    if len(same_id) != 2:
                        print('[W] Found {} Same ID: {}'.format(len(same_id), same_id))
                    same_id_list_c.append(same_id)
                    same_idx_list_c.append(same_idx)

        return same_id_list_c, same_idx_list_c

    def split_data_by_code_id(self):

        test_idx = np.argsort(self.code_id_test)
        self.code_id_test = self.code_id_test[test_idx]
        self.x_test = self.x_test[test_idx]
        self.id_test = self.id_test[test_idx]

        x_test_i = []
        id_test_i = []
        code_id_i = []
        test_idx_i = []
        x_test_list = []
        id_test_list = []
        code_id_list = []
        test_idx_list = []
        code_id_tag = self.code_id_test[0]

        for idx, code_id in enumerate(self.code_id_test):

            if idx == len(self.code_id_test) - 1:
                x_test_i.append(self.x_test[idx])
                id_test_i.append(self.id_test[idx])
                code_id_i.append(self.code_id_test[idx])
                test_idx_i.append(test_idx[idx])
                x_test_list.append(x_test_i)
                id_test_list.append(id_test_i)
                code_id_list.append(code_id_i)
                test_idx_list.append(test_idx_i)
            elif code_id_tag == code_id:
                x_test_i.append(self.x_test[idx])
                id_test_i.append(self.id_test[idx])
                code_id_i.append(self.code_id_test[idx])
                test_idx_i.append(test_idx[idx])
            else:
                code_id_tag = code_id
                x_test_list.append(x_test_i)
                id_test_list.append(id_test_i)
                code_id_list.append(code_id_i)
                test_idx_list.append(test_idx_i)
                x_test_i = [self.x_test[idx]]
                id_test_i = [self.id_test[idx]]
                code_id_i = [self.code_id_test[idx]]
                test_idx_i = [test_idx[idx]]

        return x_test_list, id_test_list, code_id_list, test_idx_list

    def main(self):

        x_test_list, id_test_list, code_id_list, test_idx_list = self.split_data_by_code_id()

        same_id_list = []
        same_idx_list = []

        for i in tqdm.trange(len(id_test_list)):

            x_test_i, id_test_i, code_id_i, test_idx_i = \
                x_test_list[i], id_test_list[i], code_id_list[i], test_idx_list[i]

            same_id_list_c, same_idx_list_c = self.get_same_id_list(x_test_i, id_test_i, test_idx_i)
            same_id_list.extend(same_id_list_c)
            same_idx_list.extend(same_idx_list_c)

        print('------------------------------------------------------')
        print('Same Code Pairs: {}'.format(len(same_idx_list)))
        print('------------------------------------------------------')
        print('Saving Data to csv File...'.format(len(same_idx_list)))
        same_idx = np.concatenate(np.array(same_idx_list)).tolist()
        test_f = pd.read_csv(test_path, header=0, dtype=np.float64)
        df = test_f.iloc[same_idx]
        df.loc['code_id'] = np.concatenate(np.array(same_id_list), axis=0)
        cols = ['code_id', *['feature{}'.format(i) for i in range(97)], 'group1', 'group2', 'id']
        df = df.loc[:, cols]
        df.to_csv(preprocess_path + 'same_test_pairs.csv', sep=',', index=False)
        utils.save_data_to_pkl(same_idx_list, preprocess_path + 'same_test_idx_pairs.p')

if __name__ == "__main__":

    start_time = time.time()

    print('======================================================')
    print('Searching Same ID Pairs...')
    print('------------------------------------------------------')

    SSI = SearchSameID()
    SSI.main()

    print('------------------------------------------------------')
    print('Done!')
    print('Total Time: {:.2f}s'.format(time.time() - start_time))
    print('======================================================')
