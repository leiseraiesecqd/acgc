import preprocess
import tqdm
import time
import numpy as np
from models import utils

preprocess_path = preprocess.preprocessed_path
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

    def get_same_id_list(self, x_test, id_test):

        same_id_list_c = []

        for i, (x, idx) in enumerate(zip(x_test, id_test)):

            if i != len(id_test)-1:

                same_id = [idx]

                for row, idx_row in zip(x_test[i+1:], id_test[i+1:]):

                    if self.compare(x, row):
                        same_id.append(idx_row)

                if same_id != [idx]:
                    if len(same_id) != 2:
                        print('[W] Found {} Same ID: {}'.format(len(same_id), same_id))
                    same_id_list_c.append(same_id)

        return same_id_list_c

    def split_data_by_code_id(self):

        sort_idx = np.argsort(self.code_id_test)
        self.code_id_test = self.code_id_test[sort_idx]
        self.x_test = self.x_test[sort_idx]
        self.id_test = self.id_test[sort_idx]

        x_test_i = []
        id_test_i = []
        code_id_i = []
        x_test_list = []
        id_test_list = []
        code_id_list = []
        code_id_tag = self.code_id_test[0]

        for idx, code_id in enumerate(self.code_id_test):

            if idx == len(self.code_id_test) - 1:
                x_test_i.append(self.x_test[idx])
                id_test_i.append(self.id_test[idx])
                code_id_i.append(self.code_id_test[idx])
                x_test_list.append(x_test_i)
                id_test_list.append(id_test_i)
                code_id_list.append(code_id_i)
            elif code_id_tag == code_id:
                x_test_i.append(self.x_test[idx])
                id_test_i.append(self.id_test[idx])
                code_id_i.append(self.code_id_test[idx])
            else:
                code_id_tag = code_id
                x_test_list.append(x_test_i)
                id_test_list.append(id_test_i)
                code_id_list.append(code_id_i)
                x_test_i = [self.x_test[idx]]
                id_test_i = [self.id_test[idx]]
                code_id_i = [self.code_id_test[idx]]

        return x_test_list, id_test_list, code_id_list

    def main(self):

        x_test_list, id_test_list, code_id_list = self.split_data_by_code_id()

        same_id_list = []

        for i in tqdm.trange(len(id_test_list)):

            x_test_i, id_test_i, code_id_i = x_test_list[i], id_test_list[i], code_id_list[i]

            same_id_list_c = self.get_same_id_list(x_test_i, id_test_i)
            same_id_list.extend(same_id_list_c)
        print('------------------------------------------------------')
        print('Same Code Pairs: {}'.format(len(same_id_list)))
        print('------------------------------------------------------')
        utils.save_data_to_pkl(same_id_list, preprocess_path + 'same_id_pairs.p')


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
