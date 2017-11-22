import random
import tqdm
import time
import numpy as np
import pandas as pd
import preprocess
from models import utils

base_fake_result_path = './results/fake_results/'
tampered_pred_path = './results/tampered_results/'
preprocessed_data_path = preprocess.preprocessed_path
test_path = preprocess.test_csv_path


class GenerateTamperedData(object):

    def __init__(self, seed):

        utils.check_dir([tampered_pred_path])
        np.random.seed(seed)

        base_result = pd.read_csv(base_fake_result_path + '0.5_fake_result.csv', header=0, dtype=np.float64)
        self.prob = np.array(base_result['proba'], dtype=np.float64)
        self.id_test = utils.load_pkl_to_data(preprocessed_data_path + 'id_test.p')
        self.same_idx_list = utils.load_pkl_to_data(preprocessed_data_path + 'same_test_idx_pairs.p')
        self.code_id_train, self.code_id_test = utils.load_preprocessed_code_id(preprocessed_data_path)

        self.test_id_to_idx_dict = {}
        for idx, id_ in enumerate(self.id_test):
            self.test_id_to_idx_dict[id_] = idx

    def generate_many_fake_result(self, loc):

        tampered_idx_ = np.concatenate(np.array(self.same_idx_list), axis=0).tolist()

        others_idx = []
        for i in range(len(self.id_test)):
            if i not in tampered_idx_:
                others_idx.append(i)
        tampered_idx = np.random.choice(others_idx, len(tampered_idx_)).tolist()

        for ii in tampered_idx:
            if ii in tampered_idx_:
                raise ValueError

        # tampered_id_list_ = list(range(start_id, stop_id))
        #
        # tampered_idx = []
        # for i, code_id in enumerate(code_id_test):
        #     if code_id in tampered_id_list:
        #         tampered_idx.append(i)

        # tampered_idx = tampered_idx_

        tampered_prob = np.random.normal(loc=loc, size=len(tampered_idx), scale=0.0002)
        for i, idx in enumerate(tampered_idx):
            self.prob[idx] = tampered_prob[i]

        # loc = 0.60
        # prob = np.random.normal(loc=loc, size=len(index), scale=0.0002)

        utils.save_pred_to_csv(tampered_pred_path + 'tampered_', self.id_test, self.prob)

    def save_tampered_results(self, i, prob, x1, x2, idx1, idx2, id1, id2, data_path):

        tampered_prob = prob.copy()
        tampered_prob[idx1] = x1
        tampered_prob[idx2] = x2

        df = pd.DataFrame({'id': self.id_test, 'proba': tampered_prob})

        df.to_csv(data_path + '{}_{}-{}_{}-{}_result.csv'.format(i+1, id1, x1, id2, x2), sep=',', index=False)

    def get_test_idx(self, id_):

        return self.test_id_to_idx_dict[id_]

    def tamper_result(self, idx_pair_list):

        print('------------------------------------------------------')
        print('Generating Tampered Results...')
        print('------------------------------------------------------')

        for i in tqdm.trange(len(idx_pair_list)):

            idx1, idx2 = idx_pair_list[i]
            id1, id2 = self.id_test[idx1], self.id_test[idx2]

            data_path = tampered_pred_path + '{}_{}_{}/'.format(i+1, id1, id2)
            utils.check_dir([data_path])

            self.save_tampered_results(i, self.prob, 1, 0, idx1, idx2, id1, id2, data_path)
            self.save_tampered_results(i, self.prob, 0, 1, idx1, idx2, id1, id2, data_path)
            self.save_tampered_results(i, self.prob, 1, 1, idx1, idx2, id1, id2, data_path)

    def generate_all_tampered_results(self):

        print('------------------------------------------------------')
        print('Generating All Tampered Results...')

        self.tamper_result(self.same_idx_list)

    def generate_big_weight_tampered_results(self, n_pairs):

        w_train = utils.load_pkl_to_data(preprocessed_data_path + 'w_train.p')
        same_test_df = pd.read_csv(preprocessed_data_path + 'same_test_pairs.csv', header=0, dtype=np.float64)
        same_test_code_id = same_test_df['code_id']
        same_test_id = same_test_df['id']

        print('------------------------------------------------------')
        print('Calculating Big Weight Same Pairs...')
        print('------------------------------------------------------')
        print('Sorting...')
        sorted_by_weight_idx = np.argsort(w_train)[:-n_pairs*3:-1]
        sorted_w_train = w_train[sorted_by_weight_idx]
        sorted_code_id_train = self.code_id_train[sorted_by_weight_idx]

        print('Deduplicating...')
        big_weight_code_id = []
        big_weight_w_train = []
        for idx, code_id in enumerate(sorted_code_id_train):
            if code_id not in big_weight_code_id:
                if code_id in set(same_test_code_id):
                    big_weight_code_id.append(code_id)
                    big_weight_w_train.append(sorted_w_train[idx])

        print('Generating Pairs...')
        w_train_col = []
        code_id_col = []
        id_col = []
        big_weight_idx_pair_list = []
        exit_flag = False
        for code_id_bw, w_train_bw in zip(big_weight_code_id, big_weight_w_train):
            big_weight_idx_pair = []
            is_first = True
            for idx_s, code_id_s in enumerate(same_test_code_id):
                if code_id_bw == code_id_s:
                    # print('is_first: {} | {}'.format(is_first, same_test_df.iloc[idx_s]['feature0']))
                    id_bw = same_test_id[idx_s]
                    test_idx_bw = self.get_test_idx(id_bw)
                    big_weight_idx_pair.append(test_idx_bw)
                    if is_first:
                        is_first = False
                    else:
                        big_weight_idx_pair_list.append(big_weight_idx_pair)
                        big_weight_idx_pair = []
                        is_first = True
                    w_train_col.append(w_train_bw)
                    code_id_col.append(code_id_bw)
                    id_col.append(id_bw)
                    if len(big_weight_idx_pair_list) >= n_pairs:
                        exit_flag = True
                        break
            if exit_flag:
                break

        print('------------------------------------------------------')
        print('Number of Big Weight Same Pairs: {}'.format(len(big_weight_idx_pair_list)))
        print('Saving big_weight_idx_pairs.p...')
        utils.save_data_to_pkl(big_weight_idx_pair_list, preprocessed_data_path + 'big_weight_idx_pairs.p')

        print('------------------------------------------------------')
        print('Saving big_weight_tampered_log.csv...')
        index = []
        for i in range(1, len(big_weight_idx_pair_list)+1):
            index.extend([i, i])
        df_log = pd.DataFrame({'index': np.array(index, dtype=int),
                               'weight': np.array(w_train_col),
                               'code_id': np.array(code_id_col, dtype=int),
                               'id': np.array(id_col, dtype=int)})
        cols = ['index', 'weight', 'code_id', 'id']
        df_log = df_log.loc[:, cols]
        df_log.to_csv(tampered_pred_path + 'big_weight_tampered_log.csv', sep=',', index=False)

        print('------------------------------------------------------')
        print('Saving big_weight_same_pairs.csv...')
        big_weight_idx = np.concatenate(np.array(big_weight_idx_pair_list)).tolist()
        test_f = pd.read_csv(test_path, header=0, dtype=np.float64)
        df = test_f.iloc[big_weight_idx]
        cols = ['code_id', *['feature{}'.format(i) for i in range(97)], 'group1', 'group2', 'id']
        df = df.loc[:, cols]
        df.to_csv(preprocessed_data_path + 'big_weight_same_pairs.csv', sep=',', index=False)

        print('------------------------------------------------------')
        print('Generating Big Weight Tampered Results...')

        self.tamper_result(big_weight_idx_pair_list)

    def generate_absent_tampered_results(self, n_pairs):

        diff_code_id_test = utils.load_pkl_to_data(preprocessed_data_path + 'diff_code_id_test.p')
        same_test_df = pd.read_csv(preprocessed_data_path + 'same_test_pairs.csv', header=0, dtype=np.float64)
        same_test_code_id = same_test_df['code_id']
        same_test_id = same_test_df['id']

        print('------------------------------------------------------')
        print('Calculating Absent Same Pairs...')
        print('------------------------------------------------------')
        print('Sorting...')
        diff_code_id_test = np.sort(diff_code_id_test)
        absent_code_id = diff_code_id_test[:-n_pairs*3:-1]

        print('Generating Pairs...')
        code_id_col = []
        id_col = []
        absent_idx_pair_list = []
        exit_flag = False
        for code_id_ab in absent_code_id:
            absent_idx_pair = []
            is_first = True
            for idx_s, code_id_s in enumerate(same_test_code_id):
                if code_id_ab == code_id_s:
                    # print('is_first: {} | {}'.format(is_first, same_test_df.iloc[idx_s]['feature0']))
                    id_ab = same_test_id[idx_s]
                    test_idx_ab = self.get_test_idx(id_ab)
                    absent_idx_pair.append(test_idx_ab)
                    if is_first:
                        is_first = False
                    else:
                        absent_idx_pair_list.append(absent_idx_pair)
                        absent_idx_pair = []
                        is_first = True
                    code_id_col.append(code_id_ab)
                    id_col.append(id_ab)
                    if len(absent_idx_pair_list) >= n_pairs:
                        exit_flag = True
                        break
            if exit_flag:
                break

        print('------------------------------------------------------')
        print('Number of Absent Same Pairs: {}'.format(len(absent_idx_pair_list)))
        print('Saving absent_idx_pairs.p...')
        utils.save_data_to_pkl(absent_idx_pair_list, preprocessed_data_path + 'absent_idx_pairs.p')

        print('------------------------------------------------------')
        print('Saving absent_tampered_log.csv...')
        index = []
        for i in range(1, len(absent_idx_pair_list)+1):
            index.extend([i, i])
        df_log = pd.DataFrame({'index': np.array(index, dtype=int),
                               'code_id': np.array(code_id_col, dtype=int),
                               'id': np.array(id_col, dtype=int)})
        cols = ['index', 'code_id', 'id']
        df_log = df_log.loc[:, cols]
        df_log.to_csv(tampered_pred_path + 'absent_tampered_log.csv', sep=',', index=False)

        print('------------------------------------------------------')
        print('Saving absent_same_pairs.csv...')
        absent_idx = np.concatenate(np.array(absent_idx_pair_list)).tolist()
        test_f = pd.read_csv(test_path, header=0, dtype=np.float64)
        df = test_f.iloc[absent_idx]
        cols = ['code_id', *['feature{}'.format(i) for i in range(97)], 'group1', 'group2', 'id']
        df = df.loc[:, cols]
        df.to_csv(preprocessed_data_path + 'absent_same_pairs.csv', sep=',', index=False)

        print('------------------------------------------------------')
        print('Generating Absent Tampered Results...')

        self.tamper_result(absent_idx_pair_list)

    @staticmethod
    def compare(x1, x2):
        is_same = True
        for i_x1, i_x2 in zip(x1, x2):
            if i_x1 != i_x2:
                is_same = False
        return is_same

    def generate_custom_tempered_result(self, tamper_list_path=None, base_result_path=None,
                                        append_info=None, check=True):

        print('Loading {}...'.format(tamper_list_path))
        tamper_list_df = pd.read_csv(tamper_list_path, header=0, dtype=np.float64)
        tamper_id = tamper_list_df['id']
        tamper_prob = tamper_list_df['proba']

        if check:
            print('Loading {}...'.format(test_path))
            test_df = pd.read_csv(test_path, header=0, dtype=np.float64)

        if base_result_path is None:
            tampered_prob = self.prob.copy()
        else:
            base_result = pd.read_csv(base_result_path, header=0, dtype=np.float64)
            tampered_prob = np.array(base_result['proba'], dtype=np.float64)

        print('------------------------------------------------------')
        print('Generating Custom Tampered Result...')
        is_first = True
        check_feature_list = [13, 16, 49]
        check_list_1 = []
        check_list_2 = []
        for i, (id_, prob_) in enumerate(zip(tamper_id, tamper_prob)):
            test_idx = self.get_test_idx(id_)
            tampered_prob[test_idx] = prob_
            if check:
                if is_first:
                    is_first = False
                    for feature_idx in check_feature_list:
                        check_list_1.append(test_df.loc[test_idx, 'feature'+str(feature_idx)])
                else:
                    for feature_idx in check_feature_list:
                        check_list_2.append(test_df.loc[test_idx, 'feature'+str(feature_idx)])
                    if not self.compare(check_list_1, check_list_2):
                        raise ValueError("[E] Not Same! (ID: {}-{})".format(tamper_id[i-1], id_))
                    else:
                        is_first = True
                        check_list_1 = []
                        check_list_2 = []

        print('------------------------------------------------------')
        tampered_pred_path_ = './results/custom_tampered_results/'
        utils.check_dir([tampered_pred_path_])
        if append_info is not None:
            tampered_pred_path_ += append_info + '_'
        print('Saving {}...'.format(tampered_pred_path_ + 'tampered_result.csv'))
        df = pd.DataFrame({'id': self.id_test, 'proba': tampered_prob})
        df.to_csv(tampered_pred_path_ + 'tampered_result.csv', sep=',', index=False)


if __name__ == '__main__':

    start_time = time.time()

    print('======================================================')

    global_seed = random.randint(0, 500)
    GTD = GenerateTamperedData(global_seed)

    # GTD.generate_many_fake_result(loc=0.1)
    # GTD.generate_all_tampered_results()
    # GTD.generate_big_weight_tampered_results(300)
    # GTD.generate_absent_tampered_results(300)

    """Generate Custom Tampered Result"""
    base_result_path_ = './results/fake_results/0.5_fake_result.csv'
    tamper_list_path_ = './tamper_list.csv'
    GTD.generate_custom_tempered_result(tamper_list_path=tamper_list_path_, append_info='custom',
                                        base_result_path=base_result_path_, check=True)

    print('------------------------------------------------------')
    print('Done!')
    print('Total Time: {:.2f}s'.format(time.time() - start_time))
    print('======================================================')
