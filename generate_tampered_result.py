import random
import tqdm
import time
import numpy as np
import pandas as pd
import preprocess
from models import utils

base_model_path = './results/fake_results/'
tampered_pred_path = './results/tampered_results/'
preprocessed_data_path = preprocess.preprocessed_path


class GenerateTamperedData(object):

    def __init__(self, seed):

        utils.check_dir([tampered_pred_path])
        np.random.seed(seed)

        base_result = pd.read_csv(base_model_path + '0.5_fake_result.csv', header=0, dtype=np.float64)
        self.prob = np.array(base_result['proba'], dtype=np.float64)
        self.id_test = utils.load_pkl_to_data(preprocessed_data_path + 'id_test.p')
        self.same_idx_list = utils.load_pkl_to_data(preprocessed_data_path + 'same_test_idx_pairs.p')
        self.code_id_train, self.code_id_test = utils.load_preprocessed_code_id(preprocessed_data_path)

    def generate_single_fake_result(self, loc):

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

    def tamper_result(self, idx_pair_list):

        print('------------------------------------------------------')
        print('Generating Tampered Results...')
        print('------------------------------------------------------')

        for i in tqdm.trange(len(idx_pair_list)):

            idx1, idx2 = self.same_idx_list[i]
            id1, id2 = self.id_test[idx1], self.id_test[idx2]

            data_path = tampered_pred_path + '{}_{}_{}/'.format(i + 1, id1, id2)
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
        sorted_by_weight_idx = np.argsort(w_train)[:-n_pairs*5:-1]
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
                    big_weight_idx_pair.append(idx_s)
                    if is_first:
                        is_first = False
                    else:
                        big_weight_idx_pair_list.append(big_weight_idx_pair)
                        big_weight_idx_pair = []
                        is_first = True
                    w_train_col.append(w_train_bw)
                    code_id_col.append(code_id_bw)
                    id_col.append(same_test_id[idx_s])
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
        df = same_test_df.iloc[big_weight_idx]
        df.to_csv(preprocessed_data_path + 'big_weight_same_pairs.csv', sep=',', index=False)

        print('------------------------------------------------------')
        print('Generating Big Weight Tampered Results...')

        self.tamper_result(big_weight_idx_pair_list)

if __name__ == '__main__':

    start_time = time.time()

    print('======================================================')

    global_seed = random.randint(0, 500)
    GTD = GenerateTamperedData(global_seed)

    # GTD.generate_single_fake_result(loc=0.1)
    # GTD.generate_all_tampered_results()
    GTD.generate_big_weight_tampered_results(300)

    print('------------------------------------------------------')
    print('Done!')
    print('Total Time: {:.2f}s'.format(time.time() - start_time))
    print('======================================================')
