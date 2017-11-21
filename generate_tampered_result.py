import random
import tqdm
import time
import numpy as np
import pandas as pd
import preprocess
from models import utils

base_model_path = './results/fake_results/'
fake_pred_path = './results/tampered_results/'
preprocessed_data_path = preprocess.preprocessed_path


class GenerateTamperedData(object):

    def __init__(self, seed):

        utils.check_dir([fake_pred_path])
        np.random.seed(seed)

        base_result = pd.read_csv(base_model_path + '0.5_fake_result.csv', header=0, dtype=np.float64)
        self.prob = np.array(base_result['proba'], dtype=np.float64)
        self.id_test = utils.load_pkl_to_data(preprocessed_data_path + 'id_test.p')
        self.same_idx_list = utils.load_pkl_to_data(preprocessed_data_path + 'same_test_idx_pairs.p')

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

        tampered_prob = np.random.normal(loc=loc, size=len(tampered_idx), scale=0.0002)
        for i, idx in enumerate(tampered_idx):
            self.prob[idx] = tampered_prob[i]

        # loc = 0.60
        # prob = np.random.normal(loc=loc, size=len(index), scale=0.0002)

        utils.save_pred_to_csv(fake_pred_path + 'tampered_', self.id_test, self.prob)

    def save_tampered_result(self, i, prob, x1, x2, idx1, idx2, id1, id2, data_path):

        tampered_prob = prob.copy()
        tampered_prob[idx1] = x1
        tampered_prob[idx2] = x2

        df = pd.DataFrame({'id': self.id_test, 'proba': tampered_prob})

        df.to_csv(data_path + '{}_{}-{}_{}-{}_result.csv'.format(i+1, id1, x1, id2, x2), sep=',', index=False)

    def generate_fake_result(self):

        print('------------------------------------------------------')
        print('Generating Tampered Results...')
        print('------------------------------------------------------')

        for i in tqdm.trange(len(self.same_idx_list)):

            idx1, idx2 = self.same_idx_list[i]
            id1, id2 = self.id_test[idx1], self.id_test[idx2]

            data_path = fake_pred_path + '{}_{}_{}/'.format(i+1, id1, id2)
            utils.check_dir([data_path])

            self.save_tampered_result(i, self.prob, 1, 0, idx1, idx2, id1, id2, data_path)
            self.save_tampered_result(i, self.prob, 0, 1, idx1, idx2, id1, id2, data_path)
            self.save_tampered_result(i, self.prob, 1, 1, idx1, idx2, id1, id2, data_path)


if __name__ == '__main__':

    start_time = time.time()

    print('======================================================')

    global_seed = random.randint(0, 500)
    GTD = GenerateTamperedData(global_seed)

    # GTD.generate_single_fake_result(loc=0.1)
    GTD.generate_fake_result()

    print('------------------------------------------------------')
    print('Done!')
    print('Total Time: {:.2f}s'.format(time.time() - start_time))
    print('======================================================')