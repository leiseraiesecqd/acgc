import random
import numpy as np
import pandas as pd
import preprocess
from models import utils

fake_pred_path = './results/tampered_results/'
preprocessed_data_path = preprocess.preprocessed_path


def generate_fake_result(seed):

    start_id = 2000
    stop_id = 2500
    loc = 0.9

    utils.check_dir([fake_pred_path])

    np.random.seed(seed)

    index = pd.read_pickle(preprocessed_data_path + 'id_test.p')
    _, code_id_test = utils.load_preprocessed_code_id(preprocessed_data_path)
    code_id_set = list(set(code_id_test))
    print('Code ID Range: ', code_id_set[0], '-', code_id_set[-1])

    base_result = pd.read_csv(fake_pred_path + 'base_result.csv', header=0, dtype=np.float64)
    prob = np.array(base_result['proba'], dtype=np.float64)

    tampered_id_list = list(range(start_id, stop_id))

    tampered_idx = []
    for i, code_id in enumerate(code_id_test):
        if code_id in tampered_id_list:
            tampered_idx.append(i)

    tampered_prob = np.random.normal(loc=loc, size=len(tampered_idx), scale=0.0002)
    for i, idx in enumerate(tampered_idx):
        prob[idx] = tampered_prob[i]

    # loc = 0.60
    # prob = np.random.normal(loc=loc, size=len(index), scale=0.0002)

    utils.save_pred_to_csv(fake_pred_path + str(start_id) + '-' + str(stop_id) + '_', index, prob)

if __name__ == '__main__':

    global_seed = random.randint(0, 500)

    generate_fake_result(global_seed)