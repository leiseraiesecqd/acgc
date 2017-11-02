import utils
import random
import numpy as np
import pandas as pd

fake_pred_path = './results/fake_result/'


def generate_fake_result(seed):

    utils.check_dir([fake_pred_path])

    np.random.seed(seed)

    index = pd.read_pickle('./data/preprocessed_data/id_test.p')
    prob = np.random.normal(loc=0.55, size=len(index), scale=0.05)

    utils.save_pred_to_csv(fake_pred_path, index, prob)


if __name__ == '__main__':

    global_seed = random.randint(0, 500)

    generate_fake_result(global_seed)
