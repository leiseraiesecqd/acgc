import numpy as np
import utils
import preprocess
from MulticoreTSNE import MulticoreTSNE as TSNE

preprocessed_data_path = preprocess.preprocessed_path

class MultiCoreTSNE:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

    def train(self, parameters):

        tsne = TSNE(**parameters)
        tsne_outputs = tsne.fit_transform(self.x_train)

        utils.save_np_to_pkl(tsne_outputs, self.pred_path + 'era_sign_test_pickle/era_sign_test.p')


def train_tsne():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    parameters = {'n_components': 2,
                  'perplexity': 30.0,
                  'early_exaggeration': 12.0,
                  'learning_rate': 200.0,
                  'n_iter': 1000,
                  'n_iter_without_progress': 300,
                  'min_grad_norm': 1e-07,
                  'metric': 'euclidean',
                  'init': 'random',
                  'verbose': 1,
                  'random_state': None,
                  'method': 'barnes_hut',
                  'angle': 0.5}

    MTSNE = MultiCoreTSNE(x_train, y_train, w_train, e_train, x_test, id_test)

    MTSNE.train(parameters=parameters)



