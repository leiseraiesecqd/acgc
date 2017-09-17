import model

# HyperParameters
hyper_parameters = {'version': '1.0',
                    'epochs': 10,
                    'layers_number': 10,
                    'unit_number': [200, 400, 800, 800, 800, 800, 800, 800, 400, 200],
                    'learning_rate': 0.01,
                    'keep_probability': 0.75,
                    'batch_size': 512,
                    'display_step': 100,
                    'save_path': './checkpoints/',
                    'log_path': './log/'}

pickled_data_path = './preprocessed_data/'

print('Loading data set...')
tr, tr_y, tr_w, val_x, val_y, val_w = model.load_data(pickled_data_path)

dnn = model.DNN(tr, tr_y, tr_w, val_x, val_y, val_w, hyper_parameters)
dnn.train()

print('Done!')