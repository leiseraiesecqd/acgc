import numpy as np
import utils
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GroupKFold


class CrossValidation:
    """
        Cross Validation
    """
    def __init__(self):

        self.trained_cv = []

    @staticmethod
    def random_split_with_weight(x, y, w, e, n_valid, n_cv, n_era, seed=None, era_list=None):

        test_size = n_valid / n_era
        valid_era = []
        ss_train = StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size, random_state=seed)

        for train_index, valid_index in ss_train.split(x, y):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]
            e_train = e[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]
            e_valid = e[valid_index]

            yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    @staticmethod
    def sk_k_fold_with_weight(x, y, w, n_splits, n_cv, seed=None):

        if seed is not None:
            np.random.seed(seed)

        if n_cv % n_splits != 0:
            raise ValueError('n_cv must be an integer multiple of n_splits!')

        n_repeats = int(n_cv / n_splits)
        era_k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

        for train_index, valid_index in era_k_fold.split(x, y):

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]

            yield x_train, y_train, w_train, x_valid, y_valid, w_valid

    @staticmethod
    def sk_group_k_fold(x, y, e, n_cv):

        era_k_fold = GroupKFold(n_splits=n_cv)

        for train_index, valid_index in era_k_fold.split(x, y, e):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]

            yield x_train, y_train, x_valid, y_valid

    @staticmethod
    def sk_group_k_fold_with_weight(x, y, w, e, n_cv):

        era_k_fold = GroupKFold(n_splits=n_cv)

        for train_index, valid_index in era_k_fold.split(x, y, e):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]

            yield x_train, y_train, w_train, x_valid, y_valid, w_valid

    @staticmethod
    def era_k_fold_split_all_random(e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        trained_cv = []

        for i in range(n_cv):

            if era_list is None:
                era_list = range(1, n_era + 1)

            valid_era = np.random.choice(era_list, n_valid, replace=False)
            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                print('This CV split has been chosen, choosing another one...')
                valid_era = np.random.choice(era_list, n_valid, replace=False)

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):

                if ele in valid_era:
                    valid_index.append(ii)
                else:
                    train_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)
            trained_cv.append(set(valid_era))

            yield train_index, valid_index

    @staticmethod
    def era_k_fold_with_weight_all_random(x, y, w, e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        trained_cv = []

        for i in range(n_cv):

            if era_list is None:
                era_list = range(1, n_era + 1)

            valid_era = np.random.choice(era_list, n_valid, replace=False)
            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                print('This CV split has been chosen, choosing another one...')
                valid_era = np.random.choice(era_list, n_valid, replace=False)

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):
                if ele in valid_era:
                    valid_index.append(ii)
                else:
                    train_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]
            e_train = e[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]
            e_valid = e[valid_index]

            trained_cv.append(set(valid_era))

            yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    @staticmethod
    def era_k_fold_split(e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse
        trained_cv = []

        for epoch in range(n_epoch):

            if era_list is None:
                era_list = range(1, n_era + 1)

            era_idx = [era_list]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(range(1, n_era+1), n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):
                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    trained_cv.append(set(valid_era))

                    yield train_index, valid_index

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        trained_cv.append(set(valid_era))

                        yield train_index, valid_index

                    else:

                        era_idx_else = [t for t in list(range(1, n_era + 1)) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        trained_cv.append(set(valid_era))

                        yield train_index, valid_index

    @staticmethod
    def era_k_fold_with_weight(x, y, w, e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse
        trained_cv = []

        for epoch in range(n_epoch):

            if era_list is None:
                era_list = range(1, n_era + 1)

            era_idx = [era_list]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(era_list, n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):
                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    # Training data
                    x_train = x[train_index]
                    y_train = y[train_index]
                    w_train = w[train_index]
                    e_train = e[train_index]

                    # Validation data
                    x_valid = x[valid_index]
                    y_valid = y[valid_index]
                    w_valid = w[valid_index]
                    e_valid = e[valid_index]

                    trained_cv.append(set(valid_era))

                    yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        e_train = e[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        e_valid = e[valid_index]

                        trained_cv.append(set(valid_era))

                        yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

                    else:

                        era_idx_else = [t for t in list(era_list) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        e_train = e[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        e_valid = e[valid_index]

                        trained_cv.append(set(valid_era))

                        yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    def era_k_fold_for_stack(self, x, y, w, e, x_g, n_valid, n_cv, n_era, seed=None, return_train_index=False):

        if seed is not None:
            np.random.seed(seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse

        for epoch in range(n_epoch):

            era_idx = [list(range(1, n_era + 1))]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if self.trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(range(1, n_era+1), n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):
                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    # Training data
                    x_train = x[train_index]
                    y_train = y[train_index]
                    w_train = w[train_index]
                    x_g_train = x_g[train_index]

                    # Validation data
                    x_valid = x[valid_index]
                    y_valid = y[valid_index]
                    w_valid = w[valid_index]
                    x_g_valid = x_g[valid_index]

                    self.trained_cv.append(set(valid_era))

                    if return_train_index is True:
                        yield x_train, y_train, w_train, x_g_train, x_valid, \
                              y_valid, w_valid, x_g_valid, train_index, valid_index, valid_era
                    else:
                        yield x_train, y_train, w_train, x_g_train, x_valid, \
                              y_valid, w_valid, x_g_valid, valid_index, valid_era

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if self.trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        x_g_train = x_g[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        x_g_valid = x_g[valid_index]

                        self.trained_cv.append(set(valid_era))

                        if return_train_index is True:
                            yield x_train, y_train, w_train, x_g_train, x_valid, \
                                  y_valid, w_valid, x_g_valid, train_index, valid_index, valid_era
                        else:
                            yield x_train, y_train, w_train, x_g_train, x_valid, \
                                  y_valid, w_valid, x_g_valid, valid_index, valid_era

                    else:

                        era_idx_else = [t for t in list(range(1, n_era + 1)) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(
                            np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        x_g_train = x_g[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        x_g_valid = x_g[valid_index]

                        self.trained_cv.append(set(valid_era))

                        if return_train_index is True:
                            yield x_train, y_train, w_train, x_g_train, x_valid, \
                                  y_valid, w_valid, x_g_valid, train_index, valid_index, valid_era
                        else:
                            yield x_train, y_train, w_train, x_g_train, x_valid, \
                                  y_valid, w_valid, x_g_valid, valid_index, valid_era

    @staticmethod
    def era_k_fold_with_weight_balance(x, y, w, e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        trained_cv = []

        for i in range(n_cv):

            if era_list is None:
                era_list = range(1, n_era + 1)

            valid_era = np.random.choice(era_list, n_valid, replace=False)
            while utils.check_bad_cv(trained_cv, valid_era):
                valid_era = np.random.choice(era_list, n_valid, replace=False)

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):
                if ele in valid_era:
                    valid_index.append(ii)
                else:
                    train_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]
            e_train = e[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]
            e_valid = e[valid_index]

            trained_cv.append(set(valid_era))

            yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era


if __name__ == '__main__':

    pass
