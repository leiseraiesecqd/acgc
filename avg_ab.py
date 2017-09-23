import pandas as pd

ab1 = pd.read_csv('./ab_cv_1_result.csv', header=0)
ab2 = pd.read_csv('./ab_cv_2_result.csv', header=0)
ab3 = pd.read_csv('./ab_cv_3_result.csv', header=0)
ab4 = pd.read_csv('./ab_cv_4_result.csv', header=0)
ab5 = pd.read_csv('./ab_cv_5_result.csv', header=0)
ab6 = pd.read_csv('./ab_cv_6_result.csv', header=0)
ab7 = pd.read_csv('./ab_cv_7_result.csv', header=0)
ab8 = pd.read_csv('./ab_cv_1_1_result.csv', header=0)
ab9 = pd.read_csv('./ab_cv_2_1_result.csv', header=0)
ab10 = pd.read_csv('./ab_cv_3_1_result.csv', header=0)


id = ab1['id']

proba = 0.1 * ab1['proba'] + 0.1 * ab2['proba'] + 0.1 * ab3['proba'] \
        + 0.1 * ab4['proba'] + 0.1 * ab5['proba'] + 0.1 * ab6['proba'] \
        + 0.1 * ab7['proba'] + 0.1 * ab8['proba'] + 0.1 * ab9['proba'] \
        + 0.1 * ab10['proba']

stack = pd.DataFrame({'id': id, 'proba': proba})
stack.to_csv('./ab_result.csv', sep=',', index=False, float_format='%.6f')