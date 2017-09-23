import pandas as pd

xgb = pd.read_csv('./xgb_result.csv', header=0)
lgb = pd.read_csv('./lgb_result.csv', header=0)
# ab = pd.read_csv('./ab_result.csv', header=0)

id = xgb['id']

proba = 0.4 * xgb['proba'] + 0.6 * lgb['proba']

stack = pd.DataFrame({'id': id, 'proba': proba})
stack.to_csv('./result.csv', sep=',', index=False, float_format='%.6f')