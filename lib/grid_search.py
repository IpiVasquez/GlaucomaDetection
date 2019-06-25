# from sys import stdout

# import pandas as pd

# from .evaluator import evaluate
# from sklearn.ensemble import AdaBoostClassifier


# def get_best_params(model, values, target, grid={}, verbose=True):
#     """Looks for the best param combinations for a model provided."""
#     if verbose:
#         stdout.write(f' => Getting best params for {model}\n')

#     results = pd.DataFrame(columns=(list(grid.keys()) + ['Accuracy']))

#     # if verbose:
#     #     stdout.write(f'\r ==> AdaBoost LRates .... {i + 1}/{len(l_rates)}')
#     print(combination({
#         'LOL1': [1, 2, 3],
#         'LOL2': [4, 5, 6],
#         'LOL3': [7, 8, 9]
#     }))
#     # evaluation = evaluate(model, values, target, params={
#     #     'learning_rate': lr
#     # })
#     # evaluation.update({'Learning rate': lr})
#     # results = results.append(evaluation, ignore_index=True).round(4)

#     if verbose:
#         stdout.write(f'\x1b[2k\r => AdaBoost, {len([])} combs tested!\n')

#     return results


# def combination(grid, cell={}):
#     keys = list(grid.keys())
#     this = keys[0]
#     params = keys[1:]

#     if not params:
#         print(cell)
#     else:

#         print('?')
