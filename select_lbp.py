import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from lib import rimone
from lib.evaluator import evaluate
from lib.extractor import get_lbp

POINTS = [6, 7, 8, 9, 10]
RADIUS = [1, 2, 3]

ds = rimone.dataset()

Y = ds.Y

res = pd.DataFrame()
for p in POINTS:
    for r in RADIUS:
        print(f'p: {p}, r: {r}')
        X = StandardScaler().fit_transform(get_lbp(ds.cups, radius=r, points=p))
        evaluation = evaluate(GaussianNB, X, Y)
        evaluation.update({
            'radius': r,
            'point': p
        })
        res = res.append(evaluation, ignore_index=True).sort_values('Score', ascending=False)
        res.to_csv('results/lbp_cups.csv', index=False)

print(res.sort_values('Score', ascending=False))

POINTS = [6, 7, 8, 9, 10]
RADIUS = [1, 2, 3]

res = pd.DataFrame()
for p in POINTS:
    for r in RADIUS:
        print(f'p: {p}, r: {r}')
        X = StandardScaler().fit_transform(get_lbp(ds.discs, radius=r, points=p))
        evaluation = evaluate(GaussianNB, X, Y)
        evaluation.update({
            'radius': r,
            'point': p
        })
        res = res.append(evaluation, ignore_index=True).sort_values('Score', ascending=False)
        res.to_csv('results/lbp_discs.csv', index=False)

print(res.sort_values('Score', ascending=False))
