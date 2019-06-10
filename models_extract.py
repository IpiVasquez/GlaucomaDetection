from lib.features import features_models
import pandas as pd
models = features_models.models()

ind = 1
print(" => writing csv files")
for m in models:
    pd.DataFrame(m).to_csv('results/models/model{}.csv'.format(ind))
    ind += 1