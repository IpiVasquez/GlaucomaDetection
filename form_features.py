from lib.features import form
import numpy as np
from lib import rimone

print(' => Reading RIMONE meta-data')
ds = rimone.dataset()
cols = ['Perimeter','Area','Compacity','X centroid','Y centroid']

features = []
for i in range(len(ds.disc_masks)):
    features.append(np.concatenate([form.form_descriptors(ds.disc_masks[i]), form.form_descriptors(ds.cup_masks[i])]))
print(features)
