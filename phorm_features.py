from lib.features import phorm
import numpy as np
from lib import rimone

print(' => Reading RIMONE meta-data')
ds = rimone.dataset()
cols = ['perimetro','area', 'compacidad', 'centroide x', 
'centroide y','longitud eje a','longitud eje b','angulo entre ejes',
'Box ax', 'Box ay', 'Box bx', 'Box by']

print(' => Calculating CDR')
features = []
for i in range(len(ds.disc_masks)):
    features.append(np.concatenate([phorm.phorm(ds.disc_masks[i]), phorm.phorm(ds.cup_masks[i])]))
print(features)
