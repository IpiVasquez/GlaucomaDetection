"""Specifies a set of constants to use around the project."""

SELECTION_CRITERIA = 'Sensibility'
FEATURES_URI = 'results/extracted_features.csv'
BACKUP_DIR = 'datasets/rimone'
VALID_IMAGE_REGEXP = r'^(N|G)-\d+-(L|R)\.jpg$'

HARALICK_NAMES = [
    'energy', 'contrast', 'correlation', 'variance', 'homogeneity', 'sum avg',
    'sum var', 'sum ent', 'entropy', 'diff var', 'diff ent', 'IC I',
    'IC II'
]
HH_DISC = list(map(lambda x: 'Disc ' + x, HARALICK_NAMES))
HH_CUP = list(map(lambda x: 'Cup ' + x, HARALICK_NAMES))

FORM_NAMES = ['perimeter', 'area', 'compacity', 'centroid x', 'centroid y']
FH_DISC = list(map(lambda x: 'Disc ' + x, FORM_NAMES))
FH_CUP = list(map(lambda x: 'Cup ' + x, FORM_NAMES))
