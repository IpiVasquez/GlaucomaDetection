"""Specifies a set of constants to use around the project."""

RANDOM_STATE = 97
SELECTION_CRITERIA = 'Sensibility'
FEATURES_URI = 'results/extracted_features.csv'
TRAIN_URI = 'datasets/train.csv'
TEST_URI = 'datasets/test.csv'
BACKUP_DIR = 'datasets/rimone'
VALID_IMAGE_REGEXP = r'^(N|G)-\d+-(L|R)\.jpg$'

# The names of the Haralick features
HARALICK_NAMES = [
    'energy', 'contrast', 'correlation', 'variance', 'homogeneity', 'sum avg',
    'sum var', 'sum ent', 'entropy', 'diff var', 'diff ent', 'IC I',
    'IC II'
]
# The names of the Form features
FORM_NAMES = ['perimeter', 'area', 'compacity', 'centroid x', 'centroid y']
# Haralick names for cup & disc
HH_DISC = list(map(lambda x: 'Disc ' + x, HARALICK_NAMES))
HH_CUP = list(map(lambda x: 'Cup ' + x, HARALICK_NAMES))
# Form names for cup & disc
FH_DISC = list(map(lambda x: 'Disc ' + x, FORM_NAMES))
FH_CUP = list(map(lambda x: 'Cup ' + x, FORM_NAMES))
