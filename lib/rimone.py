"""Contains functions to work with RIMONE."""
import cv2
import numpy as np
import os
import re

VALID_IMAGE_REGEXP = '^(N|G)-\d+-(L|R)\.jpg$'

def raw_data(path='rimone', verbose=True):
    """Returns the RIMONE dataset contained at the path given.
    
    This function assumes there are images at the path given meeting the
    following conditions:

    - 3 images per entry:
      - Original image named: 'D-#-S.jpg', where D is the diagnosis of the eye,
        # is a patient identifier and S is the eye (left or right). For example:
        G-12-L.jpg would contain the left eye of a person with glaucoma.
        N-1-R.jpg would contain the right eye of a healthy person.
      - Disc mask image named: 'D-#-S-1-Disc-exp1.jpg' formatted as the original
        image.
      - Cup mask image named: 'D-#-S-1-Cup-exp1.jpg' formatted as the original
        image.
    """
    if path[-1] == '/':
        path = path[:-1]
    eyes_diagnosis = []
    eyes_images = []
    eyes_id = []
    # Getting available entries (Files that met the format mentioned above)
    for file in os.listdir(path):
        if not re.search(VALID_IMAGE_REGEXP, file):
            continue
        if verbose:
            print(f'=> {len(eyes_id)} images found', end='\r')
        eye = file[:-4]
        eyes_images.append({
            'original': cv2.imread(f'{path}/{eye}.jpg'),
            'disc_mask': cv2.imread(f'{path}/{eye}-1-Disc-exp1.jpg'),
            'cup_mask': cv2.imread(f'{path}/{eye}-1-Cup-exp1.jpg')
        })
        eyes_diagnosis.append(1 if eye[0] == 'G' else 0) # G | N
        eyes_id.append(eye)
    print(f'=> {len(eyes_id)} images found')

    return {
        'images': eyes_images,
        'Y': np.array(eyes_diagnosis),
        'ids': eyes_id,
        'classes_name': ['Normal', 'Glaucoma']
    }
