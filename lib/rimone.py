"""Contains functions to work with RIMONE."""
import cv2
import numpy as np
import pickle
import os
import re
from .constants import VALID_IMAGE_REGEXP, BACKUP_DIR

META_BACKUP = 'meta.pkl'
ORIGINAL_BACKUP = 'original.pkl'
DISC_MASK_BACKUP = 'disc_mask.pkl'
CUP_MASK_BACKUP = 'cup_mask.pkl'
DISC_BACKUP = 'disc.pkl'
CUP_BACKUP = 'cup.pkl'


class RimoneDataset:
    """RIMONE -r3 Dataset.

    This class assumes there are images at the path given meeting the following
    conditions:

    - 3 images per entry:
      - Original image named: 'D-#-S.jpg', where D is the diagnosis of the eye,
        # is a patient identifier and S is the eye (left or right). For example:
        G-12-L.jpg would contain the left eye of a person with glaucoma.
        N-1-R.jpg would contain the right eye of a healthy person.
      - Disc mask image named: 'D-#-S-1-Disc-exp1.jpg' formatted as the original
        image.
      - Cup mask image named: 'D-#-S-1-Cup-exp1.jpg' formatted as the original
        image.

    This class also stores backups for each propety, so the images are never
    recalculated.
    """

    def __init__(self, path='rimone', verbose=True):
        """Inits RIMONE."""
        # Knows where to look
        self.path = path[:-1] if path[-1] == '/' else path
        # Verbosity
        self.verbose = verbose
        # Find the eyes at the Dir
        self.find_eyes()

    def find_eyes(self):
        """Looks for eyes at path given."""
        try:
            # Check for backup
            with open(f'{BACKUP_DIR}/{META_BACKUP}', 'rb') as file:
                self._Y, self._ids = pickle.load(file)
        except FileNotFoundError:
            # Not found, create meta-data
            ids = []
            Y = []
            for file in os.listdir(self.path):
                if not re.search(VALID_IMAGE_REGEXP, file):
                    continue
                if self.verbose:
                    print(f' => {len(ids)} images found', end='\r')
                eye = file[:-4]
                ids.append(eye)
                Y.append(1 if eye[0] == 'G' else 0)
            if self.verbose:
                print(f' => {len(ids)} images found')
            self._Y = np.array(Y)
            self._ids = ids
            # Store backup
            with open(f'{BACKUP_DIR}/{META_BACKUP}', 'wb+') as file:
                pickle.dump((self._Y, self._ids), file)

    @property
    def ids(self):
        """Gets IDS."""
        return self._ids

    @property
    def Y(self):
        """Gets targets."""
        return self._Y

    @property
    def original_images(self):
        """Gets the original images as numpy arrays."""
        try:
            # Check for backup
            with open(f'{BACKUP_DIR}/{ORIGINAL_BACKUP}', 'rb') as file:
                imgs = pickle.load(file)
        except FileNotFoundError:
            # Read all images
            imgs = np.array([
                cv2.imread(f'{self.path}/{eye}.jpg') for eye in self.ids
            ])
            # Store backup
            with open(f'{BACKUP_DIR}/{ORIGINAL_BACKUP}', 'wb+') as file:
                pickle.dump(imgs, file)
        return imgs

    @property
    def disc_masks(self):
        """Gets the disc masks from experts as numpy arrays."""
        try:
            # Check for backup
            with open(f'{BACKUP_DIR}/{DISC_MASK_BACKUP}', 'rb') as file:
                imgs = pickle.load(file)
        except FileNotFoundError:
            # Read all disc masks
            imgs = np.array([
                cv2.imread(f'{self.path}/{eye}-1-Disc-exp1.jpg')[:, :, 0]
                for eye in self.ids
            ])
            # Store backup
            with open(f'{BACKUP_DIR}/{DISC_MASK_BACKUP}', 'wb+') as file:
                pickle.dump(imgs, file)
        return imgs

    @property
    def cup_masks(self):
        """Gets the cup masks from experts as numpy arrays."""
        try:
            # Check for backup
            with open(f'{BACKUP_DIR}/{CUP_MASK_BACKUP}', 'rb') as file:
                imgs = pickle.load(file)
        except FileNotFoundError:
            # Read all cup masks
            imgs = np.array([
                cv2.imread(f'{self.path}/{eye}-1-Cup-exp1.jpg')[:, :, 0]
                for eye in self.ids
            ])
            # Store backup
            with open(f'{BACKUP_DIR}/{CUP_MASK_BACKUP}', 'wb+') as file:
                pickle.dump(imgs, file)
        return imgs

    @property
    def discs(self):
        """Returns as a numpy array the images of the fundus with disc mask applied."""
        try:
            # Check for backup
            with open(f'{BACKUP_DIR}/{DISC_BACKUP}', 'rb') as file:
                imgs = pickle.load(file)
        except FileNotFoundError:
            # Read images & apply masks
            imgs = np.array([
                cv2.bitwise_and(img, img, mask=mask)
                for img, mask in zip(self.original_images, self.disc_masks)
            ])
            # Store backup
            with open(f'{BACKUP_DIR}/{DISC_BACKUP}', 'wb+') as file:
                pickle.dump(imgs, file)
        return imgs

    @property
    def cups(self):
        """Returns as a numpy array the images of the fundus with cup mask applied."""
        try:
            # Check for backup
            with open(f'{BACKUP_DIR}/{CUP_BACKUP}', 'rb') as file:
                imgs = pickle.load(file)
        except FileNotFoundError:
            # Read images & apply masks
            imgs = np.array([
                cv2.bitwise_and(img, img, mask=mask)
                for img, mask in zip(self.original_images, self.cup_masks)
            ])
            # Store backup
            with open(f'{BACKUP_DIR}/{CUP_BACKUP}', 'wb+') as file:
                pickle.dump(imgs, file)
        return imgs

    @property
    def classes_names(self):
        """Classes for each target."""
        return ['Normal', 'Glaucoma']


def dataset():
    return RimoneDataset()
