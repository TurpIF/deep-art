import scipy
import scipy.misc
import numpy as np


def load(path):
    img = scipy.misc.imread(path)

## TODO check what is the possible returned shapes
    if img.shape[-1] == 1: # grey image
        img = np.array([img, img, img])
    elif img.shape[-1] == 4: # alpha component
        img = img[:,:,:3]
    return img


def save(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)
