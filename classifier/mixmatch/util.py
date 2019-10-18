"""
Environment wrappers and helpful functions that do not fit
nicely into any other file.
"""

import os
import random

from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF


def mirror_obs(obs):
    """
    Mirror an observation.
    """
    obs = obs.copy()
    obs[:] = obs[:, ::-1]
    return obs


def atomic_save(obj, path):
    """
    Save a model to a file, making sure that the file will
    never be partly written.

    This prevents the model from getting corrupted in the
    event that the process dies or the machine crashes.
    """
    torch.save(obj, path + '.tmp')
    os.rename(path + '.tmp', path)


class Augmentation:
    """
    A collection of settings indicating how to slightly
    modify an image.
    """

    def __init__(self):
        self.brightness = random.random() * 0.1 + 0.95
        self.contrast = random.random() * 0.1 + 0.95
        self.gamma = random.random() * 0.1 + 0.95
        self.hue = random.random() * 0.1 - 0.05
        self.saturation = random.random() * 0.1 + 0.95
        self.translation = (random.randrange(-2, 3), random.randrange(-2, 3))

    def apply(self, image):
        return Image.fromarray(self.apply_np(np.array(image)))

    def apply_np(self, np_image):
        content = Image.fromarray(np_image)
        content = TF.adjust_brightness(content, self.brightness)
        content = TF.adjust_contrast(content, self.contrast)
        content = TF.adjust_gamma(content, self.gamma)
        content = TF.adjust_hue(content, self.hue)
        content = TF.adjust_saturation(content, self.saturation)
        content = TF.affine(content, 0, self.translation, 1.0, 0)
        result = np.array(content)
        return result

