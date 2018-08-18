import numpy as np
from image_tools import transform_image

"""
Data Augmentation:
  For each image, generate 5 images with random image transformations.
  Transformation range: [rotation, translation, shear]
  The image will be normalized and added random brightness.
"""
def augment_data(x, y, transformations=[40, 5, 5]):
    images = []
    labels = []
    assert(len(x) == len(y))

    for i in range(x.shape[0]):
        for j in range(3):
            images.append(transform_image(x[i], transformations))
            labels.append(y[i])

    return np.array(images), np.array(labels)
