import numpy as np
import cv2

def preprocess(image):
    # histogram equalize (increase contrast of image)
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    # normalize
    image = image / 255. - 0.5
    return image

def rotate(image, _range):
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    angle = np.random.uniform(_range) - _range / 2
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    return cv2.warpAffine(image, rotation_mat, (width, height))

def translate(image, _range):
    height, width = image.shape[:2]
    cx = _range * np.random.uniform() - _range / 2
    cy = _range * np.random.uniform() - _range / 2
    translation_mat = np.array([[1, 0, cx],
                                [0, 1, cy]], dtype=np.float32)
    return cv2.warpAffine(image, translation_mat, (width, height))

def shear(image, _range):
    height, width = image.shape[:2]
    pts1 = np.float32([[5,5], [20,5], [5,20]])
    pt1 = 5 + _range * np.random.uniform() - _range / 2
    pt2 = 20 + _range * np.random.uniform() - _range / 2
    pts2 = np.float32([[pt1,5], [pt2,pt1], [5,pt2]])
    shear_mat = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, shear_mat, (width, height))

def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
    random_bright = .5 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_bright
    image[:,:,2][image[:,:,2] > 255] = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def transform_image(image, transformations):
    image = rotate(image, transformations[0])
    image = translate(image, transformations[1])
    image = shear(image, transformations[2])
    image = random_brightness(image)
    return preprocess(image)
