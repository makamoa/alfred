import numpy as np
from shapes import Canvas
import gzip

def from_labels_to_x(labels, input_shape=[100, 100]):
    x = np.zeros(len(labels) * 4)
    nx = input_shape[0]
    ny = input_shape[1]
    for i, label in enumerate(labels):
        xy_c, wx, wy = label['rectangle']
        x[i * 4:(i + 1) * 4] = xy_c[0] / nx, xy_c[1] / ny, wx / nx, wy / ny
    return np.array(x)

def from_x_to_labels(x, input_shape=[100, 100]):
    labels = []
    nx = input_shape[0]
    ny = input_shape[1]
    x = x.reshape(-1, 4)
    for i, rect in enumerate(x):
        xy_c = tuple(np.rint([rect[0] * nx, rect[1] * ny]).astype(int))
        wx, wy = np.rint([rect[2] * nx, rect[3] * ny]).astype(int)
        label = {'rectangle': [xy_c, wx, wy]}
        labels.append(label)
    return labels

def from_labels_to_image(labels, input_shape=[100, 100]):
    mask = np.zeros([100, 100], dtype=np.int32)
    canvas = Canvas.from_labels(mask, labels)
    return canvas.mask

def from_x_to_image(x, input_shape=[100, 100]):
    labels = from_x_to_labels(x, input_shape=input_shape)
    image = from_labels_to_image(labels, input_shape=input_shape)
    return image

def prepare_data(datadir='data/'):
    pass