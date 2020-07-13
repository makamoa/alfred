import numpy as np
from skimage.draw import polygon
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from skimage.draw import circle, ellipse, rectangle
from shapely.affinity import skew, affine_transform, translate, rotate, scale
from shapely.ops import triangulate
from skimage.draw import random_shapes

shape_types = ['rectangle', 'square', 'circle', 'ellipse', 'ring', 'shoe', 'polygon']

def to_shapely_coords(coords):
    x0, x1 = coords[0]
    y0, y1 = coords[1]
    return [(int((x0+x1)/2), int((y0+y1)/2)), x1-x0, y1-y0]

def convert_labels(labels, mesh, units=False):
    for i, label in enumerate(labels):
        coords = label[1]
        coords = to_shapely_coords(coords)
        if units:
            coords = [mesh[x,y] for x, y in coords]
        labels[i] = [label[0],coords]
    return labels

def generate_rect_mask(mesh, min_shapes=1, max_shapes=1, skimage_labels=False, multichannel=False, intensity_range=((1, 1)), units=False, shape='rectangle',**kargs):
    image, labels = random_shapes(mesh.shape, min_shapes=min_shapes, shape='rectangle', max_shapes=max_shapes, multichannel=multichannel, intensity_range=intensity_range, **kargs)
    idx = image == 255
    image[idx] = 0
    if not skimage_labels:
        labels = convert_labels(labels, mesh, units=units)
    return image, labels
