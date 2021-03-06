import numpy as np
from skimage.draw import polygon, rectangle, ellipse
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from skimage.draw import circle, ellipse, rectangle, rectangle_perimeter
from shapely.affinity import skew, affine_transform, translate, rotate, scale
from shapely.ops import triangulate
from skimage.draw import random_shapes
from skimage import measure
import matplotlib.pyplot as plt
from shape_generators import generate_rect_mask
from skimage.morphology import convex_hull_image
from shapely.errors import TopologicalError

MIN_FEATURE_SIZE=5

shape_types = ['rectangle', 'square', 'circle', 'ellipse', 'ring', 'shoe', 'polygon']

class Canvas():
    def __init__(self, canvas, shapes=None, material_idx=1):
        self.mask = canvas
        self.shapes=[]
        if shapes:
            self.add_shapes(shapes, material_idx=material_idx)
            
    @staticmethod
    def labels_to_shapely_format(labels):
        res = []
        for label in labels:
            for key in label:
                res.append([key, label[key]])
        return res

    @classmethod
    def from_labels(cls, canvas, labels, material_idx=1, shapely=True):
        shapes = []
        labels = Canvas.labels_to_shapely_format(labels)
        for label in labels:
            if shapely:
                shape = ShapelyShape(canvas.shape,label)
            else:
                shape = Shape(canvas.shape, label)
            shapes.append(shape)
        return cls(canvas, shapes, material_idx=material_idx)

    def embed_shape(self, shape):
        return shape

    def add_shape(self, shape, material_idx=1):
        mask = shape.mask == 1
        self.mask[mask] = material_idx
        self.shapes.append(shape)

    def add_shapes(self, shapes, material_idx=1):
        for shape in shapes:
            self.add_shape(shape, material_idx)

    def show(self):
        plt.imshow(self.mask)
        plt.show()

    def find_shapes(self, shapely=False):
        pass
    
    def __str__(self):
        if hasattr(self,'shapes'):
            return '-'.join([shape.type for shape in self.shapes])
        else:
            'empty'

    def save(self, fname, save_labels=True):
        self.mask.tofile(fname+'-mask.bin')
        if save_labels:
            labels = []
            for shape in self.shapes:
                label = {shape.type:shape.params}
                labels.append(label)
            np.save(fname+'-labels',np.array(labels))

    def get_labels(self):
        labels=[]
        for shape in self.shapes:
            label = {shape.type:shape.params}
            labels.append(label)
        return labels



def split_ext_int(polygons):
    output = []
    status = ['']*len(polygons)
    for i, poly1 in enumerate(polygons):
        holes_idx = [j for j, poly in enumerate(polygons)
                     if poly1.contains(poly) and i!=j]
        if holes_idx:
            status[i] = 'shell'
            for hole_idx in holes_idx:
                poly1 = poly1.symmetric_difference(polygons[hole_idx])
                status[hole_idx] = 'hole'
            output.append(poly1)
    output += [poly for (poly,state) in zip(polygons,status) if state=='']
    return output

class Shape():
    def __init__(self, img_shape, labels, mode='run'):
        self.type, self.params = labels[0], list(labels[1])
        self.size = img_shape[:2]
        self.mask = np.zeros(img_shape[:2], dtype=np.int32)
        self.add_shape()
        self.find_contours()


    @classmethod
    def from_polygon(cls):
        pass

    @classmethod
    def embedded_into_rectangle(cls, img_shape, rect_labels, shape_type, mode='debug',**kargs):
        # when debug mode save the rectangular mask perimeter
        if mode == 'debug':
            # from (wx_c), wx, wy to start,extent
            xy_c, wx, wy = rect_labels
            start = (xy_c - np.array([wx,wy]) / 2.).astype(np.int)
            rr, cc = rectangle_perimeter(start=start, extent=(wx, wy))
            cls.rectangle_mask_contour = np.stack([rr,cc],axis=-1)
        params = embed_to_rectangle(rect_labels, shape_type, **kargs)
        labels = [shape_type, params]
        return cls(img_shape, labels)

    def find_contours(self):
        self.contours = measure.find_contours(self.mask, 0.95, positive_orientation='low')

    def add_shape(self):
        if self.type == 'rectangle':
            self.add_rectangle(*self.params)
        elif self.type == 'ellipse':
            self.add_ellipse(*self.params)
        elif self.type == 'circle':
            self.add_circle(*self.params)
        elif self.type == 'square':
            self.add_square(*self.params)
        elif self.type == 'ring':
            self.add_ring(*self.params)
        elif self.type == 'polygon':
            self.add_polygon(*self.params)
        elif self.type == 'shoe':
            self.add_shoe(*self.params)
        else:
            raise AttributeError('Could not find such a shape in a dictionary!')

    def add_rectangle(self, xy_c, wx, wy, idx=1):
        start = (np.array(xy_c) - np.array([wx,wy]) / 2.).astype(np.int)
        rr, cc = rectangle(start=start, extent=(wx,wy), shape = self.mask.shape)
        self.mask[rr,cc] = idx

    def add_square(self, xy_c, w, idx=1):
        self.add_rectangle(xy_c, w, w, idx=idx)

    def add_ellipse(self, xy_c, wx, wy, idx=1):
        rr, cc = ellipse(*xy_c, wx, wy, shape=self.mask.shape)
        self.mask[rr, cc] = idx

    def add_ring(self, xy_c, wxy_e, wxy_i):
        """Interior domain is greater than exterior"""
        if np.any(wxy_e < wxy_i):
            raise ValueError("Interior domain is greater than exterior")
        self.add_ellipse(xy_c, *wxy_e)
        self.add_ellipse(xy_c, *wxy_i, idx=0)

    def add_circle(self, xy_c, w, idx=1):
        self.add_ellipse(xy_c, w, w, idx=idx)

    def add_polygon(self, points, idx=1):
        for point in points:
            self.mask[point[0], point[1]] = idx
        self.mask = convex_hull_image(self.mask)

    def add_shoe(self, xy_c, wxy_e, wxy_i):
        try:
            wy_e = wxy_e[1]
        except IndexError:
            wy_e = wxy_e[0]
        try:
            wy_i = wxy_i[1]
        except IndexError:
            wy_i = wxy_i[0]
        """Interior domain is greater than exterior"""
        if np.any(wxy_e < wxy_i):
            raise ValueError("Interior domain is greater than exterior")
        self.add_rectangle(xy_c, *wxy_e)
        self.add_rectangle(np.array(xy_c)+ np.array([0,(wy_e-wy_i)/2+1]).astype(np.int), *wxy_i, idx=0)

    @staticmethod
    def to_shapely(mesh, canvas):
        contours = measure.find_contours(canvas, 0.95, positive_orientation='low')
        polygons = []
        for contour in contours:
            contour = np.round(contour).astype(np.int)
            contour = [mesh[ix, iy] for ix, iy in contour]
            points = MultiPoint(contour)
            pol = Polygon(points)
            polygons.append(pol)
        polygons = split_ext_int(polygons)
        return polygons

    def __repr__(self):
        pass

    def show(self):
        plt.imshow(self.mask)
        for contour in self.contours:
            plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
            if hasattr(self,'rectangle_mask_contour'):
                rect_contour = self.rectangle_mask_contour
                plt.plot(rect_contour[:, 1], rect_contour[:, 0], 'g', linewidth=2)
        plt.axis('image')
        plt.show()

    def __str__(self):
        output = 'Shape Type: %s, Parameters: %s' % (self.type, str(self.params))

class ShapelyShape(Shape):
    def __init__(self, img_shape, labels):
        super(ShapelyShape,self).__init__(img_shape, labels)
        pass

    @staticmethod
    def ellipse(xy_c, wx, wy=None, rotation=0):
        if wy is None:
            wy = wx
        pol = Point(xy_c).buffer(1.0)
        pol = scale(pol, wx, wy)
        pol = rotate(pol, rotation, origin='centroid')
        return pol

    @staticmethod
    def rectangle(xy_c, wx, wy=None, rotation=0):
        if wy is None:
            wy = wx
        x = np.zeros(4, dtype=float)
        y = np.zeros(4, dtype=float)
        x[0], y[0] = np.array(xy_c) - np.array([wx / 2., wy / 2.])
        x[1], y[1] = np.array([x[0], y[0] + wy])
        x[2], y[2] = np.array([x[0] + wx, y[0] + wy])
        x[3], y[3] = np.array([x[0] + wx, y[0]])
        pol = Polygon(MultiPoint(list(zip(x, y))))
        pol = rotate(pol, rotation, origin='centroid')
        return pol

    @staticmethod
    def to_numpy(size, shape, output=None):
        if output is None:
            output = np.zeros(size[:2], np.int)
        for ix, iy in np.ndindex(size[:2]):
            point = Point([ix, iy])
            output[ix, iy] = shape.contains(point)
        return output

    @staticmethod
    def to_numpy2(size, shape, interiors=[], output=None):
        if output is None:
            output = np.zeros(size[:2], np.int)
        exterior = np.rint(np.array(shape.exterior))
        r, c = np.array(exterior)[:, 0], np.array(exterior)[:, 1]
        rr, cc = polygon(r, c, shape=size)
        output[rr, cc] = 1
        for interior in shape.interiors:
            interior = np.rint(np.array(interior))
            r, c = np.array(interior)[:, 0], np.array(interior)[:, 1]
            rr, cc = polygon(r, c, shape=size)
            output[rr, cc] = 0
        return output

    def add_rectangle(self,xy_c, wx, wy=None, idx=1):
        super(ShapelyShape,self).add_rectangle(xy_c,wx,wy, idx=idx)
        if wy is None:
            wy = wx
        x = np.zeros(4, dtype=float)
        y = np.zeros(4, dtype=float)
        x[0], y[0] = np.array(xy_c) - np.array([wx / 2., wy / 2.])
        x[1], y[1] = np.array([x[0], y[0] + wy])
        x[2], y[2] = np.array([x[0] + wx, y[0] + wy])
        x[3], y[3] = np.array([x[0] + wx, y[0]])
        pol = Polygon(MultiPoint(list(zip(x, y))))
        self.polygon = pol

    def add_ellipse(self, xy_c, wx, wy=None, idx=1):
        super(ShapelyShape, self).add_ellipse(xy_c, wx, wy, idx=idx)
        if wy is None:
            wy = wx
        pol = Point(xy_c).buffer(1.0)
        pol = scale(pol, wx, wy)
        self.polygon = pol

    def add_ring(self, xy_c, wxy_e, wxy_i, rotation=0):
        super(ShapelyShape, self).add_ring(xy_c, wxy_e, wxy_i)
        """Interior domain is greater than exterior"""
        if np.any(wxy_e < wxy_i):
            raise ValueError("Interior domain is greater than exterior")
        exterior = self.ellipse(xy_c, *wxy_e, rotation=rotation)
        interior = self.ellipse(xy_c, *wxy_i, rotation=rotation)
        try:
            pol = exterior.symmetric_difference(interior)
        except TopologicalError:
            pol = exterior
            self.params[2] = [0,0]
        self.polygon = pol

    def add_polygon(self, points, idx=1):
        super(ShapelyShape, self).add_polygon(points, idx=idx)
        points = MultiPoint(points)
        pol = Polygon(points.convex_hull)
        self.polygon = pol

    def add_shoe(self, xy_c, wxy_e, wxy_i):
        super(ShapelyShape, self).add_shoe(xy_c, wxy_e, wxy_i)
        try:
            wy_e = wxy_e[1]
        except IndexError:
            wy_e = wxy_e[0]
        try:
            wy_i = wxy_i[1]
        except IndexError:
            wy_i = wxy_i[0]
        """Interior domain is greater than exterior"""
        if np.any(wxy_e < wxy_i):
            raise ValueError("Interior domain is greater than exterior")
        exterior = self.rectangle(xy_c, *wxy_e)
        interior = self.rectangle(xy_c, *wxy_i)
        interior = translate(interior, 0, (wy_e - wy_i) / 2.)
        try:
            pol = exterior.symmetric_difference(interior)
        except TopologicalError:
            pol = exterior
            self.params[2] = [0, 0]
        self.polygon = pol

def embed_to_rectangle(rect, shape_type, randomize=True):
    xy_c, wx, wy = rect
    if shape_type == 'rectangle':
        return (xy_c, wx, wy)
    elif shape_type == 'square':
        w = min(wx,wy)
        if randomize:
            shift = np.array((np.random.random()*(w-wx) - (w-wx) / 2., np.random.random()*(w-wy) - (w-wy) / 2.)).astype(np.int)
            xy_c = np.array(xy_c) + shift
        return [xy_c, w]
    elif shape_type == 'ellipse':
        return [xy_c, wx/2, wy/2.]
    elif shape_type == 'circle':
        w = min(wx,wy)
        if randomize:
            shift = np.array((np.random.random()*(w-wx) - (w-wx) / 2., np.random.random()*(w-wy) - (w-wy) / 2.)).astype(np.int)
            xy_c = np.array(xy_c) + shift
        return [xy_c, w/2.]
    elif shape_type == 'ring':
        xy_c, wx_e, wy_e=embed_to_rectangle(rect, shape_type='ellipse')
        try:
            wx_i = np.random.randint(low=0,high=max(0,wx_e-MIN_FEATURE_SIZE))
            wy_i = np.random.randint(low=0,high=max(0,wx_e-MIN_FEATURE_SIZE))
        except ValueError:
            wx_i = max(0,wx_e-MIN_FEATURE_SIZE)
            wy_i = max(0,wy_e-MIN_FEATURE_SIZE)
        return xy_c, [wx_e, wy_e], [wx_i, wy_i]
    elif shape_type == 'polygon':
        x0, y0 = (np.array(xy_c) - np.array([wx, wy]) / 2.).astype(np.int)
        x1, y1 = (np.array(xy_c) + np.array([wx, wy]) / 2.).astype(np.int)
        random_rr = np.random.randint(low=x0,high=x1,size=5)
        random_cc = np.random.randint(low=y0, high=y1, size=5)
        points = np.stack([random_rr, random_cc], axis=-1)
        return [points]
    elif shape_type == 'shoe':
        wx_e, wy_e = wx, wy
        if randomize:
            wx_i = np.random.randint(low=0, high=max(0,wx_e-MIN_FEATURE_SIZE))
            wy_i = np.random.randint(low=0, high=max(0,wx_e-MIN_FEATURE_SIZE))
        else:
            wx_i = int(wx_e/2)
            wy_i = int(wy_e/2)
        return xy_c, [wx_e, wy_e], [wx_i, wy_i]

def generate_random_shapes(mesh, types='all', shapely=True, **kargs):
    global shape_types
    image, labels = generate_rect_mask(mesh, **kargs)
    canvas = Canvas(mesh)
    shapes = []
    if types != 'all':
        shape_types = types
    for label in labels:
        shape_type = np.random.choice(shape_types)
        if shapely:
            shape = ShapelyShape.embedded_into_rectangle(mesh.shape, label[1], shape_type=shape_type)
        else:
            shape = Shape.embedded_into_rectangle(mesh.shape, label[1], shape_type=shape_type)
        shapes.append(shape)
    canvas.add_shapes(shapes)
    return canvas

if __name__=='__main__':
    grid = np.zeros([100,100], dtype=np.uint8)
    mask = generate_random_shapes(grid, max_shapes=5, min_size=MIN_FEATURE_SIZE)
    mask.show()

# if __name__ == '__main__':
#     grid = np.zeros([100,100], dtype=np.uint8)
#     canvas = Canvas(grid)
#     image, labels = generate_rect_mask(grid, min_size=MIN_FEATURE_SIZE, max_shapes=5)
#     plt.imshow(image)
#     plt.show()
#     shapes = []
#     shape_types = ['ring']
#     for label in labels:
#         shape_type = np.random.choice(shape_types)
#         shape = ShapelyShape.embedded_into_rectangle(grid.shape, label[1], shape_type=shape_type)
#         shapes.append(shape)
#
#     canvas.add_shapes(shapes)
#     canvas.show()
#     poly = rotate(shape.polygon, 45,origin='centroid')
#     poly = ShapelyShape.to_numpy(grid, poly)
#     plt.imshow(poly)
#     plt.show()






