import numpy as np
import matplotlib.pyplot as plt
from shapes import Shape, Canvas, ShapelyShape, generate_random_shapes
from shapely.affinity import scale
from shapely.ops import cascaded_union

class Grid():
    def __init__(self, grid_shape, shapes=None, material_idx=1,coords='index', bottom=None):
        self.mask = np.zeros(grid_shape, dtype=np.int32)
        self.shape = grid_shape
        if bottom:
            self.z0 = bottom['h']
            self.mask[:,:,:self.z0] = bottom['material_idx']
        if shapes:
            for shape in shapes:
                self.add_shape(shape, material_idx=material_idx)

    def add_shape(self, shape, material_idx=1):
        if shape['type'] == 'rectangle':
            self.add_rectangle(*shape['params'], material_idx=material_idx)
        elif shape['type'] == 'cylinder':
            self.add_cylinder(*shape['params'], material_idx=material_idx)
        elif shape['type'] == 'canvas':
            self.add_canvas(*shape['params'], material_idx=material_idx)
        elif shape['type'] == 'pyramid':
            self.add_pyramid(*shape['params'], material_idx=material_idx)
        elif shape['type'] == 'slab':
            self.add_slab(*shape['params'], material_idx=material_idx)
        else:
            raise ValueError('Shape type was not recognized!')

    def add_rectangle(self, xy_c, wx, wy, h, z0=None, material_idx=1):
        if not z0:
            z0 = self.z0
        for z in range(z0, z0+h):
            shape = Shape(self.shape, ['rectangle',[xy_c,wx,wy]])
            mask = shape.mask == 1
            self.mask[mask,z] = material_idx

    def add_cylinder(self, xy_c, wx, wy, h, z0=None, material_idx=1):
        if not z0:
            z0 = self.z0
        for z in range(z0, z0 + h):
            shape = Shape(self.shape, ['ellipse', [xy_c, wx, wy]])
            mask = shape.mask == 1
            self.mask[mask, z] = material_idx

    def add_canvas(self, canvas, h, z0=None, material_idx=1):
        mask = canvas.mask
        if not z0:
            z0 = self.z0
        for z in range(z0, z0 + h):
            mask = mask == 1
            self.mask[mask, z] = material_idx

    def add_slab(self, h, z0=None, material_idx=1):
        if not z0:
            z0 = self.z0
        for z in range(z0, z0 + h):
            self.mask[:,:,z] = material_idx

    def add_pyramid(self, canvas, h, factor=1e-3, z0=None, material_idx=1):
        if not z0:
            z0 = self.z0
        xyfactors = np.linspace(1,factor,h)
        for z, factor in zip(range(z0,z0+h),xyfactors):
            polygons = []
            for shape in canvas.shapes:
                polygon = shape.polygon
                polygon = scale(polygon, xfact=factor, yfact=factor)
                mask = ShapelyShape.to_numpy2(self.mask.shape,polygon)
                mask = mask == 1
                self.mask[mask, z] = material_idx
    
    def add_upml(self, upml_gap=0):
        idx = self.mask[0, 0, 0]
        self.mask = np.pad(self.mask,[[upml_gap,upml_gap],[upml_gap,upml_gap],[0,0]],'constant')
        if hasattr(self,'z0'):
            self.mask[:,:,:self.z0] = idx
            pass

from skimage.draw import polygon
import os

class Geometry2d():
    def __init__(self, ec):
        self.grids = {}
        self.grids['c'] = ec.copy()
        self.grids['x'] = self.get_Ex(ec)
        self.grids['z'] = self.get_Ez(ec)

    @staticmethod
    def get_Ex(ec):
        return np.pad(ec, [(0, 0), (1, 0)], 'edge')

    @staticmethod
    def get_Ez(ec):
        return np.pad(ec, [(1, 0), (0, 0)], 'edge')

class Geometry3d():
    def __init__(self, ec):
        self.grids = {}
        self.grids['c'] = ec.copy()
        self.grids['x'] = self.get_Ex(ec)
        self.grids['z'] = self.get_Ez(ec)
        self.grids['y'] = self.get_Ey(ec)

    @staticmethod
    def get_Ex(ec):
        return np.pad(ec,[(0,0),(0,1),[1,0]],'edge')

    @staticmethod
    def get_Ey(ec):
        return np.pad(ec,[(0,1),(0,0),[1,0]],'edge')

    @staticmethod
    def get_Ez(ec):
        return np.pad(ec, [(0, 1), (1, 0), [0, 0]], 'edge')

if __name__ == '__main__':
    grid_shape = [224,224,200]
    canvas = np.zeros(grid_shape[:2], np.uint8)
    canvas = generate_random_shapes(canvas, types=['rectangle', 'circle'], min_size=10, max_shapes=10)
    canvas.show()
    calc_grid = Grid(grid_shape=grid_shape,
         shapes=[{'type':'pyramid', 'params':[canvas,50, 0.5]}],
         material_idx=2,
         bottom={'h':50, 'material_idx' : 1}).mask

    plt.imshow(calc_grid[:,:,50])
    plt.show()
    plt.imshow(calc_grid[:,:,80])
    plt.show()
