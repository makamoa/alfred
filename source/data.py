"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from processor import process_image
from tensorflow.keras.utils import to_categorical
import pandas as pd
from scipy.interpolate import interp1d
from skimage.transform import resize
from skimage.transform import AffineTransform, warp, rotate
import pickle
from shapes import generate_random_shapes, Canvas, ShapelyShape as Shape

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

def random_shift(image, range):
    x_shift, y_shift = np.random.randint(low=-range//2,high=range//2,size=2)
    transform = AffineTransform(translation=[x_shift,y_shift])
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    shifted = shifted.astype(image.dtype)
    return shifted

class DataSet():
    def __init__(self, one_thickness=True, npoints=40, xmin=0.3, xmax=1.0, spectra_type=1,input_shape=(250,250)):
        self.data_dir = os.environ['DATADIR'] + 'deepnano/transmissions/'
        self.project_dir = os.environ['DATADIR'] + 'deepnano/'
        self.one_thickness=one_thickness
        self.npoints = npoints
        self.xmin = xmin
        self.xmax = xmax
        self.spectra_type = spectra_type
        self.input_shape=input_shape
        self.data = self.get_data()
        #split train val test
        self.train = self.data[self.data.status == 'train']
        self.test = self.data[self.data.status == 'test']
        self.val = self.data[self.data.status == 'val']

    def get_data(self):
        """Load our data from file."""
        path = os.path.join(self.project_dir, 'clean_data.csv')
        dataset = pd.read_csv(path, index_col=0)
        if self.one_thickness:
            dataset = dataset[dataset.thick_idx==0]
        self.number_of_thick = dataset.thick_idx.unique().__len__()
        return dataset

    def get_geometry(self, item):
        fname = os.path.join(self.data_dir,item.geometry)
        e = np.fromfile(fname, dtype=np.int32).reshape(*self.input_shape)
        if self.one_thickness==True:
            idx = e == 2
            e[idx] = 1
            e = np.expand_dims(e,axis=-1)
        else:
            # encode thickness
            idx = e == 2
            e = np.empty((*self.input_shape,self.number_of_thick))
            e[idx] = self.get_thick_one_hot(item.thick_idx)
        return e

    def get_spectra(self, item):
        fname = os.path.join(self.data_dir, item.spectra)
        s = np.fromfile(fname).reshape(-1,5)
        wl = s[:,0]
        t = s[:,self.spectra_type]
        idx = (wl + 0.1 > self.xmin) * (wl - 0.3 < self.xmax)
        f = interp1d(wl[idx],t[idx])
        wl = np.linspace(self.xmin,self.xmax,self.npoints)
        return f(wl)

    def clean_data(self):
        pass

    def get_thick_one_hot(self,thick_id):
        thick = np.zeros(self.number_of_thick)
        thick[thick_id] = 1.0
        return thick

    def get_all_sequences_in_memory(self, train_val_test, with_indices=False,**kargs):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        if train_val_test == 'train':
            data = self.train
        elif train_val_test == 'val':
            data = self.val
        elif train_val_test == 'test':
            data = self.test

        print("Loading %d samples into memory for %sing." % (len(data), train_val_test))

        X, y, indices = [], [], []
        for idx, row in data.iterrows():
            geometry = self.get_geometry(row,**kargs)
            if self.equispaced:
                spectra = self.get_spectra(row)
            else:
                spectra = self.get_spectra2(row)
            X.append(geometry)
            y.append(spectra)
            indices.append(idx)
        if with_indices:
            return np.array(indices), np.array(X), np.array(y)
        else:
            return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_val_test, random_orientation=False, random_translation=False, input_mode=False,**kargs):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset.
        if train_val_test == 'train':
            data = self.train
        elif train_val_test == 'val':
            data = self.val
        elif train_val_test == 'test':
            data = self.test

        X = np.zeros([batch_size, *self.input_shape, 1], dtype=np.int32)
        y = np.zeros([batch_size, self.npoints], dtype=np.float32)
        print("Creating %s generator with %d samples." % (train_val_test, len(data)))
        while 1:
            # Generate batch_size samples.
            sample = data.sample(batch_size)
            for i,(_,row) in enumerate(sample.iterrows()):
                # Reset to be safe.
                sequence = None
                orientation = 0
                if random_orientation:
                    orientation = np.random.randint(0,2)
                X[i] = self.get_geometry(row, orientation=orientation, random_translation=random_translation,**kargs)
                if not input_mode:
                    if self.equispaced:
                        y[i] = self.get_spectra(row, channel=orientation)
                    else:
                        y[i] = self.get_spectra2(row, channel=orientation)
            if input_mode:
                yield X, X
            else:
                yield X, y

class OfflineData(DataSet):
    datadir = {'shapes_same_period' : 'data150919',
               'rect_same_period' : 'data300919',
               'rect_various_period' : 'data190919',
               'rect_same_period2' : 'data131019',
               'rect_same_period_100_100' : 'data251019',
               'rect_same_period_100_100_corrected' : 'data251019_corrected',
               'data_rect_50_50' : 'data_50_50','data_rect_100_100' : 'data_100_100','data_rect_150_150' : 'data_150_150',
               'toydata' : 'toydata'}
    def __init__(self, one_thickness=True, thick_idx=0,npoints=40, xmin=0.3, xmax=1.0, spectra_type=1,input_shape=(224,224), equispaced=True, datafile='shapes_same_period'):
        if type(datafile) is not list:
            datafile=[datafile]
        self.datafiles = [file+'.csv' for file in datafile]
        self.thick_idx = thick_idx
        self.metadir = os.environ['DATADIR'] + 'deepnano/' + 'metadata/'
        super(OfflineData, self).__init__(one_thickness=one_thickness, npoints=npoints, xmin=xmin, xmax=xmax, spectra_type=spectra_type,input_shape=input_shape)
        #if equispaced - equispaced in wavelenght, otherwise in frequency
        self.equispaced = equispaced

    def load_metadata(self):
        fname = '_'.join([file[:-4] for file in self.datafiles]) + '-' + str(self.thick_idx) + '.pkl'
        dict_f = pickle.load(open(os.path.join(self.metadir,fname), "rb"))
        return dict_f

    def generate_metadata(self, data):
        print('Generating metadata...')
        dict_f = {}
        for i, item in data.iterrows():
            fname = os.path.join(item.datadir, item.name + '-spectra.bin')
            try:
                nchannels = 5
                s = np.fromfile(fname).reshape(-1, nchannels)
            except ValueError:
                nchannels=3
                s = np.fromfile(fname).reshape(-1, nchannels)
            wl = s[:, 0]
            idx = (wl + 0.1 > self.xmin) * (wl - 0.3 < self.xmax)
            t_f = []
            for channel in range(1,nchannels):
                t = s[:,channel]
                t_f.append(interp1d(wl[idx], t[idx]))
            dict_f[item.name] = t_f.copy()
        if dict_f.__len__() != data.__len__():
            raise ValueError("Not all samples were succesfully interpolated")
        dir = os.environ['DATADIR'] + 'deepnano/'
        fname = '_'.join([file[:-4] for file in self.datafiles]) + '-' + str(self.thick_idx) + '.pkl'
        pickle.dump(dict_f, open(os.path.join(self.metadir,fname), "wb" ) )

    def get_metadata(self, data):
        try:
            metadata = self.load_metadata()
        except FileNotFoundError:
            self.generate_metadata(data)
            metadata = self.load_metadata()
        return metadata

    def get_data(self):
        """Load our data from file."""
        datasets = []
        for datafile in self.datafiles:
            datadir = os.environ['DATADIR'] + 'deepnano/' + OfflineData.datadir[datafile[:-4]]
            path = os.path.join(self.project_dir, datafile)
            currdata = pd.read_csv(path, index_col=0)
            currdata['datadir'] = datadir
            datasets.append(currdata.copy())
        if (datasets.__len__() > 1) and (datasets[0] is datasets[1]):
            raise ValueError('Using the same dataset multiple times!')
        dataset = pd.concat(datasets).sample(frac=1)
        unique_thick = sorted(dataset.dz.unique())
        self.number_of_thick = unique_thick.__len__()
        if self.number_of_thick > 1 and self.thick_idx is None:
            raise ValueError('Please specify thickness to fit')
        else:
            dataset = dataset[dataset.dz==unique_thick[self.thick_idx]]
        assert dataset.dz.unique().__len__() == 1
        self.interp_f = self.get_metadata(dataset)
        return dataset

    def get_geometry(self, item, pad=True, orientation=0, random_translation=False, resized=False):
        fname = os.path.join(item.datadir,item.name + '-mask.bin')
        dimx = int(item.nx)
        dimy = int(item.ny)
        mask = np.fromfile(fname, dtype=np.int32).reshape(dimx, dimy)
        if orientation:
            mask = rotate(mask,90,preserve_range=True).astype(np.int32)
        if random_translation:
            mask = random_shift(mask, range=mask.shape[0])
        if pad:
            mask = np.pad(mask,pad_width=[(0,self.input_shape[0]-mask.shape[0]),
                                   (0,self.input_shape[0]-mask.shape[1])],
                                    mode='constant', constant_values=-1)
            mask = np.expand_dims(mask,axis=-1)
        elif resized:
            image_resized = resize(mask.astype(np.uint8), (self.input_shape[0], self.input_shape[1]),anti_aliasing=True)
            image_resized *= 255
            image_resized = image_resized.astype(np.uint8)
            mask = np.expand_dims(image_resized,axis=-1)
        return mask

    def get_spectra(self, item, channel=0):
        # get spectra equispaced in time domain
        f = self.interp_f[item.name][channel]
        wl = np.linspace(self.xmin, self.xmax, self.npoints)
        return f(wl)

    def get_spectra2(self, item, channel=0):
        # get spectra equispaced in frequency domain
        f = self.interp_f[item.name][channel]
        freq = np.linspace(1 / self.xmax, 1 / self.xmin, self.npoints)
        wl = 1 / freq
        return f(wl)

class DataSetVarPeriod(OfflineData):
    def __init__(self,*pargs,**kargs):
        super(DataSetVarPeriod, self).__init__(*pargs,**kargs)

    def get_geometry(self, item, orientation=0):
        img = super(DataSetVarPeriod, self).get_geometry(item, orientation=orientation)
        mesh = self.get_mesh(img)
        geometry = np.concatenate([img,mesh], axis=-1)
        return geometry

    def get_geometry2(self, item, ):
        img = super(DataSetVarPeriod, self).get_geometry(item, pad=False)
        mesh = self.get_mesh(img)
        image_resized = resize(img.astype(np.uint8), (self.input_shape[0], self.input_shape[1]),
                               anti_aliasing=True, )
        image_resized *= 255
        image_resized = image_resized.astype(np.uint8)
        image_resized = np.expand_dims(image_resized,axis=-1)
        geometry = np.concatenate([image_resized,mesh], axis=-1)
        return geometry

    def get_mesh(self, img, normalize=False):
        nx, ny = img.shape[:2]
        xx = np.linspace(0, 1, self.input_shape[0])
        yy = np.linspace(0, 1, self.input_shape[1])
        vx, vy = np.meshgrid(xx, yy)
        mesh = np.stack([vx, vy], axis=-1)
        if normalize:
            mesh /= np.array(self.input_shape)
        return mesh

class OnlineData():
    def __init__(self, predictor=None, input_shapes=[100]):
        self.predictor = predictor
        self.input_shapes = np.zeros([len(input_shapes),2], dtype=np.int)
        for i,shape in enumerate(input_shapes):
            try:
                self.input_shapes[i,:] = shape[:]
            except TypeError:
                self.input_shapes[i, :] = [shape, shape]

    def get_geometry_label(self, mode='random', **kargs):
        if mode == 'random':
            geometry, label = self.generate_random_geometry(**kargs)
        else:
            raise ValueError("Unreckognized geometry mode!")
        return geometry, label

    def generate_random_geometry(self, shape=[100,100], save=False, file=None, **kargs):
        canvas = np.zeros(shape, np.int32)
        canvas = generate_random_shapes(canvas, **kargs)
        if save:
            canvas.save(file)
        return canvas.mask, canvas.get_labels()

    def predict_spectra(self):
        pass

    @threadsafe_generator
    def geometry_label_generator(self, batch_size, input_mode=False,**kargs):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:
        """
        while 1:
            # Generate batch_size samples.
            X = []
            y = []
            # different shapes are going to different input-output channels (keras)
            for shape in self.input_shapes:
                Xs = np.zeros([batch_size,*shape],dtype=np.int32)
                ys = np.zeros(batch_size, dtype=object)
                for i in range(batch_size):
                    # Reset to be safe.
                    sequence = None
                    orientation = 0
                    Xs[i], ys[i] = self.get_geometry_label(shape=shape,**kargs)
                X.append(Xs.copy())
                y.append(ys.copy())
            if self.input_shapes.__len__() == 1:
                X = X[0]
                y = y[0]
            if input_mode:
                yield X, X
            else:
                yield X, y