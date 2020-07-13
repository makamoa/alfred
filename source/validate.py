"""
Basically just runs a test generator on
about the same number of videos as we have in our test set.
"""
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import OfflineData as DataSet
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


data_dir = os.environ['DATADIR'] + 'deepnano/'

def validate(model, saved_model, npoints=80, datafile='rect_same_period',pad=True,resized=False,**kargs):
    now = datetime.now()
    date = now.strftime("%d:%m:%Y-%H:%M")
    data = DataSet(npoints=npoints, datafile=datafile, **kargs)
    rm = ResearchModels(model, npoints=npoints, saved_model=saved_model)
    indices, X, y = data.get_all_sequences_in_memory('test',with_indices=True,pad=pad,resized=resized)
    eval = rm.model.evaluate(X,y)
    pred = rm.model.predict(X)
    print(eval)
    np.save('.tmp/indices-%s-%s-%s' % (model, datafile, os.path.basename(saved_model)),indices)
    np.save('.tmp/prediction-%s-%s-%s' % (model,datafile,os.path.basename(saved_model)),pred)
    np.save('.tmp/true-%s-%s-%s' % (model,datafile,os.path.basename(saved_model)),y)

if __name__=='__main__':
    model_name, model_file = sys.argv[1:]
    datafile = ['data_rect_50_50']
    model = model_name
    saved_model = os.path.join(data_dir,'checkpoints/', model, model_file)
    validate(model, saved_model=saved_model, npoints=80, equispaced=True,datafile=datafile, input_shape=[100,100], thick_idx=7, pad=False, resized=True)
