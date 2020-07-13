from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import OfflineData as DataSet
import time
import os.path
import argparse
from tensorflow.keras.callbacks import Callback
import numpy as np
import sys
from sklearn.utils import class_weight
from datetime import datetime

data_dir = os.environ['DATADIR'] + 'deepnano/'

def train(model, load_to_memory=True, datafile='rect_same_period', batch_size=None, nb_epoch=1000, npoints=60, random_orientation=False, random_translation=False, pad=True, resized=False, **kargs):
    # Helper: Save the model.
    if not os.path.isdir(os.path.join(data_dir,'checkpoints', model)):
        os.mkdir(os.path.join(data_dir,'checkpoints', model))
    now = datetime.now()
    date = now.strftime("%d:%m:%Y-%H:%M")
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(data_dir, 'checkpoints', model, model + '-' + '%s-%d-{val_loss:.3f}.hdf5' % (datafile,kargs['thick_idx'])),
        verbose=1,
        save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(data_dir, 'logs', model))
    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20)
    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(data_dir, 'logs', model+'_'+date+'-'+'training-'+\
        str(timestamp) + '.log'))

    data = DataSet(npoints=npoints, datafile=datafile,**kargs)
    rm = ResearchModels(model, npoints=npoints)

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train')
        X_val, y_val = data.get_all_sequences_in_memory('val')
    else:
        # Get generators.
        k = 1
        if random_orientation:
            k *= 2
        if random_translation:
            k *= 3
        steps_per_epoch = k * len(data.train) // batch_size
        validation_steps = 0.5*len(data.val) // batch_size
        generator = data.frame_generator(batch_size, 'train', random_orientation=random_orientation, random_translation=random_translation, pad=pad, resized=resized)
        val_generator = data.frame_generator(batch_size, 'val', pad=pad, resized=resized)

    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=validation_steps)

if __name__=='__main__':
    input_shape=[100,100]
    model, thick_idx = sys.argv[1:]
    one_thickness = True
    if model == 'primanet':
        one_thickness = True
    elif model == 'alexnet':
        one_thickness = False
    elif model == 'primanet2':
        one_thickness = False
    elif model == 'vgg':
        one_thickness = True
    elif model == 'vgg2':
        one_thickness = True
        from data import DataSet2 as DataSet
    elif model=='resnet18_112_112':
        input_shape = [128,128]
    elif model=='effnet':
        input_shape = [100, 100]
    elif model=='effnet_b1':
        input_shape = [100, 100]
    train(model,
          one_thickness=True,
          load_to_memory=False,
          thick_idx=int(thick_idx),
          npoints=80,
          batch_size=64,
          datafile=['data_rect_50_50'],
          random_orientation=True,
          random_translation=False,
          input_shape=input_shape,
          pad=False,
          resized=True
          )
