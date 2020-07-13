from tensorflow.keras.models import load_model, Model
from sklearn.metrics import mean_squared_error as mse
from skimage.transform import resize
from skimage.transform import AffineTransform, warp, rotate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from models import MultipleModels
from collections.abc import Iterable

def mimo(f):
    def wrapped(self, X, *args, **kargs):
        if not isinstance(X,list):
            X = [X]
        output = f(self,X,*args,**kargs)
        if not isinstance(output,list):
            output = [output]
        return output
    return wrapped

def show_mulitple_outputs(f):
    def wrapped(self,*pargs,channel=None,ax=None,**kargs):
        if ax is None:
            axes = [None] * self.n_outputs
        elif not isinstance(ax,Iterable):
            ax = [ax]
        else:
            axes = ax
        if channel is not None:
            f(self,*pargs,channel=channel,ax=axes[0],**kargs)
        else:
            for i in range(self.n_outputs):
                cax = axes[i]
                f(self,*pargs,channel=i,ax=cax,**kargs)
    return wrapped

class NNSimulator():
    def __init__(self, saved_model, th=0.005, xmin=0.3, xmax=1.0, **kargs):
        if isinstance(saved_model,list):
            self.model = MultipleModels(model_files=saved_model, **kargs).model
        else:
            self.model = load_model(saved_model)
        self.n_inputs = self.model.inputs.__len__()
        self.n_outputs = self.model.outputs.__len__()
        self.input_shape = self.model.inputs[0].shape
        self.th_ = th
        self.xmin = xmin
        self.xmax = xmax

    @mimo
    def predict(self, X):
        output = self.model.predict(X)
        return output

    @mimo
    def predict_tm(self, X):
        #rotate each input channel separately
        rotated = []
        for i in range(len(X)):
            tmp = self.rotate(X[i].copy())
            rotated.append(tmp)
        return self.predict(rotated)

    def predict_te_tm(self, X):
        te_pred = self.predict(X)
        tm_pred = self.predict_tm(X)
        return te_pred, tm_pred

    def rotate(self, X):
        mask = np.ones_like(X)
        for i in range(len(X)):
            mask[i] = rotate(X[i],90,preserve_range=True)
        return mask
    @mimo
    def eval(self, X, y, store_summary=True):
        if store_summary:
            self.X_=X.copy()
            self.pred_ = self.predict(X)
            if not isinstance(y,list):
                y = [y]
            self.y_=y.copy()
            mse_ = []
            for i,(pred,true) in enumerate(zip(self.pred_,self.y_)):
                batch_size = len(pred)
                #mse over i output channel
                mse_oi = np.zeros([batch_size, len(pred)])
                for j in range(len(y)):
                    #mse over i output channel and j sample
                    mse_oi[i,j] = mse(self.y_[j], self.pred_[j])
                mse_.append(mse_oi)
            self.mse_ = np.squeeze(np.array(mse_))
        return self.model.evaluate(X,y)

    @show_mulitple_outputs
    def show_summary(self, channel=0, ax=None):
        """
        show summary for individual output channel
        :return:
        """
        print('Average MSE', self.mse_.mean())
        print('Predictions with MSE lower than a treshhold (>%.3f)' % self.th_, (self.mse_[self.mse_<self.th_]).__len__() / self.mse_.__len__()*100, "%")
        i = np.random.randint(len(self.X_[channel]),size=4)
        mse_ = self.mse_[channel][i]
        pred = self.pred_[channel][i]
        y = self.y_[channel][i]
        X = self.X_[channel][i]
        wl = np.linspace(self.xmin, self.xmax, pred.shape[1])
        left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
        fig, axes = plt.subplots(2, 2, figsize=[10, 10])
        for i in range(4):
            m,n = divmod(i,2)
            ax = axes[m][n]
            ax.set_title('MSE = %.5f' % mse_[i])
            ax.plot(wl, pred[i], 'b', label='pred')
            ax.plot(wl, y[i], 'r', label='true')
            ax.set_ylim(0,1)
            axins = inset_axes(ax, width="20%", height="20%", loc=2)
            axins.imshow(np.squeeze(X[i]))
            axins.set_xticks([])
            axins.set_yticks([])
            ax.legend(loc=1)
        plt.show()

    @show_mulitple_outputs
    def show_prediction(self, x, y=None,ax=None, channel=0, idx=0, **kargs):
        """
        prediction for individual intput channel
        :param x:
        :param y:
        :param plot:
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots(1,1,**kargs)
        te,tm = self.predict_te_tm(x)
        te = te[channel]
        tm = tm[channel]
        wl = np.linspace(self.xmin, self.xmax, te.shape[-1])
        ax.plot(wl, te[idx], 'r', label='TE pred')
        ax.plot(wl, tm[idx], 'b', label='TM pred')
        if y is not None:
            if not isinstance(y, list):
                y = [y]
            ax.plot(y[channel][:,0], y[channel][:,1], 'ro', label='TE true')
            ax.plot(y[channel][:,0], y[channel][:,2], 'bo', label='TM true')
        ax.set_xlim(wl.min(), wl.max())
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.set_xlabel('wl, um')
        ax.set_ylabel('transmission')
        if ax is None:
            fig.show()

    @show_mulitple_outputs
    @mimo
    def show_mask(self, x, channel=0, ax=None,idx=0,**kargs):
        if ax is None:
            fig, ax = plt.subplots(1,1,**kargs)
        ax.imshow(np.squeeze(x[channel][idx]))
        ax.set_xticks([])
        ax.set_yticks([])
        if ax is None:
            fig.show()
    
    def show_mask_and_prediction(self, x, y=None, channel=None, axes=None, idx=0):
        if channel is not None:
            rows = 1
        else:
            rows = self.n_outputs
        cols = 2
        if axes is None:
            fig, axes = plt.subplots(rows,cols,figsize=[5*cols,5*rows])
        axes = np.array(axes).reshape(-1,cols)
        self.show_prediction(x, y, ax=axes[:,0], channel=channel,idx=idx)
        self.show_mask(x,ax=axes[:,1], channel=channel, idx=idx)
        if axes is None:
            fig.show()


class FirstPrincipleSimulator():
    def __init__(self, exe=None):
        pass

    def predict(self):
        pass

    def evaluate(self, X, y):
        pass