import numpy as np
import sys, os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import hilbert
import math
from colors import color

def PolyArea(x,y):
    """
    shoelace formula
    :param x:
    :param y:
    :return:
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def color_cost(te_xy,tm_xy,b1_xy,b2_xy):
    res = np.stack([te_xy,tm_xy,b1_xy,b2_xy],axis=1)
    return PolyArea(*res)

def color_4points_cost(wl,te,tm,te2,tm2):
    te_xy = color(np.stack([wl,te],axis = 1)).xy
    tm_xy = color(np.stack([wl,tm],axis = 1)).xy
    te_xy_2 = color(np.stack([wl,te2],axis = 1)).xy
    tm_xy_2 = color(np.stack([wl,tm2],axis = 1)).xy
    cost = color_cost(te_xy,tm_xy,te_xy_2,tm_xy_2)
    return -cost

def phase_cost(wavelengths,datadir='.', srcdir=None, nstep=25000):
    if srcdir is None:
        srcdir = datadir
    output = []
    for i, wavelength in enumerate(wavelengths):
        te_src = os.path.join(srcdir, 'harm0_te_src-%i.bin' % i)
        tm_src = os.path.join(srcdir, 'harm0_tm_src-%i.bin' % i)
        te_ful = os.path.join(datadir, 'harm0_te_ful-%i.bin' % i)
        tm_ful = os.path.join(datadir, 'harm0_tm_ful-%i.bin' % i)
        #TE src
        data = np.fromfile(te_src, dtype=np.float64)
        poy_te_src = np.abs(data[1::4] * data[3::4] - data[::4] * data[2::4])
        poy_te_src = np.max(poy_te_src[int(nstep/2)::])
        ey = data[2::4]  # Ey
        outh=hilbert(ey)
        ph_te_src=np.unwrap(np.angle(outh))
        #TM src
        data = np.fromfile(tm_src, dtype=np.float64)
        poy_tm_src = np.abs(data[1::4] * data[3::4] - data[::4] * data[2::4])
        poy_tm_src = np.max(poy_tm_src[int(nstep/2)::])
        ey = data[1::4]  # Ex
        outh=hilbert(ey)
        ph_tm_src=np.unwrap(np.angle(outh))
        #TE ful
        data = np.fromfile(te_ful, dtype=np.float64)
        ey = data[2::4]  # Ey
        poy_te=np.abs(data[1::4]*data[3::4]-data[::4]*data[2::4])
        poy_te=np.max(poy_te[int(nstep/2)::])
        outh=hilbert(ey)
        ph_te=np.unwrap(np.angle(outh))
        #TM ful
        data = np.fromfile(tm_ful, dtype=np.float64)
        ey = data[1::4]  # Ex
        poy_tm=np.abs(data[1::4]*data[3::4]-data[::4]*data[2::4])
        poy_tm=np.max(poy_tm[int(nstep/2)::])
        outh=hilbert(ey)
        ph_tm=np.unwrap(np.angle(outh))
        dph_te = np.mod(ph_te[int(nstep / 2)] - ph_te_src[int(nstep / 2)], 2 * math.pi)  # delta phase in [0,2pi]
        te = poy_te / poy_te_src
        dph_tm = np.mod(ph_tm[int(nstep / 2)] - ph_tm_src[int(nstep / 2)], 2 * math.pi)  # delta phase in [0,2pi]
        tm = poy_tm / poy_tm_src
        output.append([wavelength,te,tm,dph_te,dph_tm])
    return np.array(output)

if __name__ == '__main__':
    te, tm = spl_cost('res')
    wl = te.wl
    ttm = tm.transm
    tte = te.transm
    color_tm = tm.get_color()
    color_te = te.get_color()
    color_cost(color_te,color_tm)
    te.color.plot_cie()
    tm.color.plot_cie()
    plt.ylim(0,1)
    plt.xlim(0.4,0.8)
    plt.plot(wl,tte)
    plt.plot(wl,ttm)
    #plt.plot(te.f,te.poy_sr)
    plt.show()
