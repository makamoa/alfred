import matplotlib.pylab as plt
import colour
from colour.plotting import *
import numpy as np
from scipy import fftpack
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import os
from numpy.fft import fft
from math import sqrt
from scipy.interpolate import interp1d
from scipy import linalg as la
import scipy.integrate
import pickle
import sys
colour.utilities.filter_warnings(True, False)

class color:
    def __init__(self, data, units = 1e-9):
        self.scale = units / 1e-9
        self.calculate_cie(data)


    def calculate_cie(self,data):
        sample_spd_data = data[:,[0,1]]
        sample_spd_data[:,0] *= self.scale
        #####
        const = scipy.integrate.trapz(sample_spd_data[:,1],sample_spd_data[:,0])
        sample_spd_data[:,1] = sample_spd_data[:,1]
        sample_spd_data = {l: f for l, f in sample_spd_data}
        spd=colour.SpectralPowerDistribution(sample_spd_data, name="transmission")
        spd_copy = spd.copy()
        spd_copy.interpolate(colour.SpectralShape(400.0,800.0,1))
        #spd_copy.normalise()
        cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 2012 2 Degree Standard Observer']
        illuminant = colour.ILLUMINANTS_RELATIVE_SPDS['D65']
        # Calculating the sample spectral power distribution *CIE XYZ* tristimulus values.
        XYZ = colour.spectral_to_XYZ(spd_copy, cmfs, illuminant)
        RGB = colour.XYZ_to_sRGB(XYZ / 100)
        XYZ = XYZ / 100
        xyY = colour.XYZ_to_xyY(XYZ)
        for i, color in enumerate(RGB):
            if color > 1.0:
                RGB[i] = 1.0
            elif color < 0:
                RGB[i] = 0.0
        self.RGB = RGB
        self.xy = colour.xyY_to_xy(xyY)
        return self.xy

    def plot_rgb(self):
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        ax.add_patch(Circle((0.5, 0.5), radius=0.5, facecolor=tuple(self.RGB)))
        ax.annotate('TE', (0.43, 0.45), color='gray', fontsize=32)
        ax.set_aspect(1)
        ax.axis('off')
        plt.show()

    def plot_cie(self):
        plot_chromaticity_diagram_CIE1931(standalone=False)
        plt.plot(*self.xy, 'o-', color='black')
        render(
            standalone=True,
            limits=(-0.1, 0.9, -0.1, 0.9),
            x_tighten=True,
            y_tighten=True);


if __name__ == '__main__':
        wl = np.linspace(300,800,10)
        values = np.abs(np.sin(300*wl))
        a = np.stack([wl,values],axis = 1)
        col = color(a)
        col.plot_rgb()
