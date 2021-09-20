from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats
import pickle
from collections import OrderedDict, defaultdict
import pymc3 as pm
import theano.tensor as tt
from itertools import product
import matplotlib
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap
from distinctipy import distinctipy

print("=== Setting up matplotlib ===")
from matplotlib import rc, pyplot as plt
import os
plt.style.use("ggplot")
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Bitstream Charter"
plt.rcParams["font.size"] = 12
plt.rcParams['axes.titley'] = 1.03
plt.rcParams['axes.titlepad'] = 0.0
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams["legend.fontsize"] = 10

# define colormap based on Blues_r that becomes gradually more transparent
r = np.linspace(1.0,0.0,matplotlib.cm.Blues_r.N).reshape((-1,1))
customcolors = matplotlib.cm.Blues_r(np.arange(matplotlib.cm.Blues_r.N))*r + (1-r)*np.array([0.9,0.9,0.9,1.0]).reshape((1,-1))
customcolors[:,-1] = r.ravel()

basis_colors = [
    "#b1457b",
    "#69ab54",
    "#6971d7",
    "#bd5538",
    "#c3a63e",
    "#5b3788",
    "#45c097",
    "#c972c4",
    "#3ed5ce",
    "#e16275",
    "#bb7c35"
]


bluetogray = LinearSegmentedColormap.from_list("BlueToGray", customcolors)
plt.register_cmap(cmap=bluetogray)

wavelength_start = 400
wavelength_stop = 1350
spectrogram_wavelengths = range(wavelength_start,wavelength_stop+1)

csv_format = {"sep": ";", "decimal": '.', "encoding": 'utf-8', "date_format": "%Y-%m-%d %H:%M:%S"}

def quantile_plot(ax, x, Y, num_quantiles = 20, **kwargs):
    """
        quantile_plot(ax, x, Y, num_quantiles = 20, **kwargs)
    
    Plots the quantiles of `Y` (computed along axis 1) over `x` into axis `ax` 
    with a resolution of `num_quantiles` quantiles. 
    Additional `**kwargs` are passed along to the underlying `tripcolor` plot.
    """
    quantile_levels = np.linspace(0,1,num_quantiles)
    quantiles = np.quantile(Y, quantile_levels, axis=1)
    colors = 2*abs(np.tile(quantile_levels.reshape(-1,1), (1,Y.shape[0]))-0.5)
    x_rep = np.tile(x, (num_quantiles,1))
    all_triangles = np.zeros(((num_quantiles-1)*(len(x)-1)*2,3), dtype=int)
    i = 0
    for row in range(num_quantiles-1):
        for col in range(len(x)-1):
            all_triangles[i,:] = [row*len(x)+col, (row+1)*len(x)+col, (row+1)*len(x)+col+1]
            i+=1
            all_triangles[i,:] = [row*len(x)+col, (row+1)*len(x)+col+1, row*len(x)+col+1]
            i+=1

    triangles = Triangulation(x_rep.ravel(), quantiles.ravel(), triangles=all_triangles)
    return ax.tripcolor(triangles, colors.ravel(), alpha=None, **kwargs)#, shading='gouraud'


def load_data(path=os.path.join(os.path.dirname(__file__),"..", "data", "data_preprocessed.pkl")):
    """
        load_data(path)

    Utility to load data.

    Arguments:
    ==========
        path:  file path
    """
    with open(path, "rb") as file:
        data = pickle.load(file)
    
    return data

# because the default forest plot is not flexible enough #sad
def forestplot(posterior, var_labels=None, var_args={}, fig=None, sp=GridSpec(1,1)[:,:], combine=False, credible_interval=0.95, label_position="ylabel", one_based=False):
    if fig == None:
        fig = plt.gcf()

    if var_labels == None:
        var_labels = posterior.data_vars.keys()
    
    var_args = defaultdict(lambda: {"color": "C1", "label": None, "interquartile_linewidth": 2, "credible_linewidth": 1}, **var_args)
    
    num_groups = len(var_labels)
    
    # create indices
    for i,var_label in enumerate(var_labels):
        name = var_label if isinstance(var_label, str) else var_label[0]
        
        cart = list(product(*(range(s) for s in posterior[name].shape[2:])))

        if isinstance(var_label, str):
            var_labels[i] = (var_label, list(map(np.squeeze,cart)), (cart))
        else:
            var_labels[i] = tuple(var_label) + (cart,)

    def plot_var_trace(ax, y, var_trace, credible_interval, credible_linewidth, interquartile_linewidth, **args):
        endpoint = (1 - credible_interval) / 2
        qs = np.quantile(var_trace, [endpoint, 1.0-endpoint, 0.25, 0.75])
        ax.plot(qs[:2],[y, y], linewidth=credible_linewidth, **args)
        ax.plot(qs[2:],[y, y], linewidth=interquartile_linewidth, **args)
        ax.plot([np.mean(var_trace)], [y], "o", **args)
    
    grid = GridSpecFromSubplotSpec(num_groups,1,sp, height_ratios=[np.prod(posterior[name].shape[2:])+2 for (name,idxs,carts) in var_labels])
    axes = []
    for j,(name,idxs,carts) in enumerate(var_labels):
        # if len(tp[name])==0:
        #     continue
            
        ax = fig.add_subplot(grid[j])
        args = var_args[name]

        yticks = []
        yticklabels = []
        # plot label
        # plot variable stats
        

        for i,(idx,cart) in enumerate(zip(idxs,carts)):
            yticks.append(-i)

            if np.array(idx).size==0:
                yticklabels.append("")
            elif np.array(idx).size==1:
                if one_based:
                    yticklabels.append(str(idx+1))
                else:
                    yticklabels.append(str(idx))
            else:
                if one_based:
                    yticklabels.append(str(tuple(cc+1 for cc in idx)))
                else:
                    yticklabels.append(str(idx))

            if combine:
                var_trace = posterior[name][(slice(None),slice(None))+cart]
                plot_var_trace(ax, -i, var_trace, credible_interval, **args)
            else:
                for chain in posterior.chain:
                    var_trace = posterior[name][(chain,slice(None))+cart]
                    plot_var_trace(ax, -i+0.25-chain/(len(posterior.chain)-1) * 0.5, var_trace, credible_interval, **args)


        ax.set_yticks(yticks)
        ax.set_ylim([yticks[-1]-1, 1])
        ax.set_yticklabels(yticklabels)
        # if j == 1:
        #     ax.set_yticklabels([])
        # else:
        #     ax.set_yticklabels(yticklabels)

        label = args["label"]
        if label == None:
            label = name
        if label_position == "ylabel":
            ax.set_ylabel(label)
        else:
            ax.set_title(label)
    
        # ax.set_frame_on(False)
        axes.append(ax)

        if name == "w":
            plt.vlines(0,-50,50, color="grey", alpha=0.5)
    return axes, grid
