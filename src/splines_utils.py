
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

# following code (augknt, spcol, spline, N) is copied from nwilming's ocupy/spline_base.py
# see https://github.com/nwilming
# the rest is copied from the Decoding neural spiking activity (interdisciplinary blockcourse WS14/15)
# and follows the paper:
# Gordon Pipa, Zhe Chen, Sergio Neuenschwander, Bruss Lima, Emery N. Brown: Mapping of Visual Receptive Fields by Tomographic Reconstruction,
# Neural Comput. 2012 October ; 24(10): 2543–2578. doi:10.1162/NECO_a_00334.

# Spline model with adaptive knot vector
# We want to distribute a fixed number of points (the knots) in such a way that the firing rate changes by the same absolute amount between any two successive knots.
# In other words, the knots should be close to each other in timespans with a high absolute derivative, less dense elswhere.
#
# Mathematically this means, that given the Gauss-filtered spiketrain $\tilde s(t)$ we want to compute the function
# $D(t) = \int_0^t |\frac{d}{d \tau} \tilde s(\tau)| dt$
# (or rather, for vectors in discrete time, we want to calculate the vector
# $D[t] = \sum_0^t |\frac{d}{d \tau} \tilde s[\tau]| dt$).
#
# In programming-terms this means that for a spike vector $s$ we want to calculate the so called
# cumulative sum of the vector of absolute values of $s$ after it has been convolved with a derivative-of-Gaussian-filter.
#
# The function $D(t)$ (or the vector $D[t]$) tells us how much *absolute change* in the neuron's firing rate has accumulated until time $t$.
# The inverse of that function, $t=D^{-1}(x)$ tells us the time $t$ before which an absolute rate-change of $x$ has accumulated.
# We want to place our knots such that between each two knots the rate changes by the same absolut amount,
# i.e. we place our knots at positions $k_i = D^{-1}(\frac{i}{n}) \text{ for } i\in\{0,\dots,n\}$.
# Intuitively, that means that each of the knots has to account for the same amount of variability in the neuron's firing rate over time.

def augknt(knots,order):
    """Augment knot sequence such that some boundary conditions
    are met."""
    a = []
    [a.append(knots[0]) for t in range(0,order)]
    [a.append(k) for k in knots]
    [a.append(knots[-1]) for t in range(0,order)]
    return np.array(a)


def spcol(x,knots,spline_order):
    """Computes the spline colocation matrix for knots in x.

    The spline collocation matrix contains all m-p-1 bases
    defined by knots. Specifically it contains the ith basis
    in the ith column.

    Input:
        x: vector to evaluate the bases on
        knots: vector of knots
        spline_order: order of the spline
    Output:
        colmat: m x m-p matrix
            The colocation matrix has size m x m-p where m
            denotes the number of points the basis is evaluated
            on and p is the spline order. The colums contain
            the ith basis of knots evaluated on x.
    """
    columns = len(knots) - spline_order - 1
    colmat = np.nan*np.ones((len(x), columns))
    for i in range(columns):
        colmat[:,i] = spline(x, knots, spline_order, i)
    return colmat

def spline(x,knots,p,i=0.0):
    """Evaluates the ith spline basis given by knots on points in x"""
    assert(p+1<len(knots))
    return np.array([N(float(u),float(i),float(p),knots) for u in x])

def N(u,i,p,knots):
    """
    u: point for which a spline should be evaluated
    i: spline knot
    p: spline order
    knots: all knots

    Evaluates the spline basis of order p defined by knots
    at knot i and point u.
    """
    if p == 0:
        if knots[int(i)] < u and u <=knots[int(i+1)]:
            return 1.0
        else:
            return 0.0
    else:
        try:
            k = ((float((u-knots[int(i)])) / float((knots[int(i+p)] - knots[int(i)]) ))
                    * N(u,i,p-1,knots))
        except ZeroDivisionError:
            k = 0.0
        try:
            q = ((float((knots[int(i+p+1)] - u)) / float((knots[int(i+p+1)] - knots[int(i+1)])))
                    * N(u,i+1,p-1,knots))
        except ZeroDivisionError:
            q  = 0.0
        return float(k + q)

def calc_cum_abs_deriv(y, sigma=10, order=1, axis=-1):
    """ Calculates the curvature of the spike count vector `y`"""
    return np.cumsum(np.abs(gaussian_filter1d(y, sigma=sigma, order=order, axis=axis)), axis=axis)

def find_percentiles(y, T, num_percentiles, return_thresholds=False):
    """ Finds `num_percentiles` equally spaced percentiles of `y` (a monotonically increasing vector),
    i.e. it approximates the inverse `x` of the function `y=f(x)`
    at `num_percentiles` equally spaced `y`-values between 0 and `y[-1]`.

    Arguments:
        y:                  a sequence of monotonically increasing function values
        T:                  length of a trial
        num_percentiles:    number of percentiles to find between 0 and `y[-1]`.
        return_threshold:   boolean value to indicated whether or not to return the thresholds, too
    Returns:
        percentiles:       `y`-values of the percentiles
        thresholds:        `x`-values of the percentiles]
    """
    thresholds = np.linspace(0,y[-1],num_percentiles+1)
    percentiles = np.zeros_like(thresholds)

    current = 1
    for step in range(T):
        if y[step] > thresholds[current]:
            percentiles[current] = step
            current +=1

    percentiles[-1] = len(y)
    if return_thresholds:
        return percentiles, thresholds
    else:
        return percentiles


def plot_filtered(y, sigma=10, abs=True):
    """
    Plot the result of convolving three filtered signals (a Gaussian curve, its first  and second derivative) with a single spike.
    """
    # Calculate filtered signals for order 0,1,2 and scale them up for plotting
    filtered_O0 = gaussian_filter1d(y, sigma=sigma, order=0)
    filtered_O1 = gaussian_filter1d(y, sigma=sigma, order=1)*sigma
    filtered_O2 = gaussian_filter1d(y, sigma=sigma, order=2)*sigma**2

    plt.figure(figsize=(25,15))

    #plt.vlines(np.argwhere(y),0,0.04,'k',alpha=0.5)
    plt.plot(y, 'k', alpha=0.5, label='Input signal')
    #plt.plot(np.zeros(y.shape),'k',alpha=0.5,label='input signal')
    plt.plot(filtered_O0, "b", alpha=0.5, label='Gaussian filtered')
    plt.plot(filtered_O1, "g:" if abs else "g", label='1st deriv. Gaussian')
    plt.plot(filtered_O2, ":r" if abs else "r", label='2nd deriv. Gaussian')

    # If intended, also plot the absolute value of the filtered derivatives
    if abs:
        plt.plot(np.abs(filtered_O1), "g", label='abs. 1st deriv. Gaussian')
    if abs:
        plt.plot(np.abs(filtered_O2), "r", label='abs. 2nd deriv. Gaussian')
    plt.legend(fontsize=14)

    plt.title("Results of convolution of an input signal with Gaussian filters", fontsize=20)
    plt.ylim([min(filtered_O0.min(),filtered_O1.min(),filtered_O2.min())*1.5,max(filtered_O0.max(),filtered_O1.max(),filtered_O2.max())*1.5])
