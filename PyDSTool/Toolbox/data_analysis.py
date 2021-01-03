"""
Data analysis utilities

Robert Clewley, August 2006
"""


from PyDSTool.Points import Point, Pointset
from PyDSTool.Interval import Interval
from PyDSTool.utils import intersect, filteredDict
from PyDSTool.errors import *
from PyDSTool.common import _num_types, _int_types, _seq_types, args, \
     sortedDictLists, sortedDictValues, sortedDictKeys
import numpy as np
from numpy import array, dot, zeros, transpose, shape, extract, mean, std, \
     vstack, hstack, eye, ones, zeros, linalg, concatenate, \
     newaxis, r_, flipud, convolve, matrix, asarray, size
from numpy.linalg import norm
from scipy.signal import lfilter, butter
from scipy.optimize import minpack, optimize

try:
    from mdp.nodes import PCANode
except:
    def PCANode(*args, **kwargs):
        raise ImportError("MDP must be installed for this functionality")
from PyDSTool.matplotlib_import import *
from copy import copy


_PCA_utils = ['doPCA', 'get_evec', 'pca_dim', 'get_residual',
              'plot_PCA_residuals', 'plot_PCA_spectrum']

_metric_utils = ['log_distances', 'log_distances_with_D']

_analysis_utils = ['find_knees', 'whiten', 'find_closest_val', 'out_of_seq',
                   'find_nearby_ball', 'find_nearby_annulus', 'data_bins',
                   'find_central_point', 'find_diameter', 'find_from_sorted',
                   'dist_between_datasets', 'find_recurrences']

_fit_utils = ['get_linear_regression_residual', 'fitline']

_filter_utils = ['lfilter_zi', 'filtfilt', 'lfilter', 'butter', 'rectify']

__all__ = _PCA_utils + _metric_utils + _analysis_utils + _fit_utils + \
        _filter_utils

# -------------------------------------------------------------------------

# Filter utils (reproduced from SciPy Cookbook)

def lfilter_zi(b,a):
    #compute the zi state from the filter parameters. see [Gust96].

    #Based on:
    # [Gust96] Fredrik Gustafsson, Determining the initial states in forward-backward
    # filtering, IEEE Transactions on Signal Processing, pp. 988--992, April 1996,
    # Volume 44, Issue 4

    n=max(len(a),len(b))

    zin = (eye(n-1) - hstack( (-a[1:n,newaxis],
                                 vstack((eye(n-2), zeros(n-2))))))

    zid=  b[1:n] - a[1:n]*b[0]

    zi_matrix=linalg.inv(zin)*(matrix(zid).transpose())
    zi_return=[]

    #convert the result into a regular array (not a matrix)
    for i in range(len(zi_matrix)):
        zi_return.append(float(zi_matrix[i][0]))

    return array(zi_return)




def filtfilt(b,a,x):
    #For now only accepting 1d arrays
    ntaps=max(len(a),len(b))
    edge=ntaps*3

    if x.ndim != 1:
        raise ValueError("Filiflit is only accepting 1 dimension arrays.")

    #x must be bigger than edge
    if x.size < edge:
        raise ValueError("Input vector needs to be bigger than 3 * max(len(a),len(b).")

    if len(a) < ntaps:
        a=r_[a,zeros(len(b)-len(a))]

    if len(b) < ntaps:
        b=r_[b,zeros(len(a)-len(b))]

    zi=lfilter_zi(b,a)

    #Grow the signal to have edges for stabilizing
    #the filter with inverted replicas of the signal
    s=r_[2*x[0]-x[edge:1:-1],x,2*x[-1]-x[-1:-edge:-1]]
    #in the case of one go we only need one of the extremes
    # both are needed for filtfilt

    (y,zf)=lfilter(b,a,s,-1,zi*s[0])

    (y,zf)=lfilter(b,a,flipud(y),-1,zi*y[-1])

    return flipud(y[edge-1:-edge+1])


# Principal Component Analysis utilities

def doPCA(in_pts, D, tol=.8):
    pts=copy(in_pts)
    pts.shape = (len(in_pts),D)
    p=PCANode(output_dim=tol)
    p.train(pts)
    p.stop_training()
    return p


def get_evec(p, dim):
    # extra call to array() to make it contiguous
    v = array(transpose(p.v)[dim-1])
    v.shape = (totdim, 3)
    return v


def pca_dim(pca, covering, data, refpt, tol=0.8):
    if type(refpt) != int:
        refpt = refpt[0]
    pca[refpt] = doPCA(nhd(data, covering, refpt), tol)
    print("PCA eigenvalues for nhd #%i:"%refpt)
    print(pca[refpt].d)
    return len(pca[refpt].d)


def get_residual(pcanode, pts):
    pvals = pcanode.execute(pts, pcanode.get_output_dim())
    pproj = mat(pvals)*mat(transpose(pcanode.v))
    res = pts-array(pproj) # orthogonal complement of projection onto PCs
    resval_part = norm(res)
    resval = norm(resval_part)
    return resval


def plot_PCA_residuals(data, D=None, newfig=True, marker='o',
                       silent=False):
    if D is None:
        D = shape(data)[1]
    p=doPCA(data, D, D)
    spec=zeros((D,1),float)
    for i in range(1,D):
        spec[i] = norm(p.d[:i])
    res = transpose(sqrt(spec[-1]**2-spec**2)/spec[-1])[0]
    if not silent:
        print("L2-norm of PCA spectrum =", spec[-1])
        if newfig:
            figure()
            style='k'+marker
        else:
            style=marker
        semilogy(res,style)
    ##    title('PCA residuals')
        xlabel(r'$\rm{Dimension}$',fontsize=20)
        ylabel(r'$\rm{PCA \ residual}$',fontsize=20)
    return p, res


def plot_PCA_spectrum(data, D):
    p=doPCA(data, D, D)
    spec=zeros((D,1),float)
    for i in range(1,D):
        spec[i] = norm(p.d[:i])
    figure();plot(spec,'ko')
    title('L2-norm of spectrum of singular values')
    xlabel('D')
    ylabel('|s|')
    return p

# -----------------------------------------------------------------------------


def log_distances(m, sampleix=0, doplot=True, quiet=True, logv=None,
              plotstyle=None, return_ixs=False):
    """Log distances (L2-norm) of data points from a reference point (sampleix).

    For multiple calls to this function on the same data, it's faster to
    pass logv precomputed.

    return_ixs=True also returns the point indices from the sort to
    be returned in this argument in a third return value.
    """
    npts = size(m,0)
    assert sampleix >= 0
    d = zeros((npts-1,), 'd')
    if np.ndim(m) == 3:
        for i in range(npts):
            if sampleix != i:
                try:
                    d[i] = norm(m[sampleix,:,:]-m[i,:,:])
                except IndexError:
                    # catch case when index is too large for npts-1
                    # so use the empty i=pix position (ordering doesn't matter)
                    d[sampleix] = norm(m[sampleix,:,:]-m[i,:,:])
    elif np.ndim(m) == 2:
        for i in range(npts):
            if sampleix != i:
                try:
                    d[i] = norm(m[sampleix,:]-m[i,:])
                except IndexError:
                    # catch case when index is too large for npts-1
                    # so use the empty i=pix position (ordering doesn't matter)
                    d[sampleix] = norm(m[sampleix,:]-m[i,:])
    else:
        raise ValueError("Rank of input data must be 2 or 3")
    if return_ixs:
        # return sorted indices in that list argument
        # (assumed to be an empty list)
        ixs = array(argsort(d))
        d = d[ixs]
    else:
        # just sort
        d.sort()
    if not quiet:
        print("Chose reference point %i"%sampleix)
        print("Min distance = %f, max distance = %f"%(d[0], d[-1]))
    logd = log(d).ravel()
    if logv is None:
        logv = log(list(range(1,len(d)+1)))
    nan_ixs = isfinite(logd)  # mask, not a list of indices
    logd = extract(nan_ixs, logd)
    logv = extract(nan_ixs, logv)
    if doplot:
        if plotstyle is None:
            plot(logd,logv)
        else:
            plot(logd,logv,plotstyle)
    if return_ixs:
        ixs = extract(nan_ixs, ixs)
        return (logv, logd, ixs)
    else:
        return (logv, logd)


# used for log_distances_with_D
dict_tol = array([1e-8])


def log_distances_with_D(m, sampleix, logv=None, return_ixs=False):
    """Log distances (L2-norm) of data points from a reference point (sampleix).
    Additionally, it returns a list of the inter-point distances,
    relative to the reference point, and the inverse of this mapping.

    For multiple calls to this function on the same data, it's faster to
    pass logv precomputed.

    return_ixs=True returns the point indices from the sort in a fifth
    return value. This identifies the points in the returned
    list of inter-point distances and its inverse.
    """
    npts = size(m,0)
    dk = zeros((npts-1,), 'd')
    d_inv = {}
    # do in two stages: ix 0->sampleix, sampleix+1->npts-1
    # but offset second stage indices in dk so there's a total of n-1
    # this is basically a slightly unrolled loop to make it a little faster
    if np.ndim(m) == 3:
        for i in range(sampleix):
            dk[i] = norm(m[sampleix,:,:]-m[i,:,:])
            if dk[i] in d_inv:
                # ensure we find a unique distance for dictionary key
                # (doesn't matter if we adjust it at very fine resolution
                # to become unique)
                done = 0
                u = dk[i]
                sgn = 1
                lim=40
                while done < lim:
                    u = dk[i] + sgn*done*dict_tol
                    if u[0] not in d_inv:
                        done = lim+1  # avoids exception below
                    else:
                        if sgn == -1:
                            sgn = 1
                        else:
                            done += 1
                            sgn = -1
                if done == lim:
                    raise ValueError("Non-unique distance found")
                dk[i] = u
                d_inv[u[0]] = i
            else:
                d_inv[dk[i]] = i
        for i in range(sampleix+1, npts):
            ki = i-1
            dk[ki] = norm(m[sampleix,:,:]-m[i,:,:])
            if dk[ki] in d_inv:
                # ensure we find a unique distance for dictionary key
                # (doesn't matter if we adjust it at very fine resolution
                # to become unique)
                done = 0
                u = dk[ki]
                sgn = 1
                lim=40
                while done < lim:
                    u = dk[ki] + sgn*done*dict_tol
                    if u[0] not in d_inv:
                        done = lim+1  # avoids exception below
                    else:
                        if sgn == -1:
                            sgn = 1
                        else:
                            done += 1
                            sgn = -1
                if done == lim:
                    raise ValueError("Non-unique distance found")
                dk[ki] = u
                d_inv[u[0]] = i
            else:
                d_inv[dk[ki]] = i
    elif np.ndim(m) == 2:
        for i in range(sampleix):
            dk[i] = norm(m[sampleix,:]-m[i,:])
            if dk[i] in d_inv:
                # ensure we find a unique distance for dictionary key
                # (doesn't matter if we adjust it at very fine resolution
                # to become unique)
                done = 0
                u = dk[i]
                sgn = 1
                lim=40
                while done < lim:
                    u = dk[i] + sgn*done*dict_tol
                    if u[0] not in d_inv:
                        done = lim+1  # avoids exception below
                    else:
                        if sgn == -1:
                            sgn = 1
                        else:
                            done += 1
                            sgn = -1
                if done == lim:
                    raise ValueError("Non-unique distance found")
                dk[i] = u
                d_inv[u[0]] = i
            else:
                d_inv[dk[i]] = i
        for i in range(sampleix+1, npts):
            ki = i-1
            dk[ki] = norm(m[sampleix,:]-m[i,:])
            if dk[ki] in d_inv:
                # ensure we find a unique distance for dictionary key
                # (doesn't matter if we adjust it at very fine resolution
                # to become unique)
                done = 0
                u = dk[ki]
                sgn = 1
                lim=40
                while done < lim:
                    u = dk[ki] + sgn*done*dict_tol
                    if u[0] not in d_inv:
                        done = lim+1  # avoids exception below
                    else:
                        if sgn == -1:
                            sgn = 1
                        else:
                            done += 1
                            sgn = -1
                if done == lim:
                    raise ValueError("Non-unique distance found")
                dk[ki] = u
                d_inv[u[0]] = i
            else:
                d_inv[dk[ki]] = i

    else:
        raise ValueError("Rank of input data must be 2 or 3")
    d = copy(dk)  # dk remains unsorted
    if return_ixs:
        # return sorted indices in that list argument
        # (assumed to be an empty list)
        ixs = array(argsort(d))
        d = d[ixs]
    else:
        # just sort
        d.sort()
    logd = log(d).ravel()
    if logv is None:
        logv = log(list(range(1,len(d)+1)))
    nan_ixs = isfinite(logd)  # mask, not a list of indices
    logd = extract(nan_ixs, logd)
    logv = extract(nan_ixs, logv)
    d = extract(nan_ixs, d)
##    for i in range(len(d)):
##        if nan_ixs[i] == 0:
##            try:
##                del d_inv[dk[i][0]]
##                print "Deleted key %i @ %f"%(i,dk[i][0])
##            except KeyError:
##                print "Failed to delete key %i @ %f"%(i,dk[i][0])
####                raise
    if return_ixs:
        ixs = extract(nan_ixs, ixs)
        return (logv, logd, d, d_inv, ixs)
    else:
        return (logv, logd, d, d_inv)

# ----------------------------------------------------------------------------

def rectify(xa, full=False):
    """Half-wave (default) or full-wave signal rectification: specify using
    optional full Boolean flag.
    xa must be an array"""
    if full:
        pos = np.asarray(xa>0,int)
        return pos*xa - (1-pos)*xa
    else:
        return np.asarray(xa>0,int)*xa



def get_linear_regression_residual(pfit, x, y, weight='', w=0):
    """Also see scipy.stat.linregress for a more sophisticated version
    """
    res = 0
    if weight == 'lo':
        for i in range(len(x)):
            res += ((y[i]-(pfit[0]*x[i]+pfit[1]))/(i+1)/w)**2
    elif weight == 'hi':
        l = len(x)
        for i in range(l):
            res += ((y[i]-(pfit[0]*x[i]+pfit[1]))/(l-i+1)/w)**2
    else:
        for i in range(len(x)):
            res += (y[i]-(pfit[0]*x[i]+pfit[1]))**2
    return sqrt(res)


def fitline(x, y, lo=0, hi=1, doplot=True, quiet=True, linewidth=2):
    """fitline takes the position of low and high fit limits and
    returns the slope. Integer indices in the x data set can be
    specified for the low and high limits, otherwise use a fraction
    between 0 and 1.

    In application to fractal dimension estimation, if
    x = log d (radius, a.k.a. inter-point distance) and
    y = log v (index, a.k.a. estimate of volume at a given radius)
    then the slope is the dimension estimate of the data set.

    Also see scipy.stat.linregress for a more sophisticated version
    """
    lendat = len(x)
    assert hi > lo
    if lo >= 0 and hi <= 1:
        loix = int(lendat*lo)
        hiix = int(lendat*hi)
    elif lo >=0 and hi > 0:
        loix = int(lo)
        hiix = int(hi)
    else:
        raise ValueError("Invalid low and high fit limits")
    if not quiet:
        print("lo ix %d, hi ix %d, max ix %d"%(loix, hiix, lendat-1))
    pfit = polyfit(x[loix:hiix],y[loix:hiix],1)
#    print pfit
    if doplot:
        plot([x[loix],x[hiix]],
             [x[loix]*pfit[0]+pfit[1],x[hiix]*pfit[0]+pfit[1]],
             linewidth=linewidth)
    return pfit[0]



# ----------------------------------------------------------------------------

# Data analysis utilities

class data_bins(object):
    """Class for data binning. Indexed by bin number 0 - (num_bins-1).

    Initialization arguments:
            coordinate -- name of coordinate that occupies bins
            bin_ords   -- ordered sequence of n bin ordinates (integers or floats)
                          defining n-1 bins
            valuedict  -- optional dictionary containing initial values for
                          bins: keys = ordinate values,
                                values = coordinate values
            tolerance  -- for resolving bin edges with finite precision
                          arithmetic (defaults to 1e-12)
    """

    def __init__(self, coordinate, bin_ords, valuedict=None, tolerance=1e-12):
        assert isincreasing(bin_ords), \
                "Bin ordinates must be given as a sequence in increasing order"
        # bins are referenced internally by integer index
        self.num_bins = len(bin_ords)-1
        assert self.num_bins > 0, "Must be more than 2 ordinates to define bins"
        self.bin_ords = array(bin_ords)
        self.bins = dict(zip(range(self.num_bins),[0]*self.num_bins))
        self.coordinate = coordinate
        self.tolerance = tolerance
        self.intervals = [Interval('bin_%s'%i, float,
                                (bin_ords[i],bin_ords[i+1]), \
                                abseps=tolerance) for i in range(self.num_bins)]
        self.midpoints = (self.bin_ords[1:]+self.bin_ords[:-1])/2.
        if valuedict is not None:
            for k, v in valuedict.items():
                self[self.resolve_bin_index(k)] = v

    def resolve_bin_index(self, ordinate):
        """Find bin number (index) associated with ordinate"""
        try:
            # return first bin index for which ordinate in bin interval is
            # 'uncertain' or 'contained'
            return [interval.contains(ordinate) is not notcontained \
                       for interval in self.intervals].index(True)
        except ValueError:
            # no bin found
            raise ValueError("No corresponding bin for this ordinate")

    def __call__(self, ordinate):
        """Return content of the bin that ordinate resolves to.
        """
        return self.bins[self.resolve_bin_index(ordinate)]

    def __getitem__(self, bin_index):
        if isinstance(bin_index, _int_types):
            # bin_index is a single index, so...
            # return first bin for which ordinate in bin interval is
            # 'uncertain' or 'contained'
            return self.bins[bin_index]
        elif isinstance(bin_index, slice):
            if bin_index.step is not None:
                if bin_index.step != 1:
                    raise ValueError("Cannot step in slice of bins")
            if bin_index.start is None:
                start = 0
            else:
                start = bin_index.start
            if bin_index.stop is None:
                stop = self.num_bins
            else:
                stop = bin_index.stop
            assert start < stop, \
                       "Slice must go in increasing order"
            new_bin_ords = self.bin_ords[start:stop+1]
            b = bin(self.coordinate, new_bin_ords,
                    tolerance=self.tolerance)
            for x in new_bin_ords[:-1]:
                bin_ix = self.resolve_bin_index(x+self.tolerance)
                b.bins[bin_ix] = self.bins[bin_ix]
            return b
        else:
            raise ValueError("Invalid indexing of bins")

    def __setitem__(self, bin_index, value):
        """Set value of a given bin number (index)."""
        self.bins[bin_index] = value

    def increment(self, ordinate, inc=1):
        """Increment the bin corresponding to the ordinate argument.
        Optional argument inc sets increment amount (default 1).
        """
        self.bins[self.resolve_bin_index(ordinate)] += inc

    def bin(self, values):
        """Increment bin contents according to values passed."""
        if isinstance(values, _seq_types):
            for v in values:
                self.increment(v)
        else:
            self.increment(values)

    def clear(self):
        """Reset all bins to have zero contents."""
        self.bins = dict(zip(range(self.num_bins),[0]*self.num_bins))

    def __str__(self):
        return "Binned data with intervals %s:\n By bin number: %s"%(str([i.get() \
                     for i in self.intervals]), str(self.bins))

    __repr__ = __str__

    def to_pointset(self):
        """Convert to a pointset"""
        bin_ixs = range(self.num_bins)
        return Pointset(indepvararray=list(range(self.num_bins)),
                        coorddict={self.coordinate: [self.bins[ix] \
                                                        for ix in bin_ixs],
                                   'bin_limit_lo': [self.intervals[ix][0] \
                                                        for ix in bin_ixs],
                                   'bin_limit_hi': [self.intervals[ix][1] \
                                                        for ix in bin_ixs]
                                    })

    def to_arrays(self):
        """Convert to two arrays of bin indices and values"""
        x, y = sortedDictLists(self.bins, byvalue=False)
        return (array(x), array(y))

    def values(self):
        return sortedDictValues(self.bins)

    def keys(self):
        return sortedDictKeys(self.bins)

    def mean(self):
        """Mean of binned data"""
        a = self.midpoints*array(sortedDictValues(self.bins))
        return sum(a) / sum(self.bins.values())

    def std(self):
        """Standard deviation of binned data"""
        vals = array(sortedDictValues(self.bins))
        a = self.midpoints*vals
        sum_bin_vals = sum(self.bins.values())
        mean_val = sum(a) / sum_bin_vals
        return sqrt(sum(((self.midpoints-mean_val)*vals)**2) / sum_bin_vals)


def find_closest_val(x, a, eps=1e-7, which_norm=2):
    """Find one value in a closer than eps to x"""
    for aval in a:
        if norm(x-aval, which_norm) < eps:
            return aval
    return None


def find_nearby_ball(data, refix, r, which_norm=2, include_ref=False):
    """Return indices of all points in data that are inside a ball of
    radius r from the reference point (not included)."""
    nearby = []
    refpt = data[refix]
    if include_ref:
        for i in range(size(data,0)):
            if norm(refpt-data[i], which_norm) <= r:
                nearby.append(i)
    else:
        for i in range(size(data,0)):
            if i != refix:
                if norm(refpt-data[i], which_norm) <= r:
                    nearby.append(i)
    return nearby

def find_nearby_annulus(data, refix, rlo, rhi, which_norm=2):
    """Return a list containing input data's indices of all neighbours of the
    reference point within the range of distances d such that dlo < d < dhi
    """
    assert rlo > 0 and rhi > rlo
    nearby = []
    for i in range(size(data,0)):
        if i != refix:
            d = norm(data[refix]-data[i], which_norm)
            if d > rlo and d < rhi:
                nearby.append(i)
    return nearby

def find_central_point(data, which_norm=2):
    """Find the index of a point closest to the 'centre' of a data set.
    The centre is defined to be where the mean of the points' position
    vectors relative to the centre is minimized in each direction."""
    dim = data.shape[1]
    centre = mean(data,0)
    ds = array([norm(p, which_norm) for p in data-centre])
    ds_sort_ixs = ds.argsort()
    return ds_sort_ixs[0]


def find_recurrences(data, refix, r, times=None, ignore_small=0,
                     which_norm=2):
    """Find recurrences of a trajectory in a ball of radius r centred at the
    reference point given by refix in data.

    If recurrence times are desired, pass an associated time array corresponding
    to the points in the data.

    If ignore_small > 0, recurrences of fewer than this number of data points
       will be treated as anomalous and ignored completely.

    Returns a structure (class common.args) having attributes:
        partitions -- pairs of entry and exit indices for each recurrence in
                the ball
        ball_ixs -- all consecutive indices into data for each recurrence
        partitions_lengths -- number of indices in each partition
        rec_times -- times to next recurrence in the set from previous
    """
    if times is not None:
        assert len(times) == len(data), "Times must have same length as data"
    # ball_ixs is already sorted
    ball_ixs = find_nearby_ball(data, refix, r, which_norm, include_ref=True)
    result = args(ball_ixs=[], partitions=[], partition_lengths=[],
                  rec_times=None)
    if len(ball_ixs) == 0:
        # No points found in ball
        return result
    # find boundaries of each partition between contiguous indices
    partition_bds = out_of_seq(ball_ixs)
    if len(partition_bds) == 0:
        result.ball_ixs = ball_ixs
        result.partitions = [(ball_ixs[0], ball_ixs[-1])]
        result.partition_lengths = [ball_ixs[-1] - ball_ixs[0] + 1]
        if times is not None:
            # only update this if times present to be consistent with other
            # changes of rec_times from default None
            result.rec_times = [Inf]
        return result
    else:
        plen = ball_ixs[partition_bds[0]-1] - ball_ixs[0] + 1
        if plen > ignore_small:
            partitions = [(ball_ixs[0], ball_ixs[partition_bds[0]-1])]
        else:
            partitions = []
    # find contiguous indices (and corresponding times)
    for i in range(len(partition_bds)-1):
        plo = ball_ixs[partition_bds[i]]
        phi = ball_ixs[partition_bds[i+1]-1]
        plen = phi - plo + 1
        if plen > ignore_small:
            partitions.append((plo, phi))
    plo = ball_ixs[partition_bds[-1]]
    phi = ball_ixs[-1]
    plen = phi - plo + 1
    if plen > ignore_small:
        partitions.append((plo, phi))
    result.ball_ixs = ball_ixs
    result.partitions = partitions
    result.partition_lengths = [p[1]-p[0]+1 for p in partitions]
    if times is None:
        return result
    else:
        result.rec_times = recurrence_times(times, partitions)
        return result


def find_diameter(data, eps, which_norm=2):
    """Find approximate diameter of data set, up to tolerance defined by eps > 0.

    Data assumed to be a rank-2 array.
    """
    assert eps > 0, "Tolerance must be positive"
    num_pts = len(data)
    if num_pts > 400:
        # If data well distributed, assume that 100 sample points is a good
        # initial set to test
        sample_rate = int(ceil(num_pts/100.))
    else:
        # small enough data set to do by brute force on all points
        sample_rate = 1
    old_diam = -Inf
    delta = Inf  # change in estimate of radius on each iteration
    while delta > eps:
        #print "Sample rate = ", sample_rate
        downsampled = data[::sample_rate]
        mind, diam = dist_between_datasets(downsampled, downsampled,
                                             which_norm)
        if sample_rate == 1:
            # can go no further!
            delta = 0
            break
        else:
            sample_rate = int(ceil(sample_rate/2.))
        delta = diam - old_diam
        old_diam = diam
        #print old_diam, delta
    return diam


def dist_between_datasets(data1, data2, which_norm=2):
    maxd = -Inf
    mind = Inf
    for v1 in data1:
        for v2 in data2:
            d = norm(v1-v2, which_norm)
            if d > maxd:
                maxd = d
            elif d < mind:
                mind = d
    return (mind, maxd)


def out_of_seq(a, inc_only=False):
    """Determine if and where an integer array a is not increasing or
    decreasing in sequence by 1.
    """
    v_old = a[0]
    i = 0
    out = []
    dirn = 0
    for v in a[1:]:
        i +=1
        if v == v_old+1:
            if dirn == -1:
                out.append(i)
            dirn = 1
        elif v == v_old-1:
            if dirn == 1:
                out.append(i)
            dirn = -1
        else:
            dirn = 0
            out.append(i)
        v_old = v
    return out


def whiten(data):
    """Subtract mean and divide by standard deviation in each column of data"""
    wdata=zeros(shape(data),Float)
    for d in range(shape(data)[1]):
        x = data[:,d]
        wdata[:,d] = (x-mean(x))/std(x)
    return wdata


def find_knees(data, tol=1., inlog=False, verbose=False):
    """
    Find one or more 'knees' in data, according to high second derivatives.

    inlog option finds knees in the logs of the data entries.
    tol=1. works well if inlog=False
    tol=0.3 works well if inlog=True
    """
    knee_ixs = []
    for i in range(1,len(data)-1):
        if inlog:
            frac = log(data[i+1])+log(data[i-1])-2*log(data[i])
        else:
            d2 = data[i+1]+data[i-1]-2*data[i]
            try:
                frac = d2/data[i]
            except ZeroDivisionError:
                frac = Inf
        if verbose:
            print(i, data[i], frac)
        if frac > tol and frac < Inf:
            knee_ixs.append((i,frac))
    if verbose:
        print("High second derivatives at: ", knee_ixs, "\n")
    knees = []
    found = False  # in a contiguous segment of high second derivatives
    curr_kixs = []
    old_kix = None
    for i in range(len(knee_ixs)):
        process = False
        kix, frac = knee_ixs[i]
        if verbose:
            print("Processing knee at index", kix)
        if kix-1 == old_kix:
            if i == len(knee_ixs)-1:
                # this is the last index so have to process
                process = True
            if found:
                curr_kixs.append(i)
            else:
                curr_kixs = [i-1, i]
                found = True
        else:
            process = old_kix is not None
        if verbose:
            print(old_kix, found, curr_kixs)
        if process:
            if found:
                found = False
                if verbose:
                    print("Current knee indices:", curr_kixs, end=' ')
                    print([knee_ixs[k] for k in curr_kixs])
                all_d2 = array([knee_ixs[k][1] for k in curr_kixs])
                ixs_sort = argsort(all_d2)
                max_ix = ixs_sort[-1]
                knees.append(knee_ixs[curr_kixs[max_ix]][0])
                curr_kixs = []
            else:
                if verbose:
                    print("Appending knee index", old_kix)
                knees.append(old_kix)
        old_kix = kix
    # add final singleton index of high derivative, if any
    if not found and old_kix not in knees:
        if verbose:
            print("Appending knee index", old_kix)
        knees.append(old_kix)
    return knees


# -----------------------------------------------------------------------------

# Internal helper functions

def recurrence_times(ts, partitions):
    """Internal function for use by find_recurrences"""
    rtimes = []
    if len(partitions) == 1:
        return [Inf]
    for i in range(len(partitions)-1):
        (plo, phi) = partitions[i]
        rtimes.append(ts[partitions[i+1][0]]-ts[partitions[i][1]])
    return rtimes


def find_from_sorted(x, v, next_largest=1, indices=None):
    if isinstance(v, _seq_types):
        # sequence v
        return [find(array(x), y, next_largest, indices) for y in v]
    else:
        # singleton v
        return find(array(x), v, next_largest, indices)

def colormap(mag, cmin, cmax):
    """
    Return a tuple of floats between 0 and 1 for the red, green and
    blue amplitudes.
    Function originally named floatRgb and written by Alexander Pletzer,
    from the Python Cookbook.
    """

    try:
        # normalize to [0,1]
        x = float(mag-cmin)/float(cmax-cmin)
    except:
        # cmax = cmin
        x = 0.5
    blue = min((max((4*(0.75-x), 0.)), 1.))
    red  = min((max((4*(x-0.25), 0.)), 1.))
    green= min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
    return (red, green, blue)


def dump_progress(x, tot, filename='progress_info.dat'):
    """Send progress report to a file"""
    f=open(filename, 'w')
    f.write("Progress: %d / %d \n"%(x,tot))
    f.write("Date and time of update: " + " ".join([str(x) for x in localtime()[:-3]]) + "\n\n")
    f.close()


def second_diff(data, i):
    """Naive second difference formula at position i of data"""
    return data[i+1]+data[i-1]-2*data[i]


def _adjust_ix(i, n):
    """Internal helper function"""
    if i >= n:
        return i+1
    else:
        return i
