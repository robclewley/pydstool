"""Fractal dimension estimates for analysis of datasets having rank 2 or 3.

A journal publication concerning the use of these functions appears in
IEEE Transactions on BioMedical Engineering, 2008.

(c) 2005, 2006. Robert Clewley, John Guckenheimer.
"""

from __future__ import division, absolute_import, print_function
from PyDSTool import *
from PyDSTool.common import _seq_types
from PyDSTool.Toolbox.data_analysis import *
from numpy import extract, mean, std, ndarray
from scipy.linalg import norm
from PyDSTool.matplotlib_import import *
try:
    from matplotlib import cm
    DEFAULT_CMAP = cm.hot
except ImportError:
    DEFAULT_CMAP = None
from time import localtime
from copy import copy

__all__ = ['corr_dim', 'corr_dim_with_progress',
           'find_cover', 'find_outliers', 'get_cover_radii',
           'do_stats', 'nhd', 'timeseq', 'get_rV_curves',
           'get_filtered_ixs', 'plot_radius_distribution', 'filter_by_radius',
           'plot_dim_by_thickness', 'plot_thickness_by_dim', 'find_nhd',
           'prep_secant_figure', 'PD_E', 'slope_range', 'scatterplot_slopes',
           'scatter_histo', 'rescatter', 'sorted_by_slope', 'sorted_by_radius']

# -----------------------------------------------------------------------------


def corr_dim(data, which_norm=2):
    """Correlation dimension of a data set.
    """
    npts = size(data,0)
    d = zeros((npts*(npts-1)/2,), 'd')  # size of upper triangle (no diagonal) of m vs. m matrix
    # max ix for each i in expression for d is (i-1)*(i-2)/2 + (i-2)
    for i in range(1,npts):
        for j in range(i-1):
            d[(i-1)*(i-2)/2+j] = norm(data[i]-data[j], which_norm)
    d.sort()
    logd = log(d).ravel()
    logv = log(range(1,len(d)+1))
    nan_ixs = isfinite(logd)
    logd = extract(nan_ixs, logd)
    logv = extract(nan_ixs, logv)
    return (logv, logd)


def corr_dim_with_progress(data, inforate=10000, which_norm=2):
    """Version of correlation dimension with text file progress dump,
    at a sample rate of 'inforate', because the calculation typically
    takes a long time.
    """
    npts = size(data,0)
    dsize = npts*(npts-1)/2
    d = zeros((dsize,), 'd')  # size of upper triangle (no diagonal) of m vs. m matrix
    # max ix for each i in expression for d is (i-1)*(i-2)/2 + (i-2)
    for i in range(1,npts):
        for j in range(i-1):
            ix = (i-1)*(i-2)/2+j
            if mod(ix,inforate) == 0:
                dump_progress(ix, dsize)
            d[ix] = norm(data[i]-data[j])
    d.sort()
    logd = log(d).ravel()
    logv = log(range(1,len(d)+1))
    nan_ixs = isfinite(logd)
    logd = extract(nan_ixs, logd)
    logv = extract(nan_ixs, logv)
    return (logv, logd)


def timeseq(time, covering, refpt):
    """Extract time sequence from a covering"""
    ts = time.take(covering[refpt][3], 0)
    print("time sequence min = %f, max = %f"%(min(ts),max(ts)))
    return ts



#### Version 2 of regression residual method
#### This one starts at sqrt(N)
def find_cover(data, remain_tol=5,
               step=None, initref=None, wlo=1, whi=1,
               check_large_nhds=True, large_nhd=300,
               min_nhd_size=50, min_ix=25,
               max_res=1.0, max_std=0.2, quiet=True):
    """
    OUTPUTS:

     covering is a dict:
        reference point -> (dim, low radius, high radius, covered_indices)
     covered is a dict:
        point -> reference points whose neighbourhoods include it
    """
    covering = {}
    covered = {}.fromkeys(range(len(data)),None)
    not_covered = list(covered.keys())
    if remain_tol > len(data):
        remain_tol = 0
    makerefpt_ixs = []
    if initref is not None:
        refptix = initref
    else:
        refptix = 0   # initial value
    N = size(data,0)
    start_ix = int(round(sqrt(N)))
    print("N = ", N, "start ix = ", start_ix)
    logv_raw = log(range(1,N+1))
    done = False
    unsuccessful = []
    if step is None:
        ixstep = max([int(around(len(data)/2000.)),1])
    elif step > 0:
        if step < 1:
            # treat as %age of # data points
            ixstep = int(math.ceil(len(d)*step))
            if not quiet:
                print("Using index step %i out of %i points"%(ixstep, len(d)))
        else:
            ixstep = step
    else:
        raise ValueError("Invalid step argument")
    while not done:
        if not quiet:
            print("\n*************\nCalculating distances and point-wise dimension around pt %i ..."%refptix)
        logv, logd, d, d_inv_dict = log_distances_with_D(data, refptix, logv_raw)
        d_inv_keys = sortedDictKeys(d_inv_dict)
        d_inv_vals = sortedDictValues(d_inv_dict)
        d_inv = Pointset(indepvararray=d_inv_keys, indepvartype=Float,
                         coordarray=d_inv_vals, coordtype=Int, tolerance=5e-5)
        if not quiet:
            print("Finding self-consistent neighbourhood", end=' ')
        # search from min_ix -> end, increasing hiix from loix+1 until dimensionality
        # (slope) stabilizes to within an absolute tolerance and does not grow larger
        # than a different tolerance
        err=-1
        len_logd = len(logd)
        old_dim = None
        hiix = start_ix
        if not quiet:
            print("Start ix = ", start_ix)
        loix=start_ix-2
        nhd_too_small = False
        ref_dim = -1
        all_res = []
        in_zone = False
        # Grow West!
        while not in_zone and loix > min_ix+ixstep:
            loix -= ixstep
            pfit=polyfit(logd[loix:hiix],logv[loix:hiix],1)
            dim = pfit[0]
            residual = get_linear_regression_residual(pfit, logd[loix:hiix],logv[loix:hiix],
                                                      weight='')
            all_res.append(residual)
            if not quiet and mod(hiix,50) == 0:
                print(".", end=' ')
            if old_dim is not None:
                sd = std(all_res)
##                if residual > max_res:
##                    in_zone = True
                if sd > max_std:
                    in_zone = True
            old_dim = dim
        # Grow East!
        all_res = [] #[all_res[-1]]
        not_done = True
        while not_done and hiix < len_logd-1-ixstep:
            hiix += ixstep
            # do a straight-line best fit
            pfit=polyfit(logd[loix:hiix],logv[loix:hiix],1)
            dim = pfit[0]
            residual = get_linear_regression_residual(pfit, logd[loix:hiix],logv[loix:hiix],
                                                      weight='lo', w=wlo)
            all_res.append(residual)
            if not quiet:
                print("(%i,%i)"%(loix,hiix), end=' ')
                print("dim=%.4f, residual = %.4f"%(dim,residual))
            if not quiet and mod(hiix,50) == 0:
                print(".", end=' ')
            try:
                sd = std(all_res)
            except:
                sd = 0
##            if residual > max_res:
##                if not quiet:
##                    print "residual > max_res"
##                not_done = False
            if sd > max_std:
                if not quiet:
                    print("residuals s.d. > max_std")
                not_done = False
            old_dim = dim
        nhd_too_small = hiix-loix < min_nhd_size
        if not quiet:
            print("nhd_too_small = ", nhd_too_small)
        if nhd_too_small:
            if not quiet:
                print("Neighbourhood too small. Moving to a different reference point...")
            unsuccessful.append(refptix)
        else:
            if not quiet:
                print("\nDimension = %f"%dim)
                print("Found best fit line from relative ix %i to %i (radius %f)"%(loix, hiix, d[hiix]))
            # consolidate results in terms of global index positions in data
            covered_ixs = [d_inv(d[ix])[0] for ix in range(loix, hiix+1)]
            covered_ixs.append(refptix)
            # if nhd very large, recheck one in every 100 points because ptwise
            #  dim estimate may occasionally vary between points in that nhd.
            if hiix-loix > large_nhd and check_large_nhds:
                num_recheck = int(around((hiix-loix)/large_nhd))
                for i in range(num_recheck):
                    ix = random.randrange(0, hiix-loix)
                    if ix not in makerefpt_ixs and ix != refptix:
                        makerefpt_ixs.append(ix)
            covering[refptix] = (dim, d[loix], d[hiix], covered_ixs)
            for ix in covered_ixs:
                if covered[ix] is None:
                    covered[ix] = [refptix]
                else:
                    covered[ix].append(refptix)
            not_covered = remain(not_covered, covered_ixs)
            # only assert this as a consistency check the first time through the loop
##            assert len(not_covered) + len(covered_ixs) + 1 == lendat
        # find new ref pt in not_covered and repeat
        num_uncovered = len(not_covered)
##        if not quiet:
        print("%i points left to cover"%(num_uncovered+len(makerefpt_ixs)))
        if num_uncovered < remain_tol:
            if len(makerefpt_ixs) == 0:
                done = True
            else:
                # check these ref pts that were part of large nhds
                refptix = makerefpt_ixs.pop()
                check_large_nhds = False   # switch this off now!
        else:
            pick_unsuccessful = 0
            while pick_unsuccessful < num_uncovered and pick_unsuccessful >= 0:
                refptix = not_covered[random.randrange(0,num_uncovered)]
                if refptix in unsuccessful:
                    pick_unsuccessful += 1
                else:
                    pick_unsuccessful = -1
            if pick_unsuccessful == num_uncovered:
                done = True
    return (covered, covering)



def find_nhd(data, tol_up=.2, tol_down=0.05, quiet=True, max_res=0.2, max_std=0.1,
             step=None, initref=None, min_ix=10, min_nhd_size=20,
             max_radius_ratio=1.5):
    """Find a single neighbourhood around a reference point."""
    covering = {}
    covered = {}.fromkeys(range(len(data)),None)
    if initref is not None:
        refptix = initref
    else:
        refptix = 0   # initial value
    logv_raw = log(range(1,size(data,0)+1))
    if not quiet:
        print("\n*************\nCalculating distances and point-wise dimension around pt %i ..."%refptix)
        doplot = 1
    else:
        doplot = 0
    logv, logd, d, d_inv_dict = log_distances_with_D(data, refptix, logv_raw)
    d_inv_keys = sortedDictKeys(d_inv_dict)
    d_inv_vals = sortedDictValues(d_inv_dict)
    d_inv = Pointset(indepvararray=d_inv_keys, indepvartype=Float,
                     coordarray=d_inv_vals, coordtype=Int, tolerance=5e-5)
    if not quiet:
        print("Finding self-consistent neighbourhood", end=' ')
    # search from min_ix -> end, increasing hiix from loix+1 until dimensionality
    # (slope) stabilizes to within an absolute tolerance and does not grow larger
    # than a different tolerance
    err=-1
    len_logd = len(logd)
    old_dim = None
    loix=min_ix
    hiix=loix+1
    nhd_too_small = (len_logd < min_ix+10)
    if step is None:
        ixstep = max([int(around(len(data)/2000.)),1])
    elif step > 0:
        if step < 1:
            # treat as %age of # data points
            ixstep = int(math.ceil(len(d)*step))
            if not quiet:
                print("Using index step %i out of %i points"%(ixstep, len(d)))
        else:
            ixstep = step
    else:
        raise ValueError("Invalid step argument")
    if not quiet:
        plot(logd,logv)
    not_done = not nhd_too_small
    in_zone = False
    ref_dim = -1
    log_max_radius_ratio = log(max_radius_ratio)
    all_res = []
    while not_done and hiix < len_logd-1-ixstep:
        hiix += ixstep
        # do a straight-line best fit
        pfit = polyfit(logd[loix:hiix],logv[loix:hiix],1)
        dim = pfit[0]
        residual = get_linear_regression_residual(pfit, logd[loix:hiix],logv[loix:hiix])
##        if hiix - loix > min_nhd_size:
        all_res.append(residual)
        if not quiet:
            print("Dim = %.3f in [%i,%i], Residual = %.4f "%(dim,loix,hiix,residual), end=' ')
            plot([logd[loix],logd[hiix]],[logd[loix]*pfit[0]+pfit[1],logd[hiix]*pfit[0]+pfit[1]])
##            if mod(hiix,50) == 0:
##                print ".",
        if old_dim is not None:
##            err = abs(dim-old_dim)/dim
            sd = std(all_res)
            if not quiet:
                print("S.d. = %.4f "%sd)
            if residual > max_res:
                print("residual > max_res")
                not_done = False
            if sd > max_std:
                print("residuals s.d. > max_std")
                not_done = False
##            if err > max_err and hiix-loix > min_nhd_size/4:
##                if not quiet:
##                    print "\nMaxed out successive difference in dimension estimate"
##                    print " ... current dim = %.3f, old dim = %.3f, error = %.3f"%(dim,old_dim,err)
##                not_done = False
##                continue
##            if in_zone:
##                zone_err = abs(dim-ref_dim)/ref_dim
##                if in_zone and zone_err > tol_up:
##                    not_done = False  # end neighbourhood growth
##                if logd[hiix]-logd[loix] > log_max_radius_ratio:
##                    not_done = False  # end neighbourhood growth
##                    if not quiet:
##                        print "Radius too large: %.4f/%.4f"%(exp(logd[hiix]),exp(logd[loix]))
##            elif err < tol_down:
##                in_zone = True
##                ref_dim = dim
##                if not quiet:
##                    print " - entered zone at ix", hiix, " - with ref dim ", ref_dim
        old_residual = residual
        old_dim = dim
    nhd_too_small = nhd_too_small or hiix-loix < min_nhd_size
    if nhd_too_small:
        print("Neighbourhood too small. Try a different starting index or a new reference point ...")
        print("Dim found over ixs [%i, %i] = %.4f"%(loix,hiix,dim))
        raise RuntimeError
    else:
        if not quiet:
            print("\nDimension = %f"%dim)
            print("Found best fit line from relative ix %i to %i (radius %f)"%(loix, hiix, d[hiix]))
        # consolidate results in terms of global index positions in data
        covered_ixs = [d_inv(d[ix])[0] for ix in range(loix, hiix+1)]
        covered_ixs.append(refptix)
        covering[refptix] = (dim, d[loix], d[hiix], covered_ixs)
        for ix in covered_ixs:
            if covered[ix] is None:
                covered[ix] = [refptix]
            else:
                covered[ix].append(refptix)
    return (covered, covering)



def do_stats(covering, covered, maxD, bin_width=1,
             fignum=1, num_panels=4, s_frac=0.5,
             nhd_size_thresh=None, nhd_max_plot_size=None, minD=None,
             D_estimate_method="absolute", weight_by_size=False,
             force_xaxis_limits=False, radius_yaxis_limits=None,
             save=''):
    """Produces statistics of pointwise dimension estimates of a data set,
    including a histogram of the distribution of neighbourhood dimensions.

    R. Clewley, June 2006.

    INPUTS:

    maxD sets the maximum dimension to cater for.

    bin_width selects the width of bins for dimensions (default 1).

    fignum forces a particular figure number to be used for display.
        (Use fignum=0 to suppress plotting of the results and just return
         the statistics.)

    num_panels argument must be 2 (for just nhd size and bin histo) or
       4 (also include min and max nhd radius).

    s_frac selects the fraction (or multiple) of the standard deviation of
       neighbourhood sizes to use as the cutoff for the "bulk" of the data
       set, in order to avoid including large outliers in determining
       nhd_size_threshold (default 0.5).

    nhd_size_threshold selects the smallest neighbourhood size to be
       included in histogram of neighbourhood dimensionalities
       (default None => automatic selection, i.e., the mean of the first
        s_frac of the neighbourhood sizes' standard deviation).
       Set to zero to prevent it being used or plotted.

    nhd_max_plot_size selects the y-axis limit on the plot of neighbourhood
       sizes (default None => automatic selection).

    D_estimate_method can be 'absolute' (uses cutoff of size = 1 to estimate D
       in histogram tail), or 'relative' (uses a cutoff = %age of peak binned
       neighbourhoods)

    force_xaxis_limits (Boolean) determines whether x-axis limits should be
       forced to be the value of the (minD,maxD) parameters.

    minD sets the minimum dimension to cater for (default to auto-choice).

    radius_yaxis_limits (pair) sets the r_min and r_max y-axis upper limits,
       respectively.

    save (string) sets a filename to use for saving the figure in available
       formats (so don't include a file extension).


    OUTPUTS:

    d_stats: dictionary {"D_tail": estimated D from tail,
                         "D_mean": histo mean,
                         "D_std": histo std dev}

    dbins: Pointset of D bins

    cover_by_dimix: dictionary of covers by index of dimensions binned
       (i.e., [0, bin_width, 2*bin_width, ..., maxD+1])

    cover_by_size: list of neighbourhood coverings ordered by their size

    cbu: array indicating how many points in overlap between neighbourhoods
        (array index corresponds to a number of points shared,
         value = number of nhds with that number of points shared)

    ixs: list of indices of neighbourhoods of size > nhd_size_threshold

    (integral: internal diagnostic feature)
    """
    dx = zeros((len(covering),1),'f')
    ly = zeros((len(covering),1),'f')
    rloy = zeros((len(covering),1),'f')
    rhiy = zeros((len(covering),1),'f')
    drange = arange(0, maxD+1, bin_width, 'd').tolist()
    dbins = data_bins('D', drange)
    cover_by_dimix = {}.fromkeys(range(len(drange)))
    for dix in range(len(drange)):
        cover_by_dimix[dix] = []
    cover_by_size_dict = {}.fromkeys(range(len(covered)+1))
    ix = 0
    integral = 0
    largest = 0
    try:
        for p, (d, rlo, rhi, l) in covering.items():
            dx[ix] = d
            rloy[ix] = rlo
            rhiy[ix] = rhi
            lenl = len(l)
            if lenl > largest:
                largest = lenl
            try:
                cover_by_size_dict[lenl].append(p)
            except AttributeError:
                cover_by_size_dict[lenl] = [p]
            ly[ix] = lenl
            integral += lenl
            ix += 1
    except ValueError:
        # No max radius information available!
        # compatible with old version of covering that does not return rhi
        for p, (d, rlo, l) in covering.items():
            dx[ix] = d
            rloy[ix] = rlo
            rhiy[ix] = 0
            lenl = len(l)
            if lenl > largest:
                largest = lenl
            try:
                cover_by_size_dict[lenl].append(p)
            except AttributeError:
                cover_by_size_dict[lenl] = [p]
            ly[ix] = lenl
            integral += lenl
            ix += 1
    print("\nNeighbourhood statistics:")
    print("There are %i neighbourhoods to this covering"%len(covering))
    print("Largest neighbourhood has %i points"%largest)
    cover_by_size = [di for di in cover_by_size_dict.items() if di[1] is not None]
    cover_by_size.sort()
    csizes = [c[0] for c in cover_by_size]
    if nhd_size_thresh is None:
        s = std(csizes)
        sm = s_frac*s
        m = mean(csizes)
        print("Std. dev. of cover_by_size =", s)
        print("Max size found =", max(csizes))
        print("Mean size found =", m)
##        # find largest index of set of covering nhds such that its size
##        # is smaller than s_frac% of the std deviation of the sizes
##        for i in range(len(cover_by_size)):
##            if csizes[i] > sm:
##                break
##        try:
##            nhd_size_thresh = mean(csizes[:i])
##        except ZeroDivisionError:
##            nhd_size_thresh = 0
##        print "N'hood size threshold used = mean(cover_by_sizes restricted to %.3f of 1st std. dev.)"%s_frac
##        print "                           =", nhd_size_thresh
        nhd_size_thresh = m-sm
        print("N'hood size threshold used = mean - %.3f of std. dev."%s_frac)
        print("                           =", nhd_size_thresh)
    else:
        if nhd_size_thresh > 0:
            print("N'hood size threshold set by user =", nhd_size_thresh)
    if nhd_max_plot_size is None:
##        # 2 * original set sizes mean
##        try:
##            nhd_max_plot_size = 2*mean(csizes)
##        except ZeroDivisionError:
##            nhd_max_plot_size = largest
##        print "Using max plot nhd size = 2*mean(cover_by_sizes) =", nhd_max_plot_size
        nhd_max_plot_size = largest
        print("Using max plot nhd size of largest neighbourhood found =", nhd_max_plot_size)
    else:
        print("Using max plot nhd size set by user =", nhd_max_plot_size)
    # reverse so that largest first for returning to user
    cover_by_size.reverse()
    filtered_ixs = []
    try:
        for p, (d, rlo, rhi, l) in covering.items():
            if d <= maxD:
                dix = dbins.resolve_bin_index(d)
                if len(l) > nhd_size_thresh:
                    if weight_by_size:
                        inc = len(l)/largest
                    else:
                        inc = 1
                    dbins.increment(d, inc)
                    filtered_ixs.append(p)
                cover_by_dimix[dix].append(p)
    except ValueError:
        # compatible with old version of covering that does not return rhi
        for p, (d, rlo, l) in covering.items():
            if d <= maxD:
                dix = dbins.resolve_bin_index(d)
                if len(l) > nhd_size_thresh:
                    if weight_by_size:
                        inc = len(l)/largest
                    else:
                        inc = 1
                    dbins.increment(d, inc)
                    filtered_ixs.append(p)
                cover_by_dimix[dix].append(p)
    # decide on single dimension estimate
    ks=dbins.midpoints
    d_est = 0   # initial value
    if D_estimate_method == "absolute":
        # based on largest D > 1 in
        # tail of dbins histogram (assumes unimodal)
        if len(covering) < 3:
            dbin_thresh = 0
        else:
            dbin_thresh = 1
        for k in ks:
            n = dbins(k)
            if d_est > 0 and n == 0:
                # stop before a second spike might occur
                break
            if n > dbin_thresh:
                d_est = k
    elif D_estimate_method == "relative":
        peak = 0
        last_peak = 0
        for k in ks:
            n = dbins(k)
            if n > peak and n > 1:
                peak = n
                last_peak = n
            if d_est > 0 and n == 0:
                # stop before a second spike might occur
                break
            if peak > 0:
                # only applies if peak already found
                if n > .2*last_peak:
                    # discard bins if smaller than 20% of last peak
                    d_est = k
                    # only change last peak record if new one is actually smaller!
                    last_peak = n
    else:
        raise ValueError("Invalid dimension estimation method name")
    if fignum > 0:
        fontmath = args(fontsize=22,fontname='Times')
        fonttext = args(fontsize=18,fontname='Times')
        # make plot
        print("Plotting histograms to figure", fignum)
        assert num_panels in [2,4], "num_panels argument must be 2 or 4"
        figure(fignum)
        if num_panels == 4:
            subplot(2,2,1)
            plot(dx, rhiy, 'ro')
            glines = getp(gca(), 'xticklines')+getp(gca(), 'yticklines')
            setp(glines, 'linewidth', 2)
##            title('nhd max radius')
            xlabel(r'$\rm{Dimension}$', fonttext)
            ylabel(r'$r_{max}$', fontmath)
        if num_panels == 4:
            subplot(2,2,2)
        else:
            subplot(2,1,1)
        plot(dx, ly, 'ko')
        glines = getp(gca(), 'xticklines')+getp(gca(), 'yticklines')
        setp(glines, 'linewidth', 2)
        dmin, dmax = figure(fignum).axes[0].get_xlim()
        if dmax > maxD or force_xaxis_limits:
            dmax = maxD
        if dmin < minD or force_xaxis_limits:
            if minD is not None:
                dmin = minD
        if nhd_size_thresh > 0:
            plot([dmin,dmax],[nhd_size_thresh,nhd_size_thresh],'k--',linewidth=1.2)
##        title('size')
        xlabel(r'$\rm{Dimension}$', fonttext)
        ylabel(r'$\rm{N-hood \ size}$', fonttext)
        if num_panels == 4:
            subplot(2,2,3)
            plot(dx, rloy, 'go')
            glines = getp(gca(), 'xticklines')+getp(gca(), 'yticklines')
            setp(glines, 'linewidth', 2)
##            title('minimum radius')
            xlabel(r'$\rm{Dimension}$', fonttext)
            ylabel(r'$r_{min}$', fontmath)
        if num_panels == 4:
            subplot(2,2,4)
        else:
            subplot(2,1,2)
        dbx = dbins.midpoints.tolist()
        dby = list(dbins.values())
        bar(dbx, dby, color='b', width=dbx[1]-dbx[0])
        for panel in range(num_panels):
            figure(fignum).axes[panel].set_xlim([dmin,dmax])
##        title('number binned')
        xlabel(r'$\rm{Dimension}$', fonttext)
        ylabel(r'$\rm{Number \ binned}$',fonttext)
    # Calculate overlap info:
    # how many times does a point appear in the neighbourhoods?
    # assumes will not be more than 100 times!
    cbins = zeros((100,),'i')
    max_used = 0
    for clist in covered.values():
        try:
            n = len(clist)
        except TypeError:
            # a point that was not included in a covering
            n = 0
        if n > max_used:
            max_used = n
        if n < maxD+1:
            cbins[n] = cbins[n] + 1
##    cran = range(max_used+1)
    cbu = cbins[0:max_used+1]
    if fignum > 0:
##        subplot(2,2,4)
##        bar(cran,cbu)
##        title('nhd overlap')
##        xlabel('# pts in common')
##        ylabel('# nhds')
        if num_panels == 4:
            figure(fignum).axes[0].set_position([0.1, 0.55, 0.37, 0.35])
            figure(fignum).axes[1].set_position([0.57, 0.55, 0.37, 0.35])
            figure(fignum).axes[2].set_position([0.1, 0.09, 0.37, 0.35])
            figure(fignum).axes[3].set_position([0.57, 0.09, 0.37, 0.35])
            figure(fignum).axes[1].set_ylim([0,nhd_max_plot_size])
            if radius_yaxis_limits is not None:
                assert radius_yaxis_limits[1] >= radius_yaxis_limits[0], \
                       "Specify radius y-axis limits in increasing order"
                ylo1 = figure(fignum).axes[0].get_ylim()[0]
                ylo0 = figure(fignum).axes[2].get_ylim()[0]
                figure(fignum).axes[0].set_ylim([ylo1,radius_yaxis_limits[1]])
                figure(fignum).axes[2].set_ylim([ylo0,radius_yaxis_limits[0]])
        else:
            figure(fignum).axes[0].set_position([0.125, 0.57, 0.775, 0.364])
            figure(fignum).axes[1].set_position([0.125, 0.08, 0.775, 0.364])
            figure(fignum).axes[0].set_ylim([0,nhd_max_plot_size])
        draw()
        if save != '':
            savefig(save+'.png')
            savefig(save+'.svg')
            savefig(save+'.eps')
    # don't assert the following if re-checking points in large nhds (the default)
    #assert sum([i*cbins[i] for i in cran]) == integral
    tail_ix = dbins.resolve_bin_index(d_est)
    if isinstance(tail_ix, tuple):
        tail_ix = tail_ix[1]
    try:
        m = dbins[:tail_ix].mean()
        s = dbins[:tail_ix].std()
    except:
        print("Tail found at index %i and D = %.3f"%(tail_ix,d_est))
        print(" -- problem computing mean and std dev for this mode")
        print(" so calculating for whole data set")
        m = dbins.mean()
        s = dbins.std()
    print("Estimate dimension to be (to resolution of histogram bin width):", d_est)
    print("  with histogram mean D = %.5f and std dev D = %.5f"%(m, s))
    return {"D_tail": d_est, "D_mean": m, "D_std": s}, \
           dbins, cover_by_dimix, cover_by_size, cbu, filtered_ixs, integral


def get_filtered_ixs(cover_by_dimix):
    """Returns the list of indices in covering for which the neighbourhood
    size was larger than the threshold for dimension binning."""
    fixs=[]
    for p, ixlist in cover_by_dimix.items():
        fixs.extend(ixlist)
    fixs.sort()
    return fixs


def find_outliers(data, plotdata, cluster_dim, cluster_width, max_central=50):
    outliers = []
    central = []
    assert cluster_width > 0 and cluster_dim - cluster_width > 1
    for i, (dlo, dhi, refix) in enumerate(plotdata):
        refix = int(refix)
        if dlo < 1:
            outliers.append(refix)
        if dlo > cluster_dim-cluster_width and dlo < cluster_dim+cluster_width:
            if len(central) < max_central:
                central.append(refix)
    num_central = len(central)
    central_data = array([data[cix] for cix in central])
    if max_central > 0:
        c_mean = mean(central_data,0)
    else:
        c_mean = central_data[0]
    do = zeros((len(outliers),),float)
    dc = zeros((num_central,),float)
    for i, oix in enumerate(outliers):
        do[i]=norm(c_mean-data[oix])
    for i, cdata in enumerate(central_data):
        dc[i]=norm(c_mean-cdata)
    ds = []
    for i in range(len(data)):
        if i not in outliers:
            ds.append(norm(data[i]-data[outliers[0]]))
    ds = array(ds)
    if num_central > 1:
        mean_dc = mean(dc)
        std_dc = std(dc)
    else:
        mean_dc = dc[0]
        std_dc = 0
    #print "Found %i outliers and %i central points"%(len(outliers),num_central)
    return (mean(do), std(do), mean_dc, std_dc, mean(ds), std(ds), do)


def get_cover_radii(covering, ixs=None, method='divide'):
    """Return the radius of a covering"""
    if method == 'divide':
        rfun = lambda rh, rl: rh/rl
    elif method == 'subtract':
        rfun = lambda rh, rl: rh-rl
    radii = {}
    if ixs is None:
        for p, (d, rlo, rhi, l) in covering.items():
            radii[p] = rfun(rhi,rlo)
    else:
        for p, (d, rlo, rhi, l) in covering.items():
            if p in ixs:
                radii[p] = rfun(rhi,rlo)
    return radii


def filter_by_radius(covering, radii, rlo=None, rhi=None):
    """Returns list of filtered (index, radius) pairs and corresponding
    sub-coverings' dimensions"""
    if rhi is None:
        assert rlo is not None
        test = lambda r: r > rlo
    else:
        if rlo is None:
            test = lambda r: r < rhi
        else:
            test = lambda r: r > rlo and r < rhi
    rfilt = [(ix,r) for (ix, r) in radii.items() if test(r)]
    return rfilt, [covering[rinfo[0]][0] for rinfo in rfilt]


def plot_radius_distribution(radii, max_radius, radius_bin_width):
    """Expects radii dictionary, as returned by get_cover_radii.
    Returns figure handle."""
    rads = array(radii.values())
    dbins = data_bins('r', arange(0, max_radius+radius_bin_width, radius_bin_width))
    dbins.bin(rads)
    dbx, dby = dbins.to_array()
    f = figure()
    bar(dbx,dby,width=rbin)
    title('Distribution of nhd radii')
    xlabel('r')
    ylabel('#')
    return f


def plot_dim_by_thickness(covering, radii, rwidth=0, plotstyle='k.'):
    """Plot of dimensions of neighbourhoods with thickness r, using
    binned data of bin width rwidth. r is defined as the neighbourhood's
    r_max - r_min.

    If rwidth=0 (default) then no data binning is done.
    Returns figure handle."""
    rlo, rhi = extent(radii.values())
    assert rlo > 0 and rhi > rlo
    f=figure()
    if rwidth > 0:
        assert rwidth < rhi-rlo
        for r in arange(0, rhi+rwidth, rwidth):
            rs,ds=filter_by_radius(covering, radii, r, r+rwidth)
            if ds != []:
                plot([r], ds, plotstyle)
        title('Distribution of dimension by radius bin (width '+str(rwidth)+')')
    else:
        rs,ds=filter_by_radius(covering, radii, 0, rhi)
        if ds != []:
            plot([r[1] for r in rs], ds, plotstyle)
##        title('Distribution of dimension by radius')
    xlabel(r'$r_{max} \ - \ r_{min}$',fontsize=22)
    ylabel(r'$\rm{Dimension}$',fontsize=20)
    return f


def plot_thickness_by_dim(covering, radii, plotstyle='k.'):
    """Plot of thickness of neighbourhoods as a function of their
    estimated dimension. r is defined as the neighbourhood's
    r_max - r_min.

    Returns figure handle."""
    rlo, rhi = extent(radii.values())
    assert rlo > 0 and rhi > rlo
    f = figure()
    rs, ds = filter_by_radius(covering, radii, 0, rhi)
    if ds != []:
        plot(ds, [r[1] for r in rs], plotstyle)
    ylabel(r'$r_{max} \ - \ r_{min}$',fontsize=22)
    xlabel(r'$\rm{Dimension}$',fontsize=20)
    return f


def nhd(data, covering, refpt):
    return data.take(covering[refpt][3], 0)

# ---------------------------------------------------------------------------


def get_rV_curves(data, refpts, doplot=True, show_seq=False):
    """Get r-V curves for pointwise dimension estimate of data"""
    if doplot:
        figure()
    rV = {}
    for refpt in refpts:
        if show_seq:
            logv, logd, ixs_sorted = log_distances(data, refpt, doplot,
                                                   return_ixs=True)
        else:
            logv, logd = log_distances(data, refpt, doplot,
                                                   return_ixs=False)
        if show_seq:
            # get points that are not in increasing or decreasing sequence
            notinseq = out_of_seq(ixs_sorted)
            inseq = remain(ixs_sorted, notinseq)
        else:
            notinseq = []
            inseq = []
        rV[refpt] = (logv, logd, inseq, notinseq)
    return rV


def make_secant_fig_dict(items):
    return {
        'Dim': items[0],
        'refpt': int(items[1]),
        'r_min_minslope': items[2],
        'r_min_maxslope': items[3],
        'index': int(items[4])
        }

def prep_secant_figure(data, plotdata, ssorted, radii, ixdata, sstats,
                       start_ix, stop_ix, Delta, fignum, save='',
                       show_seq=False):
    sms=sstats['min_slope']
    res_info = {'min_slope':{}, 'max_slope': {}}
    keyslopes_min=[sms['max'], sms['min'], sms['mean'], sms['mean']+sms['std']]
    if sms['mean']-sms['std'] > 0:
        keyslopes_min.append(sms['mean']-sms['std'])
    ixs_min = [int(i) for i in find_from_sorted(plotdata[:,0], keyslopes_min)]
    refpts_min = [ixdata[ix][1] for ix in ixs_min]
    rm=res_info['min_slope']
    rm['max'] = make_secant_fig_dict([sms['max'],
                                  refpts_min[0]]+radii[ixs_min[0]].tolist())
    rm['min'] = make_secant_fig_dict([sms['min'],
                                  refpts_min[1]]+radii[ixs_min[1]].tolist())
    rm['mean'] = make_secant_fig_dict([sms['mean'],
                                   refpts_min[2]]+radii[ixs_min[2]].tolist())
    rm['mean_plus_std'] = make_secant_fig_dict([sms['mean']+sms['std'], \
                                    refpts_min[3]]+radii[ixs_min[3]].tolist())
    if sms['mean']-sms['std'] > 0:
        rm['mean_minus_std'] = make_secant_fig_dict([sms['mean']-sms['std'], \
                                    refpts_min[4]]+radii[ixs_min[4]].tolist())

    sms=sstats['max_slope']
    keyslopes_max=[sms['max'], sms['min'], sms['mean'], sms['mean']+sms['std']]
    if sms['mean']-sms['std'] > 0:
        keyslopes_max.append(sms['mean']-sms['std'])
    ixs_max = [int(i) for i in find_from_sorted(plotdata[:,1], keyslopes_max)]
    refpts_max = [ixdata[ix][1] for ix in ixs_max]
    rm=res_info['max_slope']
    rm['max'] = make_secant_fig_dict([sms['max'],
                                  refpts_max[0]]+radii[ixs_max[0]].tolist())
    rm['min'] = make_secant_fig_dict([sms['min'],
                                  refpts_max[1]]+radii[ixs_max[1]].tolist())
    rm['mean'] = make_secant_fig_dict([sms['mean'],
                                   refpts_max[2]]+radii[ixs_max[2]].tolist())
    rm['mean_plus_std'] = make_secant_fig_dict([sms['mean']+sms['std'], \
                                    refpts_max[3]]+radii[ixs_max[3]].tolist())
    if sms['mean']-sms['std'] > 0:
        rm['mean_minus_std'] = make_secant_fig_dict([sms['mean']-sms['std'], \
                                    refpts_max[4]]+radii[ixs_max[4]].tolist())

    rV=get_rV_curves(data, refpts_min+refpts_max, False, show_seq)
    res_info['min_slope']['refpts'] = refpts_min
    res_info['max_slope']['refpts'] = refpts_max
    res_info['rV'] = rV
    cols_min=['b','r','y']  # min_slope [max, min, mean]
    cols_max=['c','m','g']  # max_slope [max, min, mean]
    figure(fignum)
    subplot(1,2,1)
    xlim=[Inf,-Inf]
    ylim=[Inf,-Inf]
    for i in [0,1,2]:
        logv, logd, inseq, notinseq = rV[refpts_min[i]]
        plot(logd, logv, cols_min[i], linewidth=2)
        if show_seq:
            if len(inseq) > len(notinseq):
                # plot not in seq points in red
                plot(logd[notinseq], logv[notinseq], 'r^')
            else:
                # plot seq points in green
                plot(logd[inseq], logv[inseq], 'gv')
        minlogd = min(logd)
        maxlogd = max(logd)
        minlogv = min(logv)
        maxlogv = max(logv)
        if minlogd < xlim[0]:
            xlim[0] = minlogd
        if maxlogd > xlim[1]:
            xlim[1] = maxlogd
        if minlogv < ylim[0]:
            ylim[0] = minlogv
        if maxlogv > ylim[1]:
            ylim[1] = maxlogv
        logv, logd, inseq, notinseq = rV[refpts_max[i]]
        plot(logd, logv, cols_max[i], linewidth=2)
        if show_seq:
            if len(inseq) > len(notinseq):
                # plot not in seq points in red
                plot(logd[notinseq], logv[notinseq], 'r^')
            else:
                # plot seq points in green
                plot(logd[inseq], logv[inseq], 'gv')
        minlogd = min(logd)
        maxlogd = max(logd)
        minlogv = min(logv)
        maxlogv = max(logv)
        if minlogd < xlim[0]:
            xlim[0] = minlogd
        if maxlogd > xlim[1]:
            xlim[1] = maxlogd
        if minlogv < ylim[0]:
            ylim[0] = minlogv
        if maxlogv > ylim[1]:
            ylim[1] = maxlogv
    # plot line marking where start_ix'th point lies
    plot(xlim, [logv[start_ix], logv[start_ix]], 'k--', linewidth=2)
    # if stop_ix < N, plot line marking where stop_ix'th point lies
    if stop_ix < len(data):
        plot(xlim, [logv[stop_ix], logv[stop_ix]], 'k--', linewidth=2)
    # New loop, to ensure markers are on top of all lines:
    # Mark r_min and r_max for secant of both max and min slope,
    # for each r-V curve plotted
    for i in [0,1,2]:
        logv, logd, inseq, notinseq = rV[refpts_min[i]]
        loix_min = ixdata[ixs_min[i]][0][0]
        hiix_min = find(logv, logv[loix_min]+log(Delta))
        plot([logd[loix_min]], [logv[loix_min]], cols_min[i]+'o',
             markersize=10)
        plot([logd[hiix_min]], [logv[hiix_min]], cols_min[i]+'o',
             markersize=10)
        loix_min = ixdata[ixs_min[i]][0][1]
        hiix_min = find(logv, logv[loix_min]+log(Delta))
        plot([logd[loix_min]], [logv[loix_min]], cols_min[i]+'s',
             markersize=10)
        plot([logd[hiix_min]], [logv[hiix_min]], cols_min[i]+'s',
             markersize=10)
        logv, logd, inseq, notinseq = rV[refpts_max[i]]
        loix_max = ixdata[ixs_max[i]][0][0]
        hiix_max = find(logv, logv[loix_max]+log(Delta))
        plot([logd[loix_max]], [logv[loix_max]], cols_max[i]+'o',
             markersize=10)
        plot([logd[hiix_max]], [logv[hiix_max]], cols_max[i]+'o',
             markersize=10)
        loix_max = ixdata[ixs_max[i]][0][1]
        hiix_max = find(logv, logv[loix_max]+log(Delta))
        plot([logd[loix_max]], [logv[loix_max]], cols_max[i]+'s',
             markersize=10)
        plot([logd[hiix_max]], [logv[hiix_max]], cols_max[i]+'s',
             markersize=10)
    # plot bar for Delta
    bar_x = 0.1*(xlim[1]-xlim[0])+xlim[0]
    bar_y = 0.5*(ylim[1]-ylim[0])+ylim[0]
    plot([bar_x, bar_x],[bar_y, bar_y+log(Delta)],'k-',linewidth=10)
    # labels
    fontmath_h = args(fontsize=22,fontname='Times',
                      verticalalignment='top')
    fontmath_v = args(fontsize=22,fontname='Times',
                      horizontalalignment='right')
    xlabel(r'$\rm{log \ } r_k$',fontmath_h)
    ylabel(r'$\rm{log \ } k$',fontmath_v)
    subplot(1,2,2)
    rescatter(plotdata, array([0.2,0.2,0.2]), newfigure=False)
    # replot selected points
    for i in [0,1,2]:
        rescatter(array([plotdata[ixs_min[i]]]), cols_min[i],
                  marker='o', marker_size=150, newfigure=False)
        rescatter(array([plotdata[ixs_max[i]]]), cols_max[i],
                  marker='s', marker_size=225, newfigure=False)
    figure(fignum).axes[0].set_xlim(xlim)
    figure(fignum).axes[0].set_ylim(ylim)
    figure(fignum).axes[0].set_position([0.07, 0.11, 0.41, 0.85])
    figure(fignum).axes[1].set_position([0.54, 0.11, 0.41, 0.85])
    figure(fignum).set_figwidth(14.5)
    draw()
    if save != '':
        figure(fignum).savefig(save+'.png')
        figure(fignum).savefig(save+'.svg')
        figure(fignum).savefig(save+'.eps')
    return res_info


def slope_range(data, refptix, step=2, Delta=10, startix=10, stopix=None,
                doplot=False, maxD=Inf):
    """Slope range by secants over a range of log(Delta) ball volumes"""
    N=size(data,0)
    assert Delta > 1
    assert N > Delta*startix
    logv_raw = log(range(1,N+1))
    logv, logd, d, d_inv_dict = log_distances_with_D(data, refptix, logv_raw)
    hi_slope = 0
    lo_slope = 1e10
    loix = startix
    hiix = loix*Delta
    if stopix is None:
        stopix = N
    assert stopix <= N
    assert hiix < stopix
    logDelta = log(Delta)
    slope_by_loix = {}
    while True:
        # slope by secant
        slope = logDelta/(logd[hiix]-logd[loix])
        if slope > hi_slope and slope < maxD:
            hi_slope = slope
            hi_slope_ix = loix
        if slope < lo_slope and slope < maxD:
            lo_slope = slope
            lo_slope_ix = loix
        slope_by_loix[loix] = slope
        loix += step
        hiix = Delta*loix
        if hiix >= stopix:
            break
    if doplot:
        plot(sortedDictValues(slope_by_loix))
    return (slope_by_loix, lo_slope, lo_slope_ix,
            hi_slope, hi_slope_ix, logv, logd)



def scatterplot_slopes(data, num_samples=None, startix=10, stopix=None,
                       Delta=10, marker='o', marker_size=40, cmap=DEFAULT_CMAP,
                       color_source=None, step=5, fignum=None, maxD=Inf):
    """fignum = 0 switches off plotting (just return statistics)"""
    if num_samples is None:
        num_samples = len(data)/5
        range_step = 5
    else:
        range_step = int(len(data)/num_samples)
    if range_step > 0:
        refpts = arange(0,len(data),range_step,'i')
    else:
        raise ValueError("too many sample points specified")
    num_refpts = len(refpts)
    plotdata = zeros((num_refpts,3),Float)
    ixdata = {}
    # colors indicate either min or max radii
    colors = zeros((num_refpts,3),Float)
    if color_source == 'lo':
        cix=0
        title_str = 'min'
    elif color_source == 'hi':
        cix=1
        title_str = 'max'
    elif color_source is None:
        title_str = ''
        col = array([0.3,0.3,0.3])
    else:
        raise ValueError("invalid color source")
    print("Expect %i dots when finished:"%int(num_refpts/50))
    for i in range(num_refpts):
        if mod(i,50) == 49:
            print(".", end=' ')
        slope_by_loix, lo, loix, hi, hiix, logv, logd = slope_range(data,
                            refpts[i], startix=startix, stopix=stopix,
                            Delta=Delta, step=step, maxD=maxD)
        plotdata[i,:] = array([lo,hi,refpts[i]])
        ixdata[i] = [(loix,hiix),refpts[i]]
        colors[i,:] = array([exp(logd[loix]),exp(logd[hiix]),refpts[i]])
    if fignum is None:
        figure()
        dofig = True
    elif fignum == 0:
        dofig = False
    else:
        figure(fignum)
        dofig = True
    if dofig:
        if color_source is None:
            handle=scatter(plotdata[:,0],plotdata[:,1],marker=marker,c=col,
                    s=marker_size)
        else:
            handle=scatter(plotdata[:,0],plotdata[:,1],marker=marker,c=colors[:,cix],
                    cmap=cmap,s=marker_size)
            colorbar()
        fonttext = args(fontsize=20,fontname='Times')
        xlabel(r'$\rm{min \ slope}$',fonttext)
        ylabel(r'$\rm{max \ slope}$',fonttext)
        if title_str != '':
            title('Color-coded by r_min of '+title_str+' slope')
    else:
        handle=None
    s=(sorted_by_slope(plotdata,0),sorted_by_slope(plotdata,1))
    if color_source is None:
        c = ()
    else:
        c=(sorted_by_radius(colors,0),sorted_by_radius(colors,1))
    stats = {'min_slope': {}, 'max_slope': {}}
    stats['min_slope']['mean']=mean(plotdata[:,0])
    stats['max_slope']['mean']=mean(plotdata[:,1])
    stats['min_slope']['std']=std(plotdata[:,0])
    stats['max_slope']['std']=std(plotdata[:,1])
    stats['min_slope']['min']=min(plotdata[:,0])
    stats['max_slope']['min']=min(plotdata[:,1])
    stats['min_slope']['max']=max(plotdata[:,0])
    stats['max_slope']['max']=max(plotdata[:,1])
    print("\n")
    return (plotdata, s, colors, c, ixdata, stats, handle)


def sorted_by_slope(plotdata, select=0):
    """select: 0 = rmin, 1 = rmax"""
    ixs = argsort(plotdata, 0)[:,select]
    return list(zip(take(plotdata, ixs, 0)[:,select], ixs))

def sorted_by_radius(colors, select=0):
    """select: 0 = rmin, 1 = rmax"""
    ixs = argsort(colors, 0)[:,select]
    return list(zip(take(colors, ixs, 0)[:,select], ixs))

def find_from_sorted(x, v, next_largest=1, indices=None):
    if isinstance(v, _seq_types):
        a=array(x)
        res = [find(a, y, next_largest, indices) for y in v]
        return res
    else:
        return find(array(x), v, next_largest, indices)


def rescatter(plotdata, colors, color_source=None, marker='o', marker_size=40,
              cmap=DEFAULT_CMAP, newfigure=True):
    if color_source == 'lo':
        cix=0
        title_str = 'min'
    elif color_source == 'hi':
        cix=1
        title_str = 'max'
    elif color_source is None:
        title_str = ''
        if not isinstance(colors, str):
            assert shape(colors) == (3,)
    else:
        raise ValueError("invalid color source")
    if newfigure:
        figure()
    if color_source is None:
        handle=scatter(plotdata[:,0],plotdata[:,1],marker=marker,
                c=colors,s=marker_size)
    else:
        handle=scatter(plotdata[:,0],plotdata[:,1],marker=marker,
                c=colors[:,cix],cmap=cmap,s=marker_size)
        colorbar()
    fonttext_v = args(fontsize=22,fontname='Times',
                    horizontalalignment='right')
    fonttext_h = args(fontsize=22,fontname='Times',
                    verticalalignment='top')
    xlabel(r'$\rm{min \ slope}$',fonttext_h)
    ylabel(r'$\rm{max \ slope}$',fonttext_v)
    if title_str != '':
        title('Color-coded by r_min of '+title_str+' slope')
    return handle


def scatter_histo(plotdata,select=1,bin_width=1,maxD=None,
                  xlim=None,ylim=None):
    ds = array(plotdata[:,select])
    ds.shape = (len(plotdata),)
    if maxD is None:
        maxD = int(ceil(max(ds[:,0])))
    drange = arange(0, maxD+1+bin_width, bin_width, typecode='d')
    dbins = data_bins('D', drange)
    dbins.bin(ds)
    dbx, dby = dbins.to_array()
    f=figure()
    b=bar(dbx, dby, color='b', width=dbx[1]-dbx[0])
    if xlim is not None:
        f.axes[0].set_xlim(xlim)
        draw()
    if ylim is not None:
        f.axes[0].set_ylim(ylim)
        draw()


def PD_E(a, secfig=5, verbose=False, saveplot=True, force=False):
    """Pointwise dimension estimation (PD-E) using secants of r-V curves.
    See:

    J. Guckenheimer, "Dimension Estimates for Attractors",
    Contemporary Mathematics, Vol. 28, 1984.

    R. Clewley, J. Guckenheimer, F. Valero-Cuevas, "Estimating effective
    degrees of freedom in motor control", IEEE Transactions in Biomedical
    Engineering, 2007.
    """
    pde_name = 'PD_E-'+a.name
    if 'N' in a.keys():
        if 'stopix' in a.keys():
            if a.stopix is not None:
                a.stopix = int(round(0.7*a.N))
        else:
            a.stopix = int(round(0.7*a.N))
        a.num_samples = int(round(0.2*a.N))
        a.step = int(round(0.005*a.N))
    else:
        a.N = None
        if 'stopix' not in a.keys():
            a.stopix = None
    try:
        if force:
            raise ValueError
        data, pde_args, plotdata, ssorted, colors, \
              csorted, ixdata, sstats = loadObjects(pde_name)
        print("Loaded data set and stats for %s"%a.name)
        # don't check the data field
        if filteredDict(a, ['data','bin_width'], neg=True) != \
                 filteredDict(pde_args, ['data','bin_width'], neg=True):
            raise "Incompatible argument struct from file %s"%pde_name
        if a.data is None:
            a.data = data
        elif not all(a.data == data):
            raise "Incompatible data set loaded from file %s"%pde_name
    except ValueError:
        if a.data is None:
            try:
                data = eval(a.data_gen_str, globals(), a.data_gen_fun)
            except:
                print("Problem re-calculating data from given information")
        else:
            data = a.data
        print("Recalculating PD-E for %s"%a.name)
        try:
            csource=a.color_source
        except AttributeError:
            csource=None
        if verbose:
            scatter_fig = None
        else:
            scatter_fig = 0
        if a.data is None:
            a.data = data
        plotdata, ssorted, colors, csorted, ixdata, sstats, handle = \
                  scatterplot_slopes(data,
                            startix=a.startix, stopix=a.stopix,
                            num_samples=a.num_samples,
                            Delta=a.Delta, color_source=csource, step=a.step,
                            maxD=a.maxD, fignum=scatter_fig)
        saveObjects([data, a, plotdata, ssorted, colors, csorted, ixdata, sstats],
                    pde_name, force=True)
    if verbose:
        scatter_histo(plotdata,bin_width=a.bin_width,select=0)
        scatter_histo(plotdata,bin_width=a.bin_width,select=1)
        info(sstats)
    if saveplot:
        save_info='result_'+a.name
    else:
        save_info=''
    if 'show_seq' in a.keys():
        show_seq = a['show_seq']
    else:
        show_seq = False
    more_info=prep_secant_figure(data, plotdata, ssorted, colors, ixdata, sstats,
                   a.startix, a.stopix, a.Delta, secfig, save=save_info,
                   show_seq=show_seq)
    try:
        more_info['handle']=handle
    except UnboundLocalError:
        more_info['handle']=None
    more_info['stats']=sstats
    more_info['args']=a
    return more_info
