
import numpy as np
from PyDSTool import Events, Variable, Pointset, Trajectory
from PyDSTool.common import args, metric, metric_L2, metric_weighted_L2, \
     metric_float, remain, fit_quadratic, fit_exponential, fit_diff_of_exp, \
     smooth_pts, nearest_2n_indices, make_poly_interpolated_curve, simple_bisection
from PyDSTool.Trajectory import numeric_to_traj
from PyDSTool.ModelContext import *
from PyDSTool.Toolbox.data_analysis import butter, filtfilt, rectify
from PyDSTool.errors import PyDSTool_KeyError
import copy

# Test this on a single spike with global max at spike and minima at endpoints
# Test this on a mexican hat type spike with global min and max at spike peak and trough
# Test this on monotonic data for worst case scenario!! Should return None for max and min
# Also test on noisy monotonic data
# Return value of Nones to a feature evaluator should suggest to it to change window size for defining pts
def find_internal_extrema(pts, noise_tol=0):
    """
    Find an interior (local) maximum and minimum values of a 1D pointset, away from the endpoints.
    Returns a dictionary mapping 'local_max' -> (index_max, xmax), 'local_min' -> (index_min, xmin),
    whose values are None if the pointset is monotonic or is close enough so that the global extrema
    are at the endpoints.

    Use noise_tol > 0 to avoid getting a local extremum right next to an endpoint because of noise.

    Also returned in the dictionary for reference:
    'first' -> (0, <start_endpoint_value>), 'last' -> (last_index, <last_endpoint_value>),
    'global_max' -> (index, value), 'global_min' -> (index, value)

    Assumes there is only one interior (max, min) pair in pts, otherwise will return an arbitrary choice
    from multiple maxima and minima."""

    assert pts.dimension == 1
    # convert all singleton points to floats with [0] selection
    x0 = pts[0][0]
    x1 = pts[-1][0]
    # need last_ix explicitly for index test below
    last_ix = len(pts)-1
    end_ixs = (0, last_ix)

    max_val_ix = np.argmax(pts)
    min_val_ix = np.argmin(pts)
    glob_xmax = pts[max_val_ix][0]
    glob_xmin = pts[min_val_ix][0]

    no_local_extrema = {'local_max': (None, None), 'local_min': (None, None),
                        'first': (0, x0), 'last': (last_ix, x1),
                        'global_max': (max_val_ix, glob_xmax),
                        'global_min': (min_val_ix, glob_xmin)
                    }

    max_at_end = max_val_ix in end_ixs
    min_at_end = min_val_ix in end_ixs
    if max_at_end:
        if min_at_end:
            # No detectable turning points present (this is criterion for ignoring noisy data)
            return no_local_extrema
        else:
            # interior minimum found
            index_min = min_val_ix
            xmin = pts[index_min]
            # find associated interior local maximum
            max_val_ix1 = np.argmax(pts[:min_val_ix])
            max_val_ix2 = np.argmax(pts[min_val_ix:])+min_val_ix
            if max_val_ix1 in end_ixs:
                if max_val_ix2 in end_ixs:
                    index_max = None
                    xmax = None
                else:
                    index_max = max_val_ix2
                    xmax = pts[index_max][0]
            else:
                # assumes only one local max / min pair in interior!
                index_max = max_val_ix1
                xmax = pts[index_max][0]
    else:
        # interior maximum found
        index_max = max_val_ix
        xmax = pts[index_max][0]
        # find associated interior local minimum
        min_val_ix1 = np.argmin(pts[:max_val_ix])
        xmin1 = pts[min_val_ix1][0]
        min_val_ix2 = np.argmin(pts[max_val_ix:])+max_val_ix
        xmin2 = pts[min_val_ix2][0]
        if min_val_ix1 in end_ixs or abs(xmin1-x0)<noise_tol or abs(xmin1-x1)<noise_tol:
            if min_val_ix2 in end_ixs or abs(xmin1-x0)<noise_tol or abs(xmin1-x1)<noise_tol:
                index_min = None
                xmin = None
            else:
                index_min = min_val_ix2
                xmin = xmin2
        else:
            # assumes only one local max / min pair in interior!
            index_min = min_val_ix1
            xmin = xmin1

    return {'local_max': (index_max, xmax), 'local_min': (index_min, xmin),
            'first': (0, x0), 'last': (last_ix, x1),
            'global_max': (max_val_ix, glob_xmax),
            'global_min': (min_val_ix, glob_xmin)}


class get_spike_model(ql_feature_leaf):
    """Qualitative test for presence of spike in model trajectory data
     using events to identify spike times. Also records salient spike
     information for quantitative comparisons later."""

    def evaluate(self, traj):
        # function of traj, not target
        pts = traj.sample(coords=[self.super_pars.burst_coord],
                                      tlo=self.pars.tlo,
                          thi=self.pars.tlo+self.pars.width_tol)
        loc_extrema = find_internal_extrema(pts)
        if self.pars.verbose_level > 0:
            print(loc_extrema)
        max_val_ix, xmax = loc_extrema['local_max']
        global_max_val_ix, global_xmax = loc_extrema['global_max']
        min_val_ix, xmin = loc_extrema['local_min']
        global_min_val_ix, global_xmin = loc_extrema['global_min']

        # could split these tests into 3 further sub-features but we'll skip that here for efficiency
        if xmax is None:
            self.results.ixmax = None
            self.results.tmax = None
            test1 = test2 = test3 = False
        else:
            test1 = max_val_ix not in (loc_extrema['first'][0], loc_extrema['last'][0])
            test2 = np.linalg.norm(global_xmin-xmax) > self.pars.height_tol
            try:
                test3 = np.linalg.norm(xmin-xmax) > self.pars.height_tol
            except:
                # fails if xmin is None, i.e. no interior minimum
                # allow no local minimum present, in which case use the other endpoint for test
                # ... we don't know which is the one alread tested in test2, so test both ends again,
                # knowing that they are both lower than the interior maximum found in this case
                xmin = max([global_xmin, loc_extrema['last'][1], loc_extrema['first'][1]])
                test3 = np.linalg.norm(xmin-xmax) > self.pars.height_tol
            self.results.ixmax = max_val_ix
            self.results.tmax = pts.indepvararray[max_val_ix]
        self.results.spike_pts = pts
        return test1 and test2 and test3

    def finish(self, traj):
        self.results.spike_time = self.results.tmax
        self.results.spike_val = self.results.spike_pts[self.results.ixmax][self.super_pars.burst_coord]


class get_spike_data(ql_feature_leaf):
    """Qualitative test for presence of spike in noisy data. Also records salient spike information
     for quantitative comparisons later.

    Criteria: ensure a maximum occurs, and that this is away from endpoints of traj
     "Uniqueness" of this maximum can only be determined for noisy data using a height
     tolerance.

    Assumes spikes will never bunch up too much so that more than spike occurs in the
     spacing_tol window.

    Finds maximum position using a quadratic fit.
    """
    def _local_init(self):
        # avoids recreating this object for every test
        self.quadratic = fit_quadratic(verbose=self.pars.verbose_level>0)

    def evaluate(self, traj):
        # function of traj, not target
        event_args = {'name': 'spike_thresh',
               'eventtol': self.pars.eventtol,
               'eventdelay': self.pars.eventtol*.1,
               'starttime': 0,
               'active': True}
        if 'coord' not in self.pars:
            self.pars.coord = self.super_pars.burst_coord
        # update thi each time b/c tlo will be different
        self.pars.thi = self.pars.tlo+self.pars.width_tol
        self.pars.ev = Events.makePythonStateZeroCrossEvent(self.pars.coord,
                                                            "thresh", 0,
                                    event_args, traj.variables[self.pars.coord])
        pts = traj.sample(coords=[self.pars.coord], tlo=self.pars.tlo,
                              thi=self.pars.thi)
        if pts.indepvararray[-1] < self.pars.thi:
            self.pars.thi = pts.indepvararray[-1]
        loc_extrema = find_internal_extrema(pts, self.pars.noise_tol)
        if self.pars.verbose_level > 0:
            print(loc_extrema)
            # from PyDSTool import plot, show
            ## plot spike and quadratic fit
            #plot(pts.indepvararray, pts[self.super_pars.burst_coord], 'go-')
            #show()
        max_val_ix, xmax = loc_extrema['local_max']
        global_max_val_ix, global_xmax = loc_extrema['global_max']
        min_val_ix, xmin = loc_extrema['local_min']
        global_min_val_ix, global_xmin = loc_extrema['global_min']

        # could split these tests into 3 further sub-features but we'll skip that here for efficiency
        test1 = max_val_ix not in (loc_extrema['first'][0], loc_extrema['last'][0])
        test2 = np.linalg.norm(global_xmin-xmax) > self.pars.height_tol
        try:
            test3 = np.linalg.norm(xmin-xmax) > self.pars.height_tol
        except:
            # fails if xmin is None, i.e. no interior minimum
            # allow no local minimum present, in which case use the other endpoint for test
            # ... we don't know which is the one already tested in test2, so test both ends again,
            # knowing that they are both lower than the interior maximum found in this case
            xmin = max([global_xmin, loc_extrema['last'][1], loc_extrema['first'][1]])
            test3 = np.linalg.norm(xmin-xmax) > self.pars.height_tol
        # generate a suitable threshold from local maximum
        try:
            thresh_pc = self.pars.thresh_pc
        except:
            # default value of 15%
            thresh_pc = 0.15
        thresh = (xmin + thresh_pc*(xmax-xmin))
        if self.pars.verbose_level > 0:
            print("xmin used =", xmin)
            print("thresh = ", thresh)
        # Define extent of spike for purposes of quadratic fit ...
        evs_found = self.pars.ev.searchForEvents(trange=[self.pars.tlo,
                                                         self.pars.thi],
                                       parDict={'thresh': thresh})
        tlo = evs_found[0][0]
        thi = evs_found[1][0]
        tmax = pts.indepvararray[max_val_ix]
        symm_dist = np.min([abs(tmax-tlo), abs(thi-tmax)])
        # HACK! Ensure dt value will not cause us to hit an index directly, otherwise
        # have to catch case from Pointset.find method when return value is a single
        # integer index rather than a pair of indices
        if symm_dist > self.pars.fit_width_max/2.000000007:
            dt = self.pars.fit_width_max/2.000000007
        else:
            dt = symm_dist*1.0000000007
        tlo = tmax-dt
        thi = tmax+dt
        ixlo = pts.find(tmax-dt, end=0)
        ixhi = pts.find(tmax+dt, end=1)
        if self.pars.verbose_level > 0:
            print("ixlo =", ixlo, "ixhi =", ixhi)
            print("tlo =",tmax-dt, "thi =",tmax+dt)
            print(pts[ixlo], pts[ixhi])
            print("\nget_spike tests:", test1, test2, test3)
        self.results.ixlo = ixlo
        self.results.ixhi = ixhi
        self.results.ixmax = max_val_ix
        self.results.tlo = tlo
        self.results.thi = thi
        self.results.tmax = tmax
        self.results.spike_pts = pts[ixlo:ixhi]
        return test1 and test2 and test3

    def finish(self, traj):
        # function of traj, not target
        if self.pars.verbose_level > 0:
            print("Finishing spike processing...")
        pts = self.results.spike_pts
        coord = self.pars.coord
        xlo = pts[0][0]
        # xmax is just an estimate of the max value
        xmax = pts[self.results.ixmax-self.results.ixlo][0]
        estimate_quad_coeff = -(xmax-xlo)/((self.results.tmax - \
                                            self.results.tlo)**2)
        estimate_intercept = xlo - \
              ((xmax-xlo)/(self.results.tmax-self.results.tlo))*self.results.tlo
        res = self.quadratic.fit(pts.indepvararray, pts[coord],
                            pars_ic=(estimate_quad_coeff,0,estimate_intercept),
                            opts=args(peak_constraint=(self.results.ixmax - \
                                                       self.results.ixlo,xmax,
                                self.pars.weight*len(pts)/(self.results.tmax - \
                                                           self.results.tlo),
                                self.pars.weight*len(pts)/(xmax-xlo))))
        tval, xval = res.results.peak
        self.results.spike_time = tval
        self.results.spike_val = xval
        self.results.pars_fit = res.pars_fit
        if self.pars.verbose_level > 0:
            from PyDSTool import plot, show
            # plot spike and quadratic fit
            dec = 10
            plot(pts.indepvararray, pts[coord], 'go-')
            plot(tval, xval, 'rx')
            ts = [pts.indepvararray[0]]
            for i, t in enumerate(pts.indepvararray[:-1]):
                ts.extend([t+j*(pts.indepvararray[i+1]-t)/dec for j in range(1,dec)])
            ts.append(pts.indepvararray[-1])
            plot(ts, [res.results.f(t) for t in ts], 'k:')
            # temp
            if self.pars.verbose_level > 1:
                show()



class get_burst_duration(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_float()
        self.metric_len = 1

    def postprocess_ref_traj(self):
        on_t = self.super_pars.ref_spike_times[0] - self.pars.t_lookback
        self.pars.ref_burst_on_time = on_t
        # find associated V for ref_on_thresh
        pts = self.super_pars.ref_burst_coord_pts
        x = pts[self.super_pars.burst_coord]
        on_ix = pts.find(on_t, end=1)
        ix_lo, ix_hi = nearest_2n_indices(x, on_ix, 2)
        t = pts.indepvararray
        on_res = smooth_pts(t[ix_lo:ix_hi+1],
                            x[ix_lo:ix_hi+1], self.super_pars.quadratic)
        self.pars.ref_on_thresh = on_res.results.f(on_t)
        #
        off_t = self.super_pars.ref_spike_times[-1] + self.pars.t_lookforward
        self.pars.ref_burst_off_time = off_t
        off_ix = pts.find(off_t, end=0)
        ix_lo, ix_hi = nearest_2n_indices(x, off_ix, 2)
        off_res = smooth_pts(t[ix_lo:ix_hi+1],
                             x[ix_lo:ix_hi+1], self.super_pars.quadratic)
        self.pars.ref_off_thresh = off_res.results.f(off_t)
        self.pars.ref_burst_duration = off_t - on_t
        self.pars.ref_burst_prop = (off_t - on_t)/self.super_pars.ref_period

    def evaluate(self, target):
        traj = target.test_traj
        varname = self.super_pars.burst_coord
        pts = self.super_pars.burst_coord_pts
        on_t = self.super_results.spike_times[0] - self.pars.t_lookback
        self.results.burst_on_time = on_t
        x = pts[varname]
        on_ix = pts.find(on_t, end=1)
        ix_lo, ix_hi = nearest_2n_indices(x, on_ix, 2)
        pp = make_poly_interpolated_curve(pts[ix_lo:ix_hi+1], varname,
                                          target.model)
        thresh = pp(on_t)
        self.results.on_thresh = thresh
        #
        # don't find "off" based on last spike time because
        # when new spikes suddenly appear this value will jump
        # instead, use a threshold event search, assuming that
        # only one period is "in view"
        t = pts.indepvararray
        x_rev = x[:ix_hi:-1]
        t_rev = t[:ix_hi:-1]
        off_ix = len(x) - np.argmin(np.asarray(x_rev < thresh, int))
        ix_lo, ix_hi = nearest_2n_indices(x, off_ix, 2)
        pp = make_poly_interpolated_curve(pts[ix_lo:ix_hi+1], varname,
                                          target.model)
        # bisect to find accurate crossing point
        tlo = t[ix_lo]
        thi = t[ix_hi]
        off_t = simple_bisection(tlo, thi, pp, self.pars.t_tol)
        self.results.burst_duration = off_t - on_t
        self.results.burst_prop = (off_t - on_t) / self.super_results.period
        return self.metric(self.results.burst_prop,
                           self.super_pars.ref_burst_prop) < self.pars.tol


class get_burst_active_phase(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_float()
        self.metric_len = 1

    def postprocess_ref_traj(self):
        self.pars.ref_active_phase = self.super_pars.ref_spike_times[0] / \
            self.super_pars.ref_period

    def evaluate(self, target):
        self.results.active_phase = self.super_results.spike_times[0] / \
            self.super_results.period
        return self.metric(self.results.active_phase,
                           self.pars.ref_active_phase) \
               < self.pars.tol

class get_burst_dc_offset(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_float()
        self.metric_len = 1

    def postprocess_ref_traj(self):
        # 20% of burst_on_V (i.e., on_thresh) - min_V above min_V
        self.pars.ref_baseline_V = self.super_pars.ref_min_V + \
                               0.2*(self.super_pars.ref_on_thresh - \
                                    self.super_pars.ref_min_V)
    def evaluate(self, target):
        baseline = self.super_results.min_V + 0.2*(self.super_results.on_thresh - \
                                                self.super_results.min_V)
        self.results.baseline_V = baseline - self.super_pars.ref_baseline_V
        return self.metric(baseline, self.super_pars.ref_baseline_V) < \
                               self.pars.tol


class get_burst_passive_extent(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_float()
        self.metric_len = 1

    def postprocess_ref_traj(self):
        self.pars.ref_passive_extent_V = self.super_pars.ref_max_V - \
                   self.super_pars.ref_min_V

    def evaluate(self, target):
        self.results.passive_extent_V = self.super_results.max_V - \
            self.super_results.min_V
        return self.metric(self.results.passive_extent_V,
                           self.super_pars.ref_passive_extent_V) < \
                               self.pars.tol


class burst_feature(ql_feature_node):
    """Embed the following sub-features, if desired:
    get_burst_X, where X is a number of feature types defined in this module.
    """
    def _local_init(self):
        self.pars.quadratic = fit_quadratic(verbose=self.pars.verbose_level>0)
        self.pars.filt_coeffs = butter(3, self.pars.cutoff, btype='highpass')
        self.pars.filt_coeffs_LP = butter(3, self.pars.cutoff/10)

    def postprocess_ref_traj(self):
        # single coord used as indicator
        pts = self.ref_traj.sample()
        burst_pts = self.ref_traj.sample(coords=[self.pars.burst_coord],
                            dt=self.pars.dt)
        xrs = burst_pts[self.pars.burst_coord]
        trs = burst_pts.indepvararray
        x = pts[self.pars.burst_coord]
        b, a = self.pars.filt_coeffs_LP
        xf = filtfilt(b, a, xrs)
        t = pts.indepvararray
        min_val_ix = np.argmin(xf)  # use LPF version to avoid noise artifacts
        max_val_ix = np.argmax(xf)  # use LPF version to avoid spikes
        min_ix_lo, min_ix_hi = nearest_2n_indices(xrs, min_val_ix, 30)
        max_ix_lo, max_ix_hi = nearest_2n_indices(xrs, max_val_ix, 30)
        min_res = smooth_pts(trs[min_ix_lo:min_ix_hi+1],
                             xf[min_ix_lo:min_ix_hi+1], self.pars.quadratic)
        # use LPF data for max
        max_res = smooth_pts(trs[max_ix_lo:max_ix_hi+1],
                             xf[max_ix_lo:max_ix_hi+1], self.pars.quadratic)
        min_t, min_val = min_res.results.peak
        max_t, max_val = max_res.results.peak
#        thresh1 = float(max_val-self.pars.active_frac_height*(max_val-min_val))
#        thresh2 = x[0]+3.
#        # don't make threshold smaller than initial value, assuming
#        # burst will be rising at initial condition
#        thresh = max((thresh1,thresh2))
        self.pars.ref_burst_coord_pts = pts
#        self.pars.ref_on_thresh = thresh
#        self.pars.ref_off_thresh = thresh
        self.pars.ref_min_V = min_val
        self.pars.ref_max_V = max_val
        assert self.pars.on_cross_dir in (-1,1)
        if self.pars.on_cross_dir == 1:
            self.pars.off_cross_dir = -1
        else:
            self.pars.off_cross_dir = 1
        self.pars.ref_burst_est = estimate_spiking(burst_pts[self.pars.burst_coord],
                                     burst_pts.indepvararray,
                                     self.pars.filt_coeffs)
        self.pars.ref_burst_pts_resampled = burst_pts
        # spike times will be overwritten by get_spikes_data instance, if present
        #self.pars.ref_spike_times = self.pars.ref_burst_est.spike_ts
                # to establish period, find min on other side of active phase
        if min_t < self.pars.ref_burst_est.spike_ts[0]:
            # look to the right
            start_t = self.pars.ref_burst_est.spike_ts[-1]
            start_ix = pts.find(start_t, end=1)
            other_min_ix = np.argmin(x[start_ix:])
            other_min_t = t[start_ix+other_min_ix]
        else:
            # look to the left
            start_t = self.pars.ref_burst_est.spike_ts[0]
            start_ix = pts.find(start_t, end=0)
            other_min_ix = np.argmin(x[:start_ix])
            other_min_t = t[other_min_ix]
        self.pars.ref_period = abs(other_min_t - min_t)


    def prepare(self, target):
        # single coord used as indicator
        pts = target.test_traj.sample()
        x = pts[self.pars.burst_coord]
        burst_pts = target.test_traj.sample(coords=[self.pars.burst_coord],
                            dt=self.pars.dt)
        xrs = burst_pts[self.pars.burst_coord]
        trs = burst_pts.indepvararray
        if max(x)-min(x) < 5:
            print("\n\n  Not a bursting trajectory!!")
            raise ValueError("Not a bursting trajectory")
        b, a = self.pars.filt_coeffs_LP
        xf = filtfilt(b, a, xrs)
        t = pts.indepvararray
        min_val_ix = np.argmin(x) # precise because of Model's events
        max_val_ix = np.argmax(xf)
        max_ix_lo, max_ix_hi = nearest_2n_indices(xrs, max_val_ix, 4)
        max_res = smooth_pts(trs[max_ix_lo:max_ix_hi+1],
                             xf[max_ix_lo:max_ix_hi+1], self.pars.quadratic)
        min_t = t[min_val_ix]
        min_val = x[min_val_ix]
        max_t, max_val = max_res.results.peak
        self.results.min_V = min_val
        self.results.max_V = max_val
        assert self.pars.on_cross_dir in (-1,1)
        if self.pars.on_cross_dir == 1:
            self.pars.off_cross_dir = -1
        else:
            self.pars.off_cross_dir = 1
        self.results.burst_est = estimate_spiking(burst_pts[self.pars.burst_coord],
                                     burst_pts.indepvararray,
                                     self.pars.filt_coeffs)
        # record approximate spike times - may be overwritten by
        # get_burst_spikes if done accurately
        #self.results.spike_times = self.results.burst_est.spike_ts
        if self.pars.verbose_level > 0:
            print("Spikes found at (approx) t=", self.results.burst_est.spike_ts)
        if self.results.burst_est.spike_ts[0] < self.pars.shrink_end_time_thresh:
            # kludgy way to ensure that another burst doesn't encroach
            if not hasattr(self.pars, 'shrunk'):
                # do this *one time*
                end_time = t[-1] - self.pars.shrink_end_time_amount
                target.model.set(tdata=[0,end_time])
                end_pts = pts.find(end_time, end=0)
                end_burst_pts = burst_pts.find(end_time, end=0)
                pts = pts[:end_pts]
                burst_pts = burst_pts[:end_burst_pts]
                self.pars.shrunk = True
        elif hasattr(self.pars, 'shrunk'):
            # in case period grows back reset end time *one time*
            target.model.set(tdata=[0,t[-1]+self.pars.shrink_end_time_amount])
            del self.pars.shrunk
        self.pars.burst_coord_pts = pts
        self.pars.burst_pts_resampled = burst_pts
        # to establish period, find min on other side of active phase
        if min_t < self.results.burst_est.spike_ts[0]:
            # look to the right
            start_t = self.results.burst_est.spike_ts[-1]
            start_ix = pts.find(start_t, end=1)
            other_min_ix = np.argmin(x[start_ix:])
            other_min_t = t[start_ix+other_min_ix]
            other_min_val = x[start_ix+other_min_ix]
        else:
            # look to the left
            start_t = self.results.burst_est.spike_ts[0]
            start_ix = pts.find(start_t, end=0)
            other_min_ix = np.argmin(x[:start_ix])
            other_min_t = t[other_min_ix]
            other_min_val = x[other_min_ix]
        self.results.period = abs(other_min_t - min_t)
        self.results.period_val_error = other_min_val - min_val


class get_burst_spikes(ql_feature_node):
    """Requires a get_spike_data and get_spike_model instance to be
    the only sub-features (supplied as a dict with keys 'is_spike_data'
    and 'is_spike_model').
    """
    def _local_init(self):
        assert len(self.subfeatures) == 2
        assert remain(self.subfeatures.keys(),
                      ['is_spike_data', 'is_spike_model']) == []

    def postprocess_ref_traj(self):
        # get precise spike times and record in self.results.ref_spike_times
        self.pars.ref_spike_times, self.pars.ref_spike_vals = \
              self._eval(self.ref_traj, self.super_pars.ref_burst_est,
                                 self.subfeatures['is_spike_data'])

    def evaluate(self, target):
        self.results.spike_times, self.results.spike_vals = \
               self._eval(target.test_traj, self.super_results.burst_est,
                                     self.subfeatures['is_spike_model'])
        # satisfied if all spikes determined correctly
        return len(self.results.spike_times) == \
               len(self.super_results.burst_est.spike_ixs)


    def _eval(self, traj, burst_est, is_spike):
        # isn't the next line redundant?
        is_spike.super_pars = copy.copy(self.pars)
        spike_times = []
        spike_vals = []
        satisfied = True
        for spike_num, spike_ix in enumerate(burst_est.spike_ixs):
            if self.pars.verbose_level > 0:
                print("\n Starting spike", spike_num+1)
            is_spike.super_pars.burst_coord = self.super_pars.burst_coord
            # step back 20% of estimated period
            try:
                is_spike.pars.width_tol = burst_est.ISIs[spike_num]*.8
            except IndexError:
                # one fewer ISI than spike, so just assume last one is about
                # the same
                is_spike.pars.width_tol = burst_est.ISIs[spike_num-1]*.8
            is_spike.pars.tlo = burst_est.t[spike_ix] - \
                    is_spike.pars.width_tol #/ 2.
            if self.pars.verbose_level > 0:
                print("new tlo =", is_spike.pars.tlo)
            # would prefer to work this out self-consistently...
            #is_spike.pars.fit_width_max = ?
            new_sat = is_spike(traj)
            satisfied = satisfied and new_sat
            # make recorded spike time in global time coordinates
            if new_sat:
                spike_times.append(is_spike.results.spike_time)
                spike_vals.append(is_spike.results.spike_val)
            if self.pars.verbose_level > 0:
                print("Spike times:", spike_times)
        return spike_times, spike_vals


class get_burst_peak_env(qt_feature_leaf):
    """Requires tol and num_samples parameters.
    """
    def _local_init(self):
        self.metric = metric_L2()
        self.metric_len = self.pars.num_samples

    def postprocess_ref_traj(self):
        # should really use quadratic fit to get un-biased peaks
        peak_vals = self.super_pars.ref_spike_vals
        peak_t = self.super_pars.ref_spike_times
        self.ref_traj = numeric_to_traj([peak_vals], 'peak_envelope',
                                        self.super_pars.burst_coord, peak_t,
                                        self.super_pars.ref_burst_pts_resampled.indepvarname,
                                        discrete=False)
        # discrete option false yields error if only one spike found, but error is cryptic!

        if len(peak_t) > 1:
            ref_env_ts = np.linspace(peak_t[0], peak_t[-1],
                                            self.pars.num_samples)
        else:
            ref_env_ts = np.array(peak_t)
        self.pars.ref_peak_vals = self.ref_traj(ref_env_ts,
                                   self.super_pars.burst_coord)[0]

    def evaluate(self, target):
        # ignore target
        dc_offset = self.super_results.baseline_V
        # min and max events in model mean that these are recorded
        # accurately in the pointsets already
        peak_vals = self.super_results.spike_vals - dc_offset
        peak_t = self.super_results.spike_times
        self.results.burst_peak_env = numeric_to_traj([peak_vals],
                                                      'peak_envelope',
                                        self.super_pars.burst_coord, peak_t,
                                self.super_pars.burst_pts_resampled.indepvarname,
                                discrete=False)
#        burst_est = self.super_results.burst_est
#        call_args = {}
#        try:
#            call_args['noise_floor'] = is_spike.pars.noise_tol
#        except AttributeError:
#            pass
#        try:
#            call_args['depvar'] = self.super_pars.burst_coord
#        except AttributeError:
#            pass
#        try:
#            call_args['tol'] = 1.1*burst_est.std_ISI/burst_est.mean_ISI
#        except AttributeError:
#            pass
#        call_args['make_traj'] = False
#        call_args['spest'] = burst_est
#        env = spike_envelope(burst_est.pts, burst_est.mean_ISI,
#                             **call_args)
        test_env_ts = np.linspace(peak_t[0], peak_t[-1], self.pars.num_samples)
        return self.metric(self.results.burst_peak_env(test_env_ts,
                                                   self.super_pars.burst_coord),
                           self.super_pars.ref_peak_vals) < self.pars.tol


class get_burst_trough_env(qt_feature_leaf):
    """Requires tol and num_samples parameters.
    """
    def _local_init(self):
        self.metric = metric_L2()
        self.metric_len = self.pars.num_samples

    def postprocess_ref_traj(self):
        burst_pts = self.super_pars.ref_burst_pts_resampled
        burst_est = self.super_pars.ref_burst_est
        vals = burst_pts[self.super_pars.burst_coord]
        inter_spike_ixs = [(burst_est.spike_ixs[i-1],
                            burst_est.spike_ixs[i]) \
                           for i in range(1, len(burst_est.spike_ixs))]
        # should really use quadratic fit to get an un-biased minimum
        trough_ixs = [np.argmin(vals[ix_lo:ix_hi])+ix_lo for ix_lo, ix_hi in \
                      inter_spike_ixs]
        trough_vals = [vals[i] for i in trough_ixs]
        trough_t = [burst_pts.indepvararray[i] for i in trough_ixs]
        self.ref_traj = numeric_to_traj([trough_vals], 'trough_envelope',
                                        self.super_pars.burst_coord, trough_t,
                                        burst_pts.indepvarname, discrete=False)
        ref_env_ts = np.linspace(trough_t[0], trough_t[-1],
                                            self.pars.num_samples)
        self.pars.ref_trough_vals = self.ref_traj(ref_env_ts,
                                  self.super_pars.burst_coord)

    def evaluate(self, target):
        # ignore target
        dc_offset = self.super_results.baseline_V
        burst_pts = self.super_pars.burst_coord_pts
        burst_est = self.super_results.burst_est
        vals = burst_pts[self.super_pars.burst_coord]
        ts = self.super_results.spike_times
        spike_ixs = []
        for t in ts:
            tix = burst_pts.find(t, end=0)
            spike_ixs.append(tix)
        inter_spike_ixs = [(spike_ixs[i-1],
                            spike_ixs[i]) \
                           for i in range(1, len(ts))]
        # min and max events in model mean that these are recorded
        # accurately in the pointsets already
        trough_ixs = [np.argmin(vals[ix_lo:ix_hi])+ix_lo for ix_lo, ix_hi in \
                      inter_spike_ixs]
        trough_vals = [vals[i] - dc_offset for i in trough_ixs]
        # use self.pars.trough_t for isi mid-point times
        trough_t = [burst_pts.indepvararray[i] for i in trough_ixs]
        self.results.burst_trough_env = numeric_to_traj([trough_vals],
                                                        'trough_envelope',
                                        self.super_pars.burst_coord,
                                        trough_t,
                                        burst_pts.indepvarname, discrete=False)
        test_env_ts = np.linspace(trough_t[0], trough_t[-1],
                                   self.pars.num_samples)
        self.results.trough_t = trough_t
        return self.metric(self.results.burst_trough_env(test_env_ts,
                                                self.super_pars.burst_coord),
                           self.super_pars.ref_trough_vals) < self.pars.tol


class get_burst_isi_env(qt_feature_leaf):
    """Requires tol and num_samples parameters.
    """
    def _local_init(self):
        self.metric = metric_L2()
        self.metric_len = self.pars.num_samples

    def postprocess_ref_traj(self):
        burst_pts = self.super_pars.ref_burst_pts_resampled
        ts = burst_pts.indepvararray
        burst_est = self.super_pars.ref_burst_est
        # find approximate (integer) mid-point index between spikes
        mid_isi_ixs = [int(0.5*(burst_est.spike_ixs[i-1]+burst_est.spike_ixs[i])) \
                           for i in range(1, len(burst_est.spike_ixs))]
        isi_t = [ts[i] for i in mid_isi_ixs]
        isi_vals = [ts[burst_est.spike_ixs[i]]-ts[burst_est.spike_ixs[i-1]] for \
                    i in range(1, len(burst_est.spike_ixs))]
        self.ref_traj = numeric_to_traj([isi_vals], 'isi_envelope',
                                        self.super_pars.burst_coord, isi_t,
                                        burst_pts.indepvarname, discrete=False)
        ref_env_ts = np.linspace(isi_t[0], isi_t[-1],
                                            self.pars.num_samples)
        self.pars.ref_isis = self.ref_traj(ref_env_ts,
                                   self.super_pars.burst_coord)

    def evaluate(self, target):
        # ignore target
        ts = self.super_results.spike_times
        tname = self.super_pars.burst_coord_pts.indepvarname
        isi_vals = [ts[i]-ts[i-1] for i in range(1, len(ts))]
        self.results.burst_isi_env = numeric_to_traj([isi_vals],
                                                        'isi_envelope',
                                        self.super_pars.burst_coord,
                                        self.super_results.trough_t,
                                        tname, discrete=False)
        test_env_ts = np.linspace(self.super_results.trough_t[0],
                                   self.super_results.trough_t[-1],
                                   self.pars.num_samples)
        return self.metric(self.results.burst_isi_env(test_env_ts,
                                                   self.super_pars.burst_coord),
                           self.pars.ref_isis) < self.pars.tol


class get_burst_upsweep(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_L2()
        self.metric_len = len(self.pars.t_offs)

    def postprocess_ref_traj(self):
        vname = self.super_pars.burst_coord
        ts = [self.super_pars.ref_spike_times[0] - toff for \
              toff in self.pars.t_offs]
        self.pars.ref_upsweep_V = np.array([self.ref_traj(t, vname) for \
                                             t in ts])

    def evaluate(self, target):
        dc_offset = self.super_results.baseline_V
        vname = self.super_pars.burst_coord
        all_pts = self.super_pars.burst_coord_pts
        vals = []
        for toff in self.pars.t_offs:
            target_t = self.super_results.spike_times[0] - toff
            if target_t < all_pts.indepvararray[0]:
                # out of range - return penalty
                self.metric.results = 5000*np.ones((self.metric_len,),float)
                return False
            tix = all_pts.find(target_t, end=0)
            new_var = make_poly_interpolated_curve(all_pts[tix-5:tix+5],
                                                   vname, target.model)
            vals.append(new_var(target_t))
        self.results.upsweep_V = np.array(vals) - dc_offset
        return self.metric(self.results.upsweep_V, \
                               self.pars.ref_upsweep_V) < self.pars.tol


class get_burst_downsweep(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_L2()
        self.metric_len = len(self.pars.t_offs)

    def postprocess_ref_traj(self):
        vname = self.super_pars.burst_coord
        ts = [self.super_pars.ref_spike_times[-1] + toff for \
              toff in self.pars.t_offs]
        self.pars.ref_downsweep_V = np.array([self.ref_traj(t, vname) for \
                                               t in ts])

    def evaluate(self, target):
        dc_offset = self.super_results.baseline_V
        vname = self.super_pars.burst_coord
        all_pts = self.super_pars.burst_coord_pts
        vals = []
        for toff in self.pars.t_offs:
            target_t = self.super_results.spike_times[-1] + toff
            if target_t > all_pts.indepvararray[-1]:
                # out of range - return penalty
                self.metric.results = 5000*np.ones((self.metric_len,),float)
                return False
            tix = all_pts.find(target_t, end=0)
            new_var = make_poly_interpolated_curve(all_pts[tix-5:tix+5],
                                                   vname, target.model)
            vals.append(new_var(target_t))
        self.results.downsweep_V = np.array(vals) - dc_offset
        return self.metric(self.results.downsweep_V,
                               self.pars.ref_downsweep_V) < self.pars.tol

class get_burst_num_spikes(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_float()
        self.metric_len = 1

    def evaluate(self, target):
        return self.metric(np.array(len(self.super_results.spike_times)),
                           np.array(len(self.super_pars.ref_spike_times))) == 0


class get_burst_period_info(qt_feature_leaf):
    def _local_init(self):
        self.metric = metric_weighted_L2()
        self.metric_len = 2
        # strongly penalize lack of periodicity
        self.metric.weights = np.array([1., 1000.])

    def evaluate(self, target):
        return self.metric(np.array([self.super_results.period,
                                  self.super_results.period_val_error]),
                           np.array([self.super_pars.ref_period,
                                  0.])) \
               < self.pars.tol

# --------------------------------------------


class spike_metric(metric):
    """Measures the distance between spike time and height,
    using an inherent weighting of height suited to neural voltage
    signals (0.05 of time distance)."""
    def __call__(self, sp1, sp2):
        # weight 'v' component down b/c 't' values are on a different scale
        self.results = np.array(sp1-sp2).flatten()*np.array([1,0.05])
        return np.linalg.norm(self.results)

class spike_feature(qt_feature_node):
    """pars keys: tol"""
    def _local_init(self):
        self.metric_len = 2
        self.metric = spike_metric()

    def evaluate(self, target):
        # traj will be a post-processed v trajectory ->
        #   spike time and height values
        return self.metric(target.test_traj.sample(), self.ref_traj.sample()) \
                       < self.pars.tol


class geom_feature(qt_feature_leaf):
    """Measures the residual between two 1D parameterized geometric
    curves (given as Trajectory objects).
    """
    def _local_init(self):
        self.metric = metric_L2()
        self.metric_len = len(self.pars.tmesh)

    def evaluate(self, target):
        # resample ref_traj to the tmesh we want
        return self.metric(target.test_traj(self.pars.tmesh,
                                         coords=[self.pars.depvar]),
                           self.ref_traj(self.pars.tmesh,
                                         coords=[self.pars.depvar])) < self.pars.tol

# ------------------------------------------------------------------

class estimate_spiking(object):
    """Estimate pattern of spiking in tonic or burst patterns."""

    def __init__(self, x, t, filt_coeffs, sense='up'):
        """Pass only 1D pointset.
        If spikes are in the "positive" direction of the variable,
        use sense='up', else use 'down'."""
        self.sense = sense
        self.b, self.a = filt_coeffs
        x_filt = filtfilt(self.b, self.a, x)
        self.x_just_filt = x_filt
        self.t = t
        max_x = max(x_filt)
        # retain only values larger than 10% of max to estimate burst
        # envelope
        x_filt_mask = np.asarray(x_filt>(0.1*max_x),int)
        burst_off_ix = len(t) - np.argmax(x_filt_mask[::-1])
        burst_on_ix = np.argmax(x_filt_mask)
        self.burst_on = (burst_on_ix, t[burst_on_ix])
        self.burst_off = (burst_off_ix, t[burst_off_ix])
        self.burst_duration = t[burst_off_ix] - t[burst_on_ix]
        # retain only values larger than 25% of max for actual spikes
        # FAILING: temp switch off
        x_filt_th = x_filt_mask #np.asarray(x_filt>(0.25*max_x),int)*x_filt
        # find each spike by group of positive values
        # eliminating each afterwards (separated by zeros)
        spike_ixs = []
        done = False
        n = 0  # for safety
        while not done:
            # find next group centre and eliminate it
            x_filt_th = self.eliminate_group(x_filt_th, spike_ixs)
            n += 1
            # no groups left to eliminate?
            done = max(x_filt_th) == 0 or n > 100
        spike_ixs.sort()
        self.spike_ixs = spike_ixs
        self.spike_ts = t[spike_ixs]
        self.ISIs = [self.spike_ts[i]-self.spike_ts[i-1] for \
                     i in range(1, len(spike_ixs))]
        self.mean_ISI = np.mean(self.ISIs)
        self.std_ISI = np.std(self.ISIs)
        self.num_spikes = len(spike_ixs)

    def eliminate_group(self, xf, spike_ixs):
        centre_ix = np.argmax(xf)
#        print "Current spike_ixs", spike_ixs
#        print "eliminating group at t = ", self.t[centre_ix]
        # forward half-group
        end_ix = np.argmin(xf[centre_ix:])+centre_ix
        # backward half-group
        start_ix = centre_ix-np.argmin(xf[:centre_ix+1][::-1])
        # nullify values in range!
        xf[start_ix:end_ix]=0
#        print start_ix, end_ix, xf[start_ix:end_ix]
        if self.sense == 'up':
            # x will be rising to peak, so track forwards until
            # xfilt makes zero crossing and becomes negative
            new = centre_ix+np.argmax(self.x_just_filt[centre_ix:]<0)
            if new not in spike_ixs:
                spike_ixs.append(new)
        else:
            # track backwards
            new = centre_ix-np.argmin(self.x_just_filt[:centre_ix+1]>0)
            if new not in spike_ixs:
                spike_ixs.append(new)
        return xf


class spike_envelope(object):
    """Find an amplitude envelope over a smooth 1D signal that features
    roughly periodic spikes. Input is a 1D parameterized pointset
    and the approximate period. An optional input is the tolerance (fraction)
    for finding spikes around the period (measuring uncertainty in the
    period) -- default 0.2 (20% of the period).

    Optional start_t sets where to orient the search in the independent
    variable -- default None (start at the highest point of the signal).
    It *must* be a value that is present in the independent variable
    array of the given points argument.

    Optional noise_floor sets minimum signal amplitude considered to
    be a peak (default 0 means non-noisy data assumed).

    Outside of spike times +/- tol, envelope curve will be defined as
    amplitude zero.

    adjust_rate is a fraction < 1 specifying the %age change of spike
    search interval (a.k.a. 'period'). default 0.1.

    make_traj option can be used to avoid costly creation of a Trajectory
    object representing the envelope curve, if unneeded (default True).

    When less is known in advance about the regularity or other properties
    of the spikes, pre-process using estimate_spiking() and pass the
    result as the optional argument spest.
    """
    def __init__(self, pts, per, tol=0.2, start_t=None,
                 noise_floor=0, depvar=None, adjust_rate=0.1,
                 make_traj=True, spest=None):
        try:
            self.tvals = pts.indepvararray
        except:
            raise TypeError("Parameterized pointset required")
        self.pts = pts  # store this to take advantage of index search
        if depvar is None:
            assert pts.dimension == 1
            depvar = pts.coordnames[0]
            self.vals = pts[depvar]
        else:
            try:
                self.vals = pts[depvar]
            except PyDSTool_KeyError:
                raise ValueError("Invalid dependent variable name")
        self.numpoints = len(self.vals)
        assert self.numpoints > 1
        self.per = per
        self.noise_floor = noise_floor
        assert tol < 1 and tol > 0
        self.tol = tol
        # assume that the maximum is a spike, so is a reliable
        # phase reference
        if start_t is None:
            self.start_ix = np.argmax(self.vals)
            self.start_t = self.tvals[self.start_ix]
        else:
            assert start_t in self.tvals
            self.start_t = start_t
            self.start_ix = pts.find(start_t)
        assert adjust_rate > 0 and adjust_rate < 1
        adjust_rate_up = 1+adjust_rate
        adjust_rate_down = 1-adjust_rate
        spike_ixs_lo = []
        spike_ixs_hi = []
        start_t = self.start_t
        per = self.per
        tol = self.tol
        done_dir = False
        while not done_dir:
#            print "\n======================\nDir +1"
            res = self.find_spike_ixs_dir(1, per=per, start_t=start_t,
                                          tol=tol)
            spike_ixs_hi.extend(res['spike_ixs'])
            if res['success']:
                done_dir = True
            else:
                if res['problem_dir'] == 'lo':
                    per = per * adjust_rate_down
                elif res['problem_dir'] == 'hi':
                    per = per * adjust_rate_up
                rat = per/self.per
                if rat > 2 or rat < 0.5:
                    # per is too far off, must be no more spikes
                    done_dir = True
                    continue
#                print "failed:", res['problem_dir'], res['restart_t']
                start_t = res['restart_t']
                #tol *= 1.2
        start_t = self.start_t
        per = self.per
        tol = self.tol
        done_dir = False
        while not done_dir:
#            print "\n======================\nDir -1"
            res = self.find_spike_ixs_dir(-1, per=per, start_t=start_t,
                                          tol=tol)
            spike_ixs_lo.extend(res['spike_ixs'])
            if res['success']:
                done_dir = True
            else:
                if res['problem_dir'] == 'lo':
                    per = per * adjust_rate_up
                elif res['problem_dir'] == 'hi':
                    per = per * adjust_rate_down
                rat = per/self.per
                if rat > 2 or rat < 0.5:
                    # per is too far off, must be no more spikes
                    done_dir = True
                    continue
#                print "failed:", res['problem_dir'], res['restart_t']
                start_t = res['restart_t']
                #tol *= 1.2
        spike_ixs_lo.sort()
        spike_ixs_hi.sort()
        ts = self.pts.indepvararray
        self.spike_ixs = np.array(spike_ixs_lo+spike_ixs_hi)
        self.spike_vals = np.array([self.vals[i] for i in self.spike_ixs])
        nearest_per_ix_lo = self.pts.find(ts[self.spike_ixs[0]]-per*tol, end=1)
        nearest_per_ix_hi = self.pts.find(ts[self.spike_ixs[-1]]+per*tol, end=0)
        # fill in rest of curve (outside of +/- period tolerance) with zeros
        if spike_ixs_lo == [] or spike_ixs_lo[0]==0:
            # spike right at t=0
            prepend_v = []
            prepend_t = []
        elif nearest_per_ix_lo==0:
            # -per*tol reaches to t=0
            prepend_v = [self.spike_vals[0]]
            prepend_t = [ts[0]]
        else:
            # add zeros up to -per*tol of first spike
            prepend_v = [0,0,self.spike_vals[0]]
            prepend_t = [ts[0],
                         ts[nearest_per_ix_lo-1],
                         ts[nearest_per_ix_lo]]
        if spike_ixs_hi[-1]==self.numpoints-1:
            postpend_v = []
            postpend_t = []
        elif nearest_per_ix_hi==self.numpoints-1:
            postpend_v = [self.spike_vals[-1]]
            postpend_t = [ts[-1]]
        else:
            postpend_v = [self.spike_vals[-1],0,0]
            postpend_t = [ts[nearest_per_ix_hi],
                          ts[nearest_per_ix_hi+1],
                          ts[-1]]
        curve_vals = np.array(prepend_v+[self.vals[i] for i in \
                                        self.spike_ixs]+postpend_v)
        curve_t = prepend_t + list(ts[self.spike_ixs]) \
                   + postpend_t
        #zeros_ixs_lo = xrange(0,spike_ixs_lo[0])
        #zeros_ixs_hi = xrange(spike_ixs_hi[-1],self.numpoints)
        if make_traj:
            self.envelope = numeric_to_traj([curve_vals], 'envelope',
                                        depvar, curve_t,
                                        pts.indepvarname, discrete=False)

    def find_spike_ixs_dir(self, dir=1, per=None, start_t=None,
                           tol=None):
        """Use dir=-1 for backwards direction"""
        if start_t is None:
            t = self.start_t
        else:
            t = start_t
        if per is None:
            per = self.per
        if tol is None:
            tol = self.tol
        assert dir in [-1,1]
        if dir == 1:
            # only include starting index once!
            if t == self.start_t:
                spike_ixs = [self.start_ix]
            else:
                spike_ixs = []
        else:
            spike_ixs = []

        res = {'success': False, 'problem_dir': '', 'spike_ixs': [],
               'restart_t': None}
        done = False
        hit_end = False
        while not done and not hit_end:
            t += dir * per
            t_lo = t - per*tol
            t_hi = t + per*tol
#            print "\n******************* find:"
#            print "Search from t=", t, "existing spikes=", spike_ixs
#            print "per= ", per, "t_lo=", t_lo, "t_hi=", t_hi
            lo_ix = self.pts.find(t_lo, end=0)
            if lo_ix == -1:
                # hit end!
                lo_ix = 0
            hi_ix = self.pts.find(t_hi, end=1)
            # find() will not return vals > numpoints or < 0
            hit_end = lo_ix == 0 or hi_ix == self.numpoints
            if lo_ix == hi_ix:
                done = True
                continue
            else:
                max_ix = np.argmax(self.vals[lo_ix:hi_ix])
            # now ensure that time window was large enough to capture a true
            # extremum, and not just an endpoint extremum
            room_lo = lo_ix
            room_hi = self.numpoints-hi_ix
            look_lo = min((room_lo, 5))
            look_hi = min((room_hi, 5))
            if look_lo > 0 and max(self.vals[lo_ix-look_lo:lo_ix]) > \
               self.vals[lo_ix+max_ix]:
                # then wasn't a true max - must de/increase per
                # depending on current dir
                res['success'] = False
                res['problem_dir'] = "lo"
                if dir < 0:
                    res['spike_ixs'] = spike_ixs[::-1]
                else:
                    res['spike_ixs'] = spike_ixs[:]
                res['restart_t'] = t - dir*per
                return res
            if look_hi > 0 and max(self.vals[hi_ix:hi_ix+look_hi]) > \
               self.vals[lo_ix+max_ix]:
                # then wasn't a true max - must de/increase per
                # depending on current dir
                res['success'] = False
                res['problem_dir'] = "hi"
                if dir < 0:
                    res['spike_ixs'] = spike_ixs[::-1]
                else:
                    res['spike_ixs'] = spike_ixs[:]
                res['restart_t'] = t - dir*per
                return res
            if abs(self.vals[max_ix+lo_ix]-self.vals[lo_ix]) >= self.noise_floor:
                # need equals case at endpoint when lo_ix = 0 so LHS is zero but
                # maximum is on first index
                spike_ixs.append(max_ix+lo_ix)
            # else don't treat as a spike

        if dir < 0:
            res['spike_ixs'] = spike_ixs[::-1]
        else:
            res['spike_ixs'] = spike_ixs[:]
        res['success'] = True
        return res

