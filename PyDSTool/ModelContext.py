"""Features, conditions, model contexts and interfaces
   for hybrid dynamical systems and model estimation tasks.

   Robert Clewley, September 2007.
"""


import copy
import sys, traceback
import numpy as np

# PyDSTool imports
from . import Events, ModelSpec, Symbolic, Trajectory
from . import utils, common, parseUtils
from .errors import *

# --------------------
# public exports

_classes = ['ql_feature_node', 'ql_feature_leaf',
            'qt_feature_node', 'qt_feature_leaf',
            'binary_feature', 'always_feature',
            'feature', 'condition', 'context',
            'GeneratorInterface', 'ModelInterface',
            'extModelInterface', 'intModelInterface'
            ]

_functions = []

_constants = ['LARGE_PENALTY']

__all__ = _classes + _functions + _constants

# ----------------------------------------------------------------------------

LARGE_PENALTY = 2000.

class feature(object):
    """End users of concrete sub-classes provide (required) evaluate method and
    (optional) prepare, finish methods."""
    def __init__(self, name, description='', pars=None, ref_traj=None):
        self.name = name
        self.description = description
        if pars is None:
            self.pars = common.args()
        elif isinstance(pars, dict):
            self.pars = common.args(**pars)
        elif isinstance(pars, common.args):
            self.pars = pars
        else:
            raise PyDSTool_TypeError("Invalid type for pars argument")
        if 'verbose_level' not in self.pars:
            self.pars.verbose_level = 0
        if 'debug' not in self.pars:
            self.pars.debug = False
        # penalty used if an error occurs during residual calculation
        if 'penalty' not in self.pars:
            self.pars.penalty = LARGE_PENALTY
        self.ref_traj = ref_traj
        self.results = common.args()
        self.super_pars = common.args()
        self.super_results = common.args()
        # perform any sub-class specific initializations,
        # such as providing a metric (and its output array length)
        try:
            self._local_init()
        except AttributeError:
            pass
        self.subfeatures = []

    def __hash__(self):
        return hash((self.name,common.className(self)))

    def __eq__(self, other):
        try:
            res = self.name == other.name
        except AttributeError:
            return False
        if hasattr(self, 'subfeatures'):
            if hasattr(other, 'subfeatures'):
                res = res and self.subfeatures == other.subfeatures
            else:
                return False
        elif hasattr(other, 'subfeatures'):
            return False
        return res

    def __ne__(self, other):
        return not self == other

    def _find_idx(self):
        """Internal function for finding index in trajectory meshpoints
        at which containment first failed. Defaults to returning None and
        must be overridden by a class that has access to a mesh."""
        return None

    def __call__(self, target):
        try:
            self.prepare(target)
            satisfied = self.evaluate(target)
        except KeyboardInterrupt:
            raise
        except:
            display_error = self.pars.verbose_level > 0 or self.pars.debug
            if display_error:
                exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
                print("******************************************")
                print("Problem evaluating feature:" + self.name)
                print("  %s %s" % (exceptionType, exceptionValue))
                for line in traceback.format_exc().splitlines()[-12:-1]:
                    print("   " + line)
                print("  originally on line:%d" % traceback.tb_lineno(exceptionTraceback))
                if self.pars.debug:   #and self.pars.verbose_level > 1:
                    raise
                else:
                    print("(Proceeding as 'unsatisfied')\n")
            satisfied = False
            if hasattr(self, 'metric'):
                self.metric.results = self.pars.penalty * \
                                  np.ones((self.metric_len,), float)
            for sf in self.subfeatures:
                if hasattr(sf, 'metric'):
                    sf.metric.results = self.pars.penalty * \
                      np.ones((sf.metric_len,), float)
        if satisfied:
            self.finish(target)
        self.results.satisfied = satisfied
        return satisfied

    def set_ref_traj(self, ref_traj):
        """May or may not be used by the feature. If not used,
        it will be ignored."""
        raise NotImplementedError("Override in concrete sub-class")

    def evaluate(self, target):
        raise NotImplementedError("Override in concrete sub-class")

    def prepare(self, target):
        """Operations to prepare for testing (optional).
        Override in concrete sub-class if desired"""
        pass

    def finish(self, target):
        """Operations to complete only if evaluate was True (optional).
        Override in concrete sub-class if desired"""
        pass

    def info(self):
        utils.info(self, self.name)

    def postprocess_ref_traj(self):
        """User-definable by overriding in a sub-class"""
        pass

    def validate(self):
        """NOT YET IMPLEMENTED. Would test for reachability of a feature?
        This is not easy and may be unnecessary!"""
        return True

    def reset_metric(self):
        try:
            self.metric.results = None
        except AttributeError:
            # no metric for this feature
            pass


class feature_leaf(feature):
    """Abstract super-class for feature leaf nodes.
    """
    def set_ref_traj(self, ref_traj):
        self.ref_traj = ref_traj
        self.postprocess_ref_traj()

    def _residual_info(self, feats, sizes):
        """Update feats and sizes lists in place with metric info, if any.
        """
        try:
            sizes.append(self.metric_len)
            # touch self.metric to ensure it exists!
            self.metric
        except AttributeError:
            # no metric present
            return
        else:
            feats.append(self)

    def __str__(self):
        return "Feature %s"%self.name

    __repr__ = __str__


class feature_node(feature):
    """Abstract super-class for feature regular nodes (supporting sub-features).
    """
    def __init__(self, name, description='', pars=None,
                 ref_traj=None, subfeatures=None):
        """Sub-features is an ordered sequence of QL or QT feature instances
        which are (by default) evaluated in this order on a trajectory segment
        unless evaluation method is overridden.

        For more sophisticated use of sub-features, they should be provided as
        a dictionary mapping feature names to the feature instance.
        """
        feature.__init__(self, name, description, pars, ref_traj)
        if subfeatures is None:
            self.subfeatures = ()
            self._namemap = {}
        elif isinstance(subfeatures, (list, tuple)):
            for sf in subfeatures:
                assert isinstance(sf, feature), \
                       "Only define quantitative or qualitative features"
            self.subfeatures = subfeatures
            self._namemap = dict(zip([sf.name for sf in subfeatures],
                                     subfeatures))
        elif isinstance(subfeatures, dict):
            for sfname, sf in subfeatures.items():
                assert isinstance(sf, feature), \
                       "Only define quantitative or qualitative features"
            self.subfeatures = subfeatures
            self._namemap = subfeatures
        else:
            raise TypeError("List or dictionary of sub-features expected")


    def _residual_info(self, feats, sizes):
        """Update feats and sizes lists in place with metric info, if any.
        """
        try:
            sizes.append(self.metric_len)
            # touch self.metric to ensure it exists!
            self.metric
        except AttributeError:
            # no metric present
            pass
        else:
            feats.append(self)
        # continue gathering from sub-features
        for sf in self._namemap.values():
            sf._residual_info(feats, sizes)

    def __str__(self):
        s = "Feature %s "%self.name
        if len(list(self._namemap.keys())) > 0:
            s += "- " + str(list(self._namemap.keys()))
        return s

    __repr__ = __str__

    def __getitem__(self, featname):
        """Return named sub-feature"""
        return self._namemap[featname]

    def propagate_verbosity(self, sf):
        # subfeatures inherit one lower level of verbosity
        if 'verbose_level' in self.pars:
            v = max([0,self.pars.verbose_level - 1])
            if isinstance(sf, common._seq_types):
                for sf_i in sf:
                    sf_i.pars.verbose_level = v
            else:
                sf.pars.verbose_level = v
        if 'debug' in self.pars:
            if isinstance(sf, common._seq_types):
                for sf_i in sf:
                    sf_i.pars.debug = self.pars.debug
            else:
                sf.pars.debug = self.pars.debug

    def evaluate(self, target):
        """Default method: evaluate sub-features in order (assumes they
        are stored as a list).

        Can override with more sophisticated method (e.g. for use with a
        dictionary of sub-features).
        """
        # initial value
        satisfied = True
        # this loop only works if subfeatures is a list
        # (must retain correct order for this list so don't use _namemap.values())
        for sf in self.subfeatures:
            try:
                self.propagate_verbosity(sf)
            except KeyboardInterrupt:
                raise
            except:
                if not isinstance(self.subfeatures, common._seq_types):
                    raise TypeError("You must override the evaluate method for "
                                    "dictionary-based sub-features")
                else:
                    raise
            sf.super_pars.update(self.pars)
            sf.super_results.update(self.results)
            sf.reset_metric()
            if self.pars.verbose_level > 1:
                print("feature_node.evaluate: sf=%r" % sf)
            error_raised = False
            try:
                new_result = sf(target)
            except KeyboardInterrupt:
                raise
            except:
                # catch errors in prepare or finish (evaluate was trapped
                # in __call__)
                new_result = False
                error_raised = True
                if sf.pars.debug: # and sf.pars.verbose_level > 1:
                    raise
            # have to compute new separately to ensure sf computes its results
            # for potential use by a residual function
            satisfied = satisfied and new_result
            if error_raised:
                print(" ... error raised")
                if hasattr(self, 'metric'):
                    # kludgy penalty function in lieu of something smarter
                    if sf.metric.results is None:
                        sf.metric.results = self.pars.penalty * \
                          np.ones((sf.metric_len,),float)
            else:
                self.results.update(sf.results)
        return satisfied

    def set_ref_traj(self, ref_traj):
        """May or may not be used by the feature. If not used, it will be
        ignored."""
        self.ref_traj = ref_traj
        self.postprocess_ref_traj()
        if isinstance(self.subfeatures, dict):
            sfs = list(self.subfeatures.values())
        else:
            sfs = self.subfeatures
        for sf in sfs:
            self.propagate_verbosity(sf)
            sf.super_pars.update(self.pars)
            sf.super_results.update(self.results)
            sf.set_ref_traj(ref_traj)
            self.pars.update(sf.pars)


class ql_feature_leaf(feature_leaf):
    """Qualitative feature (leaf node).
    Add description to note assumptions used for defining feature.

    input: a trajectory segment
    output: a vector of boolean-valued event detections (non-terminal events or even
      non-linked python events) or other function tests (e.g. existence of a fixed point)
      stored in a list.
    """


class qt_feature_leaf(feature_leaf):
    """Quantitative feature (leaf node).
    Add description to note assumptions used for defining feature.

    input: a trajectory segment
    output: a vector of boolean-valued tolerance tests on the discrepancies between ideal and
      actual features defined by a list of function tests
      e.g. a test returns (residual of ideal-actual) < tolerance
    """


class ql_feature_node(feature_node):
    """Qualitative feature (regular node).
    Add description to note assumptions used for defining feature.

    input: a trajectory segment
    output: a vector of boolean-valued event detections (non-terminal events or even
      non-linked python events) or other function tests (e.g. existence of a fixed point)
      stored in a list.
    """


class qt_feature_node(feature_node):
    """Quantitative feature (regular node).
    Add description to note assumptions used for defining feature.

    input: a trajectory segment
    output: a vector of boolean-valued tolerance tests on the discrepancies between ideal and
      actual features defined by a list of function tests
      e.g. a test returns (residual of ideal-actual) < tolerance
    """

# -----------------------------------------------------------------------------


class condition(object):
    """Model context condition, made up of a boolean composition of wanted and
    unwanted features.
    This is specified by a dictionary of feature objects mapping to True
    (wanted feature) or False (unwanted feature).
    """

    def __init__(self, feature_composition_dict):
        # fcd maps feature objects to True (wanted feature) or
        # False (unwanted feature)
        self.namemap = {}
        try:
            for f, c in feature_composition_dict.items():
                assert isinstance(c, bool), \
                       "Feature composition dictionary requires boolean values"
                assert isinstance(f, (ql_feature_leaf, qt_feature_leaf,
                                      ql_feature_node, qt_feature_node)), \
                        "Only define quantitative or qualitative features"
                self.namemap[f.name] = f
        except AttributeError:
            raise TypeError("Dictionary of features to Booleans expected")
        self.fcd = feature_composition_dict
        self.results = common.args()

    __hash__ = None

    def __eq__(self, other):
        try:
            return self.namemap == other.namemap
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.namemap != other.namemap
        except AttributeError:
            return True

    def keys(self):
        return self.namemap.keys()

    def values(self):
        return self.namemap.values()

    def items(self):
        return self.namemap.items()

    def __getitem__(self, name):
        return self.namemap[name]

    def __contains__(self, name):
        return name in self.namemap

    def set_ref_traj(self, ref_traj):
        """Set reference trajectory for the features (if used, otherwise will
        be ignored or overridden in feature _local_init methods).
        """
        for f,c in self.fcd.items():
            f.set_ref_traj(ref_traj)

    def evaluate(self, target):
        """Apply conditions to trajectory segments
        and returns True if all are satisfied."""
        satisfied = True
        for f,c in self.fcd.items():
            # have to call new separately to ensure f calcs its residual
            new = f(target) == c
            satisfied = satisfied and new
            self.results[f.name] = f.results
        return satisfied

    __call__ = evaluate

    def __str__(self):
        s = "Condition "
        if len(list(self.namemap.keys())) > 0:
            s += "- " + str(list(self.namemap.keys()))
        return s

    __repr__ = __str__

    def _find_idx(self):
        min_ix = np.Inf
        for f in self.fcd.keys():
            f_ix = f._find_idx()
            if f_ix is not None and f_ix < min_ix:
                min_ix = f_ix
        if np.isfinite(min_ix):
            return min_ix
        else:
            return None

    def _residual_info(self):
        """Update metric information used for residual / objective function,
        from all sub-features."""
        #feats and sizes updated in place
        feats = []
        sizes = []
        for f in self.fcd.keys():
            f._residual_info(feats, sizes)
        return {'features': dict(zip(feats,sizes)),
                'total_size': sum(sizes)}

    def collate_results(self, result_name, merge_lists=False,
                        feature_names=None):
        res = []
        if feature_names is None:
            feature_list = list(self.fcd.keys())
        else:
            feature_list = [self.namemap[f] for f in feature_names]
        for f in feature_list:
            try:
                resval = getattr(f.results, result_name)
            except AttributeError:
                # no such result name
                continue
            else:
                if merge_lists and isinstance(resval, list):
                    res.extend(resval)
                else:
                    res.append(resval)
        return res


class context(object):
    """A collection of related model interfaces that apply to a model.
    interface_pairs are a list of ModelInterface instance (test) and class (ref) pairs,
    the latter to be instantiated on a model.

    Set the debug_mode attribute at any time, or as the optional argument at initializiation,
    to ensure that any exceptions that arise from interacting model interfaces and their
    features are fully passed back to the caller of the context containing them.
    """
    def __init__(self, interface_pairs, debug_mode=False):
        self.interfaces = dict(interface_pairs)
        self.debug_mode = debug_mode
        # Determine which qt features have metrics to use to make a
        # residual function. Keep multiple views of this data for efficient
        # access in different ways.
        metric_features = {}
        res_feature_list = []
        tot_size = 0
        for test_mi, ref_mi_class in self.interfaces.items():
            # list of suitable features for each test_mi
            metric_features[test_mi] = test_mi.conditions._residual_info()
            tot_size += metric_features[test_mi]['total_size']
            res_feature_list.extend([(test_mi, f) for f in \
                                     metric_features[test_mi]['features'].keys()])
        self.metric_features = metric_features
        self.res_feature_list = res_feature_list
        self.res_len = tot_size
        # instances cleared on each evaluation of a model
        self.ref_interface_instances = []
        # default weights are all 1, and set up weights dictionary
        self.reset_weights()

    def reset_weights(self, old_weights=None):
        """Reset weights to unity, unless old_weights array
        is given, in which case reset to that.
        """
        if old_weights is None:
            self.weights = np.ones(self.res_len, 'float')
        else:
            self.weights = old_weights
        self.weight_index_mapping = {}
        self.feat_weights = {}
        ix = 0
        for test_mi, feat_dict in self.metric_features.items():
            self.weight_index_mapping[test_mi] = {}
            for f, size in feat_dict['features'].items():
                self.weight_index_mapping[test_mi][f] = (ix, ix+size)
                # weights are constant for a given feature
                self.feat_weights[(test_mi, f)] = self.weights[ix]
                ix += size

    def set_single_feat_weights(self, feat, weight=1):
        """Set weights for a single feature given as an (interface, feature)
        pair, setting all others to zero."""
        wdict = {}
        for test_mi, feat_dict in self.metric_features.items():
            if test_mi != feat[0]:
                continue
            w = {}.fromkeys(feat_dict['features'].keys(), 0)
            if feat[1] in w:
                w[feat[1]] = weight
            wdict[test_mi] = w
        self.set_weights(wdict)

    def set_weights(self, weight_dict):
        """Update weights with a dictionary keyed by test_mi, whose values are
        either:
         (1) dicts of feature -> scalar weight.
         (2) a scalar which will apply to all features of that model interface
        Features and model interfaces must correspond to those declared for the
        context.
        """
        for test_mi, fs in weight_dict.items():
            try:
                flist = list(self.metric_features[test_mi]['features'].keys())
            except KeyError:
                raise AssertionError("Invalid test model interface")
            if isinstance(fs, common._num_types):
                feat_dict = {}.fromkeys(flist, fs)
            elif isinstance(fs, dict):
                assert np.alltrue([isinstance(w, common._num_types) for \
                            w in fs.values()]), "Invalid scalar weight"
                assert np.alltrue([f in flist for f in fs.keys()]), \
                       "Invalid features given for this test model interface"
                feat_dict = fs
            for f, w in feat_dict.items():
                self.feat_weights[(test_mi, f)] = w
                # update weight value
                start_ix, end_ix = self.weight_index_mapping[test_mi][f]
                self.weights[start_ix:end_ix] = w

    def show_res_info(self, resvec):
        """Show detail of feature -> residual mapping for a given residual
        vector."""
        i = 0
        for test_mi, feat_dict in self.metric_features.items():
            print("Test model interface:", test_mi)
            for f in feat_dict['features']:
                if self.feat_weights[(test_mi, f)] == 0:
                    continue
                ix0, ix1 = self.weight_index_mapping[test_mi][f]
                len_w = ix1-ix0
                f_str = "  "+f.name
                # '  unweighted:' is 13 chars long
                extra_space_w = " "*max([0, 13-len(f_str)])
                extra_space_unw = " "*max([0, len(f_str)-13])
                print(f_str + extra_space_w  + "%r" % resvec[i:i+len_w])
                try:
                    print("  unweighted:" + extra_space_unw +
                           "%r" % (resvec[i:i+len_w]/self.weights[ix0:ix1]))
                except ZeroDivisionError:
                    print("  (unweighted values unavailable)")
                i += len_w

    def _map_to_features(self, x):
        """Utility to map 1D array x onto the model interface's
        features with non-zero weights, returning a dictionary.

        x is assumed to have correct length.
        """
        out = {}
        i = 0
        for test_mi, feat_dict in self.metric_features.items():
            for f in feat_dict['features']:
                if self.feat_weights[(test_mi, f)] == 0:
                    continue
                ix0, ix1 = self.weight_index_mapping[test_mi][f]
                len_w = ix1-ix0
                try:
                    out[test_mi][f] = x[i:i+len_w]
                except KeyError:
                    out[test_mi] = {f: x[i:i+len_w]}
                i += len_w
        return out

    def evaluate(self, model):
        """Evaluate whole context on a model instance, returning a single
        Boolean.
        """
        result = True
        # typically, test_mi is an external interface (e.g., for data)
        # and ref_mi is an internal interface (e.g., for a model)
        self.ref_interface_instances = []
        for test_mi, ref_mi_class in common.sortedDictItems(self.interfaces, byvalue=False):
            # evaluate test_mi on model, via the associated ref_mi
            ref_mi = ref_mi_class(model)
            self.ref_interface_instances.append(ref_mi)
            try:
                new_result = test_mi(ref_mi)
            except KeyboardInterrupt:
                raise
            except:
                if self.debug_mode:
                    raise
                else:
                    print("******************************************")
                    print("Problem evaluating interface %s on %s" % (test_mi,ref_mi))
                    print("  %s %s" % (sys.exc_info()[0], sys.exc_info()[1]))
                    new_result = False
            # must create new_res first, to ensure all interfaces are
            # evaluated (to create their results for possible post-processing)
            result = result and new_result
        return result

    def residual(self, model, include_raw=False):
        """Evaluate whole context on a model instance, returning an array
        of residual error between quantitative features in the model trajectory
        and their target values.

        Residual array will be weighted if one was set. Any weights set to zero
        will cause those features to *not appear* in the residual.

        Provide include_raw=True argument to also return the raw, unweighted residual.
        (Mainly for internal use.)
        """
        # discard the boolean, just compute the residuals through the calls to
        # metric, and access them through the feature list
        self.evaluate(model)
        raw_residual = np.concatenate(tuple([mf[1].metric.results for \
                                     mf in self.res_feature_list]))
        residual = process_raw_residual(raw_residual, self.weights)
        if include_raw:
            return residual, raw_residual
        else:
            return residual


def process_raw_residual(raw_residual, weights):
    ixs = np.nonzero(weights)
    residual = (weights*raw_residual)[ixs]
    nan_ixs = np.where(np.asarray(np.isnan(residual),int))
    for ix in nan_ixs:
        residual[ix] = 100.
    return residual


# -----------------------------------------------------------------

class always_feature(ql_feature_leaf):
    """Use this for a single vector field model that uses discrete
    event mappings."""
    def evaluate(self, target):
        return True

class binary_feature(ql_feature_leaf):
    """Use this as a binary switch feature, toggled
    by a given variable name 'varname' that is supplied
    in the pars dict at initialization."""
    def evaluate(self, target):
        try:
            pts = target.test_traj.sample(coords=[self.pars.varname])
        except AttributeError:
            raise AttributeError("No variable name given for switch")
        except KeyboardInterrupt:
            raise
        except:
            print("Failed to find trajectory values for given variable name: %s"%self.pars.varname)
            raise
        self.results.output = pts
        return all(self.results.output==1)

    def _find_idx(self):
        if self.results.satisfied:
            # Trajectory satisfied contraint!
            return None
        res = self.results.output
        if res[0] == 1:
            adjusted_res = list((res - 1) != 0)
        else:
            if 1 not in res:
                # never goes to excited state so no index to return
                raise RuntimeError
            adjusted_res = list(res != 0)
        # find first index at which value is non-zero
        # should never raise ValueError because this method is
        # only run if there was a sign change found
        return adjusted_res.index(True)


# ----------------------------------------------------------------------------
## ModelInterface-related classes

# Private class
class dsInterface(object):
    """Generic and abstract interface class for dynamical systems."""
    # _getkeys shown for reference
    #    _getkeys = ['indepvariable', 'algparams', 'funcspec',
    #                   'diagnostics', 'variables',
    #                   'pars', 'inputs',
    #                   'eventstruct', 'globalt0', 'Rhs', 'Jacobian',
    #                   'JacobianP', 'MassMatrix', 'AuxVars']
    _setkeys = ['globalt0', 'tdata', 'pars', 'algparams', 'inputs']
    # the query key list is copied from Model.Model
    _querykeys = ['pars', 'parameters', 'events', 'submodels',
                  'ics', 'initialconditions', 'vars', 'variables',
                  'auxvariables', 'auxvars',  'vardomains', 'abseps']

    def get_test_traj(self):
        raise NotImplementedError("Only call this on a concrete sub-class")

    def query(self, querykey=''):
        return self.model.query(querykey)

    @property
    def name(self):
        try:
            return self.__name__
        except AttributeError:
            return self.__class__.__name__

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def __le__(self, other):
        return self.name <= other.name

    def __ge__(self, other):
        return self.name >= other.name

    def __hash__(self):
        return hash(self.name)


class GeneratorInterface(dsInterface):
    """Wrapper for Generator (for non-hybrid models) that shares similar API
    with ModelInterface for use in HybridModel objects."""

    def __init__(self, model, FScompatibleNames=None,
                 FScompatibleNamesInv=None):
        """model argument must be a Generator only"""
        self.model = model
        if FScompatibleNames is None:
            if self.model._FScompatibleNames is None:
                self.model._FScompatibleNames = symbolMapClass()
        else:
            self.model._FScompatibleNames = FScompatibleNames
        if FScompatibleNamesInv is None:
            if self.model._FScompatibleNamesInv is None:
                self.model._FScompatibleNamesInv = symbolMapClass()
        else:
            self.model._FScompatibleNamesInv = FScompatibleNamesInv
        self.eventstruct = Events.EventStruct()
        #self.diagnostics = common.Diagnostics()

    def get(self, key, ics=None, t0=0):
        # self.model is a Generator
        return self.model.get(key)

    def set(self, key, value, ics=None, t0=0):
        if key in self._setkeys:
            self.model.set(**{key:value})
        else:
            raise KeyError("Invalid or unsupported 'set' key: %s"%key)

    def Rhs(self, t, xdict, pdict):
        """Direct access to a generator's Rhs function."""
        return self.model.Rhs(t, xdict, pdict)

    def Jacobian(self, t, xdict, pdict, idict=None):
        """Direct access to a generator's Jacobian function (if defined)."""
        return self.model.Jacobian(t, xdict, pdict)

    def JacobianP(self, t, xdict, pdict):
        """Direct access to a generator's JacobianP function (if defined)."""
        return self.model.JacobianP(t, xdict, pdict)

    def MassMatrix(self, t, xdict, pdict):
        """Direct access to a generator's MassMatrix function (if defined)."""
        return self.model.MassMatrix(t, xdict, pdict)

    def AuxVars(self, t, xdict, pdict):
        """Direct access to a generator's auxiliary variables
        definition (if defined)."""
        return self.model.AuxVars(t, xdict, pdict)


class ModelInterface(dsInterface):
    """Model constraints expressed as a uni-directional interface to another
    formal system model:
    - Made up of conditions imposed on the other system's test trajectory.
    - Defines evaluation criteria for any view (e.g. from experimental data and
    test conditions).
    This is an abstract superclass for the 'internal' and 'external'
    sub-classes.
    """
    _trajname = 'test_iface_traj'

    def __init__(self):
        # Cache (3-tuple) for the ics, t0 and initiator last specified
        self._initiator_cache = None
        self.eventstruct = Events.EventStruct()
        #self.diagnostics = common.Diagnostics()

    def _get_initiator_cache(self, ics=None, t0=0):
        """
        Return initating object for given initial conditions and start time.
        For internal use only.
        "initiator" is the ModelInterface or GeneratorInterface object that
        will be chosen to begin an integration at the specified ICs and t0.
        """
        if self._initiator_cache is None:
            if ics is None:
                raise ValueError("Must pass initial conditions")
            else:
                initiator = self.model._findTrajInitiator(None, 0,
                                                t0, dict(ics))[0]
                self._initiator_cache = (ics, t0, initiator)
        else:
            if np.alltrue(self._initiator_cache[0] == ics) and \
               self._initiator_cache[1] == t0:
                ics, t0, initiator = self._initiator_cache
            elif ics is None:
                raise ValueError("Must pass initial conditions")
            else:
                # initial conditions or t0 don't match -- don't use cache
                initiator = self.model._findTrajInitiator(None, 0,
                                                t0, dict(ics))[0]
                self._initiator_cache = (ics, t0, initiator)
        return (ics, t0, initiator)

    def set(self, key, value, ics=None, t0=0):
        # set own conditions then propagate them down to sub-models
        self.model.set(**{key:value})
        ics, t0, initiator = self._get_initiator_cache(ics, t0)
#        print "ModelInterface.set %s = %s for %s (type %s)"%(str(key), str(value), initiator.model.name, type(initiator))
        if key in self._setkeys:
            initiator.set(key, value, ics, t0)
            initiator.model.set(**{key:value})
        else:
            raise KeyError("Invalid or unsupported 'set' key: %s"%key)

    def get(self, key, ics=None, t0=0):
        ics, t0, initiator = self._get_initiator_cache(ics, t0)
#        if key in self._getkeys:
        try:
            return initiator.get(key, ics, t0)
        except AttributeError:
            raise ValueError("Invalid or unsupported 'get' key: %s"%key)

    def Rhs(self, t, xdict, pdict):
        """Direct access to a generator's Rhs function."""
        ics_ignore, t_ignore, ds = self._get_initiator_cache(xdict, t)
        try:
            return self.model.Rhs(ds._supermodel.name, t, xdict, pdict)
        except AttributeError:
            # ds is not a MI with attribute _supermodel
            return self.model.Rhs(t, xdict, pdict)

    def Jacobian(self, t, xdict, pdict, idict=None):
        """Direct access to a generator's Jacobian function (if defined)."""
        ics_ignore, t_ignore, ds = self._get_initiator_cache(xdict, t)
        try:
            return self.model.Jacobian(ds._supermodel.name, t, xdict, pdict)
        except AttributeError:
            # ds is not a MI with attribute _supermodel
            return self.model.Jacobian(t, xdict, pdict)

    def JacobianP(self, t, xdict, pdict):
        """Direct access to a generator's JacobianP function (if defined)."""
        ics_ignore, t_ignore, ds = self._get_initiator_cache(xdict, t)
        try:
            return self.model.JacobianP(ds._supermodel.name, t, xdict, pdict)
        except AttributeError:
            # ds is not a MI with attribute _supermodel
            return self.model.JacobianP(t, xdict, pdict)

    def MassMatrix(self, t, xdict, pdict):
        """Direct access to a generator's MassMatrix function (if defined)."""
        ics_ignore, t_ignore, ds = self._get_initiator_cache(xdict, t)
        try:
            return self.model.MassMatrix(ds._supermodel.name, t, xdict, pdict)
        except AttributeError:
            # ds is not a MI with attribute _supermodel
            return self.model.MassMatrix(t, xdict, pdict)

    def AuxVars(self, t, xdict, pdict):
        """Direct access to a generator's auxiliary variables
        definition (if defined)."""
        ics_ignore, t_ignore, ds = self._get_initiator_cache(xdict, t)
        try:
            return ds.model.AuxVars(ds._supermodel.name, t, xdict, pdict)
        except AttributeError:
            # ds is not a MI with attribute _supermodel
            return ds.model.AuxVars(t, xdict, pdict)

    def setup_conditions(self, conditions, traj):
        # in case the conditions use this model trajectory as a reference
        # then provide them with it
        if conditions is None:
            self.conditions = None
        else:
            # really need to copy conditions?
            self.conditions = conditions
            try:
                self.conditions.set_ref_traj(traj)
            except AttributeError:
                raise

    def evaluate(self, target, force=False):
        """Evaluate interface consistency against target interface's trajectory
        on specified conditions.

        Optional force argument forces model to recompute its test trajectory,
        e.g. because of a known change in model parameters, ics, etc.
        """
        assert isinstance(target, ModelInterface), \
               "Target argument must be another interface object"
        if len(self.compatibleInterfaces) > 0 and \
             target.__class__.__name__ not in self.compatibleInterfaces \
                and not np.sometrue([common.compareBaseClass(target, ctype) \
                                     for ctype in self.compatibleInterfaces]):
            raise ValueError("Target interface not of compatible type")
        try:
            self.conditions
        except AttributeError:
            self.setup_conditions(conditions, self.get_test_traj())
        force = force or target.test_traj is None
        if force:
            # discard returned traj here (still accessible via target.test_traj)
            target.get_test_traj(force=force)
        self.prepare_conditions(target)
        try:
            result = self.conditions(target)
        except KeyError:
            raise KeyError("Condition evaluation failed")
        return result

    __call__ = evaluate

    def postprocess_test_traj(self, test_traj):
        """Called by another interface via get_test_traj.
        Default post-processing of test trajectory is the identity
         function, i.e. no processing.

        Override this method to return a processed version of the
         trajectory or perform other post-computation clean-up, e.g.
         prepare auxiliary feature/condition-related information based
         on end state of trajectory so that HybridModel can use it to
         decide on next hybrid state to switch to.
        """
        return test_traj

    def prepare_conditions(self, target):
        """Called automatically by evaluate. Override with user-defined access
        to the target interface or processing of trajectory after return of the
        target's test trajectory.
        """
        pass


class intModelInterface(ModelInterface):
    """Interface providing internal evaluation criteria between models.
    Optional conditions (object) argument used to specify these criteria.
    """
    def __init__(self, model, conditions=None, compatibleInterfaces=None,
                 test_traj=None):
        """Set model that generates test trajectories from which the dictionary
        of conditions can be imposed on a connected model.

        If no conditions are specified then the model is trivially wrapped in
          an "empty" interface.

        Optionally, a dummy test traj can be supplied in case of a dummy interface
        for a trivial condition test that does not need to evaluate a trajectory
        to determine the result.
        """
        ModelInterface.__init__(self)
        # Avoid circular import with Model module
        #assert isinstance(model, Model.Model), "Invalid Model object passed"
        self.model = model  #copy.deepcopy(model)  # ???
        #print "TEMP: (intModelInterface.__init__) -- should model be copied?"
        self.test_traj = test_traj  # may be initially a temporary value, None

    def ensure_has_test_traj(self):
        """Cause recomputation of test trajectory if not already present in
        model, returning boolean for whether recomputation was performed.
        """
        info = self.model.current_defining_args()
        # include any interface-specific changes that would be made
        new_args = self.initialize_model()
        if new_args is not None:
            info.update(new_args)
        if self.model.has_exact_traj(self._trajname, info):
            # this verifies that the traj that would be computed
            # already exists
            return False
        else:
            try:
                self.compute_traj(need_init=False, new_args=new_args)
            except KeyboardInterrupt:
                raise
            except:
                print("Model interface compute_traj method for model " + \
                      "'%s' failed" % self.model.name)
                print("%s %s" % (sys.exc_info()[0], sys.exc_info()[1]))
                return False
            else:
                return True

    def has_test_traj(self):
        return self.test_traj is not None

    def compute_traj(self, need_init=True, new_args=None):
        if need_init:
            new_args = self.initialize_model()
        if new_args is not None and len(new_args) > 0:
            old_info = self.model.current_defining_args()
            self.model.set(**new_args)
        else:
            old_info = None
        self.model.compute(trajname=self._trajname, force=True)
        if old_info is not None:
            # restore "standard" state
            self.model.set(**dict(old_info))

    def get_test_traj(self, force=False):
        """Called by another interface.
        Return model's test trajectory, using any post-processing
        specified by user-defined process_test_traj method.

        Use force option if model is known to have changed and trajectory
        needs refreshing.
        """
        if force and not isinstance(self.test_traj, Trajectory):
            self.compute_traj()
            recomputed = True
        else:
            recomputed = self.ensure_has_test_traj()
        if recomputed or self.test_traj is None:
            self.test_traj = \
                self.postprocess_test_traj(self.model[self._trajname])
        return self.test_traj

    def initialize_model(self):
        """Return any unique model-specific settings here, as a dictionary with
        keys that can include initial conditions, parameters, tdata, algorithmic
        parameters. Use the same keys that are suitable for a call to the
        Model.set method, i.e. 'pars', 'ics', 'tdata', and 'algparams'.

        Override in a sub-class to use. This method will be called
        before any trajectory computation of the model.
        """
        pass


class extModelInterface(ModelInterface):
    """Interface from a trajectory of numerical data and test conditions
    providing external evaluation criteria for a model.
    Optional conditions (object) argument used to specify these criteria.
    """
    def __init__(self, traj=None, conditions=None, compatibleInterfaces=None):
        ModelInterface.__init__(self)
        self.setup_conditions(conditions, traj)
        self.set_test_traj(traj)
        if compatibleInterfaces is None:
            self.compatibleInterfaces = []
        else:
            self.compatibleInterfaces = compatibleInterfaces

    def set_test_traj(self, traj):
        """Do any user-defined preprocessing to the given trajectory, including
        converting it to a different type of trajectory.
        """
        self.test_traj = self.postprocess_test_traj(traj)
        # propagate ref traj to conditions if they use it
        self.conditions.set_ref_traj(self.test_traj)

    def ensure_has_test_traj(self):
        """Never needs to recompute trajectory as it is fixed, so always
        returns False.
        """
        assert self.has_test_traj(), "Test trajectory missing"
        return False

    def has_test_traj(self):
        return isinstance(self.test_traj, Trajectory.Trajectory)

    def get_test_traj(self, force=False):
        """Called by another interface.
        Optional force argument is ignored for this class, as the
         trajectory is fixed."""
        return self.test_traj



# old code for internal interface
            ## currently both 'ics' and 'initialconditions' keys are valid, but want only one
            ## present here and we'll make sure it's stored as 'ics' here
            #assert len(common.intersect(["initialconditions", "ics"], conditions.keys())) == 1, \
                   #"Conditions must include one list of initial conditions"
            #if 'ics' not in conditions:
                #self.conditions['ics'] = conditions['initialconditions']
                #del self.conditions['initialconditions']



