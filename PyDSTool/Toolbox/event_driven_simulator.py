
from PyDSTool.common import *
from PyDSTool.errors import *
from PyDSTool.utils import remain, info
from PyDSTool.Points import Point, Pointset
from numpy import array, asarray, NaN, Inf, isfinite
from PyDSTool.matplotlib_import import gca, plt
from copy import copy

_classes = ['connection', 'node', 'simulator', 'composed_map1D',
            'map', 'map1D', 'map2D', 'identity_map', 'delay_map']

_functions = ['extract_digraph']

_instances = ['idmap']

__all__ = _classes + _functions + _instances

# ------------------------------------------------------------------------------

class connection(object):
    def __init__(self, name):
        self.name = name
        # input is a node object
        self.input = None
        # output is for informational purposes only
        # (nodes keyed by name)
        self.outputs = {}
        # map is of type map (defaults to identity)
        self.map = idmap

    def poll(self):
        # poll input
        return self.map(self.input.last_t)

    def __repr__(self):
        return "connection(%s)" % self.name

    __str__ = __repr__


class node(object):
    def __init__(self, name):
        self.name = name
        # inputs are connection objects
        self.inputs = {}
        # output is for informational purposes only
        # (connections keyed by name)
        self.outputs = {}
        # projected state
        self.next_t = 0
        # last state (set by simulator.set_state)
        self.last_t = 0
        # current input values
        self.in_vals = {}
        # map attribute will be of type map
        self.map = None

    def poll(self, state):
        #print "\n ** ", self.name, "received state:", state
        self.in_vals.update(state)
        # poll inputs
        for name, connxn in self.inputs.items():
            self.in_vals[connxn.input.name] = connxn.poll()
        #print "... passing input values", self.in_vals
        self.next_t = self.map(self.in_vals)
        return self.next_t

    def __repr__(self):
        return "node(%s)" % self.name

    __str__ = __repr__


class FIFOqueue_uniquenode(object):
    """Only one entry per node is allowed.
    !! Does not allow for simultaneous events !!
    """
    def __init__(self, node_names):
        self.nodes = node_names
        self.reset()

    def push(self, t, node_name):
        old_t = self.by_node[node_name]
        if t != old_t:
            self.by_node[node_name] = t
            self.by_time[t] = node_name
            if isfinite(old_t):
                del self.by_time[old_t]
            self.next_t = min(self.by_time.keys())

    def reset(self):
        self.by_time = {Inf: None}
        self.next_t = Inf
        self.by_node = dict.fromkeys(self.nodes, Inf)

    def pop(self):
        t = self.next_t
        if isfinite(t):
            val = (t, self.by_time[t])
            del self.by_time[t]
            self.by_node[val[1]] = Inf
            self.next_t = min(self.by_time.keys())
            return val
        else:
            raise PyDSTool_UndefinedError("Empty queue")


def extract_digraph(mspec, node_types, connxn_types):
    """Extract directed graph of connections from a ModelSpec description
    of a dynamical systems model, using the node and connection types
    provided. The name attributes of those types are used to search the
    ModelSpec.
    """
    connxn_names = mspec.search(connxn_types)
    node_names = mspec.search(node_types)

    # declare connection and node objects
    connxns = {}
    nodes = {}
    for c in connxn_names:
        cobj = mspec.components[c]
        new_connxn = connection(c)
        connxns[c] = new_connxn

    for n in node_names:
        nobj = mspec.components[n]
        new_node = node(n)
        nodes[n] = new_node

    # fill in inputs and outputs dictionaries for each type
    for cn, c in connxns.items():
        cobj = mspec.components[cn]
        targs = [head(t) for t in cobj.connxnTargets]
        c.outputs = dict(zip(targs, [nodes[t] for t in targs]))
        for t in targs:
            nodes[t].inputs[cn] = c

    for nn, n in nodes.items():
        nobj = mspec.components[nn]
        targs = [head(t) for t in nobj.connxnTargets]
        n.outputs = dict(zip(targs, [connxns[t] for t in targs]))
        for t in targs:
            connxns[t].input = n

    return nodes, connxns


def head(hier_name):
    if '.' in hier_name:
        return hier_name.split('.')[0]
    else:
        return hier_name

def tail(hier_name):
    if '.' in hier_name:
        return hier_name.split('.')[-1]
    else:
        return hier_name

# maps' inputs and output are absolute times --
# for dealing with relative times they must be passed a reference
# time value to add to a relative time.
class map(object):
    pass

class map2D(map):
    pass

class map1D(map):
    pass


class delay_map(map1D):
    def __init__(self, delay):
        self.delay = delay

    def __call__(self, t):
        return t + self.delay


class identity_map(map1D):
    def __call__(self, t):
        return t


# default instance of identity class
idmap = identity_map()


class composed_map1D(map1D):
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    def __call__(self, t):
        return self.m1(self.m2(t))

# -----------------------------------------------------------------------------

class simulator(object):
    """Mapping-based, event-driven simulator for
    dynamical systems reductions.
    """
    def __init__(self, nodes, connections, state=None):
        self.nodes = nodes
        # don't really need the connections, but just as a reference
        self.connections = connections
        if state is None:
            self.state = dict(zip(nodes.keys(), [NaN for n in nodes]))
        else:
            self.state = state
        self.curr_t = 0
        self.history = {}
        self.verbosity = 0
        self.Q = FIFOqueue_uniquenode(list(nodes.keys()))

    def set_node_state(self):
        for name, n in self.nodes.items():
            n.last_t = self.state[name]

    def validate(self):
        for name, n in self.nodes.items():
            assert isinstance(n, node)
        for name, connxn in self.connections.items():
            assert isinstance(connxn, connection)
        assert sortedDictKeys(self.state) == sortedDictKeys(self.nodes), \
               "Invalid or missing node state in history argument"

    def run(self, history, t_end):
        """history is a dictionary of t -> (event node name, {node name: state})
        values. Initial time of simulator will be the largest t in this
        dictionary.
        """
        self.curr_t = max(history.keys())
        assert t_end > self.curr_t, "t_end too small"
        self.state = history[self.curr_t][1].copy()

        # structural validation of model
        self.validate()
        node_names = sortedDictKeys(self.state)
        self.history = history.copy()
        done = False

        while not done:
            last_state = self.state.copy()
            print("\n ***", self.curr_t, self.history[self.curr_t][0], self.state)

            next_t = Inf
            iters = 1
            nodes = self.nodes.copy()
            # set the last_t of each node
            self.set_node_state()

            try:
                proj_state = self.compile_next_state(nodes)
            except PyDSTool_BoundsError:
                print("Maps borked at", self.curr_t)
                done = True
                break
            print("Projected:", proj_state)
            for node, t in proj_state.items():
                if t > self.curr_t:
                    self.Q.push(t, node)
#            self.state[next_node] = self.Q.next_t

            if self.verbosity > 0:
                print("Took %i iterations to stabilize" % iters)
                self.display()

            t, next_node = self.Q.pop()
            while self.curr_t > t:
                t, next_node = self.Q.pop()
            self.curr_t = t
            self.history[self.curr_t] = (next_node, last_state)
            ##print " * last state =", last_state
            next_state = last_state
            next_state[next_node] = self.curr_t
            # must copy here to ensure history elements don't get
            # overwritten
            self.state = next_state.copy()

            if self.verbosity > 0:
                print("Next node is", next_node, "at time ", self.curr_t)
            done = self.curr_t >= t_end
            continue

            ##

            vals = sortedDictValues(projected_states)
            filt_vals = []
            min_val = Inf
            min_ix = None
            for i, v in enumerate(vals):
                if v > self.curr_t:
                    if v < min_val:
                        min_val = v
                        min_ix = i
##                else:
##                    # do not ignore projected times that are in the past relative to
##                    # curr_t! Must retrgrade curr_t to the earliest newly projected time.
##                    if v not in self.history:
##                        if v < min_val:
##                            min_val = v
##                            min_ix = i
            if min_ix is None:
                # no further events possible
                print("No further events possible, stopping!")
                break
##            if min_val < self.curr_t:
##                # clear later history
##                print "Clearing later history items that are invalid"
##                for t, s in sortedDictItems(self.history):
##                    if t > min_val:
##                        del self.history[t]
            self.curr_t = min_val
            next_state = self.state[1].copy()
            next_node = node_names[min_ix]
            next_state[next_node] = min_val
            self.history[min_val] = (next_node, next_state)
            self.state = (next_node, next_state)

            if self.verbosity > 0:
                print("Next node is", next_node, "at time ", self.curr_t)
            done = self.curr_t >= t_end

        ts, state_dicts = sortedDictLists(self.history, byvalue=False)
        vals = []
        for (evnode, vd) in state_dicts:
            vals.append([vd[nname] for nname in node_names])
        self.result = Pointset(indepvararray=ts, indepvarname='t',
                               coordnames = node_names,
                               coordarray = array(vals).T)


    def compile_next_state(self, nodes):
        vals = {}
        for name, n in nodes.items():
            vals[name] = n.poll(self.state)
        return vals

    def display(self):
        print("\n****** t =", self.curr_t)
        info(self.state, "known state")
        print("\nNodes:")
        for name, n in self.nodes.items():
            #n.poll(self.state)
            print(name)
            for in_name, in_val in n.in_vals.items():
                print("  Input", in_name, ": ", in_val)

    def extract_history_events(self):
        node_names = list(self.nodes.keys())
        node_events = dict(zip(node_names, [None]*len(node_names)))
        ts = sortedDictKeys(self.history)
        old_node, old_state = self.history[ts[0]]
        # deal with initial conditions first
        for nn in node_names:
            node_events[nn] = [old_state[nn]]
        # do the rest
        for t in ts:
            node, state = self.history[t]
            node_events[node].append(t)
        return node_events

    def display_raster(self, new_figure=True):
        h = self.history
        ts = sortedDictKeys(h)
        node_names = list(self.nodes.keys())
        print("\n\nNode order in plot (bottom to top) is", node_names)
        if new_figure:
            plt.figure()
        t0 = ts[0]
        node, state = h[t0]
        # show all initial conditions
        for ni, n in enumerate(node_names):
            plt.plot(state[n], ni, 'ko')
        # plot the rest
        for t in ts:
            node, state = h[t]
            plt.plot(t, node_names.index(node), 'ko')
        a = gca()
        a.set_ylim(-0.5, len(node_names)-0.5)


def sequences_to_eventlist(seq_dict):
    """seq_dict maps string symbols to increasing-ordered sequences of times.
    Returns a single list of (symbol, time) pairs ordered by time."""
    out_seq = []
    symbs = list(seq_dict.keys())
    next_s = None
    indices = {}
    for s in symbs:
        indices[s] = 0
    remaining_symbs = symbs
    while remaining_symbs != []:
        #print "\n***", remaining_symbs
        to_remove = []
        #print indices
        #print out_seq, "\n"
        next_t = Inf
        for s in remaining_symbs:
            try:
                t = seq_dict[s][indices[s]]
            except IndexError:
                # no times remaining for this symbol
                #print "No more symbols for ", s
                to_remove.append(s)
            else:
                #print s, t
                if t < next_t:
                    next_s = s
                    next_t = t
                    #print "Chose ", next_s, t
        indices[next_s] += 1
        for s in to_remove:
            remaining_symbs.remove(s)
        if isfinite(next_t):
            out_seq.append( (next_s, next_t) )
    return out_seq

# ---------------------------------------------------------------------

if __name__ == '__main__':
    class testmap1D(map1D):
        def __init__(self, targ, val):
            self.targ = targ
            self.val = val

        def __call__(self, state):
            return self.val + state[self.targ]

    class testmap2D(map2D):
        def __init__(self, targ, me):
            self.targ = targ
            self.me = me

        def __call__(self, state):
            return 3 + state[self.targ] + 0.1*state[self.me]


    xmap = testmap1D('x', 2)
    ymap = testmap2D('y', 'x')
    x = node('x')
    y = node('y')

    xcy = connection('xcy')
    xcy.input = x
    xcy.outputs = {'y': y}

    ycx = connection('ycx')
    ycx.input = y
    ycx.map = delay_map(2)
    ycx.outputs = {'x': x}

    xcx = connection('xcx')
    xcx.input = x
    xcx.outputs = {'x': x}

    x.inputs['y'] = ycx
    x.inputs['x'] = xcx
    x.map = ymap
    y.inputs['x'] = xcy
    y.map = xmap

    sim = simulator({'x': x, 'y': y}, {'xcy': xcy, 'ycx': ycx,
                                       'xcx': xcx})
    sim.verbosity = 1
    sim.run({0: ('x', {'x': -5, 'y': -2}), -2: ('y', {'x': -5, 'y': -8})}, 10)
