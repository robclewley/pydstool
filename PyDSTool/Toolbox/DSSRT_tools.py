# DSSRT interface tools (incomplete)
"""DSSRT interface tools.
(strongly biased towards neural models at this time).

These utilities do not fully construct the graphical
specification for DSSRT, which still has to be done by hand.
Partial templates for the graphical specs are created.

   Robert Clewley, October 2005.

"""

from __future__ import absolute_import, print_function

from PyDSTool import *
from PyDSTool.parseUtils import symbolMapClass
from PyDSTool.Toolbox.FR import *
import os
from copy import copy
from random import uniform, gauss
import six

__all__ = ['DSSRT_info', 'plotNetworkGraph']

# ---------------------------------------------------------------------------

CFGorder = ['VARSEXT', 'VARSINT', 'INPUTS', 'DEQNS', 'CFAC', 'BOUNDS',
            'UNITBOUNDS', 'GAM1TERM', 'GAM2TERM', 'DEPAR']

nodes_order = ['NODE', 'NODELABEL', 'NODEADDSTATES', 'NODESTATEMAP_ACTS',
               'NODESTATEMAP_POTS']


class DSSRT_info(object):
    def __init__(self, name='', infty_val=1e10):
        """The optional 'infty_val' argument sets the substitute default
        finite value used to represent 'infinity', for variables that are
        not given explicit bounds."""

        self.name = name
        self.varsext = []
        self.varsint = []
        self.varsall = []
        self.deqns = {}
        self.depars = {}
        self.inputs = {}
        self.bounds = {}
        self.unitbounds = []
        self.gam1terms = {}
        self.gam2terms = {}
        self.cfac = {}
        # graphics templates
        self.nodes = []
        self.links = []
        self.vbars = []
        # for internal use
        self._CFG = {}
        self._infty_val=infty_val


    def setUnitBounds(self, coords):
        self.unitbounds = coords


    def prepInputs(self, dependencies, nameMap=None):
        if nameMap is None:
            nameMap = symbolMapClass({})
        inputs = {}.fromkeys(self.varsall)
        for (i,o) in dependencies:
            mi = nameMap(i)
            mo = nameMap(o)
            if mi != mo:
                try:
                    inputs[mi].append(mo)
                except AttributeError:
                    inputs[mi] = [mo]
        for k, v in inputs.items():
            if v is None:
                inputs[k] = []
        self.inputs = inputs


    def prepGraph(self, ics=None, vbars_int={}, vars=None):
        """Call this multiple times to re-randomize the initial vertex
        selection."""
        if self.varsall == [] or self.inputs == {}:
            raise ValueError("Prerequisite of prepGraph is varsall and inputs")
        if vars is None:
            vars = self.varsext
        # TEMP  -- only support full vars (no hidden)
        if remain(self.varsext, vars) != []:
            raise NotImplementedError("Can only make network graph with all "
                                      "external variables at this time")
        print("Finding graph layout for network...")
        V = {}
        E = {}
        # prepare vertices
        for var in vars:
            V[var] = vertex(uniform(0.2, 0.8), uniform(0.2, 0.8))
#            print "vertex %s initialized at co-ords (%.4f, %.4f)"%(var,
#                                    V[var].pos[0], V[var].pos[1])
        # valid domain, in [0, 1]
        domY = [0.08, 0.92]
        domX = [0.08, 0.96]
        # override position initial conditions for specially selected variables
        if ics is not None:
            for var, ic in ics.items():
                if ic[0] < domX[0] or ic[0] > domX[1]:
                    raise ValueError("X initial condition for %s out of "%var \
                                     + "bounds [%.3f, %.3f]"%(domX[0], domX[1]))
                if ic[1] < domY[0] or ic[1] > domY[1]:
                    raise ValueError("Y initial condition for %s out of "%var \
                                     + "bounds [%.3f, %.3f]"%(domY[0], domY[1]))
                try:
                    V[var].pos = array(ic, Float)
                except KeyError:
                    raise KeyError("Variable name %s in initial condition"%var\
                                   + " dictionary not known")
        # prepare edges
        for var in vars:
            ins = self.inputs[var]
            if len(ins) >= 1:
                for invar in ins:
                    if invar == var:
                        raise ValueError("Internal error: invar == var")
                    try:
                        e = edge(V[invar], V[var])
                    except KeyError:
                        # assume connected to an internal variable, so ignore
                        continue
                    try:
                        E[var].append(e)
                    except KeyError:
                        E[var] = [e]
        in_d = in_degree(V, E)
        out_d = out_degree(V, E)
        for v in vars:
            if in_d[v] + out_d[v] == 0:
                info(in_d, 'in-degrees')
                print("\n")
                info(out_d, 'out-degrees')
                raise ValueError("Variable %s had no associated edges"%v)
        self._V = V
        self._E = E
        if ics is None:
            fixedICs = {}
        else:
            fixedICs = dict(zip(ics.keys(),[V[k] for k in ics.keys()]))
        # call the Fruchtermann-Reingold algorithm to determine graph
        FR(V, E, domX, domY, fixed=fixedICs)
        self.nodes = []
        self.links = []
        self.vbars = []
        importance = {}
        for var in vars:
            # get all importance info first
            try:
                importance[var] = in_d[var]/float(out_d[var])
            except ZeroDivisionError:
                importance[var] = 5   # arbitrary > 3
        # generate node info for DSSRT from FR output
        for var in vars:
            if importance[var] > 3:
                # "important" variables get larger labels, nodes
                r2 = 0.04
                txtsize = 0.02
            else:
                r2 = 0.03
                txtsize = 0.015
            nx = V[var].pos[0]
            ny = V[var].pos[1]
            if txtsize == 0.02:
                lx = nx-0.25*r2*len(var)
                ly = ny-0.4*txtsize
            else:
                lx = nx-0.27*r2*len(var)
                ly = ny-0.35*txtsize
            self.nodes.append({'NODE': "%s %.4f %.4f %.3f"%(var, nx, ny, r2),
                          'NODELABEL': "%s %s %.4f %.4f %.3f"%(var, var,
                                                lx, ly, txtsize),
                          'NODEADDSTATES': "%s 0"%var,
                          'NODESTATEMAP_ACTS': var, 'NODESTATEMAP_POTS': var})
            x2 = V[var].pos[0]
            y2 = V[var].pos[1]
            for invar in self.inputs[var]:
                if invar in self.varsint:
                    continue
                if importance[invar] > 3:
                    r1 = 0.04
                else:
                    r1 = 0.03
                x1 = V[invar].pos[0]
                y1 = V[invar].pos[1]
                D = [x1-x2, y1-y2]
                sgnx = sign(D[0])
                sgny = sign(D[1])
                try:
                    theta2 = atan(abs(D[1])/abs(D[0]))
                except ZeroDivisionError:
                    theta2 = pi_/2.
                link_x1 = x1-sgnx*r1*cos(theta2)
                link_y1 = y1-sgny*r1*sin(theta2)
                link_x2 = x2+sgnx*r2*cos(theta2)
                link_y2 = y2+sgny*r2*sin(theta2)
                link_str = '%s %s %.4f %.4f %.4f %.4f'%(invar, var,
                           link_x1, link_y1, link_x2, link_y2)
                if len(self.inputs[var]) <= 1:
                    # link not shown for <=1 input
                    # link declaration needs 1 in final arg to suppress display
                    link_str += ' 1'
                self.links.append(link_str)
            try:
                bd0 = self.bounds[var][0]
                bd1 = self.bounds[var][1]
            except KeyError:
                # expect unit bounds instead
                if var in self.unitbounds:
                    bd0 = 0
                    bd1 = 1
                else:
                    print("Warning: variable %s has not been given bounds"%var)
                    print(" (assuming +/- 'infty_val' attribute)")
                    bd0 = -self._infty_val
                    bd1 = self._infty_val
            magopt = int(bd0 == 0 and bd1 == 1)
            self.vbars.append('%s %.4f %.4f %.4f %.3f %.3f %d'%(var,
                                 x2-r2*1.2, y2+0.025, r2, bd0, bd1, magopt))
        for var in remain(self.varsall, vars):
            try:
                bd0 = self.bounds[var][0]
                bd1 = self.bounds[var][1]
            except KeyError:
                # expect unit bounds instead
                if var in self.unitbounds:
                    bd0 = 0
                    bd1 = 1
                else:
                    print("Warning: variable %s has not been given bounds"%var)
                    print(" (assuming +/- 'infty_val' attribute)")
                    bd0 = -self._infty_val
                    bd1 = self._infty_val
            # these variables still need vbars
            magopt = int(bd0 == 0 and bd1 == 1)
            if var in vbars_int:
                if isinstance(vbars_int[var], six.string_types):
                    # use vertex of associated external variable
                    assocV = V[vbars_int[var]]
                    x1 = assocV.pos[0]-0.02
                    y1 = assocV.pos[1]+0.025
                    h = 0.03
                else:
                    # assume (x,y,h) given explicitly as a triple
                    x1, y1, h = vbars_int[var]
                self.vbars.append('%s %.4f %.4f %.4f %.3f %.3f %d'%(var, x1,
                                             y1, h, bd0, bd1, magopt))
            else:
                self.vbars.append('%s <x1> <y1> <h> %.3f %.3f %d'%(var, bd0,
                                                       bd1, magopt))


    def makeDefaultGraphics(self):
        for v in remain(self.varsext, self.nodes):
            self.nodes.append({'NODE': "%s <xpos> <ypos> <r>"%v,
                          'NODELABEL': "%s %s <xpos> <ypos> <size>"%(v,v),
                          'NODEADDSTATES': "%s 0"%v,
                          'NODESTATEMAP_ACTS': v, 'NODESTATEMAP_POTS': v})
        for v in remain(self.varsext, self.links):
            for i in self.inputs[v]:
                self.links.append('%s %s <x1> <y1> <x2> <y2>'%(v,i))
        for v in remain(self.varsint, self.vbars):
            try:
                bd0 = self.bounds[v][0]
                bd1 = self.bounds[v][1]
            except KeyError:
                # expect unit bounds instead
                if v in self.unitbounds:
                    bd0 = 0
                    bd1 = 1
                else:
                    raise NameError("Variable "+var+" has no declared bound")
            self.vbars.append('%s <xpos> <ypos> <height> %.3f %.3f <mag>'%(v,
                               bd0,bd1))


    def makeDSSRTcfg(self, model, gen_name, cfg_filename):
        """Make DSSRT configuration file from a PyDSTool Generator object that
        is embedded in a Model object."""

        try:
            gen = model.registry[gen_name]
        except KeyError:
            raise ValueError("Generator '%s' not found in Model '%s'"%(gen_name, model.name))
        except AttributeError:
            raise TypeError("Invalid Model object passed to makeDSSRTcfg()")
        try:
            mspec = model._mspecdict[gen_name]
        except (AttributeError, KeyError):
            raise ValueError("Model must contain ModelSpec information to proceed.")
        self.name = gen_name
        self.prepVarNames(model._FScompatibleNames(model.obsvars),
                          model._FScompatibleNames(model.intvars))
        assert remain(model._FScompatibleNames(model.allvars), self.varsext+self.varsint) == []
        assert remain(self.varsext+self.varsint, model._FScompatibleNames(model.allvars)) == []
        self.prepInputs(gen.funcspec.dependencies, model._FScompatibleNames)
        self.deqns = remain(self.varsall, model._FScompatibleNames(gen.funcspec.auxvars))
        # Bounds
        fsDict = mspec.funcSpecDict['vars']
        for v in self.varsall:
            domain_interval = copy(fsDict[model._FScompatibleNamesInv(v)].domain[2])
            if not isfinite(domain_interval[0]):
                domain_interval[0] = -self._infty_val
            if not isfinite(domain_interval[1]):
                domain_interval[1] = self._infty_val
            self.bounds[v] = domain_interval
        # Capacitance-like parameters
        cvars = model.searchForNames('soma.C')[gen_name]
        for fullname in cvars:
            parts = fullname.split('.')
            assert len(parts)==2, "Only know how to deal with two-tier hierarchical variable names"
            try:
                voltname = model._FScompatibleNames(model.searchForVars(parts[0]+'.'+'V')[0])
            except:
                print("Problem finding membrane voltage name in model spec")
                raise
            self.cfac[voltname] = model._FScompatibleNames(fullname)
        # need to take out function-specific parameters from depars --
        # assume that any parameters appearing in function definitions *only*
        # appear there
        subsFnDef, not_depars = self.prepAuxFns(mspec.flatSpec['auxfns'],
                                                mspec.flatSpec['pars'])
        # prepare DEpars
        alldepars = model._FScompatibleNames(model.query('parameters'))   # full list of pars
        depar_names = remain(alldepars, not_depars)
        self.depars = dict(zip(depar_names, [mspec.flatSpec['pars'][p] for p in depar_names]))
        # perform the textual subs for non-tau or inf function calls with > 1 argument
        raise NotImplementedError("This function is incomplete!")
        # do something with subsFnDef
        # work out gam1terms and gam2terms
        # validate CFG
        self.prepCFG()
        print("Finished preparing CFG information. Call outputCFG(filename) to output .cfg file")


    def prepCFG(self):
        """Prepare .cfg file (at least a pre-cursor for later editing by hand)"""
        # Set remaining bounds to default limits
        bd_overlap = intersect(self.bounds.keys(), self.unitbounds)
        if bd_overlap != []:
            print(bd_overlap)
            raise ValueError("Clash between variables with explicitly declared"
                             "bounds and those with unit bounds")
        for v in remain(self.varsall, list(self.bounds.keys())+self.unitbounds):
            self.bounds[v] = [-self._infty_val, self._infty_val]
        self._CFG['VARSEXT'] = [" ".join(self.varsext)]
        self._CFG['VARSINT'] = [" ".join(self.varsint)]
        self._CFG['INPUTS'] = [ename + " " + " ".join(inlist) for ename, inlist in self.inputs.items()]
        self._CFG['BOUNDS'] = [ename + " %f %f"%(bd[0],bd[1]) for ename, bd in self.bounds.items()]
        self._CFG['UNITBOUNDS'] = [" ".join(self.unitbounds)]
        self._CFG['DEQNS'] = [" ".join(self.deqns)]
        self._CFG['CFAC'] = [vname + " " + cname for (vname, cname) in self.cfac.items()]
        self._CFG['GAM1TERM'] = []
        for vname, termlists in self.gam1terms.items():
            self._CFG['GAM1TERM'].extend([vname + " " + " ".join(termlist) for termlist in termlists])
        self._CFG['GAM2TERM'] = []
        for vname, termlists in self.gam2terms.items():
            self._CFG['GAM2TERM'].extend([vname + " " + " ".join(termlist) for termlist in termlists])
        deparnames = list(self.depars.keys())
        deparnames.sort()
        self._CFG['DEPAR'] = []
        for parname in deparnames:
            pval = self.depars[parname]
            self._CFG['DEPAR'].append(parname + " " + str(pval))
        # Graphics configuration templates
        if self.nodes == [] or self.links == [] or self.vbars == []:
            self.makeDefaultGraphics()
        self._CFG['graphics'] = (self.nodes, self.links, self.vbars)


    def outputCFG(self, cfg_filename):
        if self._CFG == {}:
            self.prepCFG()
            if self._CFG == {}:
                raise RuntimeError("CFG dictionary was empty!")
        cfg_file = open(cfg_filename+".cfg", 'w')
        cfg_file.write("# Auto-generated CFG file for PyDSTool model %s\n"%self.name)
        for k in CFGorder:
            v = self._CFG[k]
            cfg_file.write("\n### %s configuration\n"%k.title())
            if v == []:
                cfg_file.write("# (EMPTY)\n")
            for ventry in v:
                cfg_file.write("%s %s\n"%(k,ventry))
        try:
            ginfo = self._CFG['graphics']
            cfg_file.write("\n### Nodes")
            for node in ginfo[0]:
                cfg_file.write("\n")
                for k in nodes_order:
                    cfg_file.write("%s %s\n"%(k,node[k]))
            cfg_file.write("\n### Links\n")
            for link in ginfo[1]:
                cfg_file.write("LINK %s\n"%link)
            cfg_file.write("\n### Vbars\n")
            for vbar in ginfo[2]:
                cfg_file.write("VBAR %s\n"%vbar)
        except KeyError:
            pass
        cfg_file.close()


    def prepVarNames(self, extvars, intvars):
        self.varsext = copy(extvars)
        self.varsext.sort()
        self.varsint = copy(intvars)
        self.varsint.sort()
        self.varsall = self.varsext+self.varsint


    def prepDEpars(self):
        updates = {}
        for par, parval in self.depars.items():
            try:
                if par[-3:]=="tau":
                    updates[par] = (par+"_recip", 1/parval)
            except IndexError:
                pass
        for par, (newpar, newval) in updates.items():
            del self.depars[par]
            self.depars[newpar] = newval


    def prepAuxFns(self, auxfndict, pardict, makeMfiles=True):
        """Prepare auxiliary functions, and create MATLAB m-files for
        those corresponding to 'tau' and 'inf' functions, unless optional
        makeMfiles==False."""
        # DSSRT cannot accept m-file fns that involve DEpars not explicitly
        # passed as an argument, hence not_depars is returned to caller
        # indicating instances of this occuring
        subsFnDef=[]
        not_depars=[]
        mfiles = {}
        allpars = list(pardict.keys())
        for fname, (fsig, fdef) in auxfndict.items():
            if len(fsig) == 1 and fname[-3:] in ['tau', 'inf']:
                fdefQS = QuantSpec("__fdefQS__", fdef)
                # fpars are the parameters used in the function -- we fetch
                # their values and subs them directly into the M file function.
                fpars = intersect(fdefQS.freeSymbols, allpars)
                fpar_defs = dict(zip(fpars,[pardict[p] for p in fpars]))
                not_depars.extend(remain(fpars, not_depars))
                if fname[-3:] == 'tau':
                    fname += "_recip"
                    finfo = {fname: str(1/fdefQS)}  # take reciprocal
                else:
                    finfo = {fname: str(fdefQS)}
                finfo.update(fpar_defs)
                mfiles[fname] = (fsig[0], finfo)
            else:
                # will textually substitute these function call into spec later
                subsFnDef.append(fname)
        # make m files if nofiles==False
        if makeMfiles:
            for fname, finfo in mfiles.items():
                makeMfileFunction(fname, finfo[0], finfo[1])
        return subsFnDef, not_depars


    def dumpTrajData(self, traj, dt, filename, precise=True):
        """If precise=True (default), this trajectory dump to file may take
        several minutes:
        Uses the 'precise' trajectory sample option for trajectories with
        variable time-steps. Use precise=False option for
        trajectories calculated at fixed time steps."""
        ptset = traj.sample(dt=dt, coords=self.varsall, precise=precise)
        exportPointset(ptset, {filename: tuple(['t']+self.varsall)})


# ------------------------------------------------------------------------

def plotNetworkGraph(dssrt_obj):
    try:
        V = dssrt_obj._V
        E = dssrt_obj._E
    except AttributeError:
        raise TypeError("Invalid DSSRT_info object for plotNetworkGraph")
    plt.figure()
    for v in V.values():
        plt.plot([v.pos[0]],[v.pos[1]],'ko')
    for elist in E.values():
        for e in elist:
            plt.plot([e.u.pos[0],e.v.pos[0]],
                       [e.u.pos[1],e.v.pos[1]],
                       'k-')
    plt.axis([0,1,0,1])
