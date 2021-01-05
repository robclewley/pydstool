"""Structured model specification classes, and associated utilities.

Robert Clewley, October 2005.


 Technical notes for ModelSpec attributes:
 _componentNameMap: object -> full (hierarchical) objname
 _registry: objname -> regObject:
                      obj, objLocalName, objClassName, objParentName
 _componentTypeMap: classname -> [objlist]
"""

# ----------------------------------------------------------------------------

from .common import *
from .utils import info as utils_info
from .parseUtils import *
from .errors import *
from .Symbolic import *

from math import *
from .utils import *
from numpy import Inf, NaN, isfinite,  mod, sum
from numpy import sometrue, alltrue
from copy import copy, deepcopy
import math, random

# ---------------------------------------------------------------------------
### Exports

_classes = ['ModelSpec', 'Component', 'LeafComponent',
            'nameResolverClass']

_functions = ['searchModelSpec', 'matchSubName', 'processMacro',
              'resolveMacroTargets']

_objects = ['nameResolver']


__all__ = _classes + _functions + _objects

# -----------------------------------------------------------------------------

class ModelSpec(object):
    """Model specification abstract class.
    Sub-classes can override targetLangs, compatibleSubcomponents,
    compatibleContainers, and compatibleGens."""

    targetLangs = ()
    compatibleSubcomponents = ()
    compatibleContainers = ()
    compatibleGens = ()

    def __init__(self, name):
        """Called by all sub-classes - do not override"""
        self._registry = {}
        self._componentNameMap = {}
        self._componentTypeMap = {}
        self.multiDefRefs = {}
        self.funcSpecDict = {}
        self.flatSpec = {'FScompatibleNames': symbolMapClass(),
                         'FScompatibleNamesInv': symbolMapClass()}
        self.variables = {}
        self.pars = {}
        self.inputs = {}
        self.auxfns = {}
        self.components = {}
        self.name = name
        self.eventPars = []
        # Find unique list of all compatible subcomponent types inherited
        # from ModelSpec and its sub-classes. Include only the most specialized
        # sub-class of each ancestry.
        self._allSubcomponentTypes = allCompTypes(accrueCompTypes(self.__class__))
        # optional store for names of components that depend on this component's
        # variables -- needed for supporting network 'graph' abstractions
        # (in particular, to support edges)
        self.connxnTargets = []
        # list of unbound symbols
        self.freeSymbols = []
        # for use by ModelLibrary classes to identify interchangeable types of spec
        self.library_tag = ''
        # in case a previous instance of this model spec existed in this
        # session, we need to clear out its associated names in
        # the nameResolverClass instance.
        nameResolver.clear(self)
        self.validate()


    def rename(self, newName):
        oldName = self.name
        for robj in self._registry.values():
            if robj.objParentName == oldName:
                robj.objParentName = newName
        self.name = newName


    def compileFuncSpec(self):
        raise NotImplementedError("Only call this method on a concrete "
                                  "sub-class")


    def flattenSpec(self, multiDefUnravel=True, globalRefs=None,
                    ignoreInputs=False, force=False):
        """Flatten structured model specification to dictionary compatible with
       FuncSpec instantiation.

       Use globalRefs option to declare global variables used in the
       definitions that are not defined in them (e.g. time as 't', although
       this one is included as globalRef by default).

       Default for multiple quantity definitions is to unravel them.
       Use multiDefUnravel=False to override this.

       Use force option to rebuild flat spec if an existing one is
       out of date.
       """
##        Use the optional multiDefUnravel dictionary to selectively retain
##        compact specification or only partially unravel the definitions
##        during the flattening process. Use {} to force all definitions to
##        remain compact, otherwise the syntax is:
##
##             key = a rootname of multiple quantity names
##         --> value = either (a): pair (i1,i2) to unravel all indices of the
##                       rootname from i1 to i2 inclusive, or
##                     (b): a list of such pairs (the union of the intervals will
##                                                be used if they overlap), or
##                     (c): an empty list, so that the definition will remain
##                       compact."""
        #self.validate()
        if not self.isDefined(True, ignoreInputs):
            raise AssertionError("Model spec not fully defined")
        if globalRefs is None:
            globalRefs = ['t']
        elif 't' not in globalRefs:
            globalRefs.append('t')
        if not self.isComplete(globalRefs):
##            if globalRefs is None:
##                print "Unresolved symbols:",  self.freeSymbols
##            else:
            print("Unresolved symbols:%r" % remain(self.freeSymbols, globalRefs))
            raise ValueError("Cannot flatten incomplete functional "
                             "specification")
        if self.funcSpecDict == {} or force:
            try:
                self.compileFuncSpec(ignoreInputs=ignoreInputs)
            except KeyboardInterrupt:
                raise
            except:
                print("compileFuncSpec() failed")
                raise
        fs = self.funcSpecDict
        # name mappings to be compatible with FuncSpec.py and python variable
        # naming rules: i.e. "a_name.b.x3" (ModelSpec) <-> "a_name_b_x3" (FuncSpec/python)
        FScompatibleNames = {}
        FScompatibleNamesInv = {}
        for name in self._registry.keys():
            FScompatibleNames[name] = replaceSep(name)
            FScompatibleNamesInv[replaceSep(name)] = name
        # store for later use
        self._FScompatibleNames = FScompatibleNames
        self._FScompatibleNamesInv = FScompatibleNamesInv
        # process multi-quantity definitions, if used
        # TEMP -- multiDefUnravel can only be True or False
        if self.multiDefRefs != {}:
            unravelDict = {}.fromkeys(self.multiDefRefs)
            if multiDefUnravel:
                for k in unravelDict:
                    # default is to entirely unravel all definitions
                    infotuple = self.multiDefRefs[k]
                    unravelDict[k] = [(infotuple[1], infotuple[2])]
##            if multiDefUnravel is None:
##                for k in unravelDict:
##                    # default is to entirely unravel all definitions
##                    infotuple = self.multiDefRefs[k]
##                    unravelDict[k] = [(infotuple[1], infotuple[2])]
##            else:
##                for k in unravelDict:
##                    unravelDict[k] = []
##                # if multiDefUnravel == {} then the following will not change
##                # unravelDict from being all []
##                for k, v in multiDefUnravel.iteritems():
##                    if isinstance(v, list):
##                        unravelDict[k] = []
##                        if v != []:
##                            # list of pairs
##                            for p in v:
##                                assert p[0] < p[1], "Pair must be ordered"
##                                unravelDict[k].append(p)
##                    else:
##                        # single pair
##                        assert len(p) == 2
##                        assert p[0] < p[1], "Pair must be ordered"
##                        unravelDict[k] = [p]
        # Any unravelDict entries with [], and those remaining after a
        # partial unravelling, should still have their funcSpecDict
        # entry show a valid multiDef name, e.g. z[i,1,5] including i1
        # and i2
        outfs = {}  # output dictionary for funcspec initialization
        quants = ['vars', 'pars', 'inputs']
        outfs['domains'] = {}
        outfs['spectypes'] = {}
        # mathNameMap maps title case math function names to all lower-case
        mathNameMap = dict(zip(allmathnames_symbolic,allmathnames))
        # ----- start of add_to_fs function -----
        def add_to_fs(k, name, carg, i=None):
            # locally-defined because refers to outfs from scope of method
            # 'flattenSpec'
            c = copy(carg)
            if 'for' in c.usedSymbols:
                # Activate macro for variable definition.
                # Correct syntax already checked in Quantity __init__
                # 'for' may have been used more than once
                c, dummy1, dummy2 = processMacro(c, self) #._registry)
                # dummy1 and 2 are unused here
            c.mapNames(mathNameMap)
            fsname = FScompatibleNames[name]
            if i is None:
                try:
                    outfs[k][fsname] = replaceSep(c.spec)
                except KeyError:
                    outfs[k] = {fsname: replaceSep(c.spec)}
            else:
                # c is a multi-quantity def and needs evaluating at i
                try:
                    outfs[k][fsname] = replaceSep(c[i])
                except KeyError:
                    outfs[k] = {fsname: replaceSep(c[i])}
            outfs['domains'][fsname] = c.domain
            if k not in ['pars', 'inputs']:
                # not necessary for pars and inputs
                # as they must be explicitly defined
                outfs['spectypes'][fsname] = c.spec.specType
        # ----- end of add_to_fs function -----
        full_mref_names = {}
        if multiDefUnravel:
            # reconstruct list of full names for mrefs, e.g. 'z[i]' from 'z'
            # otherwise just leave the dict empty
            for k, v in self.multiDefRefs.items():
                try:
                    ksubs = FScompatibleNames[k]
                except KeyError:
                    ksubs = replaceSep(k)
                    FScompatibleNames[k] = ksubs
                    FScompatibleNamesInv[ksubs] = k
                    # also add all instances of the multiref to
                    # FScompatibleNames, so add z0 ... z10 for
                    # z[i,0,10].
                    for ix in range(v[1], v[2]+1):
                        FScompatibleNames[k+str(ix)] = ksubs+str(ix)
                        FScompatibleNamesInv[ksubs+str(ix)] = k+str(ix)
                full_mref_names[ksubs+"["+v[0]+","+str(v[1])+","+str(v[2])+"]"]\
                               = ksubs
        for k, v in fs.items():
            if k in quants:
                for c in v:
                    if c.name in full_mref_names.keys():
                        # unravel defs given by pairs p in unravelDict
                        root = full_mref_names[c.name]
                        for p in unravelDict[root]:
                            for i in range(p[0],p[1]+1):
                                name = root + str(i)
                                try:
                                    if name not in outfs[k]:
                                        add_to_fs(k, name, c, i)
                                except KeyError:
                                    add_to_fs(k, name, c, i)
                    else:
                        add_to_fs(k, c.name, c)
            elif k == 'auxfns':
                for a in v:
                    afsname = FScompatibleNames[a.name]
                    acopy = copy(a)
                    # don't include the local signature names in the mapping
                    acopy.mapNames(filteredDict(mathNameMap,
                                                 a.signature,
                                                 neg=True))
                    aspec = acopy.spec
                    try:
                        outfs[k][afsname] = (a.signature, replaceSep(aspec))
                    except KeyError:
                        outfs[k] = {afsname: (a.signature, replaceSep(aspec))}
            elif k == 'complete':
                # ignore this -- it's not needed in the flattened specification
                pass
            else:
                outfs[k] = v
        outfs['FScompatibleNames'] = symbolMapClass(FScompatibleNames)
        outfs['FScompatibleNamesInv'] = symbolMapClass(FScompatibleNamesInv)
        self.flatSpec = outfs   # for future reference
        return outfs

    def isDefined(self, verbose=False, ignoreInputs=False):
        defined = True  # initial value
        if len(self.compatibleGens) == 0:
            if verbose:
                print("'%s' ill defined: empty compatibleGens"%self.name)
            return False
        if len(self.targetLangs) == 0:
            if verbose:
                print("'%s' ill defined: empty targetLangs"%self.name)
            return False
        if self.isEmpty():
            if verbose:
                print("'%s' ill defined: empty contents"%self.name)
            return False
        for v in self.variables.values():
            if not v.isDefined(verbose):
                if verbose:
                    print("... in '%s' (type %s)"%(self.name,str(self.__class__)))
                return False
        for p in self.pars.values():
            if not p.isDefined(verbose):
                if verbose:
                    print("... in '%s' (type %s)"%(self.name,str(self.__class__)))
                return False
        if not ignoreInputs:
            for i in self.inputs.values():
                if not i.isDefined(verbose):
                    if verbose:
                        print("... in '%s' (type %s)"%(self.name,str(self.__class__)))
                        return False
        for a in self.auxfns.values():
            if not a.isDefined(verbose):
                if verbose:
                    print("... in '%s' (type %s)"%(self.name,str(self.__class__)))
                return False
        return defined


    def addConnxnTarget(self, targ):
        if isinstance(targ, str):
            if targ not in self.connxnTargets:
                self.connxnTargets.append(targ)
        elif isinstance(targ, list):
            for t in targ:
                if t not in self.connxnTargets:
                    self.connxnTargets.append(t)
        else:
            raise TypeError("Invalid ModelSpec type for targ argument")


    def delConnxnTarget(self, targ):
        if isinstance(targ, str):
            try:
                self.connxnTargets.remove(targ)
            except ValueError:
                raise ValueError("Connection target %s not found"%targ)
        elif isinstance(targ, list):
            for t in targ:
                try:
                    self.connxnTargets.remove(t)
                except ValueError:
                    raise ValueError("Connection target %s not found"%t)


    def __contains__(self, obj):
        try:
            if obj.name in self._registry:
                return obj == self._registry[obj.name].obj
            elif self.multiDefInfo[0]:
                # there are multiply-defined variables here
                # if subject is a single Quantity that is defined here
                # in multiple definition form.
                return isMultiDefClash(obj, self.multiDefRefs)
            else:
                return False
        except AttributeError:
            return False


    def __eq__(self, other, diff=False):
        results = []
        try:
            results.append(type(self) == type(other))
            results.append(self.name == other.name)
            results.append(self._registry == other._registry)
        except AttributeError as e:
            if diff:
                print("Type: %s %r" % (className(self), results))
                print("  " + e)
            return False
        if diff:
            print("Type: %s %r" % (className(self), results))
        return alltrue(results)

    def __hash__(self):
        h = [hash(type(self)), hash(self.name)] + [hash(k) for k in self._registry]
        return int(sum(h))

    def __ne__(self, other):
        return not self.__eq__(other)


    def difference(self, other):
        """Print the difference between two ModelSpecs to screen."""
        self.__eq__(other, diff=True)


    def validate(self):
        assert isinstance(self.name, str), \
               "ModelSpec name must be a string"
        assert len(remain(self.targetLangs, targetLangs)) == 0, \
               "Invalid target language for '" + self.name + "'"
        if hasattr(self, 'components'):
            # leaf nodes do not have this
            for c in self.components.values():
                assert c in self._componentNameMap, "Component name map incomplete"
#                assert remain(self._componentNameMap[c],
#                              self._registry.keys()) == []
        for v in self.flatSpec['FScompatibleNamesInv'](list(self.variables.values())):
            assert v in self._componentNameMap, "Variable name map incomplete"
        for p in self.flatSpec['FScompatibleNamesInv'](list(self.pars.values())):
            if p not in self._componentNameMap and p.name not in self.eventPars:
                print("\n %s %r" % (self.name, self))
                print("Pars:%r" % p)
                print(p.name in self.flatSpec['FScompatibleNames'])
                print(p.name in self.flatSpec['FScompatibleNamesInv'])
                print(self.eventPars)
                print(self.flatSpec['FScompatibleNamesInv'](p))
                print("Component name map:%r" % self._componentNameMap)
                raise AssertionError("Parameter name map incomplete")
        i = None
        for i in self.flatSpec['FScompatibleNamesInv'](list(self.inputs.values())):
            assert i in self._componentNameMap, "External input name map incomplete"
        for a in self.flatSpec['FScompatibleNamesInv'](list(self.auxfns.values())):
            assert a in self._componentNameMap, "Auxiliary function name map incomplete"
        # !! check that compatibleGens point to known Generator types !!
        # if get this far without raising exception, then valid


    def add(self, arg, tosubcomponent=None):
        """Add object to registry, into a specified sub component if
        provided.
        """
        if isinstance(arg, list):
            for obj in arg:
                self.add(obj, tosubcomponent)
        elif tosubcomponent is not None:
            if tosubcomponent in self.components:
                c = deepcopy(self[tosubcomponent])
                self.remove(tosubcomponent)
                c.add(arg)
                self.add(c)
            elif tosubcomponent == '':
                # no subcomponent, add to top level
                self.add(arg)
            else:
                raise ValueError("Unknown sub-component %s"%tosubcomponent)
        elif isHierarchicalName(arg.name):
            # add the object to the appropriate sub-component
            obj = deepcopy(arg)
            hier_name_split = obj.name.split(NAMESEP)
            comp_name = hier_name_split[0]
            rest_name = ".".join(hier_name_split[1:])
            obj.rename(rest_name)
            self.add(obj, tosubcomponent=comp_name)
        else:
            obj = deepcopy(arg)
            objname = self._register(obj)
            if isinstance(obj, Var):
                self.variables[objname] = obj
            elif isinstance(obj, Par):
                self.pars[objname] = obj
            elif isinstance(obj, Input):
                self.inputs[objname] = obj
            elif isinstance(obj, Fun):
                self.auxfns[objname] = obj
            elif isinstance(obj, ModelSpec):
                if not issubclass(self.__class__, obj.compatibleContainers):
                    print("Component " + self.name + ": " + self.__class__.__name__)
                    print("Invalid sub-component " + objname \
                          + " (type " + className(obj) + ") to add,")
                    print("  with compatible container types:%r" % obj.compatibleContainers)
                    print("Compatible sub-component types: %r" % self._allSubcomponentTypes)
                    raise ValueError("Incompatible sub-component type"
                                     " for object '" + objname + "'")
                else:
                    self.components[objname] = obj
            else:
                raise TypeError("Invalid Quantity object to add")


    def isEmpty(self):
        raise NotImplementedError("Only call this method on a concrete "
                                  "sub-class")


    def _register(self, obj, depth=0, parent_obj=None):
        # depth = recursion depth, when this component registers its
        # subcomponents' definitions (from their _registry attributes)
        if len(obj.compatibleGens) == 0:
            # object inherits self's
            obj.compatibleGens = self.compatibleGens
        elif len(remain(obj.compatibleGens, self.compatibleGens)) > 0:
            print(remain(obj.compatibleGens, self.compatibleGens))
            raise ValueError("Incompatible generators found in component"
                             " '" + obj.name + "'")
        if len(obj.targetLangs) == 0:
            # object inherits self's
            obj.targetLangs = self.targetLangs
        elif len(remain(obj.targetLangs, self.targetLangs)) > 0:
            print(remain(obj.targetLangs, self.targetLangs))
            raise ValueError("Incompatible target language in component"
                             " '" + obj.name + "'")
        if obj.name in protected_allnames:
            raise ValueError("Name '" + obj.name + "' is a protected name")
        if not compareClassAndBases(obj, self._allSubcomponentTypes):
            print("Valid sub-component types that have been declared:%r" % \
                  self._allSubcomponentTypes)
            raise TypeError("Invalid type for object '" + obj.name + \
                            "' in component '" + self.name + "'")
        if parent_obj is None:
            parentname = None
        else:
            parentname = parent_obj.name
        if isMultiDefClash(obj, self.multiDefRefs, parentname):
            raise ValueError("Object %s defines names that already exist in"
                             " registry"%obj.name)
#        # TEMP
#        print "\n------------------------"
#        print depth, self.name, self.freeSymbols
        if hasattr(obj, '_registry'):
            # if obj is a Component, then process its registry into ours
            # but register the object itself
            do_obj_register = False
            # globalized name means we must create a local namemap
            # for obj and its local free symbols.
            # namemap for a registry object maps original (local) object name
            # to globalized name at this level.
            # tracebacknames are the names of all sub-component registry
            # names that are to be added to the registry at this level.
            namemap = {}
            tracebacknames = []
            for sub_regObj in obj._registry.values():
                subobj = sub_regObj.obj
                subobjname = self._register(subobj, depth+1, obj)
                # _register won't have created an entry yet
                namemap[subobj.name] = subobjname
                if isMultiRef(subobjname):
                    # assumes isMultiRef(objname) == sub_regObj.obj.multiDefInfo[0]
                    # Add non-definition version of name to dictionary,
                    # i.e. add z to mapping dictionary containing z[i,0,4]
                    rootname = subobj.multiDefInfo[1]
                    ixname = subobj.multiDefInfo[2]
                    namemap[rootname] = obj.name+NAMESEP+rootname
                tracebacknames.append(subobjname)
            # for all newly registered names, add their associated
            # namemap to the regObject, and map all occurrences of the
            # local names from the sub-component to their new, globalized
            # names at this level, now that we have collected up the namemap.
            #
            # First, take out symbols defined by obj (i.e. those appearing in
            # its registry) from self.freeSymbols
            mapper = symbolMapClass(namemap)
            globalized_newdefs = mapper(list(obj._registry.keys()))
            self.freeSymbols = remain(self.freeSymbols,
                                      globalized_newdefs)
            obj_freeSymbols = mapper(obj.freeSymbols)
#            # TEMP
#            print "New self.freeSymbols:", self.freeSymbols
#            print "Processing ", obj.name
#            print "Namemap"
#            info(namemap)
#            print "Obj free (orig):", obj.freeSymbols, "(mapped)", obj_freeSymbols
            # the following never executes!
            if obj_freeSymbols != obj.freeSymbols:
                print("mapped object free symbols not same as originals")
                # Test code introspection for HH_spectest.py
#            if self.name == 'cell1' and obj.name == 'chan_s21':
#                # this is None, so comment out namemap[s] = snew below!
#                print "cell1: parentname =", parentname
            for s in obj.freeSymbols:
                if s not in namemap:
                    # then it didn't get mapped, so add to namemap
                    # that will be embedded in reg object
                    if parentname is None:
                        snew = self.name+NAMESEP+s
                    else:
                        snew = parentname+NAMESEP+s
#                    namemap[s] = snew
#                    # TEMP
#                    print self.name, obj.name, "namemap[%s] = %s"%(s,snew)
            for sub_regObj in obj._registry.values():
                subobjname = namemap[sub_regObj.obj.name]
                subsplit = subobjname.split(NAMESEP)
                # resolve the first parent of objects brought up from deep
                # in the sub-component tree
                if len(subsplit) > 1:
                    actual_parentname = subsplit[-2]
                    if actual_parentname in obj.components:
                        actual_parent = obj.components[actual_parentname]
#                        print "Old parent %s"%actual_parentname
#                        print "parent is obj '%s': self=%s, subobj=%s"%(obj.name, self.name, subobjname)
#                        if obj.name == 'cell1':
#                            1/0
                    else:
                        # TEMP
#                        print "Changing parent from %s to %s"%(actual_parentname, self.name)
#                        print "for obj %s, sub-obj %s"%(obj.name, subobjname)
#                        print obj._registry.keys()
#                        print obj.components.keys()
#                        1/0
                        actual_parent = obj
#                        actual_parentname = self.name
#                        actual_parent = self
#                        print "parent is self '%s' (1): obj=%s, subobj=%s"%(self.name, obj.name, subobjname)
                else:
                    actual_parent = obj
                new_regObj = deepcopy(sub_regObj)
                if isinstance(new_regObj.obj, Quantity):
                    if new_regObj.obj.typestr == 'auxfn':
                        # do not map the signature variables
                        func_namemap = copy(namemap)
                        for s in new_regObj.obj.signature:
                            try:
                                del func_namemap[s]
                            except KeyError:
                                pass
                        new_regObj.obj.mapNames(func_namemap)
                    else:
                        new_regObj.obj.mapNames(namemap)
                self._registry[subobjname] = regObject(new_regObj.obj,
                                                       actual_parent)
                self._registry[subobjname].namemap = namemap
                # fix new registry object's local name to not include the
                # highest level name prefix
                localName = self._registry[subobjname].objLocalName
                temp = localName.split(NAMESEP)
                if len(temp) > 1:
                    localName = NAMESEP.join(temp[1:])
                self._registry[subobjname].objLocalName = localName
            objname = obj.name
            self._componentNameMap[obj] = objname #tracebacknames
            oclasses = getSuperClasses(obj, ModelSpec)
            for oclass in oclasses:
                try:
                    if obj not in self._componentTypeMap[oclass]:
                        self._componentTypeMap[oclass].append(obj)
                except KeyError:
                    self._componentTypeMap[oclass] = [obj]
            do_free_symbs = True
        else:
            # obj is a Quantity object
            do_obj_register = (depth == 0)
            do_free_symbs = do_obj_register
            try:
                if depth > 0: # and obj.scope == localscope:
                    objname = nameResolver(obj, self, parentname)
                else:
                    objname = obj.name
            except AttributeError:
                raise TypeError("Invalid object type in _register(): %s (type"
                                " %s)"%(obj.name, className(obj)))
#            # TEMP
#            print "Processing Quant:", obj.name, "(%s)"%objname
            if hasattr(obj, 'multiDefInfo'):
                # then Quantity spec may be defining multiple quantities,
                # so check it
                if obj.multiDefInfo[0]:
                    # no longer need first entry of multiDefInfo.
                    # store the info for purposes of flattening, using
                    # the *rootname* as key (e.g. 'z' and not 'z[i]').
                    # Note that the rootname has the parentname appended,
                    # if it exists
                    if parentname is None:
                        rootname = obj.multiDefInfo[1]
                    else:
                        rootname = parentname+NAMESEP+obj.multiDefInfo[1]
                    self.multiDefRefs[rootname] = obj.multiDefInfo[2:]
            self._componentNameMap[obj] = objname
            oclasses = getSuperClasses(obj)
            for oclass in oclasses:
                try:
                    if obj not in self._componentTypeMap[oclass]:
                        self._componentTypeMap[oclass].append(obj)
                except KeyError:
                    self._componentTypeMap[oclass] = [obj]
            if objname in self._registry:
                raise ValueError("Name '%s' already exists in "%objname \
                                 + "registry of object '%s'"%self.name)
            # remove free symbols from self's sub-components if
            # they are being defined here
            for fs in copy(self.freeSymbols):
                fs_split = fs.split(NAMESEP)
                if len(fs_split) > 1:
                    checkname = "".join(fs_split[1:])
                else:
                    checkname = fs
                if objname == checkname:
                    self.freeSymbols.remove(fs)
            obj_freeSymbols = obj.freeSymbols
        if do_obj_register:
            self._registry[objname] = regObject(deepcopy(obj),
                                                parent_obj or self)
            # localname = objname so no need for the next line
#            self._registry[objname].objLocalName = obj.name.split(NAMESEP)[-1]
        if do_free_symbs:
#            # TEMP
#            print "new free symbols from obj:", obj_freeSymbs
#            print "defined:", self._registry.keys()
            self.freeSymbols.extend(remain(obj_freeSymbols,
                                           list(self._registry.keys())+self.freeSymbols))
#            print "resulting self.free:", self.freeSymbols
        # return objname in case _register is being called recursively
        return objname


    def __delitem__(self, name):
        """Delete named object (Var, Par, Fun, Input, or sub-component)
        """
        self.remove(name)

    def __getitem__(self, name):
        """Return object named using the hierarchical naming format."""
        try:
            return self._registry[name].obj
        except KeyError:
            # full hierarchical name not found in registry, so
            # check for components (not kept in registry)
            name_list = name.split('.')
        try:
            c = self.components[name_list[0]]
        except KeyError:
            if name == '':
                # for compatibility with empty string index
                return self
            else:
                raise ValueError("Object %s not found in registry"%name)
        else:
            rest = name_list[1:]
            if len(rest) > 0:
                return c['.'.join(rest)]
            else:
                return c


    def remove(self, target):
        """Remove target component from specification.

       Use global hierarchical names for components if specifying a string."""

        if hasattr(target, 'name'):
            objname = target.name
        elif isinstance(target, str):
            objname = target
        elif isinstance(target, list):
            for t in target: self.remove(t)
            return
        else:
            raise TypeError("Invalid type to remove from ModelSpec "
                            "'%s'"%self.name)
        try:
            del self._registry[objname]
        except KeyError:
            # objname may refer to a component, which is not stored in registry
            pass
        if isHierarchicalName(objname):
            tempname = objname.split(NAMESEP)
            parentname = tempname[0]
            try:
                self.components[parentname].remove("".join(tempname[1:]))
            except ValueError as e:
                print(e)
                raise ValueError("Error recognizing parent object '%s' in "
                                 "sub-component '%s'"%(parentname,objname))
        else:
            if objname in self.variables:
                del self.variables[objname]
            elif objname in self.pars:
                del self.pars[objname]
            elif objname in self.inputs:
                del self.inputs[objname]
            elif objname in self.components:
                del self.components[objname]
            elif objname in self.auxfns:
                del self.auxfns[objname]
                protected_auxnamesDB.removeAuxFn(objname)
            else:
                raise ValueError("%s: Error recognizing object '%s'"%(self.name,objname))
        # This is inefficient, but simply reregister everything to
        # work out the change in free symbols!
        self.freeSymbols = []
        self.multiDefRefs = {}
        self._registry = {}
        self._componentNameMap = {}
        self._componentTypeMap = {}
        nameResolver.clear(self)
        for v in self.variables.values():
            self._register(v)
        for p in self.pars.values():
            self._register(p)
        for i in self.inputs.values():
            self._register(i)
        for a in self.auxfns.values():
            self._register(a)
        # do components, if any, last, so that self._componentNameMap is
        # built up properly (otherwise get validation error)
        if hasattr(self, 'components'):
            # leaf nodes do not have this
            for c in self.components.values():
                self._register(c)
        try:
            self.validate()
        except AssertionError as e:
            raise RuntimeError("Model spec structure inconsistent: "+str(e))


    def isComplete(self, globalRefs=None):
        if globalRefs is None:
            globalRefs = allmathnames_symbolic + \
                  ['True', 'False', 'and', 'or', 'not']
        else:
            globalRefs.extend(allmathnames_symbolic + \
                              ['True', 'False', 'and', 'or', 'not'])
        return remain(self.freeSymbols, globalRefs) == []


    def __call__(self):
        # info is defined in PyDSTool.utils
        utils_info(self.__dict__, "ModelSpec " + self.name)


    def info(self, verboselevel=1):
        if verboselevel > 0:
            # info is defined in PyDSTool.utils
            utils_info(self.__dict__, "ModelSpec " + self.name,
                       recurseDepthLimit=1+verboselevel)
        else:
            print(self.__repr__())


    def __repr__(self):
        return self._infostr(1)

    def _infostr(self, verbose=1):
        base_str = "Component " + self.name
        if verbose > 0:
            if len(self.components) > 0:
                base_str += "\n Components: ( "
                base_str += ", ".join([c._infostr(verbose-1) for c in self.components.values()]) + " )"
            if len(self.variables) > 0:
                base_str += "\n Variables: ( "
                base_str += ", ".join([str(c) for c in self.variables.values()]) + " )"
            if len(self.pars) > 0:
                base_str += "\n Parameters: ( "
                base_str += ", ".join([str(c) for c in self.pars.values()]) + " )"
            if len(self.inputs) > 0:
                base_str += "\n Inputs: ( "
                base_str += ", ".join([str(c) for c in self.inputs.values()]) + " )"
            if len(self.auxfns) > 0:
                base_str += "\n Functions: ( "
                base_str += ", ".join([str(c) for c in self.auxfns.values()]) + " )"
        return base_str


    def search(self, name, component_type_order=None):
        """Find Quantity objects containing a component named <name>,
        of type given by the hierarchical name <comptype1.comptype2. ... .name>,
        where component_type_order = [<comptype1>, <comptype2>, ... ],
        and <comptypeN> may be a component type (as a wildcard) or a specific
        component name.
        """
        return searchModelSpec(self, name, component_type_order)


    __str__ = __repr__


    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)


# ----------------------------------------------------------------------------

class Component(ModelSpec):
    """Non-leaf node sub-class of ModelSpec abstract class."""

    def compileFuncSpec(self, ignoreInputs=False):
        self.validate()
        assert self.isDefined(True, ignoreInputs=ignoreInputs), \
               "Node '" + self.name + "' is not completely defined"
        fsdict = {}
        fsdict['inputs'] = []
        fsdict['vars'] = []
        fsdict['pars'] = []
        fsdict['auxfns'] = []
        for objname, regObj in self._registry.items():
            objtype = regObj.obj.typestr+'s'
            add_obj = regObj.obj.renderForCode()
            if regObj.namemap != {}:
                if isinstance(add_obj, Fun):
                    # AuxFns may only reference pars, so raise
                    # error if the auxfn's namemap refers (i.e. is bound)
                    # to a non-parameter
                    for name in regObj.namemap:
                        mappedName = regObj.namemap[name]
                        if mappedName in add_obj.freeSymbols and not \
                           isinstance(self._registry[mappedName].obj, (Par, Fun)):
                            print("Problem with registry object:%r" % \
                                  self._registry[mappedName].obj)
                            print("in auxiliary function:" + add_obj.name)
                            print("   that has free symbols:%r" % \
                                  add_obj.freeSymbols)
                            raise TypeError("Fun '" + add_obj.name + "' "
                                            "cannot bind symbols to non-"
                                            "pars (info printed above)")
                # redundant mapping (already has been done)
#                add_obj.spec.mapNames(regObj.namemap)
            fsdict[objtype].append(add_obj)
        fsdict['complete'] = self.isComplete()
        self.funcSpecDict = fsdict


    def isEmpty(self):
        return self.components == {}


    def isDefined(self, verbose=False, ignoreInputs=False):
        # Quantity is defined if it has a specified definition
        # (even with unbound references)
        defined = ModelSpec.isDefined(self, verbose, ignoreInputs)
        for c in self.components.values():
            if not c.isDefined(verbose, ignoreInputs):
                return False
        return defined

# ----------------------------------------------------------------------------

class LeafComponent(ModelSpec):
    """Leaf node sub-class of ModelSpec abstract class."""

    def _register(self, obj, depth=0, parent_obj=None):
        if not compareBaseClass(obj, Quantity):
            print("Bad argument %s (type %s) to register"%(str(obj), type(obj)))
            raise TypeError("Not a valid Quantity type")
        return ModelSpec._register(self, obj, 0, parent_obj)


    def isEmpty(self):
        return self.variables == {} and self.pars == {} \
               and self.inputs == {}


    def compileFuncSpec(self, ignoreInputs=False):
        self.validate()
        assert self.isDefined(ignoreInputs=ignoreInputs), \
               "Node '" + self.name + "' is not completely defined"
        fsdict = {}
        fsdict['vars'] = [deepcopy(v) for v in self.variables.values()]
        fsdict['pars'] = [deepcopy(p) for p in self.pars.values()]
        fsdict['inputs'] = [deepcopy(i) for i in self.inputs.values()]
        fsdict['auxfns'] = [deepcopy(a) for a in self.auxfns.values()]
        self.funcSpecDict = fsdict


    def add(self, arg):
        if isinstance(arg, list):
            for obj in arg:
                self.add(obj)
        else:
            if isinstance(arg, ModelSpec):
                raise TypeError("Cannot add sub-components to a leaf node")
            else:
                ModelSpec.add(self, arg)

# add the valid sub-component types now that all classes have been declared
Component.compatibleSubcomponents=(Par, Var, Input, Fun,
    Component, LeafComponent)
Component.compatibleContainers=(Component,)

LeafComponent.compatibleSubcomponents=(Par, Var, Input, Fun)
LeafComponent.compatibleContainers=(Component,LeafComponent)

# ----------------------------------------------------------------------------
### Search utilities

def allCompTypes(class_list):
    """Filter full list of all super-classes' compatible sub-component
    types to remove any classes that are not over-ridden by sub-classes.

    e.g. If Par was allowed for Component type, but a sub-class of
    Component specifies only a sub-class of Par may be used, then
    Par is removed from the original list of valid component types.
    """
    filter_list = []
    for c in class_list:
        filter_list.extend([d for d in class_list if issubclass(c, d) and c!=d])
    return [c for c in class_list if c not in filter_list]



def accrueCompTypes(c):
    """Collect all super-class component types (string names) and accumulate
    recursively.
    """
    cs = list(c.compatibleSubcomponents)
    for cb in c.__bases__:
        if cb is ModelSpec:
            # don't go any deeper as there won't be any more
            # componentTypes -- this is the top of this class tree
            break
        else:
            try:
                cs.extend(accrueCompTypes(cb))
            except AttributeError:
                # cover the event of a user-defined multiple
                # inheritance with non ModelSpec classes
                pass
    return makeSeqUnique(cs)


def matchSubName(nameList, subName, position, level=0, invert=False):
    """Return a list of names from nameList matching subName at the given
   position (at the hierarchical name level given, if the name is
   hierarchical).

   e.g. For a hierarchical variable name such as 'sRM.s_cell1_cell2',
   level=0 will cause a search for a match with subName in the top-level name,
   i.e. 'sRM', at the given position. Whereas, level=1 will search for a match
   in 's_cell1_cell2' at the given position.

   invert=True will return names that did not match."""
    matchList = []
    for name in nameList:
        if isHierarchicalName(name):
            hier_parts = name.split('.')
            searchName = hier_parts[level]
        else:
            searchName = name
        subName_parts = searchName.split('_')
        if invert:
            if subName != subName_parts[position]:
                matchList.append(name)
        else:
            if subName == subName_parts[position]:
                matchList.append(name)
    return matchList


def searchModelSpec(mspec, name, component_type_order=None):
    """Find Quantity or sub-component objects containing named <name>
    of type given by the hierarchical name <comptype1.comptype2. ... .name>,
    where component_type_order = [<comptype1>, <comptype2>, ... ],
    and <comptypeN> may be a component type or a specific component name."""
    if isHierarchicalName(name):
        # transfer root names to component_type_order
        if component_type_order is None or component_type_order == []:
            # OK to continue
            parts = name.split(NAMESEP)
            return searchModelSpec(mspec, parts[-1], parts[:-1])
        else:
            raise ValueError("Cannot pass both a hierarchical name to search"
                             " and a non-empty component_type_order argument")
    if component_type_order is None or component_type_order == []:
        # we have gone as far as intended
        if name in mspec._registry:
            return [name]
        elif name in mspec.components:
            return [name]
        elif name in mspec._componentTypeMap:
            return [mspec._componentNameMap[o] for o in \
                    mspec._componentTypeMap[name]]
        else:
            return []
    else:
        ct = component_type_order[0]
        try:
            complist = mspec._componentTypeMap[ct]
        except KeyError:
            # ct not present in this generator as a type
            # see if it's an actual component
            try:
                complist = [mspec.components[ct]]
            except KeyError:
                return []
        result = []
        # special case for distinguishing RHSfuncSpec vs ExpFuncSpec vars, which
        # have to be the last component type specified
        if name in ['RHSfuncSpec', 'ExpFuncSpec'] and component_type_order[1:] == []:
            for comp in complist:
                compname = mspec._componentNameMap[comp]
                try:
                    test = mspec._registry[compname].obj.specType == name
                except (KeyError, AttributeError):
                    # invalid kind of object to have a specType (e.g. a Component)
                    test = False
                if test:
                    result.append(compname)
        else:
            for comp in complist:
                comp_result = searchModelSpec(comp, name, component_type_order[1:])
                result.extend([comp.name + NAMESEP + r for r in comp_result])
        return result

# -------------------------------------------------------------------------
### Quantity 'for' macro utilities for ModelSpecs

def processMacro(c, mspec, forSubs=False):
    """Process 'for' macro occurrences in a Quantity specification.

   forSubs = True causes any target names to be resolved to their
   global names and returned to caller so that they can be substituted
   textually into the Quantity's specification."""

    ctypestrlist = []   # component type names to remove
    targList = []
    toks = c.spec.parser.tokenized
    num_fors = toks.count('for')
    specStr = ""
    forpos = 0
    for for_ix in range(num_fors):
        old_forpos = forpos
        forpos += toks[forpos:].index('for')
        if old_forpos == 0:
            specStr += "".join(toks[:forpos])
        else:
            specStr += "".join(toks[old_forpos+1:forpos])
        qtemp = QuantSpec('macrospec', toks[forpos+1])
#        targList_new, ctStr, opStr = resolveMacroTargets(qtemp.parser.tokenized,
#                                                     registry, c.name)
        parentName = ".".join(c.name.split('.')[:-1])
        sourceList = searchModelSpec(mspec, parentName)
        if len(sourceList) == 1:
            sourceName = sourceList[0]
        elif len(sourceList) == 0:
            sourceName = ""
        else:
            print("Found: %r" % sourceList)
            raise ValueError("source list for macro resolution should have no "
                             "more than one entry")
        targList_new, ctStr, opStr = resolveMacroTargets(qtemp.parser.tokenized,
                                                         mspec, sourceName)
        ctypestrlist.append(ctStr)
        if forSubs:
            for t in targList_new:
                if t not in targList: targList.append(t)
        if targList_new == []:
            specStr += "0"
        else:
            specStr += "(" + opStr.join(targList_new) + ")"
        forpos += 1
    # add rest of original spec string
    specStr += "".join(toks[forpos+1:])
    c.spec = QuantSpec(replaceSep(c.spec.subjectToken), specStr.strip(),
                       c.spec.specType)
    # remove componentTypeStr from c.freeSymbols and
    # c.usedSymbols
    c.freeSymbols = remain(c.freeSymbols, ctypestrlist)
    c.usedSymbols = remain(c.usedSymbols, ctypestrlist+['for'])
    return (c, targList, ctypestrlist)


def resolveMacroTargets(toks, mspec, targParent):
    # opStr already checked to be either + or *
    opStr = toks[5]
    localTarget = toks[3]
    componentTypeStr = toks[1]
    if componentTypeStr in ['Var', 'Par', 'Quantity']:
        raise ValueError("Invalid component type for macro in"
                         " %s"%targParent)
    if targParent == "":
        target = componentTypeStr+'.'+localTarget
    else:
        target = targParent+'.'+componentTypeStr+'.'+localTarget
    targList = searchModelSpec(mspec, target)
    return (targList, componentTypeStr, opStr)

# ---------------------------------------------------------------------------
### Private classes

class regObject(object):
    """Registry object container."""

    def __init__(self, obj, parent, namemap={}):
        self.obj = obj
        self.objClass = obj.__class__
        self.objParentClass = parent.__class__
        self.objParentName = parent.name
        self.objLocalName = obj.name
        # maps local variable name to globalized name (if global, o/w identical)
        self.namemap = copy(namemap)

    __hash__ = None

    def __eq__(self, other):
        tests = []
        try:
            tests.append(self.objClass == other.objClass)
            tests.append(self.objParentClass == other.objParentClass)
            tests.append(self.objParentName == other.objParentName)
            tests.append(self.objLocalName == other.objLocalName)
            tests.append(self.obj == other.obj)
            tests.append(self.namemap == other.namemap)
        except AttributeError:
            return False
        return alltrue(tests)

    def __repr__(self):
        return "regObject %s (%s)"%(self.objLocalName, className(self.obj))

    def __str__(self):
        return self.obj.spec()

    def __call__(self):
        return (self.objLocalName, self.objClass, self.namemap, self.obj)

    def getGlobalName(self):
        if self.namemap == {}:
            return self.objLocalName
        else:
            try:
                return self.namemap[self.objLocalName]
            except KeyError:
                print("Namemap for '%s' is: %r"%(self.__repr__(), self.namemap))
                raise ValueError("Namemap for registry object didn't "
                                 "contain object's name "
                                 "'%s'"%self.objLocalName)


class typeCounter(object):
    def __init__(self):
        self.varcount = self.parcount = self.funcount = self.inpcount = 0

    def __getitem__(self, typelabel):
        if typelabel == 'var':
            return self.varcount
        elif typelabel == 'par':
            return self.parcount
        elif typelabel == 'input':
            return self.inpcount
        elif typelabel == 'auxfn':
            return self.funcount
        else:
            raise KeyError("Invalid key")

    def __setitem__(self, typelabel, value):
        if typelabel == 'var':
            self.varcount = value
        elif typelabel == 'par':
            self.parcount = value
        elif typelabel == 'input':
            self.inpcount = value
        elif typelabel == 'auxfn':
            self.funcount = value
        else:
            raise KeyError("Invalid key")



class nameResolverClass(object):
    """This class keeps a tab of how many times a local name has been
   used for a given specfication type ('var', 'par', 'input' or
   'auxfn'), and renames it with an appropriate globalized name
   (hierarchical according to declared parent object, with possible
   numbered suffix for multiple name declarations).

   Only one instance of this class is needed for a session."""

    def __init__(self):
        self.database = {}

    def __call__(self, obj, fspec, parentname=None):
        typelabel = className(obj).lower()
        if parentname is None:
            fullname = obj.name
        else:
            # create hierarchical name
            fullname = parentname + NAMESEP + obj.name
        if fspec.name not in self.database:
            self.database[fspec.name] = {}
        if obj.name in self.database[fspec.name]:
            self.database[fspec.name][fullname][typelabel] += 1
            namecount = self.database[fspec.name][fullname][typelabel]
        else:
            self.database[fullname] = typeCounter()
            namecount = 0
        if namecount == 0:
            globalname =  fullname
        else:
            # add number to end of name if multiple instances of the
            # same hierarchical name declared
            globalname =  fullname + NAMESEP + str(namecount)
        return globalname

    def __repr__(self):
        return "ModelSpec internal helper class: nameResolver object"

    __str__ = __repr__

    def clear(self, fspec):
        if fspec.name in self.database:
            del self.database[fspec.name]

    def clearall(self):
        self.database = {}


# use single instance of nameResolver per session
global nameResolver
nameResolver = nameResolverClass()
