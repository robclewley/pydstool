"""
    Run all PyDSTool tests and examples
"""
import os, time
from numpy import any

# Set this to be the command you want to use to invoke python
# in the tests. If None, then the $PYTHON environment variable will
# be checked first, and if empty 'python' will be used
pythonprogram = None

##### Select which sets of tests to run
# Test lists have entries on separate lines to make it easier to
# comment out the ones you don't want
#
# Basic PyDSTool modules
test_general = True
# Map generators
test_maps = True
# ODE integration with vode, no external compiler needed
test_vode = True
# Parameter estimation module
test_param_est = True
# Parameter estimation, requires external C compiler
test_param_est_C = True
# Symbolic differentiation module
test_symbolic = True
# Dopri integration; requires external C compiler
test_dopri = True
# Radau stiff integration; requires external C and fortran compilers
test_radau = True
# Continuation; no external compilers needed
test_pycont = True
# Continuation with auto; requires external compilers
test_pycont_auto = True

# -------------------------------------------------------

# Basic PyDSTool modules
general_list = [
    'test_variable_traj',
    'traj_gt0_test',
    'ModelSpec_test',
    'interp_piecewise_test',
    'objectdelete_test',
    ]

# Map generators
map_list = [
    'SLIP_2D_maps'
    ]

# ODE integration with vode, no external compiler needed
vode_list = [
    'poly_interp_test',
    'vode_withJac_test',
    'interp_vode_test',
    'fingermodel_vode',
    'fingermodel_auxvartest',
    'test_hybrid_extinputs',
    'HH_model',
    'HH_model_testbounds',
    'HH_loaded',
    'IF_model_test',
    'IF_squarespike_model',
    'IF_delaynet',
    'forced_spring'
    ]

# Parameter estimation module
param_est_list = [
    'joe_pest',
    'pest_test1',
    'pest_test2',
    'pest_test3',
    ]

# Parameter estimation, requires external C compiler
param_est_C_list = [
    'pest_test3_Cintegrator',
    'pest_test4_Cintegrator'
    ]

# Symbolic differentiation module
symbolic_list = [
    'Tutorial_SymbolicJac'
    ]

# Dopri integration; requires external C compiler
dopri_list = [
    'imprecise_event_test',
    'interp_dopri_test',
    'HH_model_Cintegrator',
    'HH_loaded_Cintegrator',
    'IF_delaynet_syn',
    'CIN',
    'HH_model_Cintegrator_testbounds',
    'Dopri_backwards_test'
    ]

# Radau stiff integration; requires external C and fortran compilers
radau_list = [
    'test_hybrid_extinputs_Cintegrator',
    'SLIP_2D_pdc',
    'DAE_example',
    'freefinger_noforce_radau',
    'sloppycell_example'
    ]

# Continuation; no external compilers needed
pycont_list = [
    'PyCont_Brusselator',
    'PyCont_Catalytic',
    'PyCont_ABCReaction',
    'PyCont_DiscPredPrey',
    'PyCont_Hopfield',
    'PyCont_hybrid_osc',
    'PyCont_LevelCurve',
    'PyCont_Logistic',
    'PyCont_NewLorenz',
    'PyCont_PredPrey',
    'PyCont_SaddleNode',
    ]

# Continuation with auto; requires external compilers
pycont_auto_list = [
    'PyCont_MorrisLecar_TypeI',
    'PyCont_MorrisLecar_TypeII',
    'PyCont_LPNeuron',
    'PyCont_HindmarshRose',
    'PyCont_ABReaction',
    'PyCont_Hamiltonian',
    'PyCont_Lorenz',
    'PyCont_vanDerPol',
    ]

do_external = any([test_dopri, test_radau, test_param_est_C, test_pycont_auto])


# ----------------------------------------------------------------------------

res = []
failed = []

def test(flist, whichpy=None, infostr=""):
    if whichpy is None or not whichpy:
        whichpy = os.getenv(key='PYTHON', default='python')

    failure = False
    for f in flist:
        fname = f+'.py'
        print("\n***** Testing script %s ****************************\n"%fname)
        try:
            e=os.system(whichpy + ' ' + fname)
        except:
            print("\n      Testing failed on test file %s"%fname)
            failed.append(fname)
            if not failure:
                res.append("%s: appears to be broken on your system"%infostr)
                failure = True
        else:
            if e in [0,3]:
                print("\n      Testing passed on test file %s"%fname)
            else:
                print("\n      Testing failed on test file %s"%fname)
                failed.append(fname)
                failure = True
        time.sleep(2)
    if failure:
        res.append("%s: appears to be broken on your system"%infostr)
    else:
        res.append("%s: appears to work on your system"%infostr)

# ---------------------------------------------------------------------------

print("***** Running all tests in order...\n")
print("Note: Depending on your settings, you may have to close matplotlib windows by hand in order to continue to the next test script\n")

if test_general:
    print("Testing general PyDSTool functions...\n")
    test(general_list, pythonprogram, "Basic PyDSTool functions")
else:
    res.append("Tests of basic PyDSTool functions: SKIPPED")

if test_maps:
    print("Testing map modules...\n")
    test(map_list, pythonprogram, "Map related modules")
else:
    res.append("Tests of map related modules: SKIPPED")

if test_vode:
    print("Testing modules using VODE integrator...\n")
    test(vode_list, pythonprogram, "VODE related modules")
else:
    res.append("Tests of VODE related modules: SKIPPED")

if test_symbolic:
    print("Testing symbolic differentiation module...\n")
    test(symbolic_list, pythonprogram, "Symbolic differentiation module")
else:
    res.append("Tests of symbolic differentiation module: SKIPPED")

if test_param_est:
    print("Testing parameter estimation module, no C compiler dependence...\n")
    test(param_est_list, pythonprogram, "Parameter estimation module")
else:
    res.append("Tests of parameter estimation module: SKIPPED")

if test_pycont:
    print("Testing PyCont module, no external compiler dependence...\n")
    test(pycont_list, pythonprogram, "PyCont")
else:
    res.append("Tests of PyCont with no external compiler dependence: SKIPPED")

if do_external:
    print("\n***** Now running tests that use external compilers...\n")

if test_dopri:
    print("Testing dopri integration; external C compiler dependence...\n")
    test(dopri_list, pythonprogram, "Dopri ODE systems")
else:
    res.append("Tests of Dopri ODE systems: SKIPPED")

if test_radau:
    print("Testing radau integration; external C, fortran compiler dependence...\n")
    test(radau_list, pythonprogram, "Radau ODE systems")
else:
    res.append("Tests of Radau ODE systems: SKIPPED")

if test_param_est_C:
    print("Testing parameter estimation module; with C compiler dependence...\n")
    test(param_est_C_list, pythonprogram, "Parameter estimation module with external compilers")
else:
    res.append("Tests of parameter estimation module with external compilers: SKIPPED")

if test_pycont_auto:
    print("Testing PyCont continuation with AUTO...\n")
    test(pycont_auto_list, pythonprogram, "PyCont interface to AUTO")
else:
    res.append("Tests of PyCont interface to AUTO: SKIPPED")

if len(failed) == 0:
    print("No test scripts failed")
else:
    print("Test scripts that failed:")
    for fname in failed:
        print("\t%s"%fname)

print("Summary:")
for r in res:
    print(r)
