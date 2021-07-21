"""
Cross-channel coupling for a biophysical neural network.
Example courtesy of Mark Olenik (Bristol University).
"""
from PyDSTool import Var, Exp, Par, Pow, args
from PyDSTool.Toolbox.neuralcomp import voltage, \
    ModelConstructor, makeSoma, channel
from matplotlib import pyplot as plt


v = Var(voltage)
# Create placeholder structs to collect together related symbols
# (not used internally by PyDSTool)
NMDA = args()
KCa = args()
# Calcium concentration through nmda channels
# Ca_nmda won't create a current but will be used for KCa.I
Ca_nmda = args()
Iapp = args()

NMDA.g = Par(0.75, 'g')
NMDA.erev = Par(0., 'erev')
KCa.g = Par(0.0072, 'g')
KCa.erev = Par(-80., 'erev')
Ca_nmda.erev = Par(20., 'Ca_erev')
Ca_nmda.rho = Par(0.0004, 'rho')
Ca_nmda.delta = Par(0.002, 'delta')
Iapp.amp = Par(0.0, 'amp')

NMDA.p = Var('p')           # nmda gating variable
Ca_nmda.c = Var('c')        # concentration

a_p = 700./1000.*Exp(v/17.)
b_p = 5.6/1000.*1.8*Exp(-v/17.)

NMDA_p_RHS = Var(a_p*(1-NMDA.p)-b_p*NMDA.p, name=NMDA.p.name,
            domain=[0, 1], specType='RHSfuncSpec')

NMDA.I = Var(NMDA.g*NMDA.p*(v-NMDA.erev), name='I', specType='ExpFuncSpec')
KCa.I = Var(KCa.g*Ca_nmda.c*(v-KCa.erev), name='I', specType='ExpFuncSpec')
Iapp.I = Var(-Iapp.amp, name='I', specType='ExpFuncSpec')

channel_NMDA = channel('NMDA')
channel_NMDA.add([NMDA.g, NMDA.erev, NMDA_p_RHS, NMDA.I])

# For Ca_nmda_c_RHS, simply writing -NMDA.p * Ca_nmda.rho  ... will not work.
# The name p will rename unresolved when building the system, since it belonged
# to the NMDA channel. In this python script, `NMDA` is simply an args() struct.
# To access the NMDA p variable from a different channel, the
# foreign channel has to be explicitly added to the name. We do that here using
# a string. As such, we can use the fact that * is defined between a string
# and a Symbolic type, but the negative sign has to be included in the string
# since python won't let you simply negate a plain string. A (longer, but more
# generic) alternative would be to create a QuantSpec object out of the string
# 'NMDA.p' first. Then, you can negate that.
# i.e. Var(-QuantSpec('NMDA.p')* ...
Ca_nmda_c_RHS = Var('-NMDA.p'*Ca_nmda.rho*(v-Ca_nmda.erev) -
                    Ca_nmda.delta*Ca_nmda.c, name=Ca_nmda.c.name,
                    domain=[0, 1000.], specType='RHSfuncSpec')

channel_KCa = channel('KCa')
channel_KCa.add([KCa.g, KCa.erev, KCa.I, Ca_nmda_c_RHS,
                 Ca_nmda.erev, Ca_nmda.rho, Ca_nmda.delta])
channel_Iapp = channel('Iapp')
channel_Iapp.add([Iapp.I, Iapp.amp])

cell = makeSoma('cell',
                channelList=[channel_NMDA,
                             channel_KCa, channel_Iapp],
                C=1.0)

alg_args = {'init_step': 0.01, 'max_pts': 5000000}

t_max = 200000
model = ModelConstructor('HH_model',
                         generatorspecs={cell.name:
                                         {'modelspec': cell,
                                          'target': 'Vode_ODEsystem',
                                          'algparams': alg_args}},
                         indepvar=('t', [0, t_max]))
cell = model.getModel()

#if __name__ == "__main__":
#cell = buildsys()
verboselevel = 2
ic_args = {'V': -75.0,
           'NMDA.p': 0.1,
           'KCa.c': 0.1}

print("Computing trajectory using verbosity level %d..." % verboselevel)
cell.compute(trajname='test',
             tdata=[0, 100],
             ics=ic_args,
             verboselevel=verboselevel)
v_dat = cell.sample('test')
plt.figure()
plt.plot(v_dat['t'], v_dat['V'], 'b')
plt.show() #block=True)

