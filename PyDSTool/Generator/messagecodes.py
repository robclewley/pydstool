"""Message code definitions for Generators
"""

## Warning message codes:
# terminals must have 1 in tens place, non-terminals with 2
# additional types from 00-09 (models make use of this format)
W_UNCERTVAL = 00
W_TERMEVENT = 10
W_TERMSTATEBD = 11
W_BISECTLIMIT = 12
W_NONTERMEVENT = 20
W_NONTERMSTATEBD = 21

## Error message codes
# computation errors have 0 in tens place, event errors have 1 in tens place
E_COMPUTFAIL    = 00
E_NONUNIQUETERM = 10

errmessages = {E_NONUNIQUETERM: 'More than one terminal event found',
               E_COMPUTFAIL: 'Computation of trajectory failed'}

errorfields = {E_NONUNIQUETERM: ['t', 'event list'],
               E_COMPUTFAIL: ['t', 'error info']}

warnmessages = {W_UNCERTVAL: 'Uncertain or boundary value computed',
              W_TERMEVENT: 'Terminal event(s) found',
              W_NONTERMEVENT: 'Non-terminal event(s) found',
              W_TERMSTATEBD: 'State variable reached bounds (terminal)',
              W_BISECTLIMIT: 'Bisection limit reached for event',
              W_NONTERMSTATEBD: 'State or input variable reached ' + \
                              'bounds (non-terminal)'}

warnfields = {W_UNCERTVAL: ['value', 'interval'],
                W_TERMEVENT: ['t', 'event list'],
                W_BISECTLIMIT: ['t', 'event list'],
                W_NONTERMEVENT: ['t', 'event list'],
                W_TERMSTATEBD: ['t', 'var name', 'var value',
                                '\n\tvalue interval'],
                W_NONTERMSTATEBD: ['t', 'var name', 'var value',
                                   '\n\tvalue interval']}
