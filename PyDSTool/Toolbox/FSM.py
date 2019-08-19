"""
Finite State Machines.

By Noah Spurrier (2002) and other contributors,
from http://code.activestate.com/recipes/146262-finite-state-machine-fsm/

This module implements a Finite State Machine (FSM). In addition to state
this FSM also maintains a user defined "memory". So this FSM can be used as a
Push-down Automata (PDA) since a PDA is a FSM + memory.

The following describes how the FSM works, but you will probably also need to
see the example function to understand how the FSM is used in practice.

You define an FSM by building tables of transitions. For a given input symbol
the process() method uses these tables to decide what action to call and what
the next state will be. The FSM has a table of transitions that associate:

        (input_symbol, current_state) --> (action, next_state)

Where "action" is a function you define. The symbols and states can be any
objects. You use the add_transition() and add_transition_list() methods to add
to the transition table. The FSM also has a table of transitions that
associate:

        (current_state) --> (action, next_state)

You use the add_transition_any() method to add to this transition table. The
FSM also has one default transition that is not associated with any specific
input_symbol or state. You use the set_default_transition() method to set the
default transition.

When an action function is called it is passed a reference to the FSM. The
action function may then access attributes of the FSM such as input_symbol,
current_state, or "memory". The "memory" attribute can be any object that you
want to pass along to the action functions. It is not used by the FSM itself.
For parsing you would typically pass a list to be used as a stack.

The processing sequence is as follows. The process() method is given an
input_symbol to process. The FSM will search the table of transitions that
associate:

        (input_symbol, current_state) --> (action, next_state)

If the pair (input_symbol, current_state) is found then process() will call the
associated action function and then set the current state to the next_state.

If the FSM cannot find a match for (input_symbol, current_state) it will then
search the table of transitions that associate:

        (current_state) --> (action, next_state)

If the current_state is found then the process() method will call the
associated action function and then set the current state to the next_state.
Notice that this table lacks an input_symbol. It lets you define transitions
for a current_state and ANY input_symbol. Hence, it is called the "any" table.
Remember, it is always checked after first searching the table for a specific
(input_symbol, current_state).

For the case where the FSM did not match either of the previous two cases the
FSM will try to use the default transition. If the default transition is
defined then the process() method will call the associated action function and
then set the current state to the next_state. This lets you define a default
transition as a catch-all case. You can think of it as an exception handler.
There can be only one default transition.

Finally, if none of the previous cases are defined for an input_symbol and
current_state then the FSM will raise an exception. This may be desirable, but
you can always prevent this just by defining a default transition.

Noah Spurrier 20020822
"""



class ExceptionFSM(Exception):

    """This is the FSM Exception class."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class FSM(object):

    """This is a Finite State Machine (FSM).
    """

    def __init__(self, initial_state, memory=None):
        """This creates the FSM. You set the initial state here. The "memory"
        attribute is any object that you want to pass along to the action
        functions. It is not used by the FSM. For parsing you would typically
        pass a list to be used as a stack. """

        # Map (input_symbol, current_state) --> (action, next_state).
        self.state_transitions = {}
        # Map (current_state) --> (action, next_state).
        self.state_transitions_any = {}
        self.default_transition = None

        self.input_symbol = None
        self.initial_state = initial_state
        self.current_state = self.initial_state
        self.next_state = None
        #self.action = None
        self.memory = memory

    def reset (self):

        """This sets the current_state to the initial_state and sets
        input_symbol to None. The initial state was set by the constructor
        __init__(). """

        self.current_state = self.initial_state
        self.input_symbol = None

    def add_transition (self, input_symbol, state, action=None, next_state=None):

        """This adds a transition that associates:

                (input_symbol, current_state) --> (action, next_state)

        The action may be set to None in which case the process() method will
        ignore the action and only set the next_state. The next_state may be
        set to None in which case the current state will be unchanged.

        You can also set transitions for a list of symbols by using
        add_transition_list(). """

        if next_state is None:
            next_state = state
        self.state_transitions[(input_symbol, state)] = (action, next_state)

    def add_transition_list (self, list_input_symbols, state, action=None, next_state=None):

        """This adds the same transition for a list of input symbols.
        You can pass a list or a string. Note that it is handy to use
        string.digits, string.whitespace, string.letters, etc. to add
        transitions that match character classes.

        The action may be set to None in which case the process() method will
        ignore the action and only set the next_state. The next_state may be
        set to None in which case the current state will be unchanged. """

        if next_state is None:
            next_state = state
        for input_symbol in list_input_symbols:
            self.add_transition (input_symbol, state, action, next_state)

    def add_transition_any (self, state, action=None, next_state=None):

        """This adds a transition that associates:

                (current_state) --> (action, next_state)

        That is, any input symbol will match the current state.
        The process() method checks the "any" state associations after it first
        checks for an exact match of (input_symbol, current_state).

        The action may be set to None in which case the process() method will
        ignore the action and only set the next_state. The next_state may be
        set to None in which case the current state will be unchanged. """

        if next_state is None:
            next_state = state
        self.state_transitions_any [state] = (action, next_state)

    def set_default_transition (self, action, next_state):

        """This sets the default transition. This defines an action and
        next_state if the FSM cannot find the input symbol and the current
        state in the transition list and if the FSM cannot find the
        current_state in the transition_any list. This is useful as a final
        fall-through state for catching errors and undefined states.

        The default transition can be removed by setting the attribute
        default_transition to None. """

        self.default_transition = (action, next_state)

    def get_transition (self, input_symbol, state):

        """This returns (action, next state) given an input_symbol and state.
        This does not modify the FSM state, so calling this method has no side
        effects. Normally you do not call this method directly. It is called by
        process().

        The sequence of steps to check for a defined transition goes from the
        most specific to the least specific.

        1. Check state_transitions[] that match exactly the tuple,
            (input_symbol, state)

        2. Check state_transitions_any[] that match (state)
            In other words, match a specific state and ANY input_symbol.

        3. Check if the default_transition is defined.
            This catches any input_symbol and any state.
            This is a handler for errors, undefined states, or defaults.

        4. No transition was defined. If we get here then raise an exception.
        """

        if (input_symbol, state) in self.state_transitions:
            return self.state_transitions[(input_symbol, state)]
        elif state in self.state_transitions_any:
            return self.state_transitions_any[state]
        elif self.default_transition is not None:
            return self.default_transition
        else:
            raise ExceptionFSM ('Transition is undefined: (%s, %s).' %
                (str(input_symbol), str(state)) )

    def process (self, input_symbol):

        """This is the main method that you call to process input. This may
        cause the FSM to change state and call an action. This method calls
        get_transition() to find the action and next_state associated with the
        input_symbol and current_state. If the action is None then the action
        is not called and only the current state is changed. This method
        processes one complete input symbol. You can process a list of symbols
        (or a string) by calling process_list(). """

        self.input_symbol = input_symbol
        (action_fn, self.next_state) = self.get_transition (self.input_symbol, self.current_state)
        if action_fn is not None:
            ret = action_fn (self)
        else:
            ret = None
        self.current_state = self.next_state
        self.next_state = None
        return ret


    def process_list (self, input_symbols):

        """This takes a list and sends each element to process(). The list may
        be a string or any iterable object. """

        for s in input_symbols:
            self.process (s)


class ObjFSM(FSM):
    '''A subclass of FSM where input_symbol may be any kind of object, even an unhashable one.
    For each input_symbol to process, the machine will try a sequence of functions;
    the first to return True determines (action, next_state).
    '''

    def add_transition(self, test, state, action, next_state):
        self.state_transitions.setdefault(state, []).append((test, action, next_state))

    def get_transition(self, input_symbol, state):
        #input_symbol arg is not used, but we keep it for compatibility with FSM class
        for (test, action, next_state) in self.state_transitions.get(state, []):
            if test(self):
                return (action, next_state)
        try:
            return self.state_transitions_any[self.current_state]
        except KeyError:
            pass
        if self.default_transition != None:
            return self.default_transition
        raise ExceptionFSM('Transition is undefined: (%s, %s).' %
            (str(input_symbol), str(self.current_state)) )
