import re
import numpy as np
import copy


# Best regexp (alarm): (0|1)*111((0111(0|1)?)|(00111)|111(0|1)?(0|1)?)$
# Best regexp (certainly not): (0|1)*(000|010|100|001)$

def csv_to_transition_table(fname):
    raw = np.genfromtxt(fname, delimiter=",", dtype=int)
    states = raw[1:, 0]
    inputs = raw[0, 1:]

    table = {}
    for i in range(len(states)):
        table[states[i]] = {inputs[j]: raw[i+1, j+1]
                            for j in range(len(inputs))}
    return table


class DFA(object):
    def __init__(self, transition_table, acceptance_states, initial,
                 buffer_size=10000):
        # transition_table = nested dict with for each state the input and the
        # next state
        self.buffer_size = 1000
        self.order = len(transition_table)
        self.table = transition_table
        self._past_states = np.empty(buffer_size, dtype="int")
        self._past_states[0] = initial
        self.state_count = 0
        self.past_inputs = np.empty(buffer_size, dtype="int")
        self.acceptance_states = acceptance_states

    @property
    def state(self):
        return self._past_states[self.state_count]

    @property
    def options(self):  # Dict with inputs for the current state
        return self.table[self.state]

    def go(self, inputs):  # Transition to new state given (list of) input(s)
        try:  # Assume it's a list
            for inp in inputs:
                self._past_states[self.state_count+1] = self.options[int(inp)]
                self.past_inputs[self.state_count] = int(inp)
                self.state_count += 1
        except TypeError:  # Single input
            self._past_states[self.state_count+1] = self.options[inputs]
            self.past_inputs[self.state_count] = inputs
            self.state_count += 1

        return self.state

    @property
    def accepted(self):
        return self.state in self.acceptance_states

    def reset_to(self, state):
        self.__init__(self.table, self.acceptance_states, state,
                      buffer_size=self.buffer_size)

    @property
    def past_states(self):
        # Includes the currents state
        return self._past_states[:self.state_count+1]


# Merges multiple DFA objects together
class Moore(object):
    def __init__(self, output_map, dfas, keep_dfa=False):
        # Expects an output map and a list of DFA's
        if not keep_dfa:
            self.dfas = list(map(copy.deepcopy, dfas))
        else:
            self.dfas = dfas
        self.output_map = output_map

    @property
    def state(self):
        return [dfa.state for dfa in self.dfas]

    @property
    def output(self):
        return self.output_map(self.state)

    def go(self, inputs):
        for dfa in self.dfas:
            dfa.go(inputs)
        return self.output

    @property
    def past_states(self):
        return list(zip(*[dfa.past_states for dfa in self.dfas]))

    def reset_to(self, states):
        for i in range(states):
            self.dfas[i].reset_to(states[i])

    @property
    def past_outputs(self):
        return list(map(self.output_map, self.past_states))


def machine_output(state):
    alarm = state[0] in [12] + list(range(14, 21))
    safety = state[1] in [7, 8, 9]

    if not alarm and not safety:
        return "D"
    elif alarm and not safety:
        return "T"
    else:
        return "O"


table_alarm = csv_to_transition_table("dfa_alarm.csv")
table_no_alarm = csv_to_transition_table("dfa_no_alarm.csv")

dfa_alarm = DFA(table_alarm, [12] + list(range(14, 21)), 1)
dfa_no_alarm = DFA(table_no_alarm, [7, 8, 9], 1)

if __name__ == "__main__":
    # Generate all possible 8-bit sequences
    byte_seq = {bin(num)[2:] for num in range(256)}
    byte_seq = {(8 - len(num))*"0" + num for num in byte_seq}

    pattern = "(0|1)*111(0){0,2}111(0|1)*"

    byte_seq_certain = {num for num in byte_seq if re.search(pattern, num)}

    inp_string = "0010011011000111011110100"

    machine = Moore(machine_output, [dfa_alarm, dfa_no_alarm])

    # Produce simulation output for report
    input_range = []
    states_1 = [str(machine.state[0])]
    states_2 = [str(machine.state[1])]
    outputs = [machine.output]

    for char in inp_string:
        machine.go(char)
        input_range.append("{:>3s}".format(char))
        states_1.append("{:>3s}".format(str(machine.state[0])))
        states_2.append("{:>3s}".format(str(machine.state[1])))
        outputs.append("{:>3s}".format(machine.output))

    print(" " + "".join(input_range))
    print("".join(states_1))
    print("".join(states_2))
    print("".join(outputs))
