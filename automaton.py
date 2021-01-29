import re
import numpy as np


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
                 bufsize=1000):
        # transition_table = nested dict with for each state the input and the
        # next state
        self.order = len(transition_table)
        self.table = transition_table
        self.past_states = np.empty(bufsize, dtype="int")
        self.past_states[0] = initial
        self.state_count = 0
        self.past_inputs = np.empty(bufsize, dtype="int")
        self.acceptance_states = acceptance_states

    @property
    def state(self):
        return self.past_states[self.state_count]

    @property
    def options(self):  # Dict with inputs for the current state
        return self.table[self.state]

    def go(self, inputs):  # Transition to new state given (list of) input(s)
        try:  # Assume it's a list
            for inp in inputs:
                self.past_states[self.state_count+1] = self.options[int(inp)]
                self.past_inputs[self.state_count] = int(inp)
        except TypeError:  # Single input
            self.past_states[self.state_count+1] = self.options[inputs]
            self.past_inputs[self.state_count] = inputs
        self.state_count += 1

        return self.state

    @property
    def accepted(self):
        return self.state in self.acceptance_states


# Merges multiple DFA objects together
class Moore(object):
    def __init__(self, output_map, dfas):
        # Expects an output map and a list of DFA's
        self.dfas = dfas
        self.output_map = output_map

    @property
    def state(self):
        return [dfa.state for dfa in self.dfas]

    @property
    def output(self):
        return self.output_map(self)

    def go(self, inputs):
        for dfa in self.dfas:
            dfa.go(inputs)
        return self.output

    @property
    def past_states(self):
        return list(zip(*[dfa.past_states for dfa in self.dfas]))


if __name__ == "__main__":
    # Generate all possible 8-bit sequences
    byte_seq = {bin(num)[2:] for num in range(256)}
    byte_seq = {(8 - len(num))*"0" + num for num in byte_seq}

    pattern = "(0|1)*111(0){0,2}111(0|1)*"

    byte_seq_certain = {num for num in byte_seq if re.search(pattern, num)}

    table_alarm = csv_to_transition_table("dfa_alarm.csv")
    table_no_alarm = csv_to_transition_table("dfa_no_alarm.csv")

    def machine_output(machine):
        if (not machine.dfas[0].accepted) and (not machine.dfas[1].accepted):
            return "D"
        elif (machine.dfas[0].accepted) and (not machine.dfas[1].accepted):
            return "T"
        else:
            return "O"

    dfa_alarm = DFA(table_alarm, [12] + list(range(14, 21)), 1)
    dfa_no_alarm = DFA(table_no_alarm, [7, 8, 9], 1)
    machine = Moore(machine_output, [dfa_alarm, dfa_no_alarm])

    inp_string = "0010011011000111011110100"

    inputs = []
    states_1 = [str(machine.state[0])]
    states_2 = [str(machine.state[1])]
    outputs = [machine.output]

    for char in inp_string:
        machine.go(char)
        inputs.append("{:>3s}".format(char))
        states_1.append("{:>3s}".format(str(machine.state[0])))
        states_2.append("{:>3s}".format(str(machine.state[1])))
        outputs.append("{:>3s}".format(machine.output))

    print(" " + "".join(inputs))
    print("".join(states_1))
    print("".join(states_2))
    print("".join(outputs))
