import re
from numpy import genfromtxt


# Best regexp (alarm): (0|1)*111((0111(0|1)?)|(00111)|111(0|1)?(0|1)?)$


def csv_to_transition_table(fname):
    raw = genfromtxt(fname, delimiter=",", dtype=int)
    states = raw[1:, 0]
    inputs = raw[0, 1:]

    table = {}
    for i in range(len(states)):
        table[states[i]] = {inputs[j]: raw[i+1, j+1]
                            for j in range(len(inputs))}
    return table


class DFA(object):
    def __init__(self, transition_table, acceptance_states, initial):
        # transition_table = nested dict with for each state the input and the
        # next state
        self.order = len(transition_table)
        self.table = transition_table
        self.past_states = [initial]
        self.past_inputs = []
        self.acceptance_states = acceptance_states

    @property
    def state(self):
        return self.past_states[-1]

    @property
    def options(self):  # Dict with inputs for the current state
        return self.table[self.state]

    def go(self, inputs):  # Transition to new state given (list of) input(s)
        try:  # Assume it's a list
            for inp in inputs:
                self.past_states.append(self.options[int(inp)])
                self.past_inputs.append(int(inp))
        except TypeError:  # Single input
            self.past_states.append(self.options[inputs])
            self.past_inputs.append(inputs)

        return self.state

    @property
    def accepted(self):
        return self.state in self.acceptance_states


# class Mealy(object):
#     def __init__(output_map, *args):


if __name__ == "__main__":
    # Generate all possible 8-bit sequences
    byte_seq = {bin(num)[2:] for num in range(256)}
    byte_seq = {(8 - len(num))*"0" + num for num in byte_seq}

    pattern = "(0|1)*111(0){0,2}111(0|1)*"

    byte_seq_certain = {num for num in byte_seq if re.search(pattern, num)}

    table = csv_to_transition_table("transition_dfa.csv")

    dfa_alarm = DFA(table, [12] + list(range(14, 21)), 1)
