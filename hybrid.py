import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.stats import geom

import parameters as par
import automaton


class HybridSimulation(object):
    def __init__(self, f_full, input_voltage, brake_force, x0, t_out, pos_fun):
        self.machine_left = automaton.Moore(automaton.machine_output,
                                            (automaton.dfa_alarm,
                                             automaton.dfa_no_alarm))
        self.machine_right = automaton.Moore(automaton.machine_output,
                                            (automaton.dfa_alarm,
                                             automaton.dfa_no_alarm))

        self.f = f_full
        self.input_voltage_default = input_voltage
        self.brake_force = brake_force
        self.override_voltage_until = 0

        self.output = np.zeros((x0.shape[0], t_out.shape[-1]))
        self.t_out = t_out
        self.dt = t_out[1] - t_out[0]
        self.pos_fun = pos_fun

        self.events = self.generate_event_list(pos_fun)
        self.start_state = x0
        self.start_time = 0
        self.start_idx = 0
        self.stop_time = np.max(self.t_out)

    def input_voltage_extended(self, t):
        if t < self.override_voltage_until:
            return 0
        else:
            return self.input_voltage_default(t)

    @staticmethod
    def generate_event_list(pos_fun):
        events = []
        # for i in range(len(pos_fun)):
        #     for j in range(len(pos_fun)):
        #         if j > i:
        #             events.append(lambda t, x:
        #                           norm(np.array(pos_fun[i](x[1:]))
        #                                - np.array(pos_fun[j](x[1:])))
        #                           - 2*par.ball_radius)

        events.append(lambda t, x: norm(pos_fun[0](x[1:]) - pos_fun[1](x[1:]))
                                   -2*par.ball_radius)

        events.append(lambda t, x: pos_fun[0](x[1:])[1] - par.ball_radius - par.ground_height)
        events.append(lambda t, x: pos_fun[1](x[1:])[1] - par.ball_radius - par.ground_height)
        events.append(lambda t, x: pos_fun[2](x[1:])[1] - par.ball_radius - par.ground_height)

        for event in events:
            event.terminal = True  # Stop simulation when event is reached
            event.direction = -1  # Only check crossing from + to -

        return events

    @staticmethod
    def find_event(sol):
        for i in range(len(sol.t_events)):
            if sol.t_events[i].size != 0:
                break
        return i

    def simulate_once(self):
        t_start_output = self.t_out[np.where(self.t_out
                                             >= self.start_time)[0][0]]
        sol = solve_ivp(lambda t, x: self.f(t, x,
                                            self.input_voltage_extended,
                                            self.brake_force),
                        (self.start_time, self.stop_time),
                        self.start_state,
                        method="RK45",
                        t_eval=np.arange(t_start_output, self.stop_time,
                                         self.dt),
                        events=self.events)

        return sol

    def simulate(self):
        finished = False
        while not finished:
            sol = self.simulate_once()
            self.output[:, self.start_idx:(self.start_idx + sol.y.shape[-1])] \
                = sol.y
            self.start_idx += sol.y.shape[-1]

            if sol.status == 1:  # Termination event has occurred
                print("Collision occured!")
                event_idx = self.find_event(sol)
                if event_idx == 0:
                    flag = 2
                else:
                    flag = 1

                self.reset_after_collision(sol.y_events[event_idx].squeeze(),
                                           flag)
                self.start_time = sol.t_events[event_idx][-1]
            elif sol.status == 0:
                finished = True


    def reset_after_collision(self, state, flag):
        self.start_state = state

        if flag == 1:  # Collision with ground
            self.start_state[-1] *= -par.restitution_coeff
            self.start_state[-2] *= -par.restitution_coeff ** 2
        if flag == 2:  # Collision between balls
            self.start_state[-1] *= -par.restitution_coeff
            # self.start_state[-2] *= -par.restitution_coeff ** 2


def simulate_hybrid(f_full, input_voltage, brake_force, x0, t_out, pos_fun):
    # Initialize Moore machines for both upper links
    machine_left = automaton.Moore(automaton.machine_output,
                                   (automaton.dfa_alarm,
                                    automaton.dfa_no_alarm))
    machine_right = automaton.Moore(automaton.machine_output,
                                    (automaton.dfa_alarm,
                                     automaton.dfa_no_alarm))
    override_voltage_until = 0

    # Initialize integrator
    def input_voltage_extended(t):
        if t < override_voltage_until:
            return 0
        else:
            return input_voltage(t)

    integrator = \
        RK45(lambda t, x: f_full(t, x, input_voltage_extended, brake_force),
             t_out[0], x0, t_out[-1])

    y = np.zeros((x0.size, t_out.size))

    idx_start = 0  # Start of integration step

    def check_for_events(states):
        # Check for collisions
        collision = check_collision(states[1:, :], pos_fun)

        # Check for alarm states
        machine_left.go(emulate_sensor(states[-1, :] + states[-2, :],
                                       par.sensor_threshold))
        machine_right.go(emulate_sensor(states[-1, :] - states[-2, :],
                                        par.sensor_threshold))

        try:
            alarm = machine_left.past_outputs[:-len(t_step)].index("T")
        except ValueError:
            alarm = None
        try:
            tmp = machine_right.past_outputs[:-len(t_step)].index("T")
            if alarm is not None:
                if alarm > tmp:
                    alarm = tmp
            else:
                alarm = tmp
        except ValueError:
            pass

        # Handle all possibilities appropriately
        if collision is None and alarm is None:
            return 0, None
        elif collision is not None and alarm is None:
            return 1, collision
        elif collision is None and alarm is not None:
            return 2, alarm
        else:
            if alarm > collision:
                return 1, alarm
            else:
                return 0, collision

    # Integration process
    while integrator.status == "running":
        integrator.step()
        time_fcn = integrator.dense_output()
        t_step = np.arange(time_fcn.t_min, time_fcn.t_max, 1./par.baud)
        states_step = time_fcn(t_step)
        idx_end = np.where(t_out <= time_fcn.t_max)[0][-1]  # End of step

        event_flag, event_idx = check_for_events(states_step)

        if event_flag == 1:  # Collision
            if len(t_step) == 1:  # Time step is too short for baud rate
                t_collision = time_fcn.t_max
                reset_state = reset(states_step[:, time_fcn(time_fcn.t_max)],
                                    flag=1)
            else:
                t_collision = t_step[event_idx]
                reset_state = reset(states_step[:, event_idx], flag=1)

            idx_end = np.where(t_out <= t_collision)[0][-1]  # in global time vector

            integrator = \
                RK45(lambda t, x: f_full(t, x, input_voltage, brake_force),
                     t_collision, reset_state, t_out[-1])

        elif event_flag == 2:  # Alarm
            if len(t_step) == 1:  # Time step is too short for baud rate
                t_alarm = time_fcn.t_max
                reset_state = states_step[:, event_idx]
            else:
                t_alarm = t_step[event_idx]
                reset_state = time_fcn(time_fcn.t_max)

            idx_end = np.where(t_out <= t_alarm)[0][-1] # in global time vector
            print("Alarm state detected")
            override_voltage_until = t_alarm + 2

            integrator = \
                RK45(lambda t, x: f_full(t, x, input_voltage_extended,
                                         brake_force),
                     t_alarm, reset_state, t_out[-1])

        y[:, idx_start:(idx_end + 1)] = \
            states_step[:, :(idx_end - idx_start + 1)]  # Set output states
        idx_start = idx_end + 1  # Reset start index for next step

    return y

def emulate_sensor(y_in, threshold):
    # Expect a Boolean array
    finished = False
    p = 0.2
    i = 0
    y_out = np.abs(y_in) > threshold

    while not finished:
        i += geom.rvs(p) + 7
        if i < len(y_in):
            y_out[i] = ~y_out[i]
        else:
            finished = True

    return y_out.astype(np.short)


# if __name__ == "__main__":
     # HybridSimulation()

