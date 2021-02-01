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
        self.voltage_breakpoint = [-2]  # This is a bit hacky tbh

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
        override_voltage = False
        for point in self.voltage_breakpoint:
            if point < t < point + 2:
                override_voltage = True

        if override_voltage:
            return 0
        else:
            return self.input_voltage_default(t)

    @staticmethod
    def generate_event_list(pos_fun):
        events = [lambda t, x: norm(pos_fun[0](x[1:]) - pos_fun[1](x[1:]))
                  - 2 * par.ball_radius,
                  lambda t, x: pos_fun[0](x[1:])[1] - par.ball_radius
                  - par.ground_height,
                  lambda t, x: pos_fun[1](x[1:])[1] - par.ball_radius
                  - par.ground_height,
                  lambda t, x: pos_fun[2](x[1:])[1]  - par.ball_radius
                   - par.ground_height]

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
                        (self.start_time, self.stop_time + self.dt),
                        self.start_state,
                        method="RK45",
                        t_eval=np.arange(t_start_output, self.stop_time,
                                         self.dt),
                        events=self.events,
                        dense_output=True)
        return sol

    def simulate(self):
        finished = False
        while not finished:
            sol = self.simulate_once()
            self.output[:, self.start_idx:(self.start_idx + sol.y.shape[-1])] \
                = sol.y

            alarm = self.check_output(sol.sol)

            if alarm[0] and sol.t[-1] > (self.voltage_breakpoint[-1] + 2):
                print(f"Alarm raised at {alarm[1]}")
                self.start_state = alarm[2]
                self.start_time = alarm[1]
                self.voltage_breakpoint.append(alarm[1])

                # Append appropriate output in t_out
                # Compute the index in the global t_out that is closest
                # approximation (from below) to t_alarm
                alarm_global_idx = np.where(self.t_out <= alarm[1])[0][-1]
                self.output[:, self.start_idx:alarm_global_idx] = \
                    sol.y[:, :(alarm_global_idx - self.start_idx)]
                self.start_idx = alarm_global_idx

            elif sol.status == 1:  # Termination event has occurred
                print("Collision occurred!")
                event_idx = self.find_event(sol)
                if event_idx == 0:
                    flag = 2
                else:
                    flag = 1

                self.reset_after_collision(sol.y_events[event_idx].squeeze(),
                                           flag)
                self.start_time = sol.t_events[event_idx][-1]

                self.start_idx += sol.y.shape[-1]

            elif sol.status == 0:  # Simulation done
                finished = True


    def reset_after_collision(self, state, flag):
        self.start_state = state

        if flag == 1:  # Collision with ground
            self.start_state[-1] *= -par.restitution_coeff
            self.start_state[-2] *= -par.restitution_coeff  #** 2
        if flag == 2:  # Collision between balls
            self.start_state[-1] *= -par.restitution_coeff
            # self.start_state[-2] *= -par.restitution_coeff ** 2

    def check_output(self, dense_solution):
        # Check output for alarm states
        # Sample output at correct rate
        t_sampled = np.arange(dense_solution.t_min, dense_solution.t_max,
                              1./par.baud)
        y_sampled = dense_solution(t_sampled)
        n_samples = t_sampled.size

        # Feed sensor data in Moore machines
        self.machine_right.go(self.emulate_sensor(y_sampled[-1, :]
                                                  + y_sampled[-2, :],
                                                  par.sensor_threshold))
        self.machine_left.go(self.emulate_sensor(y_sampled[-1, :]
                                                 - y_sampled[-2, :],
                                                 par.sensor_threshold))

        # Check for alarm states
        alarm_idx = []
        for machine in (self.machine_right, self.machine_right):
            try:
                alarm_idx.append(machine.past_outputs[-n_samples:].index("T"))
            except ValueError:
                pass

        if len(alarm_idx) >= 1:  # Alarm!
            alarm_idx = min(alarm_idx)
            t_alarm = t_sampled[alarm_idx]
            state_alarm = y_sampled[:, alarm_idx]
            return True, t_alarm, state_alarm

        else:  # No alarm
            return False, 0, 0

    def emulate_sensor(self, y_in, threshold):
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
